#!/usr/bin/env python3
"""D6 technical checks only. Reuse E2/loss/optimizer/train_step; no AU-PRO or selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gc
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
import yaml
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.adapter_factory import AdapterCandidate, AdapterFactoryConfig, ResidualAdapterFactory
from src.models.msila import E2, resolve_g2_config
from src.train.g1_e1 import module_sha256
from src.train.g2_e2 import FrozenE2Factory, trainable_modules
from src.train.optimizer import build_optimizer
from src.train.overfit16 import Overfit16Trainer
from src.train.screen_adapter import seed_everything
from src.train.screen_representation import read_yaml, save_json, sha256_file, sha256_json

PAIRS = tuple((r, d) for r in (64, 128, 256) for d in (256, 512, 768))
FIELDS = ("r", "d", "status", "shape", "params", "peak_vram_if_measured", "error",
          "gpu_status", "config_sha256", "decoder_params", "total_trainable_params", "gpu_batch_size")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolved_config(config):
    cfg = resolve_g2_config(config, root=ROOT)
    b = cfg["backbone"]
    require((b["name"], b["channels"], b["patch_size"], b["deepest_block"]) == ("dinov3_vitb16", 768, 16, 12),
            "D6 preflight requires ViT-B/16 width 768")
    require(cfg["adapter"]["gamma_init"] == 0, "E2 requires gamma_init=0")
    require(cfg["loss"]["type"] == "bce_plus_positive_mask_dice", "Use the existing E2 loss")
    require(cfg["data"]["tile_size"] == 512 and cfg["decoder"]["output_size"] == [512, 512], "G2 requires 512x512")
    require(cfg["training"]["optimizer"]["name"] == "AdamW" and cfg["training"]["amp"] is False,
            "Preflight requires existing FP32 AdamW configuration")
    cfg["decoder"]["deterministic_resize"] = True
    return cfg


def pair_config(cfg, pair):
    AdapterCandidate(*pair)
    candidate = deepcopy(cfg)
    candidate["adapter"].update(r=pair[0], d=pair[1])
    candidate.update(selection_status="pending_joint_selection_D10", adapter_pair_role="technical_candidate")
    require(sha256_json(candidate) == sha256_json(yaml.safe_load(yaml.safe_dump(candidate))), "Config round-trip drift")
    return candidate


class MockDeepestFeatures(nn.Module):
    """Small CPU input fixture only, not a replacement DINO implementation."""
    out_channels, depth, blocks = 768, 12, (12,)

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("features", torch.randn(1, 768, 4, 4))

    def forward(self, image):
        return {"b12": (self.features * self.scale).expand(image.shape[0], -1, -1, -1)}


def mock_e2(cfg):
    a = cfg["adapter"]
    return E2(MockDeepestFeatures(), adapter_bottleneck_dim=a["r"], adapter_projection_dim=a["d"],
              adapter_kernel_size=a["kernel_size"], gamma_init=a["gamma_init"], adapter_bias=a["bias"],
              hidden_channels=cfg["decoder"]["hidden_channels"], deterministic_resize=True)


def fixture_batch(size, device):
    # Normalized random RGB/rectangles exercise the model/loss, not performance.
    generator = torch.Generator(device=device).manual_seed(17017)
    image = torch.randn(size, 3, 512, 512, device=device, generator=generator)
    mask = torch.zeros(size, 1, 512, 512, device=device)
    mask[-1, :, 128:256, 160:288] = 1
    return dict(image=image, mask=mask, meta=[{"role": "technical_fixture"} for _ in range(size)])


def check_model(model, cfg, batch, *, device, audit_weights=True):
    """Call the existing train_step twice: zero gate, then active Conv gradients."""
    model.to(device).train()
    require(set(model._modules) == {"extractor", "adapter", "decoder"}, "Unexpected E2 component")
    require(not model.extractor.training, "Frozen extractor must be in eval mode")
    modules = trainable_modules(model, "E2")
    opt = cfg["training"]["optimizer"]
    optimizer, report = build_optimizer(dict(modules.items()), frozen_modules={"backbone": model.extractor},
                                        learning_rate=opt["lr"], weight_decay=opt["weight_decay"])
    ids = {id(p) for p in modules.parameters()}
    require({id(p) for g in optimizer.param_groups for p in g["params"]} == ids
            == {id(p) for p in model.parameters() if p.requires_grad}, "Optimizer must contain exactly Adapter + Decoder")
    before = module_sha256(model.extractor) if audit_weights else None
    decoder_before = module_sha256(model.decoder)
    a, r, d = model.adapter, cfg["adapter"]["r"], cfg["adapter"]["d"]
    require((a.down_proj.in_channels, a.down_proj.out_channels, a.dwconv.groups, a.mid_proj.in_channels,
             a.mid_proj.out_channels, a.out_proj.in_channels, a.out_proj.out_channels) == (768, r, r, r, d, d, 768),
            "r/d do not affect the actual architecture")
    params = sum(p.numel() for p in a.parameters() if p.requires_grad)
    require(params == a.expected_parameter_count(), "Adapter parameter-count drift")
    with torch.no_grad():
        logits, trace = model(batch, return_trace=True)
        feature = trace["dino"]["b12"]
        shape = list(trace["decoder_feature"].shape)
        require(shape == list(feature.shape) and shape[:2] == [batch["image"].shape[0], 768], "Output width/shape drift")
        require(list(logits.shape) == [batch["image"].shape[0], 1, 512, 512], "Unexpected logits shape")
        require(torch.equal(feature, trace["decoder_feature"]), "gamma=0 must be exact identity")
        # E1 feeds the same frozen feature directly into this same Decoder.
        require(torch.equal(logits, model.decoder(feature, output_size=(512, 512))), "Zero-gate E1/E2 mismatch")
    del logits, trace, feature
    criterion = AnomalySegmentationLoss(**{k: v for k, v in cfg["loss"].items() if k != "type"})
    trainer = Overfit16Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device,
                               frozen_modules={"backbone": model.extractor})
    logs = []
    for step in (1, 2):
        log = trainer.train_step(batch, step=step, epoch=0).to_dict()
        logs.append({k: log[k] for k in ("step", "loss", "bce", "dice_loss", "grad_norm")})
        require(all(math.isfinite(log[k]) for k in ("loss", "bce", "dice_loss", "grad_norm")), "Nonfinite loss/gradient norm")
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in modules.parameters()), "Nonfinite/missing gradient")
        require(a.gamma.grad.abs().item() > 0, "Gamma gradient must be nonzero")
        branch = [p for name, p in a.named_parameters() if name != "gamma"]
        if step == 1:
            require(all(torch.count_nonzero(p.grad).item() == 0 for p in branch), "Zero gate must block Conv gradients")
            require(a.gamma.detach().abs().item() > 0, "Gamma did not move")
        else:
            require(all(torch.count_nonzero(p.grad).item() > 0 for p in branch), "Every Adapter Conv must receive a gradient")
    require(all(not p.requires_grad and p.grad is None for p in model.extractor.parameters()), "Backbone received a gradient")
    require(not model.extractor.training, "Backbone left eval mode")
    if audit_weights:
        require(module_sha256(model.extractor) == before, "Frozen weights/buffers changed")
    require(module_sha256(model.decoder) != decoder_before, "Decoder did not update")
    require(all(torch.isfinite(p).all() for p in modules.parameters()), "Nonfinite updated parameter")
    return dict(shape=shape, logits_shape=[batch["image"].shape[0], 1, 512, 512], params=params,
                decoder_params=sum(p.numel() for p in model.decoder.parameters()),
                total_trainable_params=report.trainable_parameter_elements, decoder_initial_sha256=decoder_before,
                logs=logs, forward_backward_steps=2, finite_gradients=True,
                nonzero_branch_gradients_after_gate_update=True, frozen_backbone=True, exact_identity=True)


def invalid_checks(cfg):
    a = cfg["adapter"]
    factory = ResidualAdapterFactory(AdapterFactoryConfig(in_dim=768, kernel_size=a["kernel_size"],
                                                         gamma_init=a["gamma_init"], bias=a["bias"]))
    pairs = ((0, 256), (-1, 256), (64, 0), (64, -1), (True, 256), (64, False), (64.0, 256), (64, None))
    for r, d in pairs:
        try:
            factory.build_rd(r=r, d=d)
        except (ValueError, TypeError):
            continue
        raise ValueError(f"Invalid pair accepted: {(r, d)}")
    return dict(status="PASS", rejected_pairs=[list(pair) for pair in pairs])


def git_evidence():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    files = (Path(__file__), ROOT / "scripts/run_g2.py", ROOT / "src/models/msila.py", ROOT / "src/models/adapter_factory.py",
             ROOT / "src/models/residual_adapter.py", ROOT / "src/train/g2_e2.py", ROOT / "src/train/overfit16.py",
             ROOT / "src/train/optimizer.py", ROOT / "src/losses/anomaly_loss.py")
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                runtime_torch=str(torch.__version__), runtime_python=sys.version.split()[0],
                source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in files})


def save_record(record, path):
    record["record_sha256"] = sha256_json({k: v for k, v in record.items() if k != "record_sha256"})
    save_json(record, path)


def read_record(path):
    record = json.loads(Path(path).read_text())
    require(record.get("record_sha256") == sha256_json({k: v for k, v in record.items() if k != "record_sha256"}),
            f"Evidence checksum mismatch: {path}")
    return record


def write_report(root):
    root, rows = Path(root), []
    for r, d in PAIRS:
        cpu_path, gpu_path = root / "cpu" / f"r{r}_d{d}.json", root / "gpu" / f"r{r}_d{d}.json"
        try:
            cpu = read_record(cpu_path) if cpu_path.is_file() else {}
        except (OSError, ValueError) as exc:
            cpu = dict(status="FAIL", error=str(exc))
        try:
            gpu = read_record(gpu_path) if gpu_path.is_file() else {}
        except (OSError, ValueError) as exc:
            gpu = dict(status="FAIL", error=str(exc), cpu_evidence_sha256=cpu.get("evidence_sha256"))
        if not cpu or gpu.get("cpu_evidence_sha256") != cpu.get("evidence_sha256"):
            gpu = {}
        measurement = gpu.get("measurement", {}) if gpu.get("status") == "PASS" else {}
        rows.append(dict(r=r, d=d, status=cpu.get("status", "NOT RUN"), shape=json.dumps(cpu.get("shape")) if cpu.get("shape") else "",
                         params=cpu.get("params", ""), peak_vram_if_measured=measurement.get("peak_reserved_bytes", ""),
                         error=cpu.get("error", "") or gpu.get("error", ""), gpu_status=gpu.get("status", "NOT RUN"), config_sha256=cpu.get("config_sha256", ""),
                         decoder_params=cpu.get("decoder_params", ""), total_trainable_params=cpu.get("total_trainable_params", ""),
                         gpu_batch_size=gpu.get("batch_size", "")))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "adapter_preflight.csv"
    with path.with_suffix(".csv.tmp").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(path.with_suffix(".csv.tmp"), path)
    summary = dict(status="PASS" if all(row["status"] == "PASS" for row in rows) else "INCOMPLETE",
                   cpu_pass=sum(row["status"] == "PASS" for row in rows), expected=9,
                   gpu_pass=sum(row["gpu_status"] == "PASS" for row in rows),
                   gpu_status="NOT RUN" if all(row["gpu_status"] == "NOT RUN" for row in rows) else "MEASURED",
                   peak_vram_unit="bytes (torch.cuda.max_memory_reserved)",
                   selection_status="pending_joint_selection_D10", performance_evaluation="NOT RUN")
    if any(row["status"] == "FAIL" for row in rows):
        summary["status"] = "FAIL"
    save_json(summary, root / "adapter_preflight_summary.json")
    return summary


def sync_outputs(root, drive_root):
    if drive_root is None:
        return
    source, target = Path(root).resolve(), Path(drive_root).resolve()
    require(source != target and source not in target.parents and target not in source.parents, "Local/Drive roots must be disjoint")
    # Copy only preflight evidence; do not touch historical training/checkpoints.
    names = {"run_manifest.json", "adapter_preflight.csv", "adapter_preflight_summary.json",
             "setup_manifest.json", "preflight.log", "colab.log"}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_file() and (str(relative) in names or relative.parts[0] in {"cpu", "gpu"}) and path.suffix in {".json", ".log" , ".csv"}:
            dst = target / path.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            temporary = dst.with_name(dst.name + ".copying")
            shutil.copy2(path, temporary); os.replace(temporary, dst)


def run_cpu(config, root, *, drive_root=None):
    cfg, root = resolved_config(config), Path(root)
    source, invalid = git_evidence(), invalid_checks(cfg)
    save_json(dict(version="g2_d6_preflight_v1", scope="technical_only", selection_status="pending_joint_selection_D10",
                   seed=cfg["training"]["seed"], config=cfg, config_sha256=sha256_json(cfg), provenance=source,
                   cpu_fixture="mock features [B,768,4,4]; real E2/Adapter/Decoder/loss/train_step", gpu_status="NOT RUN",
                   invalid_pairs=invalid), root / "run_manifest.json")
    old_threads, old_rng = torch.get_num_threads(), torch.get_rng_state()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        for pair in PAIRS:
            candidate = pair_config(cfg, pair)
            record = dict(r=pair[0], d=pair[1], config=candidate, config_sha256=sha256_json(candidate),
                          evidence_sha256=sha256_json(dict(config=candidate, provenance=source)), device="cpu")
            try:
                seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
                record.update(check_model(mock_e2(candidate), candidate, fixture_batch(2, "cpu"), device="cpu"), status="PASS", error="")
            except (ValueError, RuntimeError, TypeError) as exc:
                record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            save_record(record, root / "cpu" / f"r{pair[0]}_d{pair[1]}.json")
            write_report(root); sync_outputs(root, drive_root)
            logging.info("CPU r=%d d=%d %s", *pair, record["status"])
    finally:
        torch.set_num_threads(old_threads); torch.set_rng_state(old_rng)
        torch.use_deterministic_algorithms(old_deterministic)
    records = [read_record(root / "cpu" / f"r{r}_d{d}.json") for r, d in PAIRS]
    if all(row["status"] == "PASS" for row in records):
        require(len({row["params"] for row in records}) == 9, "Duplicate r/d capacity")
        require(len({row["config_sha256"] for row in records}) == 9, "r/d must change config hashes")
        require(len({row["decoder_initial_sha256"] for row in records}) == 1, "Decoder initialization drift")
    result = write_report(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    manifest["gpu_status"] = result["gpu_status"]
    save_json(manifest, root / "run_manifest.json")
    sync_outputs(root, drive_root)
    return result


def batch_candidates(maximum):
    require(type(maximum) is int and maximum >= 1, "max_batch must be a positive integer")
    return sorted({maximum, *(2 ** i for i in range(maximum.bit_length()))}, reverse=True)


def vram_batch_cap(total_bytes, maximum):
    gib = total_bytes / 2**30
    return min(16 if gib < 18 else 32 if gib < 32 else 64 if gib < 64 else 128, maximum)


class BatchCapacityError(RuntimeError):
    def __init__(self, trials):
        super().__init__(f"No batch fits the GPU memory reserve: {trials}")
        self.trials = trials


def choose_batch(candidates, probe, *, memory_limit):
    trials = []
    for batch in candidates:
        try:
            measurement = probe(batch)
            fits = measurement["peak_reserved_bytes"] <= memory_limit
            trials.append(dict(batch_size=batch, status="FIT" if fits else "HEADROOM", **measurement))
            if fits:
                return batch, trials
        except torch.cuda.OutOfMemoryError as exc:
            trials.append(dict(batch_size=batch, status="OOM", error=str(exc)))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise BatchCapacityError(trials)


def requested_pairs(requested):
    if requested == ["all"]:
        return list(PAIRS)
    try:
        pairs = [tuple(map(int, item.split(":"))) for item in requested]
    except (TypeError, ValueError) as exc:
        raise ValueError("--pairs must be all or R:D") from exc
    require(bool(pairs) and len(set(pairs)) == len(pairs) and all(pair in PAIRS for pair in pairs), "Choose distinct grid pairs")
    return [pair for pair in PAIRS if pair in pairs]


def probe_gpu(factory, cfg, size, device):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model = batch = None
    try:
        seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
        model, batch = factory(cfg).to(device), fixture_batch(size, device)
        result = check_model(model, cfg, batch, device=device, audit_weights=False)
        require(result["shape"] == [size, 768, 32, 32], "Real DINO feature grid must be 32x32")
        torch.cuda.synchronize(device)
        result.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        return result
    finally:
        del model, batch
        gc.collect(); torch.cuda.empty_cache()


def run_gpu(config, root, *, pairs, max_batch=128, memory_fraction=.8, device="cuda", resume=False, drive_root=None):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise FileNotFoundError("GPU NOT RUN: select Colab GPU; no pretrained model loaded")
    require(0 < memory_fraction < 1, "memory_fraction must be between 0 and 1")
    require(bool(pairs) and len(set(pairs)) == len(pairs) and all(pair in PAIRS for pair in pairs), "Choose distinct grid pairs")
    batch_candidates(max_batch)
    cfg, root = resolved_config(config), Path(root)
    source = git_evidence()
    manifest = json.loads((root / "run_manifest.json").read_text())
    require(manifest["config_sha256"] == sha256_json(cfg) and manifest["provenance"] == source, "CPU config/source drift: rerun CPU")
    require(write_report(root)["cpu_pass"] == 9, "GPU smoke requires 9/9 CPU PASS")
    require(Path(cfg["backbone"]["weights"]).is_file(), "Real checkpoint is missing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    factory = FrozenE2Factory(cfg, device=device)
    require(factory.extractor.backbone.__class__.__module__.startswith("dinov3."), "GPU acceptance requires real DINOv3")
    before = module_sha256(factory.extractor)
    free, total = torch.cuda.mem_get_info(device)
    memory_limit = min(int(total * memory_fraction), int(torch.cuda.memory_reserved(device) + free - total * (1-memory_fraction)))
    identity = dict(checkpoint_sha256=sha256_file(cfg["backbone"]["weights"]), device=str(device),
                    gpu_name=torch.cuda.get_device_name(device), total_vram_bytes=total, max_batch=max_batch,
                    memory_fraction=memory_fraction, runtime_torch=str(torch.__version__), runtime_cuda=torch.version.cuda,
                    precision=dict(amp=False, tf32=False), provenance=source)
    repo = Path(cfg["backbone"]["repo_dir"])
    identity["dino_source_sha256"] = {str(path.relative_to(repo)): sha256_file(path) for path in sorted(repo.rglob("*.py"))}
    manifest.update(gpu_status="RUNNING", gpu=identity)
    save_json(manifest, root / "run_manifest.json")
    passed = 0
    for pair in pairs:
        cpu = read_record(root / "cpu" / f"r{pair[0]}_d{pair[1]}.json")
        path = root / "gpu" / f"r{pair[0]}_d{pair[1]}.json"
        signature = sha256_json(dict(cpu=cpu["evidence_sha256"], gpu=identity))
        if path.is_file():
            old = read_record(path)
            require(old.get("signature") == signature, "GPU resume drift: use a new output root")
            if resume and old.get("status") == "PASS":
                passed += 1
                continue
        candidate = pair_config(cfg, pair)
        record = dict(r=pair[0], d=pair[1], signature=signature, cpu_evidence_sha256=cpu["evidence_sha256"], gpu=identity)
        try:
            size, trials = choose_batch(batch_candidates(vram_batch_cap(total, max_batch)),
                                        lambda n: probe_gpu(factory, candidate, n, device), memory_limit=memory_limit)
            require(module_sha256(factory.extractor) == before, "Frozen DINO state changed")
            record.update(status="PASS", batch_size=size, trials=trials, measurement=trials[-1], error="")
            passed += 1
        except BatchCapacityError as exc:
            status = "OOM" if all(trial["status"] == "OOM" for trial in exc.trials) else "NO_SAFE_BATCH"
            record.update(status=status, trials=exc.trials, error=str(exc))
        except (ValueError, RuntimeError) as exc:
            record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        save_record(record, path); write_report(root); sync_outputs(root, drive_root)
        logging.info("GPU r=%d d=%d %s", *pair, record["status"])
    manifest["gpu_status"] = "MEASURED"
    save_json(manifest, root / "run_manifest.json")
    result = write_report(root)
    result.update(gpu_requested_pass=passed, gpu_requested=len(pairs))
    sync_outputs(root, drive_root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("cpu", "preflight", "gpu", "report"), default="cpu")
    parser.add_argument("--model-config", default="configs/g2_experiments.yaml")
    parser.add_argument("--output-root", default="outputs/G2/D6")
    parser.add_argument("--drive-root")
    parser.add_argument("--pairs", nargs="+", default=["256:768"], help="GPU subset; CPU always checks all nine")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--memory-fraction", type=float, default=.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(root / "preflight.log")],
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.action in {"cpu", "preflight"}:
        result = run_cpu(read_yaml(args.model_config), root, drive_root=args.drive_root)
    elif args.action == "gpu":
        result = run_gpu(read_yaml(args.model_config), root, pairs=requested_pairs(args.pairs), max_batch=args.max_batch,
                         memory_fraction=args.memory_fraction, device=args.device, resume=args.resume, drive_root=args.drive_root)
    else:
        result = write_report(root)
    sync_outputs(root, args.drive_root)
    print(json.dumps(result, indent=2))
    return 0 if result["cpu_pass"] == 9 and result.get("gpu_requested_pass", 0) == result.get("gpu_requested", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
