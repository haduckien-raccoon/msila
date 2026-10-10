#!/usr/bin/env python3
"""Colab orchestration for D6 only; training and selection belong to run_g2.

GPU capacity probes are disposable TRAIN steps, never screening evidence.
The chosen batch is locked before smoke/full runs and cannot change on resume.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gc
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from scripts import run_g2 as runner
from src.data.loader import G1NativeDataset, G1TileDataset, g1_tile_collate
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.contracts import validate_g2_batch
from src.train.g1_e1 import discover_sources, module_sha256
from src.train.g2_e2 import FrozenE2Factory, trainable_modules
from src.train.optimizer import build_optimizer
from src.train.overfit16 import Overfit16Trainer
from src.train.screen_adapter import seed_everything
from src.train.screen_representation import enforce_lock, read_yaml, save_json, save_yaml
from src.utils.resume import ResumeError, verify_checkpoint_sha256


def copy_atomic(source, target):
    source, target = Path(source), Path(target)
    if target.is_file() and (source.stat().st_size, source.stat().st_mtime_ns) == (
            target.stat().st_size, target.stat().st_mtime_ns):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def copy_tree(source, target):
    source, target = Path(source), Path(target)
    if source.exists():
        for path in sorted(source.rglob("*")):
            if path.is_file() and not path.name.endswith((".tmp", ".copying")):
                if '.pt.previous' in path.name or path.name.endswith('.pt.sha256'):
                    continue
                destination = target / path.relative_to(source)
                if path.suffix == '.pt':
                    copy_checkpoint(path, destination)
                else:
                    copy_atomic(path, destination)


def checkpoint_digest(path):
    try:
        verify_checkpoint_sha256(path, require_sidecar=True)
        return Path(str(path) + '.sha256').read_text().split()[0].lower()
    except (OSError, ResumeError):
        return None


def write_checkpoint_sidecar(path, digest):
    path = Path(path)
    sidecar = Path(str(path) + '.sha256')
    temporary = sidecar.with_name(sidecar.name + '.tmp')
    temporary.write_text(f'{digest}  {path.name}\n')
    os.replace(temporary, sidecar)


def copy_checkpoint(source, target):
    """Retain the previous verified generation if runtime dies between renames."""
    source, target = Path(source), Path(target)
    digest = checkpoint_digest(source)
    if digest is None:
        previous = Path(str(source) + '.previous')
        digest = checkpoint_digest(previous)
        if digest is None:
            raise ValueError(f'BLOCKED: checkpoint and backup checksums invalid: {source}')
        logging.warning('Recovering verified previous checkpoint generation: %s', source)
        source = previous
    target_digest = checkpoint_digest(target)
    if target_digest == digest:
        return
    if target_digest is not None:
        previous = Path(str(target) + '.previous')
        copy_atomic(target, previous)
        write_checkpoint_sidecar(previous, target_digest)
    copy_atomic(source, target)
    write_checkpoint_sidecar(target, digest)
    if checkpoint_digest(target) != digest:
        raise OSError(f'Checkpoint Drive copy verification failed: {target}')


def batch_candidates(maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_batch must be a positive integer")
    return sorted({maximum, *(2 ** i for i in range(maximum.bit_length()))}, reverse=True)


def vram_batch_cap(total_bytes, requested_maximum):
    gib = total_bytes / 2**30
    cap = 16 if gib < 18 else 32 if gib < 32 else 64 if gib < 64 else 128
    return min(cap, requested_maximum)


def choose_batch(candidates, probe, *, memory_limit):
    """Only genuine CUDA OOMs trigger backoff; scientific/runtime errors fail."""
    trials = []
    for batch in candidates:
        try:
            measurement = probe(batch)
            fits = measurement["peak_reserved_bytes"] <= memory_limit
            trials.append(dict(batch_size=batch, status="FIT" if fits else "HEADROOM", **measurement))
            if fits:
                return batch, trials
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch_size=batch, status="OOM"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise runner.G2Blocked(f"BLOCKED: no batch fits the GPU reserve; trials={trials}")


def probe_training_batch(factory, cfg, tiles, batch_size, device):
    """Use the actual E2/loss/optimizer/train_step, including the second gate step."""
    seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = factory(cfg).to(device)
    modules = trainable_modules(model, "E2")
    optimizer, _ = build_optimizer(dict(modules.items()), frozen_modules={"backbone": model.extractor},
                                   learning_rate=cfg["training"]["optimizer"]["lr"],
                                   weight_decay=cfg["training"]["optimizer"]["weight_decay"])
    loss_config = {k: v for k, v in cfg["loss"].items() if k != "type"}
    trainer = Overfit16Trainer(model=model, criterion=AnomalySegmentationLoss(**loss_config),
                               optimizer=optimizer, device=device, frozen_modules={"backbone": model.extractor})
    # Repetition is solely a capacity probe; full runs use the existing loader unchanged.
    batch = g1_tile_collate([tiles[index % len(tiles)] for index in range(batch_size)])
    validate_g2_batch(batch)
    logs = [trainer.train_step(batch, step=i + 1, epoch=0).to_dict() for i in range(2)]
    if not all(torch.isfinite(torch.tensor(log["loss"])) for log in logs):
        raise ValueError("Preflight loss is nonfinite")
    if (model.extractor.training or model.extractor.backbone.training
            or any(p.requires_grad or p.grad is not None for p in model.extractor.parameters())):
        raise ValueError("Preflight frozen backbone contract failed")
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in modules.parameters()):
        raise ValueError("Preflight trainable gradient contract failed")
    torch.cuda.synchronize(device)
    result = dict(peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  forward_backward_steps=2, finite_loss=True, frozen_backbone=True)
    del trainer, optimizer, modules, model, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def requested_pairs(study, requested):
    if requested == ["all"]:
        return study["pairs"]
    try:
        pairs = [tuple(int(v) for v in value.split(":")) for value in requested]
    except ValueError as exc:
        raise ValueError("pairs must be all or R:D, e.g. 64:256 128:512") from exc
    if not pairs or len(set(pairs)) != len(pairs) or any(pair not in study["pairs"] for pair in pairs):
        raise ValueError("Select distinct pairs from the declared ViT-B 3x3 grid")
    return [pair for pair in study["pairs"] if pair in pairs]


def preflight_plan(study):
    """Existing source discovery checks TRAIN/DEV leakage before truncation."""
    rows = []
    for category in runner.CATEGORIES:
        context, _ = runner.make_context(study, category, study["pairs"][-1], "adapter_screen")
        cfg = context["config"]
        rows.append(dict(category=category, train_sources=len(cfg["sources"]["train"]),
                         dev_sources=len(cfg["sources"]["dev"]), expected_steps=context["expected_steps"],
                         seed=cfg["training"]["seed"], dev_seed=cfg["training"]["dev_seed"],
                         training_batch_size=cfg["training"]["batch_size"],
                         epochs=cfg["training"]["epochs"]))
    return rows


def export_results(study):
    """Validate every artifact via TV1 APIs; missing measurements stay null."""
    valid, missing = runner.collect_results(study, "adapter_screen")
    indexed = {(row["category"], row["adapter"]["r"], row["adapter"]["d"]): row for row in valid}
    table, candidates = [], []
    for r, d in study["pairs"]:
        group = []
        for category in runner.CATEGORIES:
            context, _ = runner.make_context(study, category, (r, d), "adapter_screen")
            directory = runner.run_directory(study, category, (r, d), "adapter_screen")
            result = indexed.get((category, r, d))
            status = "PASS" if result else "NOT RUN"
            if result is None and (directory / "metrics.json").exists():
                status = "INVALID_OR_INCOMPLETE"
            elif result is None and (directory / "last.pt").exists():
                status = "INCOMPLETE"
            row = dict(category=category, r=r, d=d, status=status,
                       synthetic_dev_aupro_0_05=None if result is None else result["best_synthetic_dev"]["synthetic_dev_aupro_0_05"],
                       seed=context["config"]["training"]["seed"],
                       dev_seed=context["config"]["training"]["dev_seed"],
                       expected_steps=context["expected_steps"], config_sha256=context["config_sha256"],
                       metrics_path=str(directory / "metrics.json"), checkpoint_path=str(directory / "best.pt"))
            table.append(row)
            if result:
                group.append(result)
        summary = dict(r=r, d=d, valid_categories=len(group), expected_categories=8,
                       macro_synthetic_dev_aupro_0_05=None, adapter_trainable_parameters=None)
        if len(group) == 8:
            # Delegate even the macro/tie-break calculation to the existing selector.
            summary.update(runner.rank_candidates({**study, "pairs": [(r, d)]}, group)[0])
            summary.pop("category_scores")
        candidates.append(summary)
    lock = runner.publish_selection(study)
    if lock is not None:
        runner.validate_selection(study)
    directory = study["root"] / "full" / "colab_report"
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("screening_72", table), ("macro_9", candidates)):
        save_json(rows, directory / f"{name}.json")
        with (directory / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = dict(status="PASS" if lock else ("NOT RUN" if not valid else "INCOMPLETE"),
                   valid_runs=len(valid), expected_runs=72, missing=missing, study_sha256=study["sha256"],
                   metric="synthetic_dev_aupro_0_05", selection_rule=runner.SELECTION_RULE,
                   selected_pair=None if lock is None else lock["payload"]["selected_pair"],
                   selection_lock=None if lock is None else str(study["root"] / "full" / "adapter_selection_lock.json"))
    save_json(summary, directory / "acceptance.json")
    plot_heatmap(candidates, directory / "rd_heatmap.png")
    return summary


def plot_heatmap(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rs, ds = sorted({row["r"] for row in rows}), sorted({row["d"] for row in rows})
    values = np.full((len(rs), len(ds)), np.nan)
    for row in rows:
        if row["macro_synthetic_dev_aupro_0_05"] is not None:
            values[rs.index(row["r"]), ds.index(row["d"])] = row["macro_synthetic_dev_aupro_0_05"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#eeeeee")
    mesh = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap="viridis")
    for row in rows:
        value = row["macro_synthetic_dev_aupro_0_05"]
        label = f'{row["valid_categories"]}/8' if value is None else f"{value:.4f}"
        ax.text(ds.index(row["d"]), rs.index(row["r"]), label, ha="center", va="center",
                color="black" if value is None or value > .6 else "white")
    ax.set(xticks=range(len(ds)), xticklabels=ds, yticks=range(len(rs)), yticklabels=rs,
           xlabel="d (projection width)", ylabel="r (bottleneck width)",
           title="ViT-B/16: macro synthetic DEV AU-PRO@0.05\nBlank cells: incomplete, no score")
    fig.colorbar(mesh, ax=ax, label="AU-PRO@0.05")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "smoke", "screen", "report"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--pairs", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="null: probe before the study; positive int: fixed")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--memory-fraction", type=float, default=.8)
    return parser.parse_args(argv)


def configure(args):
    if (args.local_root.resolve().is_relative_to(args.drive_root.resolve())
            or args.drive_root.resolve().is_relative_to(args.local_root.resolve())):
        raise ValueError("Train on local disk; Drive must be a separate backup root")
    if not 0 < args.memory_fraction < 1:
        raise ValueError("memory_fraction must be in (0,1)")
    batch_candidates(args.max_batch)
    if args.batch_size is not None:
        runner.positive_int(args.batch_size, "batch-size")
    cli = runner.parse_args(["--config", args.config, "--stage", "adapter_screen", "--device", args.device,
                             "--output-root", str(args.local_root), "--categories", *args.categories])
    cfg, config, root = runner.load_config(cli)
    if cfg["backbone"]["name"] != "dinov3_vitb16":
        raise ValueError("D6-TV1 Colab screening requires ViT-B/16")
    return cfg, config, root, cli.categories


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, config, root, categories = configure(args)
    if args.action != "report" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        raise FileNotFoundError("NOT RUN: Colab preflight/smoke/screen require a usable CUDA GPU")
    root = Path(root)
    drive = args.drive_root / cfg["backbone"]["name"]
    copy_tree(drive, root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dependencies = {name: version(name) for name in
                    ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'safetensors', 'PyYAML', 'matplotlib')}
    identity = runner.sha256_json(dict(config=cfg, runner=config, commit=commit,
                                      dependencies=dependencies,
                                      checkpoint_sha256=runner.sha256_file(cfg["backbone"]["weights"])))
    profile_path = root / "colab_batch_profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
    if profile is not None and (profile["input_sha256"] != identity
            or (args.batch_size is not None and args.batch_size != profile["batch_size"])):
        raise runner.G2Blocked("BLOCKED: study settings/batch/commit changed; choose a new RUN_ID")
    if profile is None and args.action != "preflight":
        raise runner.G2Blocked("BLOCKED: run preflight to lock the batch before smoke/screen/report")
    factory = None
    if args.action != "report":
        seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runner.check_assets(cfg, args.device)
        factory = FrozenE2Factory(cfg, device=args.device)
    if args.action == "preflight":
        category = categories[0]
        probe_cfg = deepcopy(cfg)
        probe_cfg["category"] = category
        probe_cfg["synthetic_protocol"] = runner.validate_native_protocol(read_yaml(runner.absolute(cfg["synthetic_protocol"])))
        probe_cfg["adapter"].update(r=256, d=768)
        pools, _ = discover_sources(probe_cfg)
        native = G1NativeDataset(pools["train"], probe_cfg["synthetic_protocol"], seed=cfg["training"]["seed"],
                                 role="train", variants=cfg["data"]["train_variants_per_image"], fixed=False)
        tiles = G1TileDataset(native, tile_size=512, overlap=cfg["data"]["overlap"])
        free, total = torch.cuda.mem_get_info(args.device)
        memory_limit = free + torch.cuda.memory_reserved(args.device) - int(total * (1 - args.memory_fraction))
        candidates = ([profile["batch_size"]] if profile else
                      [args.batch_size] if args.batch_size else
                      batch_candidates(vram_batch_cap(total, args.max_batch)))
        before = module_sha256(factory.extractor)
        batch, trials = choose_batch(candidates, lambda b: probe_training_batch(factory, probe_cfg, tiles, b, args.device),
                                     memory_limit=memory_limit)
        if module_sha256(factory.extractor) != before:
            raise ValueError("Preflight modified the frozen DINO state")
        if profile is None:
            profile = dict(version="g2_colab_batch_v1", input_sha256=identity, commit=commit,
                           batch_size=batch, evaluation_tile_batch_size=batch, trials=trials,
                           largest_pair={"r": 256, "d": 768}, memory_fraction=args.memory_fraction,
                           gpu=torch.cuda.get_device_name(args.device), vram_bytes=total,
                           interpretation="Capacity probe only; no DEV or selection evidence")
            save_json(profile, profile_path)
            copy_atomic(profile_path, drive / profile_path.name)
        else:
            logging.info("Resume uses the locked batch %d; GPU capacity verified", batch)
    cfg["training"]["batch_size"] = profile["batch_size"]
    cfg["evaluation"]["tile_batch_size"] = profile["evaluation_tile_batch_size"]
    study = runner.prepare_study(cfg, config, root, device="cpu" if args.action == "report" else args.device)
    pairs = requested_pairs(study, args.pairs)
    plan = preflight_plan(study)
    enforce_lock(root / "full", study["protocol"])
    save_yaml(cfg, root / "colab_input_config.yaml")
    save_yaml(config, root / "colab_runner_config.yaml")
    save_json(dict(commit=commit, input_sha256=identity, study_sha256=study["sha256"], config=cfg,
                   seed=cfg["training"]["seed"], dev_seed=cfg["training"]["dev_seed"],
                   batch_profile=profile, category_budget=plan,
                   torch=str(torch.__version__), cuda=torch.version.cuda,
                   python=sys.version, dependencies=dependencies,
                   orchestration_sha256=runner.sha256_file(__file__),
                   scope="D6 Adapter screening only", test_used_for_selection=False), root / "run_manifest.json")
    for path in root.iterdir():
        if path.is_file(): copy_atomic(path, drive / path.name)
    copy_atomic(root / "full" / "protocol_lock.json", drive / "full" / "protocol_lock.json")
    if args.action == "preflight":
        save_json(dict(status="PASS", real_gpu=True, frozen_backbone=True, trainable="Adapter+Decoder",
                       gpu=torch.cuda.get_device_name(args.device), memory_limit_bytes=memory_limit,
                       plan=plan, capacity_trials=trials, study_sha256=study["sha256"]), root / "preflight.json")
        copy_atomic(root / "preflight.json", drive / "preflight.json")
        print(json.dumps(dict(status="PREFLIGHT_PASS", batch_size=profile["batch_size"], plan=plan), indent=2))
    if args.action in {"smoke", "screen"}:
        smoke = args.action == "smoke"
        if smoke and (len(categories) != 1 or len(pairs) != 1):
            raise ValueError("Smoke must select exactly one category and one pair")
        if not smoke:
            smoke_context, _ = runner.make_context(study, "rice", study["pairs"][0], "adapter_screen", smoke=True)
            smoke_dir = runner.run_directory(study, "rice", study["pairs"][0], "adapter_screen", True)
            if runner.read_valid_result(smoke_dir, smoke_context) is None:
                raise runner.G2Blocked("BLOCKED: complete smoke on rice, 64:256 before full screening")
        enforce_lock(root / ("smoke" if smoke else "full"), study["protocol"])
        for pair in pairs:
            for category in categories:
                directory = runner.run_directory(study, category, pair, "adapter_screen", smoke)
                target = drive / directory.relative_to(root)
                def backup_checkpoint(_path):
                    copy_tree(directory, target)
                    copy_tree(directory.parent.parent / "_fairness", target.parent.parent / "_fairness")
                started = time.time()
                try:
                    status = runner.execute_job(study, category, pair, "adapter_screen", smoke=smoke,
                                                resume=True, device=args.device, model_factory=factory,
                                                on_checkpoint=backup_checkpoint)
                    run_manifest = directory / "colab_run.json"
                    if status != "SKIP" or not run_manifest.exists():
                        save_json(dict(status=status, category=category, r=pair[0], d=pair[1], commit=commit,
                                       gpu=torch.cuda.get_device_name(args.device), batch_size=profile["batch_size"],
                                       started_unix=started, finished_unix=time.time()), run_manifest)
                    if status not in {"PASS", "SKIP"}:
                        raise runner.G2Blocked(f"BLOCKED: {category}/{pair} {status}; resume before continuing")
                    if smoke:
                        context, _ = runner.make_context(study, category, pair, "adapter_screen", smoke=True)
                        checked = runner.read_valid_result(directory, context)
                        payload, _ = runner.load_checkpoint_payload(directory / "last.pt", require_sha256=True)
                        losses = [entry["loss"] for entry in payload["metadata"]["history"]]
                        if checked is None or not losses or not all(torch.isfinite(torch.tensor(v)) for v in losses):
                            raise ValueError("Smoke checkpoint/loss/metric validation failed")
                        save_json(dict(status="SMOKE_PASS", selection_evidence=False, losses=losses,
                                       synthetic_dev_aupro_0_05=checked["best_synthetic_dev"]["synthetic_dev_aupro_0_05"],
                                       frozen_backbone_unchanged=checked["frozen_backbone_unchanged"],
                                       adapter_updated=checked["adapter_updated"], decoder_updated=checked["decoder_updated"]),
                                  directory / "smoke_acceptance.json")
                        print("SMOKE_PASS: forward/backward, finite loss, native DEV metric, durable checkpoint")
                except BaseException as exc:
                    save_json(dict(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAIL",
                                   reason=str(exc), category=category, r=pair[0], d=pair[1]), directory / "colab_failure.json")
                    raise
                finally:
                    backup_checkpoint(None)
                    gc.collect(); torch.cuda.empty_cache()
                if not smoke:
                    summary = export_results(study)
                    copy_tree(root / "full" / "colab_report", drive / "full" / "colab_report")
                    lock_path = root / "full" / "adapter_selection_lock.json"
                    if lock_path.exists(): copy_atomic(lock_path, drive / "full" / lock_path.name)
                    print(f"Validated screening: {summary['valid_runs']}/72")
        copy_tree(root / "smoke", drive / "smoke")
    summary = export_results(study)
    copy_tree(root / "full" / "colab_report", drive / "full" / "colab_report")
    lock_path = root / "full" / "adapter_selection_lock.json"
    if lock_path.exists(): copy_atomic(lock_path, drive / "full" / lock_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
