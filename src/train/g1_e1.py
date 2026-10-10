"""G1 orchestration using the existing train step, loss, metrics and checkpoints.

Run from the repository root: python -m src.train.g1_e1 --help
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.loader import (G1NativeDataset, G1TileDataset, g1_tile_collate,
                             scan_mvtec_ad2, normalize_dinov3)
from src.data.synthetic_anomaly import validate_native_protocol
from src.data.tiling import generate_tile_records, crop_with_padding, stitch_tiles_hann
from src.eval.evaluator import build_anomaly_map_qa_report, write_metrics_json
from src.eval.full_scale import DEVMetricAccumulator
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.backbone_registry import backbone_spec
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.models.msila import E1
from src.train.optimizer import build_optimizer
from src.train.overfit16 import Overfit16Trainer, seed_everything
from src.utils.checkpoint import save_training_checkpoint
from src.utils.resume import load_checkpoint_payload, resume_training_checkpoint
from src.utils.visualize import save_training_curves

ROOT = Path(__file__).resolve().parents[2]
G1_BACKBONES = ("dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vith16plus")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_config(args):
    path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("version") != "g1_e1_v1":
        raise ValueError("Expected version: g1_e1_v1")
    if args.category:
        cfg["category"] = args.category
    if args.smoke and args.overfit16:
        raise ValueError("Choose either smoke or Overfit-16")
    if args.smoke:
        cfg["training"].update(mode="smoke", epochs=cfg["smoke"]["epochs"],
                               max_steps=cfg["smoke"]["max_steps"])
        for key in ("max_train_sources", "max_dev_sources", "dev_variants_per_image"):
            cfg["data"][key] = cfg["smoke"][key]
    if args.overfit16:
        cfg["training"]["mode"] = "overfit16"
    for flag, section, key in (("data_root", "data", "root"), ("repo_dir", "backbone", "repo_dir"),
                               ("output_root", None, "output_root")):
        if getattr(args, flag):
            target = cfg if section is None else cfg[section]
            target[key] = getattr(args, flag)
    name = cfg["backbone"]["name"]
    backbone_spec(name)  # Explicitly reject unofficial names, including vitsh16.
    if name not in G1_BACKBONES:
        raise ValueError(f"G1 supports {G1_BACKBONES}")
    if args.weights:
        cfg["backbone"]["checkpoints"][name] = args.weights
    def absolute(value):
        value = Path(value).expanduser()
        return str((value if value.is_absolute() else ROOT / value).resolve())
    cfg["data"]["root"] = absolute(cfg["data"]["root"])
    cfg["backbone"]["repo_dir"] = absolute(cfg["backbone"]["repo_dir"])
    if name not in cfg["backbone"]["checkpoints"]:
        raise ValueError(f"No checkpoint mapping for {name}; provide its official local checkpoint")
    cfg["backbone"]["weights"] = absolute(cfg["backbone"]["checkpoints"][name])
    cfg["output_root"] = absolute(cfg["output_root"])
    cfg["synthetic_protocol_path"] = absolute(cfg["synthetic_protocol"])
    cfg["synthetic_protocol"] = validate_native_protocol(yaml.safe_load(
        Path(cfg["synthetic_protocol_path"]).read_text()))
    t, d = cfg["training"], cfg["data"]
    if t["mode"] not in {"train", "smoke", "overfit16"}:
        raise ValueError("training.mode must be train, smoke or overfit16")
    if t["seed"] == t["dev_seed"]:
        raise ValueError("TRAIN and DEV require independent seeds")
    for key in ("epochs", "batch_size"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("max_steps", "max_minutes"):
        if t[key] is not None and t[key] <= 0:
            raise ValueError(f"training.{key} must be positive or null")
    if d["tile_size"] % backbone_spec(name).patch_size or not 0 <= d["overlap"] < d["tile_size"]:
        raise ValueError("Tile size must be patch-divisible; overlap must be in [0,tile_size)")
    for key in ("max_train_sources", "max_dev_sources"):
        if d[key] is not None and (type(d[key]) is not int or d[key] < 1):
            raise ValueError(f"data.{key} must be a positive integer or null")
    for key in ("train_variants_per_image", "dev_variants_per_image"):
        if type(d[key]) is not int or d[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if cfg["evaluation"]["tile_batch_size"] < 1 or d["num_workers"] < 0:
        raise ValueError("Invalid batch size/worker count")
    return cfg


def discover_sources(cfg):
    root, category = cfg["data"]["root"], cfg["category"]
    pools = {}
    manifest = {}
    for role, split in (("train", "train"), ("dev", "validation")):
        rows = [r for r in scan_mvtec_ad2(root, split=split, categories=[category])
                if r.defect_type == "good"]
        if not rows:
            raise FileNotFoundError(f"No {category}/{split}/good images under {root}")
        manifest[role] = [dict(path=str(Path(r.image_path).resolve()), sha256=file_sha256(r.image_path))
                          for r in rows]
        pools[role] = rows
    train_hashes = {r["sha256"] for r in manifest["train"]}
    if train_hashes.intersection(r["sha256"] for r in manifest["dev"]):
        raise ValueError("TRAIN/DEV source leakage: identical image content appears in both splits")
    for role in pools:
        limit = cfg["data"][f"max_{role}_sources"]
        if limit is not None:
            pools[role] = pools[role][:limit]
            manifest[role] = manifest[role][:limit]
    return pools, manifest


def check_assets(cfg, device):
    missing = []
    for label, path in (("MVTec AD 2", cfg["data"]["root"]),
                        ("official DINOv3 hubconf.py", Path(cfg["backbone"]["repo_dir"]) / "hubconf.py"),
                        ("pretrained checkpoint", cfg["backbone"]["weights"])):
        present = Path(path).is_dir() if label == "MVTec AD 2" else Path(path).is_file()
        if not present:
            missing.append(f"{label}: {path}")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        missing.append("CUDA requested but this environment has no usable CUDA device")
    if missing:
        raise FileNotFoundError("G1 NOT RUN; missing assets:\n" + "\n".join(missing))


@torch.no_grad()
def predict_native(model, image, cfg, device):
    """Sigmoid each tile, then Hann stitch in native coordinates without resize."""
    model.eval()
    size, overlap = cfg["data"]["tile_size"], cfg["data"]["overlap"]
    records = generate_tile_records(*image.shape[-2:], size, overlap, context_size=size)
    maps = []
    batch_size = cfg["evaluation"]["tile_batch_size"]
    for start in range(0, len(records), batch_size):
        tiles = [normalize_dinov3(crop_with_padding(image, r.local_xyxy))
                 for r in records[start:start + batch_size]]
        scores = model(torch.stack(tiles).to(device)).sigmoid().float().cpu()
        maps.extend(scores[:, 0].unbind(0))
    score = stitch_tiles_hann(maps, records, tuple(image.shape[-2:]), local_size=size)
    if score.shape != image.shape[-2:] or not torch.isfinite(score).all():
        raise RuntimeError("Native inference produced invalid shape/NaN/Inf")
    return score


def save_example(sample, score, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = sample["original"].permute(1, 2, 0).numpy()
    synthetic = sample["image"].permute(1, 2, 0).numpy()
    mask = sample["mask"][0].numpy()
    score = np.asarray(score)
    Image.fromarray((original * 255).round().astype(np.uint8)).save(directory / "original.png")
    Image.fromarray((synthetic * 255).round().astype(np.uint8)).save(directory / "synthetic.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(directory / "mask.png")
    np.save(directory / "predicted_map.npy", score)
    plt.imsave(directory / "predicted_map.png", score, cmap="inferno", vmin=0, vmax=1)
    overlay = .65 * synthetic + .35 * plt.get_cmap("inferno")(score)[..., :3]
    plt.imsave(directory / "overlay.png", np.clip(overlay, 0, 1))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, value, title in zip(axes, (original, synthetic, mask, score, overlay),
                                ("Original", "Synthetic", "Exact mask", "Probability", "Prediction overlay")):
        ax.imshow(value, **({"cmap": "inferno", "vmin": 0, "vmax": 1} if value.ndim == 2 else {}))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=150)
    plt.close(fig)
    write_metrics_json(sample["meta"], directory / "metadata.json")


def evaluate_dev(model, native, cfg, device, output_dir=None, *, predict_fn=None):
    # Training inference hook: keep the existing DEV metrics/QA/protocol exact.
    predict = predict_native if predict_fn is None else predict_fn
    protocol = cfg["synthetic_protocol"]
    mixed = DEVMetricAccumulator(protocol, per_region=False, disk_backed=True)
    tiny = DEVMetricAccumulator(protocol, per_region=False, disk_backed=True)
    normal_scores = []
    examples = 0
    qa_rows = []
    with tempfile.TemporaryDirectory(prefix="g1_dev_") as scratch:
        for index in range(len(native)):
            sample = native[index]
            score = predict(model, sample["image"], cfg, device).numpy()
            mask = sample["mask"][0].numpy()
            meta = sample["meta"]
            record = dict(anomaly_map=score, gt_mask=mask, meta=dict(
                image_id=meta["sample_id"], category=cfg["category"], split="dev_synthetic",
                original_hw=meta["original_hw"], anomaly_map_space="original_image",
                gt_mask_space="original_image"))
            qa = build_anomaly_map_qa_report([record])
            if qa["summary"]["status"] != "PASS":
                raise ValueError(f"Native map QA failed: {qa}")
            qa_rows.extend(qa["per_sample"])
            synthetic = meta["synthetic"]
            mixed.add(score, mask, split="dev_mixed", image_id=meta["sample_id"])
            if not synthetic["is_anomaly"] or synthetic["size_bin"] in protocol["dev_tiny_bins"]:
                tiny.add(score, mask, split="dev_tiny", image_id=meta["sample_id"])
            if not synthetic["is_anomaly"]:
                path = Path(scratch) / f"normal_{index}.npy"
                np.save(path, score)
                normal_scores.append(np.load(path, mmap_mode="r"))
            if output_dir is not None and examples < cfg["evaluation"]["example_limit"]:
                save_example(sample, score, Path(output_dir) / "examples" / meta["sample_id"])
                examples += 1
        # Exact pixel P99 on normal-only native maps, with disk-backed workspace.
        count = sum(a.size for a in normal_scores)
        if count:
            values = np.memmap(Path(scratch) / "normal_pixels.bin", mode="w+", dtype=np.float32, shape=(count,))
            offset = 0
            for array in normal_scores:
                values[offset:offset + array.size] = array.ravel()
                offset += array.size
            p99 = float(np.quantile(values, .99, overwrite_input=True))
            del values
        else:
            p99 = None
    mixed_result, tiny_result = mixed.result()["groups"]["all"], tiny.result()["groups"]["all"]
    result = dict(dataset="mvtec_ad2_good_synthetic", split="dev_synthetic",
                  interpretation="Synthetic DEV only; no real anomaly performance claim",
                  score_orientation="higher_is_more_anomalous", score_normalization="sigmoid_no_rescaling",
                  aupro_max_fpr=.05, native_resolution=True, n_samples=len(native),
                  synthetic_dev_aupro_0_05=mixed_result["aupro_0_05"],
                  dev_tiny=tiny_result, dev_mixed=mixed_result, normal_score_p99=p99,
                  normal_p99_unit="all pixels of normal-only native DEV images",
                  qa_status="PASS", qa_samples=len(qa_rows))
    if output_dir is not None:
        write_metrics_json(dict(status="PASS", per_sample=qa_rows), Path(output_dir) / "qa_report.json")
    return result


def resume_identity(cfg):
    """Permit extending only the run budget; scientific settings stay locked."""
    identity = copy.deepcopy(cfg)
    for key in ("epochs", "max_steps", "max_minutes"):
        identity["training"].pop(key, None)
    return identity


def run(cfg, *, device, output_dir, resume=None, evaluate=None):
    check_assets(cfg, device)
    pools, manifest = discover_sources(cfg)
    cfg["sources"] = manifest
    cfg["backbone"]["checkpoint_sha256"] = file_sha256(cfg["backbone"]["weights"])
    t, d = cfg["training"], cfg["data"]
    seed_everything(t["seed"])
    extractor = DINOv3FeatureExtractor(cfg["backbone"]["repo_dir"], cfg["backbone"]["weights"],
                                     model_name=cfg["backbone"]["name"], feature_mode="deepest", check_finite=True)
    model = E1(extractor, **cfg["decoder"]).to(device)
    optimizer, report = build_optimizer({"decoder": model.decoder}, frozen_modules={"backbone": extractor},
                                       learning_rate=t["learning_rate"], weight_decay=t["weight_decay"])
    logging.info("E1 %s; optimizer=%s", extractor, report.to_dict())
    native_train = G1NativeDataset(pools["train"], cfg["synthetic_protocol"], seed=t["seed"], role="train",
                                   variants=d["train_variants_per_image"], fixed=t["mode"] == "overfit16")
    tiles = G1TileDataset(native_train, tile_size=d["tile_size"], overlap=d["overlap"],
                          overfit16=t["mode"] == "overfit16")
    native_dev = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"], seed=t["dev_seed"], role="dev",
                                 variants=d["dev_variants_per_image"], fixed=True)
    trainer = Overfit16Trainer(model=model, criterion=AnomalySegmentationLoss(**cfg["loss"]),
                              optimizer=optimizer, device=device, frozen_modules={"backbone": extractor},
                              max_grad_norm=t["max_grad_norm"])
    if evaluate:
        payload, _ = load_checkpoint_payload(evaluate, require_sha256=True)
        if resume_identity(payload["config"]) != resume_identity(cfg):
            raise ValueError("Evaluation config/source/checkpoint provenance mismatch")
        model.decoder.load_state_dict(payload["model_state"], strict=True)
        metrics = evaluate_dev(model, native_dev, cfg, device, output_dir / "evaluation")
        write_metrics_json(metrics, output_dir / "evaluation_metrics.json")
        return metrics
    history, best, start_epoch, start_batch, step = [], -float("inf"), 0, 0, 0
    initial = None
    if resume:
        payload, _ = load_checkpoint_payload(resume, require_sha256=True)
        if resume_identity(payload["config"]) != resume_identity(cfg):
            raise ValueError("Resume config/source/checkpoint provenance mismatch")
        restored = resume_training_checkpoint(resume, model=model.decoder, optimizer=optimizer, map_location=device)
        state = restored.metadata
        history, best = state["history"], state["best_metric"]
        start_epoch, start_batch, step = state["next_epoch"], state["next_batch"], restored.global_step
        initial = state["initial_overfit_loss"]
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    def loader(epoch, shuffle):
        return DataLoader(tiles, batch_size=t["batch_size"], shuffle=shuffle, num_workers=d["num_workers"],
                          collate_fn=g1_tile_collate, generator=torch.Generator().manual_seed(t["seed"] + epoch))
    if t["mode"] == "overfit16" and initial is None:
        initial = trainer.evaluate(loader(0, False)).loss
    frozen_before, decoder_before = module_sha256(extractor), module_sha256(model.decoder)
    started, stop_reason, latest_dev = time.monotonic(), "epochs_completed", None
    for epoch in range(start_epoch, t["epochs"]):
        native_train.set_epoch(epoch)
        batches = loader(epoch, True)
        next_batch = 0
        for batch_index, batch in enumerate(batches):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if ((t["max_steps"] is not None and step >= t["max_steps"]) or
                    (t["max_minutes"] is not None and time.monotonic() - started >= t["max_minutes"] * 60)):
                stop_reason = "budget_reached"
                next_batch = batch_index
                break
            step_started = time.monotonic()
            entry = trainer.train_step(batch, step=step + 1, epoch=epoch).to_dict()
            if extractor.backbone.training or not extractor.backbone_is_frozen() or any(
                    p.grad is not None for p in extractor.parameters()):
                raise RuntimeError("Frozen backbone contract violated")
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder.parameters()):
                raise RuntimeError("Decoder gradients are absent or nonfinite")
            entry["seconds"] = time.monotonic() - step_started
            history.append(entry)
            step += 1
            next_batch = batch_index + 1
            logging.info("epoch=%d step=%d loss=%.6f BCE=%.6f Dice=%.6f lr=%g time=%.3fs",
                         epoch + 1, step, entry["loss"], entry["bce"], entry["dice_loss"],
                         entry["learning_rate"], entry["seconds"])
        if not history:
            raise RuntimeError("Budget expired before any optimizer update")
        if next_batch >= len(batches):
            next_epoch, next_batch = epoch + 1, 0
        else:
            next_epoch = epoch
        latest_dev = evaluate_dev(model, native_dev, cfg, device)
        metric = latest_dev["synthetic_dev_aupro_0_05"]
        if metric is None:
            raise ValueError("Synthetic DEV has no valid anomalous regions/normal pixels")
        is_best = metric > best
        best = max(best, metric)
        metadata = dict(next_epoch=next_epoch, next_batch=next_batch, history=history,
                        best_metric=best, initial_overfit_loss=initial, dev=latest_dev)
        for name in (["last.pt", "best.pt"] if is_best else ["last.pt"]):
            save_training_checkpoint(output_dir / name, model=model.decoder, optimizer=optimizer,
                                     epoch=epoch, global_step=step, config=cfg, metadata=metadata)
            logging.info("checkpoint=%s synthetic_DEV_AU_PRO005=%.6f", output_dir / name, metric)
        if stop_reason == "budget_reached":
            break
    if not history:
        raise ValueError("No training history; resume budget must allow an update")
    final = trainer.evaluate(loader(0, False)).loss if t["mode"] == "overfit16" else None
    frozen_unchanged = module_sha256(extractor) == frozen_before
    decoder_updated = module_sha256(model.decoder) != decoder_before
    if not frozen_unchanged or (step > (restored.global_step if resume else 0) and not decoder_updated):
        raise RuntimeError("Backbone changed or decoder failed to update")
    save_training_curves(history, output_dir, metrics=("loss", "bce", "dice_loss"))
    (output_dir / "loss.png").replace(output_dir / "loss_curve.png")
    # Final artifacts always describe best.pt, selected on synthetic DEV only.
    best_path = output_dir / "best.pt"
    if not best_path.is_file() and resume:
        best_path = Path(resume).parent / "best.pt"
    payload, _ = load_checkpoint_payload(best_path, require_sha256=True)
    model.decoder.load_state_dict(payload["model_state"], strict=True)
    best_dev = evaluate_dev(model, native_dev, cfg, device, output_dir)
    result = dict(status="PASS", scope="G1 E1 synthetic DEV", mode=t["mode"],
                  checkpoint=str(best_path), best_epoch=payload["training_state"]["epoch"],
                  global_step=step, elapsed_seconds=time.monotonic() - started, stop_reason=stop_reason,
                  frozen_backbone_unchanged=frozen_unchanged, decoder_updated=decoder_updated,
                  initial_overfit_loss=initial, final_overfit_loss=final,
                  overfit_loss_decreased=None if initial is None else final < initial,
                  overfit_role="16 fixed samples; architecture QA only" if initial is not None else None,
                  best_synthetic_dev=best_dev, last_synthetic_dev=latest_dev)
    write_metrics_json(result, output_dir / "metrics.json")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g1_e1.yaml")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit16", action="store_true")
    parser.add_argument("--resume", help="Resume last.pt, including a partial epoch")
    parser.add_argument("--evaluate", help="Evaluate a decoder checkpoint on fixed native synthetic DEV")
    for name in ("data-root", "repo-dir", "weights", "output-root"):
        parser.add_argument(f"--{name}")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    output = Path(cfg["output_root"]) / cfg["category"]
    if (output / "best.pt").exists() and not (args.resume or args.evaluate):
        raise SystemExit(f"Run already exists: {output}; use --resume or choose --output-root")
    if args.resume and args.evaluate:
        raise SystemExit("Choose either --resume or --evaluate")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "best.pt").exists():
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(output / "train.log"), logging.StreamHandler()])
    try:
        result = run(cfg, device=args.device, output_dir=output, resume=args.resume, evaluate=args.evaluate)
        logging.info("G1 result: %s", json.dumps(result, allow_nan=False))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        # Never overwrite a completed run's metrics on failed resume/evaluation.
        failure_path = output / ("failure.json" if (output / "best.pt").exists() else "metrics.json")
        write_metrics_json(dict(status="NOT RUN" if isinstance(exc, FileNotFoundError) else "FAIL",
                                reason=str(exc), real_pretrained_run_verified=False), failure_path)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
