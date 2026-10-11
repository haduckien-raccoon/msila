"""One synthetic DEV evaluator. D6 implements E1/E2; E3--E5 stay reserved.

Reuse the training DEV evaluator verbatim through its native prediction hook:
sigmoid -> existing Hann stitching -> native metric/QA. Dice is the existing
pooled segmentation F1 at the protocol's predeclared threshold, not a search.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import G1NativeDataset
from src.data.synthetic_anomaly import validate_native_protocol
from src.data.tiling import generate_tile_records
from src.eval.evaluator import MVTEC_AD2_CATEGORIES, write_metrics_json
from src.metrics.segf1 import aggregate_seg_f1
from src.models.msila import build_g2_model
from src.train.g1_e1 import check_assets, discover_sources, evaluate_dev, file_sha256, predict_native
from src.train.screen_representation import sha256_json
from src.utils.resume import load_checkpoint_payload

CATEGORIES = tuple(MVTEC_AD2_CATEGORIES)
METRICS = ("synthetic_dev_aupro_0_05", "tiny_aupro_0_05", "mixed_aupro_0_05",
           "synthetic_dice", "normal_score_p99")


class Blocked(RuntimeError):
    """Missing or incompatible scientific inputs; never substitute random weights."""


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def unit_metric(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def load_trained_checkpoint(path, experiment, category):
    if experiment not in {"E1", "E2"}:
        raise Blocked(f"{experiment} is reserved; D6 implements only E1/E2")
    path = Path(path)
    if not path.is_file():
        pending = "; E2 may await D10 joint selection and main training" if experiment == "E2" else ""
        raise Blocked(f"Missing {experiment}/{category} checkpoint: {path}{pending}")
    payload, _ = load_checkpoint_payload(path, require_sha256=True)
    cfg, meta = payload["config"], payload["metadata"]
    if cfg.get("category") != category:
        raise Blocked(f"Checkpoint category mismatch: expected {category}")
    if payload["training_state"]["global_step"] <= 0 or not meta.get("history"):
        raise Blocked("Checkpoint has no optimizer updates/training history")
    if (not unit_metric(meta.get("best_metric")) or meta.get("dev", {}).get("qa_status") != "PASS"
            or meta["dev"].get("aupro_max_fpr") != .05
            or meta["dev"].get("split") != "dev_synthetic"):
        raise Blocked("Checkpoint lacks valid synthetic DEV selection evidence")
    if not cfg.get("sources", {}).get("dev") or not cfg.get("backbone", {}).get("checkpoint_sha256"):
        raise Blocked("Checkpoint lacks source/pretrained-weight provenance")
    if experiment == "E1":
        if cfg.get("version") != "g1_e1_v1" or cfg["training"].get("mode") != "train":
            raise Blocked("E1 requires a trained G1 E1 checkpoint (smoke/overfit cannot be accepted)")
    else:
        if (cfg.get("stage") != "E2" or cfg.get("mode") != "full"
                or any(type(cfg.get("adapter", {}).get(k)) is not int or cfg["adapter"][k] <= 0
                       for k in ("r", "d"))
                or meta.get("config_sha256") != sha256_json(cfg)):
            raise Blocked("E2 checkpoint stage/config/r/d provenance mismatch")
        # D6 evaluates trained checkpoints; it does not certify D10 selection.
        # Legacy selection fields are preserved in the exported checkpoint config.
    return payload


def source_identity(manifest):
    """Ignore machine-specific prefixes, preserve source order and relative names."""
    result = {}
    for role in ("train", "dev"):
        entries = []
        for row in manifest[role]:
            parts = Path(row["path"]).parts
            category_index = next((i for i, part in enumerate(parts[:-1]) if part in CATEGORIES
                                   and parts[i + 1].lower() in {"train", "validation"}), None)
            if category_index is None:
                raise Blocked(f"Source path has no category: {row['path']}")
            relative = list(parts[category_index:])
            relative[1:3] = [p.lower() for p in relative[1:3]]
            entries.append(dict(path="/".join(relative), sha256=row["sha256"]))
        result[role] = entries
    return result


def scientific_identity(cfg):
    """Normalize G1/G2 config spelling without changing any scientific variable."""
    t, d, b = cfg["training"], cfg["data"], cfg["backbone"]
    optimizer = t.get("optimizer", {})
    return dict(
        category=cfg["category"],
        backbone={k: b.get(k, True) for k in ("name", "checkpoint_sha256", "norm")},
        decoder=dict(hidden_channels=cfg["decoder"]["hidden_channels"],
                     deterministic_resize=cfg["decoder"].get("deterministic_resize", False)),
        data={k: d[k] for k in ("tile_size", "overlap", "max_train_sources", "max_dev_sources",
                               "train_variants_per_image", "dev_variants_per_image")},
        sources=source_identity(cfg["sources"]), synthetic_protocol=cfg["synthetic_protocol"],
        training={**{k: t[k] for k in ("seed", "dev_seed", "epochs", "batch_size", "max_steps",
                                      "max_minutes", "max_grad_norm")},
                  "optimizer": optimizer.get("name", "AdamW"),
                  "learning_rate": t.get("learning_rate", optimizer.get("lr")),
                  "weight_decay": t.get("weight_decay", optimizer.get("weight_decay")),
                  "amp": t.get("amp", False), "scheduler": t.get("scheduler"),
                  "train_synthetic_by_epoch": t.get("train_synthetic_by_epoch", True),
                  "dev_synthetic_fixed": t.get("dev_synthetic_fixed", True)},
        loss={k: cfg["loss"].get(k, 1e-6 if k == "dice_eps" else 1.)
              for k in ("bce_weight", "dice_weight", "dice_eps")},
        metrics=dict(max_fpr=.05, score_transform="sigmoid", stitching="hann_native_coordinates",
                     dice_threshold=cfg["synthetic_protocol"]["prediction_threshold"]),
    )


def relocate_config(saved, args):
    cfg = deepcopy(saved)
    for attr, section, key in (("data_root", "data", "root"), ("repo_dir", "backbone", "repo_dir"),
                               ("weights", "backbone", "weights")):
        value = getattr(args, attr)
        if value:
            cfg[section][key] = str(Path(value).expanduser().resolve())
    cfg["synthetic_protocol"] = validate_native_protocol(cfg["synthetic_protocol"])
    d, e = cfg["data"], cfg["evaluation"]
    if (d["tile_size"] != 512 or not 0 <= d["overlap"] < 512
            or d.get("train_split", "TRAIN/good") != "TRAIN/good"
            or d.get("dev_split", "VALIDATION/good") != "VALIDATION/good"
            or e.get("max_fpr", .05) != .05 or e.get("score_transform", "sigmoid") != "sigmoid"
            or e.get("stitching", "hann_native_coordinates") != "hann_native_coordinates"
            or cfg["synthetic_protocol"]["prediction_threshold"] != .5
            or cfg["training"]["dev_seed"] == cfg["training"]["seed"]):
        raise Blocked("Checkpoint does not use the locked G2 synthetic DEV/threshold/tiling protocol")
    check_assets(cfg, args.device)
    if file_sha256(cfg["backbone"]["weights"]) != saved["backbone"]["checkpoint_sha256"]:
        raise Blocked("Pretrained DINO checkpoint hash differs from training")
    expected_code = cfg["backbone"].get("source_sha256")
    if expected_code is not None:
        repo = Path(cfg["backbone"]["repo_dir"])
        actual = {str(p.relative_to(repo)): file_sha256(p) for p in sorted(repo.rglob("*.py"))}
        if expected_code != actual:
            raise Blocked("DINO source checkout differs from training; use the original revision")
    pools, manifest = discover_sources(cfg)
    if source_identity(saved["sources"]) != source_identity(manifest):
        raise Blocked("TRAIN/DEV sources/order/content differ from checkpoint; no TEST or substitute DEV")
    cfg["sources"] = manifest
    return cfg, pools


def validate_e1_budget(path, payload, cfg, pools):
    """Audit final budget when retained; a legacy best-only checkpoint is usable.

    Saved configuration/source hashes still bind the scientific protocol. Missing
    final artifacts do not imply that this evaluator verified full training.
    """
    directory = Path(path).parent
    if not (directory / "last.pt").is_file() or not (directory / "metrics.json").is_file():
        return False
    last, _ = load_checkpoint_payload(directory / "last.pt", require_sha256=True)
    report = json.loads((directory / "metrics.json").read_text())
    d, t = cfg["data"], cfg["training"]
    tiles = 0
    for record in pools["train"]:
        with Image.open(record.image_path) as image:
            tiles += len(generate_tile_records(image.height, image.width, d["tile_size"],
                                               d["overlap"], context_size=d["tile_size"]))
    expected = math.ceil(tiles * (d["train_variants_per_image"] + 1) / t["batch_size"]) * t["epochs"]
    if t["max_steps"] is not None:
        expected = min(expected, t["max_steps"])
    if (last["config"] != payload["config"] or last["training_state"]["global_step"] != expected
            or report.get("global_step") != expected or report.get("status") != "PASS"
            or report.get("mode") != "train" or report.get("frozen_backbone_unchanged") is not True
            or report.get("decoder_updated") is not True
            or report.get("best_epoch") != payload["training_state"]["epoch"]
            or report.get("best_synthetic_dev", {}).get("synthetic_dev_aupro_0_05") != payload["metadata"]["best_metric"]):
        raise Blocked("E1 lacks complete training budget/provenance: keep best.pt, last.pt, metrics.json and SHA sidecars")
    return True


def restore_model(payload, cfg, experiment, device):
    if experiment == "E1":
        # G1 config has no Adapter section; factory requires one even for E1.
        factory_cfg = deepcopy(cfg)
        factory_cfg["adapter"] = dict(r=1, d=1, kernel_size=3, gamma_init=0., bias=True)
    else:
        factory_cfg = cfg
    model = build_g2_model(factory_cfg, experiment=experiment, root=ROOT)
    head = model.decoder if experiment == "E1" else nn.ModuleDict({"adapter": model.adapter, "decoder": model.decoder})
    head.load_state_dict(payload["model_state"], strict=True)
    if not model.extractor.backbone.__class__.__module__.startswith("dinov3."):
        raise Blocked("API fixture or unofficial backbone cannot produce experimental results")
    if any(p.requires_grad for p in model.extractor.parameters()):
        raise Blocked("Backbone is not frozen")
    model.requires_grad_(False)
    return model.to(device).eval()


def implementation_identity():
    paths = {ROOT / p for p in ("scripts/eval_g2.py", "src/train/g1_e1.py", "src/train/screen_representation.py",
                               "src/utils/checkpoint.py", "src/utils/resume.py")}
    for directory in ("src/data", "src/eval", "src/metrics", "src/models", "src/geometry"):
        paths.update((ROOT / directory).rglob("*.py"))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


class CaptureNative:
    """Observe the existing evaluator's one pass; export exact maps and masks."""
    def __init__(self, native, directory):
        self.native, self.directory, self.index, self.rows = native, Path(directory), 0, []

    def __call__(self, model, image, cfg, device):
        sample = self.native[self.index]
        if not torch.equal(image, sample["image"]):
            raise RuntimeError("Native export order differs from evaluator input")
        score = predict_native(model, image, cfg, device)
        folder = self.directory / "maps" / sample["meta"]["sample_id"]
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "score.npy", score.numpy())
        np.save(folder / "mask.npy", sample["mask"][0].numpy().astype(np.uint8))
        row = deepcopy(sample["meta"])
        row["source"] = source_identity({"train": [], "dev": [{
            "path": row["source"], "sha256": file_sha256(row["source"])}]})["dev"][0]
        row.update(score_path=str((folder / "score.npy").relative_to(self.directory)),
                   mask_path=str((folder / "mask.npy").relative_to(self.directory)),
                   score_sha256=file_sha256(folder / "score.npy"), mask_sha256=file_sha256(folder / "mask.npy"),
                   image_sha256=hashlib.sha256(memoryview(image.contiguous().numpy()).cast("B")).hexdigest())
        self.rows.append(row)
        self.index += 1
        return score


def dice_from_exports(directory, rows, category, threshold):
    samples = [dict(category=category,
                    anomaly_map=np.load(Path(directory) / row["score_path"], mmap_mode="r"),
                    gt_mask=np.load(Path(directory) / row["mask_path"], mmap_mode="r")) for row in rows]
    return aggregate_seg_f1(samples, threshold)["per_category"][category]


def validate_metrics(result):
    if (any(result.get(k) is None for k in METRICS)
            or result.get("dev_tiny", {}).get("regions", 0) < 1
            or result.get("dev_mixed", {}).get("regions", 0) < 1):
        raise Blocked("Required category metric unavailable; insufficient synthetic regions/normal pixels")
    if (any(not unit_metric(result.get(k)) for k in METRICS) or result.get("qa_status") != "PASS"
            or result.get("native_resolution") is not True or result.get("aupro_max_fpr") != .05
            or result.get("dev_tiny", {}).get("aupro_0_05") != result["tiny_aupro_0_05"]
            or result.get("dev_mixed", {}).get("aupro_0_05") != result["mixed_aupro_0_05"]
            or result["synthetic_dev_aupro_0_05"] != result["mixed_aupro_0_05"]
            or result.get("dice_threshold") != .5
            or result.get("score_normalization") != "sigmoid_no_rescaling"
            or result.get("score_orientation") != "higher_is_more_anomalous"
            or result.get("qa_samples") != result.get("n_samples") or not result.get("n_samples")):
        raise ValueError("Missing/nonfinite metric, anomalous regions or native-map QA")


def completed_result(directory, signature, checkpoint):
    """Resume requires bound inputs plus every exported native map and mask."""
    try:
        directory = Path(directory)
        path = directory / "metrics.json"
        result = json.loads(path.read_text())
        if ((directory / "metrics.json.sha256").read_text().strip() != file_sha256(path)
                or result["evaluation_sha256"] != signature or result["checkpoint_sha256"] != file_sha256(checkpoint)
                or result["status"] not in {"PASS", "CPU_PASS"}
                or result.get("acceptance_eligible") is not (result["status"] == "PASS")):
            return None
        validate_metrics(result)
        rows = json.loads((directory / "native_maps.json").read_text())
        if len(rows) != result["n_samples"] or sha256_json(rows) != result["native_maps_sha256"]:
            return None
        if file_sha256(directory / "qa_report.json") != result["qa_report_sha256"]:
            return None
        if file_sha256(directory / "evaluation_config.json") != result["evaluation_config_sha256"]:
            return None
        examples = result["examples_sha256"]
        if not examples or not any(p.endswith("/comparison.png") for p in examples):
            return None
        for relative, digest in examples.items():
            if file_sha256(directory / relative) != digest:
                return None
        for row in rows:
            for field in ("score", "mask"):
                path = directory / row[f"{field}_path"]
                if file_sha256(path) != row[f"{field}_sha256"]:
                    return None
                if list(np.load(path, mmap_mode="r").shape) != row["original_hw"]:
                    return None
        return result
    except (OSError, KeyError, ValueError, TypeError, RuntimeError):
        return None


def prepare_job(args, experiment, category):
    if experiment not in {"E1", "E2"}:
        raise Blocked(f"{experiment} is reserved; D6 implements only E1/E2")
    path = Path(getattr(args, f"{experiment.lower()}_root")) / category / "best.pt"
    payload = load_trained_checkpoint(path, experiment, category)
    cfg, pools = relocate_config(payload["config"], args)
    budget_verified = validate_e1_budget(path, payload, cfg, pools) if experiment == "E1" else False
    identity = dict(version="g2_dev_evaluator_v2", experiment=experiment, checkpoint_sha256=file_sha256(path),
                    checkpoint_config_sha256=sha256_json(payload["config"]),
                    scientific_protocol=scientific_identity(cfg), smoke=args.smoke,
                    final_training_budget_verified=budget_verified,
                    runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=args.device,
                                 gpu=torch.cuda.get_device_name(torch.device(args.device))
                                 if torch.device(args.device).type == "cuda" else None,
                                 deterministic=True, amp=False, tf32=False),
                    implementation=implementation_identity())
    return path, payload, cfg, pools, identity


def evaluate_category(args, experiment, category):
    directory = Path(args.output_root) / experiment / category
    path, payload, cfg, pools, identity = prepare_job(args, experiment, category)
    signature = sha256_json(identity)
    if args.preflight:
        return dict(status="READY", experiment=experiment, category=category, checkpoint=str(path),
                    backbone=cfg["backbone"]["name"], training_seed=cfg["training"]["seed"],
                    dev_seed=cfg["training"]["dev_seed"],
                    checkpoint_config_sha256=sha256_json(payload["config"]),
                    pair_protocol_sha256=sha256_json(identity["scientific_protocol"]))
    if args.resume and not args.smoke:
        saved = completed_result(directory, signature, path)
        if saved is not None:
            logging.info("SKIP valid %s/%s", experiment, category)
            return saved
    torch.manual_seed(cfg["training"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = restore_model(payload, cfg, experiment, args.device)
    native = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"], seed=cfg["training"]["dev_seed"],
                             role="dev", variants=cfg["data"]["dev_variants_per_image"], fixed=True)
    if args.smoke:
        # Keep original source index/variant seeds; normal + first two defects.
        native = torch.utils.data.Subset(native, range(min(3, len(native))))
    directory.mkdir(parents=True, exist_ok=True)
    cfg["evaluation"]["example_limit"] = 3
    capture = CaptureNative(native, directory)
    started = time.monotonic()
    result = evaluate_dev(model, native, cfg, args.device, directory, predict_fn=capture)
    dice = dice_from_exports(directory, capture.rows, category, cfg["synthetic_protocol"]["prediction_threshold"])
    dev_identity = [{k: row[k] for k in ("sample_id", "source", "original_hw", "synthetic", "mask_sha256",
                                        "image_sha256")} for row in capture.rows]
    result.update(tiny_aupro_0_05=result["dev_tiny"]["aupro_0_05"],
                  mixed_aupro_0_05=result["dev_mixed"]["aupro_0_05"], synthetic_dice=dice["f1"],
                  dice_threshold=.5, dice_rule="pooled native pixels; score >= 0.5; includes normal false positives",
                  dice_counts={k: dice[k] for k in ("tp", "fp", "fn")})
    validate_metrics(result)
    write_metrics_json(capture.rows, directory / "native_maps.json")
    write_metrics_json(dict(identity=identity, resolved_config=cfg), directory / "evaluation_config.json")
    cuda = torch.device(args.device).type == "cuda"
    real = model.extractor.backbone.__class__.__module__.startswith("dinov3.")
    result.update(status="SMOKE_PASS" if args.smoke else ("PASS" if cuda and real else "CPU_PASS"), experiment=experiment, category=category,
                  verification_scope="real_pretrained" if real else "fixture", device=args.device, git_commit=git_commit(),
                  checkpoint=str(path.resolve()), checkpoint_sha256=identity["checkpoint_sha256"],
                  checkpoint_config_sha256=identity["checkpoint_config_sha256"], evaluation_sha256=signature,
                  final_training_budget_verified=identity["final_training_budget_verified"],
                  pair_protocol_sha256=sha256_json(identity["scientific_protocol"]),
                  dev_samples_sha256=sha256_json(dev_identity), native_maps_sha256=sha256_json(capture.rows),
                  qa_report_sha256=file_sha256(directory / "qa_report.json"),
                  evaluation_config_sha256=file_sha256(directory / "evaluation_config.json"),
                  examples_sha256={str(p.relative_to(directory)): file_sha256(p)
                                   for p in sorted((directory / "examples").rglob("*")) if p.is_file()},
                  elapsed_seconds=time.monotonic() - started, training_seed=cfg["training"]["seed"],
                  dev_seed=cfg["training"]["dev_seed"],
                  acceptance_eligible=not args.smoke and cuda and real)
    write_metrics_json(result, directory / "metrics.json")
    (directory / "metrics.json.sha256").write_text(file_sha256(directory / "metrics.json") + "\n")
    return result


def write_comparison(output_root, results, *, mode="full", experiments=("E1", "E2"), categories=CATEGORIES):
    directory = Path(output_root)
    directory.mkdir(parents=True, exist_ok=True)
    rows, paired = [], []
    counts = {exp: 0 for exp in ("E1", "E2")}
    for category in CATEGORIES:
        row = dict(category=category)
        records = {exp: results.get((exp, category), {"status": "NOT RUN", "reason": "Not requested"})
                   for exp in ("E1", "E2")}
        for exp, record in records.items():
            valid = record["status"] == "PASS" and record.get("acceptance_eligible") is True
            if valid:
                validate_metrics(record)
            counts[exp] += int(valid)
            row[f"{exp}_status"] = record["status"]
            row[f"{exp}_reason"] = record.get("reason", "")
            for metric in METRICS:
                row[f"{exp}_{metric}"] = record.get(metric) if valid else None
        if all(records[exp]["status"] == "PASS" and records[exp].get("acceptance_eligible") for exp in records):
            if (records["E1"]["pair_protocol_sha256"] != records["E2"]["pair_protocol_sha256"]
                    or records["E1"]["dev_samples_sha256"] != records["E2"]["dev_samples_sha256"]):
                row.update(pair_status="BLOCKED", pair_reason="E1/E2 protocol or native synthetic DEV differs")
            else:
                row.update(pair_status="PASS", pair_reason="")
                row["E2_minus_E1_aupro_0_05"] = row["E2_synthetic_dev_aupro_0_05"] - row["E1_synthetic_dev_aupro_0_05"]
                paired.append(row)
        else:
            row.update(pair_status="FAIL" if any(r["status"] == "FAIL" for r in records.values()) else "BLOCKED",
                       pair_reason="Missing valid GPU evaluation of one or both experiments")
        row.setdefault("E2_minus_E1_aupro_0_05", None)
        rows.append(row)
    complete = len(paired) == len(CATEGORIES) and mode == "full"
    macro = None
    if complete:
        macro = {key: math.fsum(row[key] for row in paired) / 8
                 for key in tuple(f"{exp}_{metric}" for exp in counts for metric in METRICS) + ("E2_minus_E1_aupro_0_05",)}
    fields = list(rows[0])
    with (directory / "metrics_E1_E2.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(directory / "metrics_E1_E2.csv.tmp", directory / "metrics_E1_E2.csv")
    selected = [results.get((exp, cat), {"status": "NOT RUN"}) for exp in experiments for cat in categories]
    statuses = {record["status"] for record in selected}
    operation_status = "BLOCKED"
    if "FAIL" in statuses:
        operation_status = "FAIL"
    elif mode == "preflight" and statuses == {"READY"}:
        operation_status = "READY"
    elif mode == "smoke" and statuses == {"SMOKE_PASS"}:
        operation_status = "SMOKE_PASS"
    elif statuses <= {"PASS", "CPU_PASS"} and selected:
        operation_status = "CPU_PASS" if "CPU_PASS" in statuses else "PASS"
        if "E1" in experiments and "E2" in experiments and any(
                row["pair_status"] != "PASS" for row in rows if row["category"] in categories):
            operation_status = "BLOCKED"
    individual_macro = {
        exp: {metric: math.fsum(row[f"{exp}_{metric}"] for row in rows) / 8 for metric in METRICS}
        if counts[exp] == 8 and mode == "full" else None for exp in counts
    }
    summary = dict(version="g2_e1_e2_evaluation_v2", status=operation_status, mode=mode,
                   comparison_status="PASS" if complete else "BLOCKED",
                   experiments=["E1", "E2"], split="dev_synthetic", max_fpr=.05,
                   requested_experiments=list(experiments), requested_categories=list(categories),
                   interpretation="Synthetic DEV comparison only; no TEST tuning or real anomaly claim",
                   expected_categories=8, coverage=counts, paired_pass=len(paired), complete=complete,
                   macro_mean_8categories=macro, per_experiment_macro_mean_8categories=individual_macro,
                   records=rows, git_commit=git_commit())
    write_metrics_json(summary, directory / "summary.json")
    write_metrics_json(dict(status=summary["status"], comparison_status=summary["comparison_status"],
                            requested_experiments=list(experiments), requested_categories=list(categories),
                            coverage=counts, paired_pass=len(paired), expected=8,
                            missing=[dict(experiment=exp, category=row["category"], status=row[f"{exp}_status"],
                                          reason=row[f"{exp}_reason"]) for row in rows for exp in counts
                                     if row[f"{exp}_{METRICS[0]}"] is None], records=rows), directory / "coverage.json")
    # Only genuine paired results produce numeric bars. Missing rows remain explicit.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    if paired:
        ax.bar([r["category"] for r in paired], [r["E2_minus_E1_aupro_0_05"] for r in paired])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("E2 - E1 AU-PRO@0.05")
    else:
        ax.text(.5, .5, "BLOCKED: 0/8 valid paired GPU results", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title(f"Synthetic DEV — coverage {len(paired)}/8" + (" (partial; no macro)" if not complete else ""))
    fig.tight_layout()
    fig.savefig(directory / "E2_minus_E1_aupro.png", dpi=150)
    plt.close(fig)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=("E1", "E2", "E3", "E4", "E5"), default=["E1", "E2"])
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--e1-root", default="outputs/G1/E1", help="Root containing <category>/best.pt and SHA sidecar")
    parser.add_argument("--e2-root", default="outputs/G2/E2",
                        help="Root containing trained <category>/best.pt and SHA sidecar; optional until E2 training")
    parser.add_argument("--data-root")
    parser.add_argument("--repo-dir", help="Exact original official DINOv3 checkout, relocated only")
    parser.add_argument("--weights", help="Exact pretrained checkpoint, SHA must match training")
    parser.add_argument("--output-root", default="outputs/G2/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Three original DEV samples per category; separate outputs, no acceptance")
    parser.add_argument("--preflight", action="store_true", help="Validate actual checkpoint/data/protocol without inference")
    parser.add_argument("--report-only", action="store_true", help="Audit existing results without running inference")
    args = parser.parse_args(argv)
    requested = [c for group in args.categories for c in group.split(",")]
    if requested == ["all"]:
        requested = list(CATEGORIES)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(CATEGORIES):
        parser.error(f"--categories must be all or distinct canonical categories: {CATEGORIES}")
    args.categories = [c for c in CATEGORIES if c in requested]
    if len(set(args.experiments)) != len(args.experiments) or sum((args.smoke, args.preflight, args.report_only)) > 1:
        parser.error("Use distinct experiments and at most one of --smoke/--preflight/--report-only")
    args.experiments = [exp for exp in ("E1", "E2", "E3", "E4", "E5") if exp in args.experiments]
    if args.smoke or args.preflight:
        args.output_root = str(Path(args.output_root) / ("smoke" if args.smoke else "preflight"))
    output = Path(args.output_root).resolve()
    for experiment in ("E1", "E2"):
        inputs = Path(getattr(args, f"{experiment.lower()}_root")).resolve()
        if output == inputs or inputs in output.parents or (output / experiment) == inputs:
            parser.error("Evaluation output must be separate from input training artifacts")
    return args


def main(argv=None):
    args = parse_args(argv)
    directory = Path(args.output_root)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "eval.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    results = {}
    try:
        for exp in args.experiments:
            for category in CATEGORIES:
                requested = category in args.categories
                row = dict(status="NOT RUN", experiment=exp, category=category, reason="Not requested")
                try:
                    if requested and not args.report_only:
                        prior = results.get(("E1", category)) if exp == "E2" else None
                        if prior is not None and prior.get("pair_protocol_sha256"):
                            _, _, _, _, candidate = prepare_job(args, exp, category)
                            if prior["pair_protocol_sha256"] != sha256_json(candidate["scientific_protocol"]):
                                raise Blocked("E1/E2 backbone, decoder, seeds, sources, loss or training budget differs")
                        row = evaluate_category(args, exp, category)
                    elif requested and args.report_only:
                        path, _, _, _, identity = prepare_job(args, exp, category)
                        saved = completed_result(directory / exp / category, sha256_json(identity), path)
                        if saved is not None:
                            row = saved
                        elif requested:
                            row["reason"] = "No valid saved evaluation; run without --report-only"
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    row.update(status="BLOCKED" if isinstance(exc, (Blocked, OSError, KeyError)) else "FAIL", reason=str(exc))
                    logging.error("%s/%s %s: %s", exp, category, row["status"], exc)
                results[exp, category] = row
                if requested and row["status"] not in {"PASS", "SMOKE_PASS", "CPU_PASS"}:
                    # Never overwrite an old result's signed metrics on a failed attempt.
                    write_metrics_json(row, directory / exp / category / "attempt_status.json")
                    if not (directory / exp / category / "metrics.json").exists():
                        write_metrics_json(row, directory / exp / category / "metrics.json")
                if row["status"] in {"PASS", "SMOKE_PASS", "READY"}:
                    logging.info("%s/%s %s", exp, category, row["status"])
        mode = "smoke" if args.smoke else ("preflight" if args.preflight else "full")
        summary = write_comparison(directory, results, mode=mode, experiments=args.experiments, categories=args.categories)
        write_metrics_json(dict(arguments=vars(args), git_commit=git_commit(), implementation=implementation_identity(),
                                environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                                 cuda_available=torch.cuda.is_available(),
                                                 gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)),
                           directory / "run_manifest.json")
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, ensure_ascii=False))
        return 0 if summary["status"] in {"PASS", "CPU_PASS", "SMOKE_PASS", "READY"} else (1 if summary["status"] == "FAIL" else 2)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def copy_atomic(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return target


def sync_tree(source, target):
    """Copy changed files atomically, ignoring in-progress writes."""
    source, target = Path(source), Path(target)
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        destination = target / path.relative_to(source)
        if (destination.is_file() and destination.stat().st_size == path.stat().st_size
                and destination.stat().st_mtime_ns == path.stat().st_mtime_ns):
            continue
        copy_atomic(path, destination)


def good_member_path(member, category):
    parts = PurePosixPath(member.name).parts
    if (PurePosixPath(member.name).is_absolute() or ".." in parts or "\\" in member.name
            or member.issym() or member.islnk()):
        raise ValueError(f"BLOCKED: unsafe archive member: {member.name}")
    if not member.isfile():
        return None
    for index, part in enumerate(parts):
        if part.lower() in {"train", "validation"} and index + 1 < len(parts) and parts[index + 1].lower() == "good":
            prefix = parts[:index]
            if any(p in MVTEC_AD2_CATEGORIES and p != category for p in prefix):
                raise ValueError(f"BLOCKED: wrong category in {member.name}")
            return Path(category, part.upper(), "good", *parts[index + 2:])
    return None


def prepare_archives(archive_dir, archive_names, local_archives, data_root):
    """Copy ALL eight Drive archives locally before opening ANY for extraction."""
    if set(archive_names) != set(MVTEC_AD2_CATEGORIES) or len(set(archive_names.values())) != 8:
        raise ValueError("BLOCKED: specify eight distinct category archives")
    sources = {c: Path(archive_dir) / archive_names[c] for c in MVTEC_AD2_CATEGORIES}
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("BLOCKED: missing archives: " + ", ".join(missing))
    local_archives, data_root = Path(local_archives), Path(data_root)
    local_archives.mkdir(parents=True, exist_ok=True)
    if sum(p.stat().st_size for p in sources.values()) > shutil.disk_usage(local_archives).free:
        raise RuntimeError("BLOCKED: insufficient local disk for all eight archives")
    local = {c: copy_atomic(source, local_archives / source.name) for c, source in sources.items()}
    # Inspect only local archives; estimate good-only extraction disk before writes.
    total, selected = 0, {}
    for category, path in local.items():
        names, destinations = [], set()
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                relative = good_member_path(member, category)
                if relative is None:
                    continue
                if relative in destinations:
                    raise ValueError(f"BLOCKED: duplicate member destination: {relative}")
                destinations.add(relative)
                names.append((member.name, relative))
                total += member.size
        selected[category] = names
    if total > shutil.disk_usage(local_archives).free:
        raise RuntimeError("BLOCKED: insufficient local disk to extract TRAIN/VALIDATION good images")
    for category, path in local.items():
        with tarfile.open(path, "r:gz") as archive:
            for name, relative in selected[category]:
                destination = data_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                with archive.extractfile(name) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(destination)
        for split in ("TRAIN", "VALIDATION"):
            if not any(p.is_file() for p in (data_root / category / split / "good").rglob("*")):
                raise FileNotFoundError(f"BLOCKED: missing {category}/{split}/good in archive")
    return data_root


def run_logged(command, *, cwd, output_root, drive_output, stage, env=None, sync_seconds=30):
    """Mirror partial results/logs to Drive during the run and on failure."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    errors = []

    def mirror():
        while not stop.wait(sync_seconds):
            try:
                sync_tree(output_root, drive_output)
            except OSError as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=mirror, daemon=True)
    thread.start()
    process = None
    try:
        with (output_root / f"{stage}.log").open("a", encoding="utf-8") as log:
            log.write("command: " + repr(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            code = process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        stop.set()
        thread.join()
        sync_tree(output_root, drive_output)
    if errors:
        print("Drive sync recovered after transient errors:", errors[-1])
    return code


if __name__ == "__main__":
    raise SystemExit(main())

