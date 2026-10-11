"""One synthetic DEV evaluator: D6 E1/E2 and D9 E5 joint-search intake.

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
from src.train.g2_context import predict_native_e4
from src.train.g2_e2 import trainable_modules
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


def load_trained_checkpoint(path, experiment, category, *, joint_record=None, joint_spec=None):
    if experiment not in {"E1", "E2", "E5"}:
        raise Blocked(f"{experiment} is reserved; D6 implements only E1/E2; D9 accepts E5 manifests")
    if experiment == "E5" and (joint_record is None or joint_spec is None):
        raise Blocked("E5 requires an explicit joint-search manifest and declared screening protocol")
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
    elif experiment == "E2":
        if (cfg.get("stage") != "E2" or cfg.get("mode") != "full"
                or any(type(cfg.get("adapter", {}).get(k)) is not int or cfg["adapter"][k] <= 0
                       for k in ("r", "d"))
                or meta.get("config_sha256") != sha256_json(cfg)):
            raise Blocked("E2 checkpoint stage/config/r/d provenance mismatch")
        # D6 evaluates trained checkpoints; it does not certify D10 selection.
        # Legacy selection fields are preserved in the exported checkpoint config.
    else:
        from src.eval.g2_joint import validate_checkpoint_record
        try:
            validate_checkpoint_record(joint_record, payload, joint_spec)
            validate_joint_budget(path, payload)
        except (ValueError, KeyError) as exc:
            raise Blocked(str(exc)) from exc
    return payload


def validate_joint_budget(path, payload):
    """Use retained TV1 best/last/report evidence; best alone proves no full budget."""
    cfg = payload["config"]
    directory = Path(path).parent
    last, _ = load_checkpoint_payload(directory / "last.pt", require_sha256=True)
    report = json.loads((directory / "metrics.json").read_text())
    expected = cfg.get("expected_steps")
    if (type(expected) is not int or expected <= 0 or last["config"] != cfg
            or last["metadata"].get("config_sha256") != sha256_json(cfg)
            or last["training_state"]["global_step"] != expected
            or report.get("status") != "PASS" or report.get("training_complete") is not True
            or report.get("config_sha256") != sha256_json(cfg)
            or report.get("expected_steps") != expected or report.get("global_step") != expected
            or report.get("verification_scope") != "real_pretrained"
            or report.get("frozen_backbone_unchanged") is not True
            or report.get("best_checkpoint_sha256") != file_sha256(path)
            or report.get("last_checkpoint_sha256") != file_sha256(directory / "last.pt")):
        raise Blocked("Joint checkpoint lacks complete training budget/best/last provenance")


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
        backbone=dict(name=b["name"], checkpoint_sha256=b.get("checkpoint_sha256"), norm=b.get("norm", True)),
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


def checkpoint_modules(model, experiment):
    if experiment == "E1":
        return model.decoder
    if experiment == "E5":
        return nn.ModuleDict({"adapters": model.adapters, "projection": model.projection,
                              "decoder": model.decoder, "fusion": model.fusion})
    return trainable_modules(model, experiment)


class EvaluationBackboneCache:
    """Keep one frozen extractor, never a trained head, across compatible jobs."""
    def __init__(self):
        self.key, self.extractor = None, None

    def build(self, cfg, experiment, device):
        b = cfg["backbone"]
        key = (b["name"], str(Path(b["repo_dir"]).resolve()), str(Path(b["weights"]).resolve()),
               b.get("checkpoint_sha256"), sha256_json(b.get("source_sha256", {})), b.get("norm", True),
               "multilayer" if experiment in {"E3", "E4", "E5"} else "deepest", str(device))
        if key != self.key:
            # Release the previous mode before allocating the next backbone.
            self.extractor, self.key = None, None
        model = build_g2_model(cfg, experiment=experiment, root=ROOT, extractor=self.extractor)
        self.extractor = model.extractor.to(device).eval()
        self.key = key
        return model


def restore_model(payload, cfg, experiment, device, *, cache=None):
    if experiment == "E1":
        # G1 config has no Adapter section; factory requires one even for E1.
        factory_cfg = deepcopy(cfg)
        factory_cfg["adapter"] = dict(r=1, d=1, kernel_size=3, gamma_init=0., bias=True)
    else:
        factory_cfg = cfg
    try:
        model = (cache.build(factory_cfg, experiment, device) if cache is not None
                 else build_g2_model(factory_cfg, experiment=experiment, root=ROOT))
    except NotImplementedError as exc:
        raise Blocked("Runtime lacks merged E5 model; run on g2/integration with TV1 D9 code") from exc
    head = (trainable_modules(model, experiment) if payload.get("config", {}).get("mode") == "main"
            else checkpoint_modules(model, experiment))
    head.load_state_dict(payload["model_state"], strict=True)
    if not model.extractor.backbone.__class__.__module__.startswith("dinov3."):
        raise Blocked("API fixture or unofficial backbone cannot produce experimental results")
    if any(p.requires_grad for p in model.extractor.parameters()):
        raise Blocked("Backbone is not frozen")
    model.requires_grad_(False)
    return model.to(device).eval()


def implementation_identity():
    paths = {ROOT / p for p in ("scripts/eval_g2.py", "scripts/summarize_g2_joint.py",
                               "src/train/g1_e1.py", "src/train/g2_context.py", "src/train/screen_representation.py",
                               "src/train/g2_main.py", "src/train/g2_e2.py",
                               "src/utils/checkpoint.py", "src/utils/resume.py")}
    for directory in ("src/data", "src/eval", "src/metrics", "src/models", "src/geometry"):
        paths.update((ROOT / directory).rglob("*.py"))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


class CaptureNative:
    """Observe the existing evaluator's one pass; export exact maps and masks."""
    def __init__(self, native, directory, *, predict_fn=predict_native):
        self.native, self.directory, self.index, self.rows = native, Path(directory), 0, []
        self.predict_fn = predict_fn

    def __call__(self, model, image, cfg, device):
        sample = self.native[self.index]
        if not torch.equal(image, sample["image"]):
            raise RuntimeError("Native export order differs from evaluator input")
        score = self.predict_fn(model, image, cfg, device)
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
    joint_record = getattr(args, "joint_record", None)
    joint_spec = getattr(args, "joint_spec", None)
    if joint_record is not None:
        if experiment != "E5" or not joint_record.get("checkpoint"):
            raise Blocked("D10 trained E5 checkpoint missing; no random-weight fallback")
        path = Path(joint_record["checkpoint"])
    else:
        if experiment not in {"E1", "E2"}:
            raise Blocked(f"{experiment} is reserved; E5 requires --joint-manifest")
        path = Path(getattr(args, f"{experiment.lower()}_root")) / category / "best.pt"
    payload = load_trained_checkpoint(path, experiment, category, joint_record=joint_record, joint_spec=joint_spec)
    cfg, pools = relocate_config(payload["config"], args)
    budget_verified = validate_e1_budget(path, payload, cfg, pools) if experiment == "E1" else experiment == "E5"
    identity = dict(version="g2_dev_evaluator_v2", experiment=experiment, checkpoint_sha256=file_sha256(path),
                    checkpoint_config_sha256=sha256_json(payload["config"]),
                    scientific_protocol=scientific_identity(cfg), smoke=args.smoke,
                    final_training_budget_verified=budget_verified,
                    runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=args.device,
                                 gpu=torch.cuda.get_device_name(torch.device(args.device))
                                 if torch.device(args.device).type == "cuda" else None,
                                 deterministic=True, amp=False, tf32=False),
                    implementation=implementation_identity())
    if joint_record is not None:
        from src.eval.g2_joint import joint_protocol_identity
        identity["scientific_protocol"] = joint_protocol_identity(cfg)
        identity.update(joint_run_id=joint_record["run_id"], joint_spec_sha256=sha256_json(joint_spec),
                        benchmark={"requested": args.benchmark, "warmup": args.benchmark_warmup,
                                   "repeats": args.benchmark_repeats})
    return path, payload, cfg, pools, identity


def evaluate_category(args, experiment, category):
    directory = Path(getattr(args, "evaluation_directory", None) or (Path(args.output_root) / experiment / category))
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
    cache = getattr(args, "backbone_cache", None)
    model = (restore_model(payload, cfg, experiment, args.device, cache=cache) if cache is not None
             else restore_model(payload, cfg, experiment, args.device))
    native = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"], seed=cfg["training"]["dev_seed"],
                             role="dev", variants=cfg["data"]["dev_variants_per_image"], fixed=True)
    if args.smoke:
        # Keep original source index/variant seeds; normal + first two defects.
        native = torch.utils.data.Subset(native, range(min(3, len(native))))
    directory.mkdir(parents=True, exist_ok=True)
    cfg["evaluation"]["example_limit"] = 3
    predict = predict_native_e4 if experiment in {"E4", "E5"} else predict_native
    capture = CaptureNative(native, directory, predict_fn=predict)
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
    if experiment == "E5":
        from src.eval.g2_benchmark import BenchmarkUnavailable, benchmark_native
        benchmark = dict(status="NOT RUN", reason="Benchmark not requested", latency=None, peak_vram=None)
        if args.benchmark and not args.smoke:
            try:
                benchmark = benchmark_native(model, native[0]["image"], cfg, args.device,
                                             predict_fn=predict, warmup=args.benchmark_warmup,
                                             repeats=args.benchmark_repeats)
            except BenchmarkUnavailable as exc:
                benchmark.update(status="BLOCKED", reason=str(exc))
        result.update(params=sum(p.numel() for p in checkpoint_modules(model, experiment).parameters()),
                      benchmark=benchmark, run_id=args.joint_record["run_id"],
                      training_git_sha=args.joint_record["git_sha"])
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


def audit_joint_row(row, spec):
    """Audit real checkpoint and signed D6 native exports; never trust a CSV score."""
    from src.eval.evaluator import build_anomaly_map_qa_report
    from src.eval.g2_joint import METRIC_FIELDS, METRIC_KEYS, joint_protocol_identity

    payload = load_trained_checkpoint(row["checkpoint"], "E5", row["category"],
                                      joint_record=row, joint_spec=spec)
    if row["checkpoint_sha256"] is not None and row["checkpoint_sha256"] != file_sha256(row["checkpoint"]):
        raise Blocked("Manifest checkpoint checksum differs")
    if row["status"] != "PASS":
        return
    directory = Path(row["evaluation_json"]).parent
    if Path(row["evaluation_json"]).name != "metrics.json":
        raise Blocked("evaluation_json must point to evaluator metrics.json")
    exported_config = json.loads((directory / "evaluation_config.json").read_text())
    identity = exported_config["identity"]
    if (identity.get("joint_run_id") != row["run_id"]
            or identity.get("joint_spec_sha256") != sha256_json(spec)
            or identity.get("checkpoint_config_sha256") != row["config_hash"]):
        raise Blocked("Evaluation identity differs from manifest/search protocol")
    result = completed_result(directory, sha256_json(identity), row["checkpoint"])
    if (result is None or result["status"] != "PASS" or result.get("acceptance_eligible") is not True
            or result.get("verification_scope") != "real_pretrained"
            or torch.device(result.get("device", "cpu")).type != "cuda"
            or result.get("experiment") != "E5" or result.get("category") != row["category"]
            or result.get("run_id") != row["run_id"] or result.get("training_git_sha") != row["git_sha"]
            or result.get("final_training_budget_verified") is not True):
        raise Blocked("No valid real-pretrained CUDA evaluation/native exports")
    if (row["protocol_hash"] != sha256_json(joint_protocol_identity(payload["config"]))
            or row["protocol_hash"] != result["pair_protocol_sha256"]
            or row["dev_hash"] != result["dev_samples_sha256"]
            or row["params"] != result.get("params")
            or any(row[key] != result[metric] for key, metric in zip(METRIC_FIELDS, METRIC_KEYS))):
        raise Blocked("CSV metrics/params/protocol differ from audited evaluation")
    native_rows = json.loads((directory / "native_maps.json").read_text())
    for native_row in native_rows:
        record = dict(anomaly_map=np.load(directory / native_row["score_path"], mmap_mode="r"),
                      gt_mask=np.load(directory / native_row["mask_path"], mmap_mode="r"),
                      meta=dict(image_id=native_row["sample_id"], category=row["category"], split="dev_synthetic",
                                original_hw=native_row["original_hw"], anomaly_map_space="original_image",
                                gt_mask_space="original_image"))
        if build_anomaly_map_qa_report([record])["summary"]["status"] != "PASS":
            raise Blocked("Native score/mask QA failed")
    benchmark = result.get("benchmark", {})
    if row["benchmark_status"] != benchmark.get("status", "NOT RUN"):
        raise Blocked("Benchmark status differs from evaluation")
    if row["latency"] is not None or row["peak_vram"] is not None:
        import statistics
        if (benchmark.get("status") != "PASS" or benchmark.get("cpu_fixture") is not False
                or torch.device(benchmark.get("device", "cpu")).type != "cuda" or not benchmark.get("gpu")
                or benchmark.get("native_hw") != native_rows[0]["original_hw"]
                or benchmark.get("latency_unit") != "milliseconds" or benchmark.get("peak_vram_unit") != "allocated_bytes"
                or row["latency"] != benchmark.get("latency") or row["peak_vram"] != benchmark.get("peak_vram")
                or len(benchmark.get("samples_ms", [])) != benchmark.get("repeats")
                or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in benchmark["samples_ms"])
                or statistics.median(benchmark["samples_ms"]) != row["latency"]):
            raise Blocked("Missing or incompatible real GPU benchmark evidence")


def joint_result_row(row, result, directory):
    """Only eligible full CUDA metrics enter the scientific joint-search CSV."""
    from src.eval.g2_joint import METRIC_FIELDS, METRIC_KEYS

    output = deepcopy(row)
    output.update(status=result["status"], reason=result.get("reason"),
                  evaluation_json=str((Path(directory) / "metrics.json").resolve()),
                  benchmark_status=result.get("benchmark", {}).get("status", "NOT RUN"))
    output.update(dict.fromkeys((*METRIC_FIELDS, "latency", "peak_vram")))
    if result["status"] == "PASS" and result.get("acceptance_eligible") is True:
        validate_metrics(result)
        output.update({key: result[metric] for key, metric in zip(METRIC_FIELDS, METRIC_KEYS)})
        output.update(checkpoint_sha256=result["checkpoint_sha256"], protocol_hash=result["pair_protocol_sha256"],
                      dev_hash=result["dev_samples_sha256"], params=result["params"])
        benchmark = result.get("benchmark", {})
        if benchmark.get("status") == "PASS":
            output.update(latency=benchmark["latency"], peak_vram=benchmark["peak_vram"])
    elif result["status"] == "PASS":
        raise Blocked("PASS without acceptance eligibility cannot enter joint manifest")
    return output


def run_joint(args):
    """D9 skeleton or D10 checkpoint intake, using this evaluator's one pipeline."""
    from src.eval.g2_joint import load_spec, read_manifest, skeleton, write_manifest, write_reports

    directory = Path(args.output_root)
    directory.mkdir(parents=True, exist_ok=True)
    if args.init_joint:
        if (directory / "joint_search_360.csv").exists():
            raise Blocked("Existing joint manifest preserved; use --joint-manifest and --resume/--report-only")
        try:
            integration_sha = subprocess.check_output(
                ["git", "rev-parse", args.integration_ref], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            integration_sha = None
        spec = load_spec(args.joint_model_config, args.joint_runner_config, root=ROOT, integration_sha=integration_sha)
        rows = skeleton(spec)
    else:
        source = Path(args.joint_manifest).resolve()
        spec = json.loads((source.parent / "joint_search_spec.json").read_text())
        rows = read_manifest(source, spec)
    if args.r is not None and [args.r, args.d] not in spec["pairs"]:
        raise Blocked("Requested r/d outside declared search")
    if not args.init_joint:
        backbone_cache = EvaluationBackboneCache()
        for i, row in enumerate(rows):
            if (row["category"] not in args.categories or args.r is not None and (row["r"], row["d"]) != (args.r, args.d)
                    or args.fusions is not None and row["fusion"] not in args.fusions):
                continue
            job = deepcopy(args)
            job.backbone_cache = backbone_cache
            job.joint_record, job.joint_spec = row, spec
            output = directory / "E5" / f"r{row['r']}_d{row['d']}" / row["fusion"] / row["category"]
            job.evaluation_directory = str(output)
            try:
                if args.report_only:
                    audit_joint_row(row, spec)
                    continue
                if spec["budget"] is None:
                    raise Blocked(spec["budget_reason"])
                if not row["checkpoint"]:
                    raise Blocked("D10 training checkpoint missing")
                if output.resolve() == Path(row["checkpoint"]).resolve().parent:
                    raise Blocked("Evaluation output must be separate from training artifacts")
                result = evaluate_category(job, "E5", row["category"])
                rows[i] = joint_result_row(row, result, output)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                blocked = deepcopy(row)
                blocked.update(status="BLOCKED" if isinstance(exc, (Blocked, OSError, KeyError)) else "FAIL",
                               reason=str(exc), benchmark_status="NOT RUN")
                from src.eval.g2_joint import METRIC_FIELDS
                blocked.update(dict.fromkeys((*METRIC_FIELDS, "latency", "peak_vram")))
                rows[i] = blocked
                write_metrics_json(blocked, output / "attempt_status.json")
            # Preserve each finished attempt for interruption/resume; never train.
            write_manifest(directory / "joint_search_360.csv", rows, spec)
            write_metrics_json(spec, directory / "joint_search_spec.json")
    coverage = write_reports(rows, spec, directory, audit=lambda row: audit_joint_row(row, spec))
    write_metrics_json(dict(arguments=vars(args), git_commit=git_commit(),
                            integration_sha=spec.get("integration_sha"), torch=str(torch.__version__),
                            cuda_available=torch.cuda.is_available(),
                            mode="skeleton" if args.init_joint else "checkpoint_intake",
                            training_performed=False, metrics_from_untrained_models=False), directory / "joint_search_run_manifest.json")
    print(json.dumps({k: v for k, v in coverage.items() if k != "records"}, indent=2))
    if args.init_joint:
        return 0  # Skeleton CODE success; scientific coverage remains explicit.
    return 0 if coverage["status"] == "PASS" else 2


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
    parser.add_argument("--final", action="store_true", help="D11: audit main exports and report all 80 logical cells")
    parser.add_argument("--joint-lock", help="D11 existing immutable D10 lock; never created here")
    parser.add_argument("--tv1-manifest", help="Original D10 TV1 180-cell evidence manifest")
    parser.add_argument("--tv2-manifest", help="Original D10 TV2 180-cell evidence manifest")
    parser.add_argument("--main-tv1-manifest", help="D11 TV1 main_training_36.csv")
    parser.add_argument("--main-tv2-manifest", help="D11 TV2 main_training_36.csv")
    parser.add_argument("--e0-root", default="outputs/G2/E0")
    parser.add_argument("--init-joint", action="store_true", help="Write a 360-cell D9 skeleton; no inference/training")
    parser.add_argument("--joint-manifest", help="Versioned E5 45x8 CSV; sibling joint_search_spec.json required")
    parser.add_argument("--joint-model-config", default="configs/g2_experiments.yaml")
    parser.add_argument("--joint-runner-config", default="configs/g2_runner.yaml")
    parser.add_argument("--integration-ref", default="g2/integration", help="Record inspected TV1 integration commit")
    parser.add_argument("--r", type=int, help="Filter joint candidates (requires --d)")
    parser.add_argument("--d", type=int, help="Filter joint candidates (requires --r)")
    parser.add_argument("--fusions", nargs="+", choices=("mean", "weighted_sum", "concat", "gated", "attention"))
    parser.add_argument("--benchmark", action="store_true", help="CUDA native-inference latency/VRAM, never CPU proxy")
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
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
    joint = args.init_joint or args.joint_manifest is not None
    if args.init_joint and (args.joint_manifest or args.smoke or args.preflight or args.report_only or args.resume or args.benchmark):
        parser.error("--init-joint only writes skeleton reports; incompatible execution flags")
    if (args.r is None) != (args.d is None) or any(v is not None and v <= 0 for v in (args.r, args.d)):
        parser.error("--r and --d must be positive integers supplied together")
    if not joint and not args.final and any((args.r is not None, args.fusions is not None, args.benchmark)):
        parser.error("Joint candidate/benchmark flags require --joint-manifest")
    final_inputs = (args.joint_lock, args.tv1_manifest, args.tv2_manifest,
                    args.main_tv1_manifest, args.main_tv2_manifest)
    if not args.final and any(final_inputs):
        parser.error("D11 manifest/lock flags require --final")
    if args.final:
        if joint or args.smoke or args.preflight or args.r is not None or args.fusions is not None:
            parser.error("--final audits locked main exports; no screening/smoke/candidate selection")
        if args.report_only and args.benchmark:
            parser.error("--report-only cannot execute a GPU benchmark")
        if args.output_root == "outputs/G2/evaluation":
            args.output_root = "outputs/G2/D11/final"
        for path in final_inputs[1:] + (args.e0_root,):
            if path:
                input_root = Path(path).resolve() if path == args.e0_root else Path(path).resolve().parent
                out = Path(args.output_root).resolve()
                if out == input_root or input_root in out.parents or out in input_root.parents:
                    parser.error("Final output must be separate from input main/E0 evidence")
    if args.benchmark_warmup < 0 or args.benchmark_repeats < 1:
        parser.error("Benchmark warmup >= 0 and repeats >= 1")
    if joint:
        args.experiments = ["E5"]
        if args.output_root == "outputs/G2/evaluation":
            args.output_root = "outputs/G2/D9"
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
    if args.final:
        from src.eval.g2_final import run_final
        return run_final(args)
    if args.init_joint or args.joint_manifest:
        try:
            return run_joint(args)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            status = "BLOCKED" if isinstance(exc, (Blocked, OSError, KeyError)) else "FAIL"
            print(json.dumps(dict(status=status, reason=str(exc)), indent=2))
            return 2 if status == "BLOCKED" else 1
    directory = Path(args.output_root)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "eval.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    results = {}
    args.backbone_cache = EvaluationBackboneCache()
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
        write_metrics_json(dict(arguments={k: v for k, v in vars(args).items() if k != "backbone_cache"}, git_commit=git_commit(), implementation=implementation_identity(),
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


def good_member_path(member, category, *, include_test_public=False):
    parts = PurePosixPath(member.name).parts
    if (PurePosixPath(member.name).is_absolute() or ".." in parts or "\\" in member.name
            or member.issym() or member.islnk()):
        raise ValueError(f"BLOCKED: unsafe archive member: {member.name}")
    if not member.isfile():
        return None
    if include_test_public:
        # Preserve the public split's defect/GT hierarchy for the existing loader.
        for index, part in enumerate(parts):
            if part.lower() == 'test_public':
                prefix = parts[:index]
                if any(p in MVTEC_AD2_CATEGORIES and p != category for p in prefix):
                    raise ValueError(f'BLOCKED: wrong category in {member.name}')
                if any(p.lower() in {'ground_truth', 'masks', 'mask', 'gt'} for p in prefix):
                    return Path(category, 'ground_truth', 'TEST_PUBLIC', *parts[index + 1:])
                return Path(category, 'TEST_PUBLIC', *parts[index + 1:])
    for index, part in enumerate(parts):
        if part.lower() in {"train", "validation"} and index + 1 < len(parts) and parts[index + 1].lower() == "good":
            prefix = parts[:index]
            if any(p in MVTEC_AD2_CATEGORIES and p != category for p in prefix):
                raise ValueError(f"BLOCKED: wrong category in {member.name}")
            return Path(category, part.upper(), "good", *parts[index + 2:])
    return None


def prepare_archives(archive_dir, archive_names, local_archives, data_root, *, include_test_public=False):
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
                relative = good_member_path(member, category, include_test_public=include_test_public)
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

