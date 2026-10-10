"""E0 orchestration: original TRAIN/good memory, fixed synthetic DEV only.

Run ``python scripts/run_g2_e0.py --help``. This baseline has no optimizer,
Adapter or Decoder. Its checkpoints contain sampled normal vectors, not DINO.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from src.data.loader import G1NativeDataset, load_rgb_native, normalize_dinov3, scan_mvtec_ad2
from src.data.synthetic_anomaly import validate_native_protocol
from src.data.tiling import crop_with_padding, generate_tile_records, stitch_tiles_hann
from src.eval.e0_memory import (DISTANCES, E0MemoryModel, NormalFeatureMemoryBank,
                                StreamingNormalMemory, distance_to_score)
from src.eval.evaluator import (MVTEC_AD2_CATEGORIES, build_anomaly_map_qa_report,
                                write_metrics_json)
from src.eval.full_scale import DEVMetricAccumulator
from src.models.backbone_registry import BACKBONES, backbone_spec
from src.models.dinov3_extractor import DINOv3FeatureExtractor

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "msila.g2.e0.v1"
SCORE_TRANSFORM = "distance_over_distance_plus_scale"


class E0Blocked(ValueError):
    """Existing artifacts cannot safely be resumed or overwritten."""


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/g2_e0.yaml"))
    parser.add_argument("--categories", nargs="+", default=["all"], help="all, names or comma-separated names")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    for flag in ("data-root", "repo-dir", "weights", "output-root"):
        parser.add_argument("--" + flag)
    parser.add_argument("--distance", choices=DISTANCES)
    parser.add_argument("--max-bank-features", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse verified bank and completed DEV metrics")
    parser.add_argument("--smoke", action="store_true", help="Reduced budget under output_root/smoke; never full coverage")
    parser.add_argument("--save-maps", action="store_true", help="Save native score/GT arrays for evaluator audit")
    return parser.parse_args(argv)


def requested_categories(tokens):
    tokens = [part for token in tokens for part in token.split(",")]
    if tokens == ["all"]:
        return list(MVTEC_AD2_CATEGORIES)
    if not tokens or len(set(tokens)) != len(tokens) or set(tokens) - set(MVTEC_AD2_CATEGORIES):
        raise ValueError(f"Categories must be distinct members of {MVTEC_AD2_CATEGORIES}, or all")
    return [category for category in MVTEC_AD2_CATEGORIES if category in tokens]


def resolve_config(args):
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    if cfg.get("version") != "g2_e0_v1" or tuple(cfg.get("categories", ())) != MVTEC_AD2_CATEGORIES:
        raise ValueError("E0 config must declare version g2_e0_v1 and all eight canonical categories")
    requested_categories(args.categories)
    for flag, section, key in (("data_root", "data", "root"), ("repo_dir", "backbone", "repo_dir"),
                               ("output_root", None, "output_root"), ("backbone", "backbone", "name"),
                               ("distance", "memory", "distance"), ("max_bank_features", "memory", "max_features")):
        if getattr(args, flag) is not None:
            (cfg if section is None else cfg[section])[key] = getattr(args, flag)
    cfg["mode"] = "smoke" if args.smoke else "full"
    if args.smoke:
        cfg["memory"]["max_features"] = cfg["smoke"]["max_features"]
        for key in ("max_train_sources", "max_dev_sources", "dev_variants_per_image"):
            cfg["data"][key] = cfg["smoke"][key]
        cfg["output_root"] = str(Path(cfg["output_root"]) / "smoke")
    if args.save_maps:
        cfg["evaluation"]["save_maps"] = True
    name = cfg["backbone"]["name"]
    backbone_spec(name)
    weights = args.weights or cfg["backbone"]["checkpoints"].get(name)
    if not weights:
        raise ValueError(f"No local checkpoint declared for {name}; use --weights")
    def absolute(value):
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else ROOT / path).resolve())
    cfg["data"]["root"] = absolute(cfg["data"]["root"])
    cfg["backbone"]["repo_dir"] = absolute(cfg["backbone"]["repo_dir"])
    cfg["backbone"]["weights"] = absolute(weights)
    cfg["output_root"] = absolute(cfg["output_root"])
    cfg["synthetic_protocol"] = validate_native_protocol(yaml.safe_load(
        Path(absolute(cfg["synthetic_protocol"])).read_text()))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    data, memory, evaluation = cfg["data"], cfg["memory"], cfg["evaluation"]
    if data["train_split"] != "TRAIN/good" or data["dev_split"] != "VALIDATION/good":
        raise ValueError("E0 uses original TRAIN/good and fixed synthetic VALIDATION/good only")
    if data["tile_size"] != 512 or type(data["overlap"]) is not int or not 0 <= data["overlap"] < 512:
        raise ValueError("E0 requires 512px tiles and overlap in [0,512)")
    for section, keys in ((memory, ("max_features", "tile_batch_size", "query_chunk_size", "bank_chunk_size")),
                          (evaluation, ("tile_batch_size",)), (data, ("dev_variants_per_image",))):
        for key in keys:
            if type(section[key]) is not int or section[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    for key in ("max_train_sources", "max_dev_sources"):
        if data[key] is not None and (type(data[key]) is not int or data[key] < 1):
            raise ValueError(f"{key} must be positive or null")
    if memory["distance"] not in DISTANCES or type(memory["normalize"]) is not bool:
        raise ValueError("Declare a supported distance and boolean feature normalization")
    for value in (memory["sampling_seed"], evaluation["dev_seed"]):
        if type(value) is not int or value < 0:
            raise ValueError("Seeds must be nonnegative integers")
    if memory["sampling_seed"] == evaluation["dev_seed"]:
        raise ValueError("Memory sampling and DEV must use independent seeds")
    if (evaluation["test_for_tuning"] is not False or evaluation["max_fpr"] != .05
            or evaluation["score_transform"] != SCORE_TRANSFORM):
        raise ValueError("E0 locks AU-PRO@0.05 and the fixed distance transform; no TEST tuning")
    if type(evaluation["save_maps"]) is not bool or type(cfg["backbone"]["norm"]) is not bool:
        raise ValueError("save_maps and backbone.norm must be booleans")
    if cfg["mode"] not in {"full", "smoke"}:
        raise ValueError("Expected full or smoke mode")
    distance_to_score(torch.zeros(1), scale=evaluation["score_scale"])
    validate_native_protocol(cfg["synthetic_protocol"])


def discover_sources(cfg, category):
    """No TEST records are selected, hashed, decoded or used for tuning."""
    pools, manifest = {}, {}
    for role, split in (("train", "train"), ("dev", "validation")):
        records = [record for record in scan_mvtec_ad2(cfg["data"]["root"], split=split, categories=[category])
                   if record.defect_type == "good"]
        if not records:
            raise FileNotFoundError(f"{category}: no {split}/good sources")
        pools[role] = records
        manifest[role] = [dict(path=record.image_path, sha256=file_sha256(record.image_path)) for record in records]
    # Check the entire good pools for leakage before applying any budget cap.
    if {row["sha256"] for row in manifest["train"]} & {row["sha256"] for row in manifest["dev"]}:
        raise ValueError(f"{category}: TRAIN/DEV source leakage")
    for role in pools:
        limit = cfg["data"][f"max_{role}_sources"]
        if limit is not None:
            pools[role], manifest[role] = pools[role][:limit], manifest[role][:limit]
    return pools, manifest


def category_seed(seed, category):
    return int.from_bytes(hashlib.sha256(f"e0|{seed}|{category}".encode()).digest()[:8], "big")


def _tile_batches(image, cfg, batch_size):
    size, overlap = cfg["data"]["tile_size"], cfg["data"]["overlap"]
    records = generate_tile_records(*image.shape[-2:], local_size=size, overlap=overlap, context_size=size)
    for start in range(0, len(records), batch_size):
        group = records[start:start+batch_size]
        yield group, torch.stack([normalize_dinov3(crop_with_padding(image, r.local_xyxy)) for r in group])


@torch.no_grad()
def fit_normal_memory(extractor, sources, cfg, category, device):
    if not sources or any(record.split != "train" or record.defect_type != "good"
                          or record.category != category for record in sources):
        raise ValueError("Normal memory accepts this category's original TRAIN/good only")
    if extractor.blocks != (extractor.depth,):
        raise ValueError("Normal memory requires only the deepest feature")
    extractor.requires_grad_(False)
    extractor.eval()
    seed = category_seed(cfg["memory"]["sampling_seed"], category)
    sampler = StreamingNormalMemory(cfg["memory"]["max_features"], seed=seed)
    source_ranges = []
    key = f"b{extractor.depth}"
    for record in sources:
        # Never instantiate the synthetic generator on TRAIN images.
        image, start = load_rgb_native(record.image_path), sampler.seen
        for group, tiles in _tile_batches(image, cfg, cfg["memory"]["tile_batch_size"]):
            features = extractor(tiles.to(device))
            if set(features) != {key}:
                raise ValueError("E0 must receive exactly one deep feature")
            feature = features[key]
            if (feature.ndim != 4 or feature.shape[0] != len(group)
                    or feature.shape[1] != extractor.out_channels
                    or tuple(feature.shape[-2:]) != (cfg["data"]["tile_size"] // extractor.patch_size,) * 2):
                raise ValueError("Deep feature shape violates the backbone contract")
            sampler.update(feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1]))
        source_ranges.append(dict(path=record.image_path, patch_start=start, patch_end=sampler.seen))
    vectors, indices = sampler.finalize()
    return dict(schema=SCHEMA, category=category, vectors=vectors, sampled_indices=indices,
                patches_seen=sampler.seen, sampling_seed=seed, source_ranges=source_ranges,
                channels=extractor.out_channels, deep_block=extractor.depth)


@torch.no_grad()
def predict_native_e0(model, image, cfg, device):
    model.eval()
    maps, records = [], []
    size = cfg["data"]["tile_size"]
    for group, tiles in _tile_batches(image, cfg, cfg["evaluation"]["tile_batch_size"]):
        score = model(tiles.to(device)).detach().float().cpu()
        if tuple(score.shape) != (len(group), 1, size, size):
            raise ValueError("E0 tile scores must be [B,1,512,512]")
        maps.extend(score[:, 0].unbind())
        records.extend(group)
    score = stitch_tiles_hann(maps, records, tuple(image.shape[-2:]), local_size=size)
    if not torch.isfinite(score).all() or (score < -1e-6).any() or (score > 1 + 1e-6).any():
        raise ValueError("Native E0 score map must be finite in [0,1]")
    return score.clamp(0, 1)


def evaluate_dev(model, sources, cfg, category, device, output_dir):
    protocol = cfg["synthetic_protocol"]
    native = G1NativeDataset(sources, protocol, seed=cfg["evaluation"]["dev_seed"],
                             role="dev", variants=cfg["data"]["dev_variants_per_image"], fixed=True)
    mixed = DEVMetricAccumulator(protocol, per_region=False, disk_backed=True)
    tiny = DEVMetricAccumulator(protocol, per_region=False, disk_backed=True)
    qa_rows, samples = [], []
    try:
        for index in range(len(native)):
            sample = native[index]
            score = predict_native_e0(model, sample["image"], cfg, device).numpy()
            mask, meta = sample["mask"][0].numpy(), sample["meta"]
            qa = build_anomaly_map_qa_report([dict(anomaly_map=score, gt_mask=mask, meta=dict(
                image_id=meta["sample_id"], category=category, split="dev_synthetic",
                original_hw=meta["original_hw"], anomaly_map_space="original_image",
                gt_mask_space="original_image"))])
            if qa["summary"]["status"] != "PASS":
                raise ValueError(f"Native E0 map QA failed: {qa}")
            qa_rows.extend(qa["per_sample"])
            mixed.add(score, mask, split="dev_mixed", image_id=meta["sample_id"])
            synthetic = meta["synthetic"]
            if not synthetic["is_anomaly"] or synthetic["size_bin"] in protocol["dev_tiny_bins"]:
                tiny.add(score, mask, split="dev_tiny", image_id=meta["sample_id"])
            samples.append(meta)
            if cfg["evaluation"]["save_maps"]:
                folder = Path(output_dir) / "maps" / meta["sample_id"]
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / "anomaly_map.npy", score, allow_pickle=False)
                np.save(folder / "gt_mask.npy", mask.astype(np.uint8), allow_pickle=False)
                write_metrics_json(meta, folder / "metadata.json")
        # Existing full-scale evaluator -> existing exact aupro_from_parts.
        # Consume only group metrics: its generic sigmoid label does not apply to E0.
        mixed_result, tiny_result = mixed.result()["groups"]["all"], tiny.result()["groups"]["all"]
    finally:
        for accumulator in (mixed, tiny):
            if accumulator._scratch is not None:
                accumulator._scratch.cleanup()
    value = mixed_result["aupro_0_05"]
    if value is None or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Synthetic DEV contains no valid AU-PRO result")
    write_metrics_json(dict(status="PASS", per_sample=qa_rows), Path(output_dir) / "qa_report.json")
    write_metrics_json(dict(split="dev_synthetic", samples=samples), Path(output_dir) / "dev_manifest.json")
    return dict(dataset="mvtec_ad2_good_synthetic", split="dev_synthetic", native_resolution=True,
                interpretation="Synthetic DEV only; no real anomaly performance claim",
                score_orientation="higher_is_more_anomalous", score_transform=SCORE_TRANSFORM,
                score_scale=cfg["evaluation"]["score_scale"], aupro_max_fpr=.05,
                score_interpretation="bounded distance score, not calibrated probability",
                n_samples=len(native), synthetic_dev_aupro_0_05=value,
                dev_mixed=mixed_result, dev_tiny=tiny_result, qa_status="PASS", qa_samples=len(qa_rows))


def asset_reasons(cfg, device):
    reasons = []
    if not Path(cfg["data"]["root"]).is_dir():
        reasons.append(f"Dataset missing: {cfg['data']['root']}")
    if not (Path(cfg["backbone"]["repo_dir"]) / "hubconf.py").is_file():
        reasons.append(f"Official DINOv3 checkout missing: {cfg['backbone']['repo_dir']}")
    if not Path(cfg["backbone"]["weights"]).is_file():
        reasons.append(f"Pretrained checkpoint missing: {cfg['backbone']['weights']}")
    target = torch.device(device)
    bank_device = torch.device(cfg["memory"]["device"])
    if (target.type == "cuda" or bank_device.type == "cuda") and not torch.cuda.is_available():
        reasons.append("CUDA unavailable")
    elif cfg["mode"] == "full" and target.type != "cuda":
        reasons.append("Full eight-category DEV requires CUDA; CPU is supported for --smoke")
    return reasons


def experiment_identity(cfg):
    settings = copy.deepcopy(cfg)
    for key in ("output_root", "smoke"):
        settings.pop(key, None)
    settings["backbone"].pop("checkpoints", None)
    files = [*sorted((ROOT / "src/data").glob("*.py")), *sorted((ROOT / "src/eval").glob("*.py")),
             *sorted((ROOT / "src/metrics").glob("*.py")), ROOT / "src/models/dinov3_extractor.py",
             ROOT / "src/models/backbone_registry.py", ROOT / "scripts/run_g2_e0.py"]
    code = {str(path.relative_to(ROOT)): file_sha256(path) for path in files}
    repo = Path(cfg["backbone"]["repo_dir"])
    producer = {str(path.relative_to(repo)): file_sha256(path) for path in sorted(repo.rglob("*.py"))}
    return dict(config_hash=object_sha256(settings), code_hash=object_sha256(code),
                producer_hash=object_sha256(producer), weights_sha256=file_sha256(cfg["backbone"]["weights"]),
                torch_version=str(torch.__version__))


def save_memory(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    sha = file_sha256(path)
    write_metrics_json(dict(schema=SCHEMA, sha256=sha), path.with_suffix(".sha256.json"))
    return sha


def load_memory(path, signature, cfg, category):
    path = Path(path)
    try:
        sha = file_sha256(path)
        if json.loads(path.with_suffix(".sha256.json").read_text())["sha256"] != sha:
            raise E0Blocked("Memory bank checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema"] != SCHEMA or payload["category"] != category or payload["signature"] != signature:
            raise E0Blocked("Memory bank config/source/producer identity mismatch")
        vectors, indices = payload["vectors"], payload["sampled_indices"]
        spec = backbone_spec(cfg["backbone"]["name"])
        if (not isinstance(vectors, torch.Tensor) or vectors.dtype != torch.float32
                or tuple(vectors.shape) != (min(payload["patches_seen"], cfg["memory"]["max_features"]), spec.channels)
                or not torch.isfinite(vectors).all() or payload["channels"] != spec.channels
                or payload["deep_block"] != spec.depth or payload["patches_seen"] < 1
                or indices.dtype != torch.int64 or indices.shape != (len(vectors),)
                or (indices < 0).any() or (indices >= payload["patches_seen"]).any()
                or (indices[1:] <= indices[:-1]).any()):
            raise E0Blocked("Invalid normal memory checkpoint tensors")
        return payload, sha
    except E0Blocked:
        raise
    except Exception as exc:
        raise E0Blocked(f"Unreadable memory bank checkpoint: {exc}") from exc


def run_category(cfg, category, device, identity, *, resume=False, extractor_factory=DINOv3FeatureExtractor):
    pools, manifest = discover_sources(cfg, category)
    signature = dict(identity, source_hash=object_sha256(manifest))
    output = Path(cfg["output_root"]) / category
    output.mkdir(parents=True, exist_ok=True)
    metrics_path, bank_path = output / "metrics.json", output / "normal_memory.pt"
    previous = None
    if metrics_path.is_file():
        try:
            previous = json.loads(metrics_path.read_text())
            if not isinstance(previous, dict):
                raise ValueError("Expected a metrics object")
        except ValueError as exc:
            raise E0Blocked(f"Unreadable existing metrics: {exc}") from exc
    if previous and previous.get("status") == "PASS":
        if not resume:
            raise E0Blocked("Completed E0 exists; use --resume to verify and skip it")
        _, sha = load_memory(bank_path, signature, cfg, category)
        value = previous.get("synthetic_dev_aupro_0_05")
        if (previous.get("signature") != signature or previous.get("bank_sha256") != sha
                or previous.get("schema") != SCHEMA or previous.get("category") != category
                or previous.get("qa_status") != "PASS" or previous.get("split") != "dev_synthetic"
                or previous.get("mode") != cfg["mode"] or type(value) not in (float, int)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise E0Blocked("Completed metrics do not match the verified bank/config or are invalid")
        for filename, checksum in previous.get("evaluation_artifacts", {}).items():
            if not (output / filename).is_file() or file_sha256(output / filename) != checksum:
                raise E0Blocked(f"Evaluation artifact mismatch: {filename}")
        if set(previous.get("evaluation_artifacts", {})) != {"qa_report.json", "dev_manifest.json"}:
            raise E0Blocked("Completed metrics have no verifiable DEV/QA provenance")
        return dict(previous, resumed=True, skipped=True)
    if bank_path.exists() and not resume:
        raise E0Blocked("Memory bank already exists; use --resume to verify and continue DEV")
    start = time.monotonic()
    extractor = extractor_factory(cfg["backbone"]["repo_dir"], cfg["backbone"]["weights"],
                                   cfg["backbone"]["name"], norm=cfg["backbone"]["norm"],
                                   feature_mode="deepest", check_finite=True).to(device)
    scope = "pretrained_dinov3" if type(extractor.backbone).__module__.startswith("dinov3.") else "fixture"
    if cfg["mode"] == "full" and scope != "pretrained_dinov3":
        raise ValueError("Full E0 requires an official pretrained DINOv3 implementation")
    frozen_hash = module_sha256(extractor)
    if bank_path.exists():
        payload, bank_sha = load_memory(bank_path, signature, cfg, category)
    else:
        payload = fit_normal_memory(extractor, pools["train"], cfg, category, device)
        payload["signature"] = signature
        bank_sha = save_memory(payload, bank_path)
    memory = cfg["memory"]
    bank = NormalFeatureMemoryBank(payload["vectors"], distance=memory["distance"], normalize=memory["normalize"],
                                   query_chunk_size=memory["query_chunk_size"], bank_chunk_size=memory["bank_chunk_size"],
                                   device=memory["device"])
    model = E0MemoryModel(extractor, bank, score_scale=cfg["evaluation"]["score_scale"])
    metrics = evaluate_dev(model, pools["dev"], cfg, category, device, output)
    unchanged = module_sha256(extractor) == frozen_hash
    if not unchanged or any(p.requires_grad or p.grad is not None for p in extractor.parameters()):
        raise ValueError("E0 backbone must remain frozen and unchanged")
    result = dict(metrics, schema=SCHEMA, category=category, status="PASS", mode=cfg["mode"],
                  verification_scope=scope, signature=signature, bank_sha256=bank_sha,
                  backbone=cfg["backbone"]["name"], channels=payload["channels"], deep_block=payload["deep_block"],
                  n_memory_features=len(payload["vectors"]), patches_seen=payload["patches_seen"],
                  sampling_seed=payload["sampling_seed"], distance=memory["distance"], normalize=memory["normalize"],
                  train_sources=len(pools["train"]), dev_sources=len(pools["dev"]), source_manifest=manifest,
                  frozen_backbone_unchanged=unchanged, trainable_parameters=0, device=str(device),
                  elapsed_seconds=time.monotonic()-start, resumed=resume, skipped=False,
                  evaluation_artifacts={name: file_sha256(output / name) for name in ("qa_report.json", "dev_manifest.json")})
    write_metrics_json(cfg, output / "resolved_config.json")
    write_metrics_json(result, metrics_path)
    return result


def _status_result(cfg, category, status, reason):
    result = dict(schema=SCHEMA, category=category, status=status, mode=cfg["mode"],
                  verification_scope="not_run", reason=reason, split="dev_synthetic")
    path = Path(cfg["output_root"]) / category / "metrics.json"
    # Preserve completed evidence when assets/configs disappear or resume is blocked.
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
        except ValueError:
            return result  # Preserve corrupt evidence; report BLOCKED in the summary.
        if not isinstance(previous, dict) or previous.get("status") == "PASS":
            return result
    write_metrics_json(result, path)
    return result


def write_summary(cfg, results):
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for category in MVTEC_AD2_CATEGORIES:
        row = results.get(category, dict(category=category, status="NOT RUN", reason="Category not requested"))
        rows.append(row)
    fields = ("category", "status", "mode", "verification_scope", "synthetic_dev_aupro_0_05", "n_samples",
              "n_memory_features", "bank_sha256", "reason")
    path = output / "e0_metrics_8categories.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    passed = [row for row in rows if row.get("status") == "PASS"]
    real = [row for row in passed if row.get("mode") == "full" and row.get("verification_scope") == "pretrained_dinov3"]
    failed = any(row["status"] == "FAIL" for row in rows)
    blocked = any(row["status"] == "BLOCKED" for row in rows)
    status = ("FAIL" if failed else "BLOCKED" if blocked else
              "SMOKE PASS" if cfg["mode"] == "smoke" and passed else
              "PASS" if len(real) == 8 else "PARTIAL" if real else "NOT RUN")
    summary = dict(schema=SCHEMA, status=status, real_categories_pass=len(real), expected_categories=8,
                   smoke_categories_pass=len(passed) if cfg["mode"] == "smoke" else 0,
                   synthetic_dev_only=True, test_used_for_tuning=False, categories=rows,
                   macro_synthetic_dev_aupro_0_05=(sum(row["synthetic_dev_aupro_0_05"] for row in real)/8
                                                  if len(real) == 8 else None))
    write_metrics_json(summary, output / "summary.json")
    return summary


def run_all(cfg, categories, device, *, resume=False):
    results = {}
    reasons = asset_reasons(cfg, device)
    if reasons:
        for category in categories:
            results[category] = _status_result(cfg, category, "NOT RUN", "; ".join(reasons))
    else:
        identity = experiment_identity(cfg)
        for category in categories:
            try:
                results[category] = run_category(cfg, category, device, identity, resume=resume)
            except (E0Blocked, FileNotFoundError) as exc:
                status = "BLOCKED" if isinstance(exc, E0Blocked) else "NOT RUN"
                results[category] = _status_result(cfg, category, status, str(exc))
            except Exception as exc:
                results[category] = _status_result(cfg, category, "FAIL", f"{type(exc).__name__}: {exc}")
            print(f"E0 {category}: {results[category]['status']}", flush=True)
    return write_summary(cfg, results)


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    summary = run_all(cfg, requested_categories(args.categories), args.device, resume=args.resume)
    print(json.dumps(dict(status=summary["status"], real_categories_pass=summary["real_categories_pass"],
                          expected_categories=8, smoke_categories_pass=summary["smoke_categories_pass"],
                          output_root=cfg["output_root"]), indent=2))
    return 0 if summary["status"] in {"PASS", "PARTIAL", "SMOKE PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
