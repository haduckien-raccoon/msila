#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, subprocess, sys, time
from contextlib import nullcontext
from src.train.day05_contract import validate_signature, validate_selection, validate_day05, validate_record_sources
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn

from src.data.cached_dataset import CachedFeatureDataset, load_training_records, make_cached_dataloader
from src.data.feature_cache import FeatureCacheReader, validate_cache
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.basic_decoder import BasicDecoder
from src.models.cached_training import cached_meta_to_aligner_geometry
from src.models.context_alignment import ContextToLocalAligner
from src.models.feature_projection import SixFeatureProjection
from src.models.feature_selector import FeatureSelector
from src.models.mean_fusion import MeanFusion
from src.models.residual_adapter import ResidualAdapter2d


class FullTrainError(RuntimeError):
    pass


def _jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def canonical_json(x):
    return json.dumps(_jsonable(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(x):
    return hashlib.sha256(canonical_json(x).encode("utf-8")).hexdigest()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(_jsonable(obj), sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)


def read_yaml(path):
    try:
        obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FullTrainError(f"Invalid YAML config {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise FullTrainError(f"Expected YAML mapping: {path}")
    return obj


def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def seed_everything(seed: int, protocol=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t = {} if protocol is None else protocol["training"]
    torch.use_deterministic_algorithms(t.get("deterministic_algorithms", True),
                                      warn_only=t.get("deterministic_warn_only", False))
    torch.backends.cudnn.benchmark = False


def record_split(r):
    value = r.get("split")
    if value is None and isinstance(r.get("meta"), Mapping):
        value = r["meta"].get("split")
    return None if value is None else str(value).strip().lower().replace("-", "_")


def filter_records(records, category: str, role: str, full_scale=False):
    out = [dict(r) for r in records if str(r.get("category", "")).lower() == category.lower()]
    if not out:
        raise FullTrainError(f"{role}: no records for category={category}")
    splits = {record_split(r) for r in out if record_split(r)}
    forbidden = {"overfit16", "test_public", "test_private", "test_private_mixed", "validation"}
    if splits & forbidden:
        raise FullTrainError(f"{role}: forbidden splits: {sorted(splits & forbidden)}")
    if role == "val" and splits != ({"dev_tiny", "dev_mixed"} if full_scale else {"dev_synthetic"}):
        raise FullTrainError(f"val must be DEV-synthetic, got {sorted(splits)}")
    if role == "train" and splits and not splits.issubset({"train", "train_core"}):
        raise FullTrainError(f"train must be TRAIN-core/train, got {sorted(splits)}")
    return out


def records_fingerprint(records):
    rows = []
    for r in records:
        rows.append({
            "image_id": str(r.get("image_id")),
            "category": str(r.get("category")),
            "mask_path": None if r.get("mask_path") is None else str(r.get("mask_path")),
            "mask_hw": r.get("mask_hw"),
            "is_anomaly": r.get("is_anomaly"),
            "split": record_split(r),
        })
    return sha256_json(rows)


def masks_fingerprint(records, mask_root):
    rows = []
    for r in records:
        raw = r.get("mask_path")
        if raw is None:
            value = {"implicit_zero": r.get("is_anomaly") is False, "hw": r.get("mask_hw")}
        else:
            path = Path(raw)
            if not path.is_absolute() and mask_root is not None:
                path = Path(mask_root) / path
            value = sha256_file(path)
        rows.append({"image_id": r["image_id"], "category": r["category"], "mask_sha256": value})
    return sha256_json(rows)


def known_positive_count(records):
    flags = []
    for r in records:
        if "is_anomaly" in r:
            flags.append(bool(r["is_anomaly"]))
        elif isinstance(r.get("meta"), Mapping) and "is_anomaly" in r["meta"]:
            flags.append(bool(r["meta"]["is_anomaly"]))
    return None if not flags else sum(flags)


def verify_cache(cache_dir: Path, train_records, val_records, expected_backbone: str, allow_unverified=False):
    validate_cache(cache_dir)
    reader = FeatureCacheReader(cache_dir, mmap=True, shard_cache_size=1)
    missing = []
    for r in list(train_records) + list(val_records):
        key = (str(r["image_id"]), str(r["category"]))
        if key not in reader:
            missing.append(key)
    if missing:
        raise FullTrainError(f"cache missing {len(missing)} required records; first={missing[:3]}")

    if allow_unverified:
        raise FullTrainError("Unverified backbone provenance is forbidden for Day-05")
    signature = reader.manifest.get("producer_signature")
    try:
        validate_signature(signature, expected_backbone=expected_backbone)
    except ValueError as exc:
        raise FullTrainError(str(exc)) from exc
    if reader.manifest.get("producer_sha256") != sha256_json(signature):
        raise FullTrainError("cache producer signature/hash mismatch")
    from src.data.feature_cache import FEATURE_KEYS
    from src.models.backbone_registry import backbone_spec
    spec = backbone_spec(expected_backbone)
    hw = [s // spec.patch_size for s in signature['model_input_size']]
    expected = (spec.channels, *hw)
    scope = 'Full-scale' if signature.get('schema') == 'msila.full_scale.cache.v2' else 'Day-05'
    for record in list(train_records) + list(val_records):
        cached = reader.get(image_id=record['image_id'], category=record['category'])
        for key in FEATURE_KEYS:
            x = cached[key]
            shape = tuple(x.shape[1:]) if x.ndim == 4 and x.shape[0] == 1 else tuple(x.shape)
            if shape != expected:
                raise FullTrainError(f'{scope} cache {record["image_id"]}/{key} shape {shape} != {expected}')
    return reader, {
        "verified": True, "producer_signature": signature,
        "producer_sha256": reader.manifest.get("producer_sha256"),
    }


def infer_in_channels(reader, record):
    sample = reader.get(image_id=str(record["image_id"]), category=str(record["category"]))
    x = sample["local_b4"]
    if x.ndim == 4:
        return int(x.shape[1])
    if x.ndim == 3:
        return int(x.shape[0])
    raise FullTrainError(f"unexpected local_b4 shape: {tuple(x.shape)}")


def move_to_device(x, device):
    if isinstance(x, Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def grad_norm(model):
    total = 0.0
    seen = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            seen = True
            g = p.grad.detach().float()
            if not bool(torch.isfinite(g).all()):
                raise FullTrainError("NaN/Inf gradient")
            total += float(g.pow(2).sum().item())
    if not seen:
        raise FullTrainError("no trainable gradient")
    return math.sqrt(total)


def snapshot_trainable(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def changed_names(before, model):
    now = dict(model.named_parameters())
    return [n for n, old in before.items() if not torch.equal(old, now[n].detach().cpu())]


class Day05RepresentationModel(nn.Module):
    def __init__(self, *, day05_config, candidate, in_channels, adapter_r, adapter_d):
        super().__init__()
        self.candidate = str(candidate).upper()
        self.selector = FeatureSelector.from_config(day05_config, self.candidate, validate=True)
        locked = day05_config["locked"]
        self.blocks = (4, 8, 12)
        self.active_blocks = (12,) if self.candidate == "R0" else self.blocks

        self.adapters = nn.ModuleDict({
            f"b{b}": ResidualAdapter2d(
                in_dim=int(in_channels),
                bottleneck_dim=int(adapter_r),
                projection_dim=int(adapter_d),
                kernel_size=int(locked["adapter"]["kernel_size"]),
                gamma_init=float(locked["adapter"]["gamma_init"]),
                bias=True,
            ) for b in self.blocks
        })
        deterministic_sampling = day05_config.get("schema") == "msila.full_scale.representation.v2"
        self.aligner = ContextToLocalAligner(check_finite=True, check_bounds=True,
                                            deterministic_sampling=deterministic_sampling)
        self.projection = SixFeatureProjection(
            in_channels=int(in_channels),
            fusion_dim=int(locked["projection"]["fusion_dim"]),
            blocks=self.blocks,
            share_across_views=bool(locked["projection"]["share_across_views"]),
            check_finite=True,
        )
        self.fusion = MeanFusion(validate=True, return_weights=True)
        self.decoder = BasicDecoder(in_channels=int(locked["projection"]["fusion_dim"]),
                                    deterministic_resize=deterministic_sampling)
        self._freeze_inactive()

    def _freeze_inactive(self):
        active = set(self.active_blocks)
        for b in self.blocks:
            if b not in active:
                self.adapters[f"b{b}"].requires_grad_(False)
        if not self.projection.share_across_views:
            raise FullTrainError("Day-05 requires share_projection_across_views=true")
        for b in self.blocks:
            if b not in active:
                self.projection.projectors[f"b{b}"].requires_grad_(False)
        if any(True for _ in self.fusion.parameters()):
            raise FullTrainError("MeanFusion must be parameter-free")
        if any(True for _ in self.aligner.parameters()):
            raise FullTrainError("Context aligner must be parameter-free")

    def forward(self, batch, retain_source_grads=False):
        local = {}
        for b in self.active_blocks:
            local[f"L{b}"] = self.adapters[f"b{b}"](batch[f"local_b{b}"])

        aligned_context = None
        if self.candidate == "R2":
            context = {
                f"C{b}": self.adapters[f"b{b}"](batch[f"context_b{b}"])
                for b in self.blocks
            }
            geometry = cached_meta_to_aligner_geometry(
                batch["meta"],
                device=local["L4"].device,
                matrix_dtype=torch.float32,
            )
            h, w = local["L4"].shape[-2:]
            aligned_context = self.aligner(context, geometry, target_hw=(int(h), int(w)))

        projected = self.projection.project_sources(
            local,
            aligned_context,
            source_keys=self.selector.source_keys,
        )
        if retain_source_grads:
            for x in projected.values():
                x.retain_grad()

        selected = self.selector(projected)
        fused, weights = self.fusion(selected)

        output_hw = batch.get("output_hw")
        if output_hw is None:
            output_hw = batch["mask"].shape[-2:]
        logits = self.decoder(fused, output_size=tuple(map(int, output_hw)))
        return logits, {
            "projected": projected,
            "selected": selected,
            "fused": fused,
            "fusion_weights": weights,
        }


def build_optimizer(model, protocol):
    cfg = protocol["training"]["optimizer"]
    if str(cfg["name"]) != "AdamW":
        raise FullTrainError(f"locked optimizer must be AdamW, got {cfg['name']}")
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(cfg["lr"]), **dict(cfg.get("kwargs", {})))


def verify_optimizer_parameters(model, optimizer):
    """Every active head tensor must belong to the optimizer exactly once."""
    named = {id(p): (name, p) for name, p in model.named_parameters()}
    expected = {key for key, (_, p) in named.items() if p.requires_grad}
    params = [p for group in optimizer.param_groups for p in group["params"]]
    actual = [id(p) for p in params]
    present = set(actual)
    missing = sorted(named[key][0] for key in expected - present)
    frozen = sorted(named[key][0] for key in present & named.keys() if key not in expected)
    foreign = len(present - named.keys())
    duplicates = len(actual) - len(present)
    if not expected or missing or frozen or foreign or duplicates:
        raise FullTrainError(f"optimizer parameter contract violated: missing={missing}, "
                             f"frozen={frozen}, foreign={foreign}, duplicates={duplicates}")
    return {
        "status": "PASS", "optimizer": type(optimizer).__name__,
        "trainable_parameters": sum(named[key][1].numel() for key in expected),
        "trainable_tensors": len(expected),
        "groups": [{"index": i, "parameter_names": [named[id(p)][0] for p in group["params"]],
                    "parameters": sum(p.numel() for p in group["params"]),
                    "lr": float(group["lr"])} for i, group in enumerate(optimizer.param_groups)],
    }


def verify_protocol(protocol):
    t = protocol["training"]
    for k in ("epochs", "batch_size", "optimizer"):
        if t.get(k) is None:
            raise FullTrainError(f"training.{k} is unresolved/null")
    if t.get("scheduler") not in (None, False):
        raise FullTrainError("non-null scheduler not implemented: lock it explicitly before training")
    if int(t["epochs"]) <= 0 or int(t["batch_size"]) <= 0:
        raise FullTrainError("epochs/batch_size must be >0")
    if int(t.get('checkpoint_interval_steps',100)) <= 0:
        raise FullTrainError('checkpoint_interval_steps must be positive')
    rule = protocol.get("checkpoint", {})
    if not ((rule.get("monitor") in {"val_loss", "val_total_loss"} and rule.get("mode") == "min")
            or (rule.get("monitor") == "dev_aupro" and rule.get("mode") == "max")):
        raise FullTrainError("Unsupported Day-04 checkpoint rule; cannot silently replace it")
    if float(t["optimizer"]["lr"]) <= 0:
        raise FullTrainError("optimizer lr must be >0")


def make_dataset(cache_dir, records, protocol, mask_root):
    d = protocol["data"]
    return CachedFeatureDataset(
        cache_dir=cache_dir,
        records=records,
        expected_producer_signature=None,
        mask_root=mask_root,
        mask_threshold=int(d.get("mask_threshold", 0)),
        mask_hw_source=str(d.get("mask_hw_source", "record")),
        allow_zero_mask_for_normal=bool(d.get("allow_zero_mask_for_normal", True)),
        squeeze_cached_batch_dim=bool(d.get("squeeze_cached_batch_dim", True)),
        feature_dtype=torch.float32,
        mmap=bool(d.get("mmap", True)),
        shard_cache_size=int(d.get("shard_cache_size", 2)),
    )


def make_loader(ds, protocol, shuffle, seed):
    d = protocol["data"]
    t = protocol["training"]
    loader = make_cached_dataloader(
        ds,
        batch_size=int(t["batch_size"]),
        shuffle=shuffle,
        num_workers=int(d.get("num_workers", 0)),
        pin_memory=d.get("pin_memory", True),
        persistent_workers=d.get("persistent_workers", False),
        prefetch_factor=int(d.get("prefetch_factor", 2)),
        drop_last=bool(d.get("drop_last", False)) if shuffle else False,
        seed=int(seed),
    )
    if shuffle:
        # Separate sample order from persistent-worker base-seed consumption.
        loader.sampler.generator = torch.Generator().manual_seed(int(seed))
    return loader


def make_criterion(day05, protocol=None):
    loss = day05["locked"]["loss"]
    if float(loss.get("illumination_weight", 0.0)) != 0.0:
        raise FullTrainError("illumination loss is forbidden in Day-05 representation ablation")
    return AnomalySegmentationLoss(
        bce_weight=float(loss["bce_weight"]),
        dice_weight=float(loss["dice_weight"]),
        dice_eps=1e-6 if protocol is None else float(protocol.get("model", {}).get("loss", {}).get("dice_eps",1e-6)),
    )


def source_usage(trace):
    out = {}
    for key, tensor in trace["projected"].items():
        g = tensor.grad
        out[key] = {
            "grad_present": g is not None,
            "finite": bool(torch.isfinite(g).all()) if g is not None else False,
            "grad_l1": float(g.detach().abs().sum().item()) if g is not None else 0.0,
        }
    return out


def amp_context(protocol, device):
    cfg = {} if protocol is None else protocol["training"].get("amp", {})
    if not cfg.get("enabled", False):
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.get("dtype"))
    if dtype != torch.bfloat16:
        raise FullTrainError("Only the locked bfloat16 AMP protocol is implemented")
    return torch.autocast(device_type=device.type, dtype=dtype)


def run_preflight(model, loader, criterion, optimizer, device, steps, protocol=None):
    if steps < 2:
        raise FullTrainError("preflight_steps must be >=2 because gamma_init=0")
    optimizer_parameters = verify_optimizer_parameters(model, optimizer)
    before = snapshot_trainable(model)
    it = iter(loader)
    reports, losses = [], []
    model.train()

    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        for k, x in batch.items():
            if k.startswith(("local_b", "context_b")) and (x.requires_grad or not torch.isfinite(x).all()):
                raise FullTrainError(f"Frozen feature contract violated: {k}")
        with amp_context(protocol, device):
            logits, trace = model(batch, retain_source_grads=True)
            lo = criterion(logits, batch["mask"])
        loss = lo["loss"]
        if not bool(torch.isfinite(loss)):
            raise FullTrainError("preflight loss NaN/Inf")
        loss.backward()
        usage = source_usage(trace)
        for key, row in usage.items():
            if not row["grad_present"] or not row["finite"] or row["grad_l1"] <= 0:
                raise FullTrainError(f"source disconnected in preflight: {key}: {row}")
        if model.candidate == "R2":
            expected = {
                "local_b4", "local_b8", "local_b12",
                "context_b4", "context_b8", "context_b12",
            }
            if set(usage) != expected:
                raise FullTrainError(f"R2 did not use exactly 6 sources: {sorted(usage)}")
        gn = grad_norm(model)
        if not math.isfinite(gn) or gn <= 0:
            raise FullTrainError(f"invalid grad_norm={gn}")
        optimizer.step()
        reports.append(usage)
        losses.append(float(loss.detach().cpu()))

    changed = changed_names(before, model)
    for prefix in ("adapters.", "projection.", "decoder."):
        if not any(n.startswith(prefix) for n in changed):
            raise FullTrainError(f"preflight: no weight changed under {prefix}")
    return {
        "status": "PASS",
        "candidate": model.candidate,
        "steps": int(steps),
        "frozen_backbone_features": True,
        "parameter_free_aligner_fusion": True,
        "optimizer_parameters": optimizer_parameters,
        "losses": losses,
        "source_usage": reports,
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
    }


def append_csv(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_checkpoint(path, model, optimizer, epoch, global_step, best_val, config_hash, progress=None):
    payload = {
        "schema": "msila.day05.full_train.checkpoint.v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": best_val,
        "resolved_config_sha256": config_hash,
        "progress": progress,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if progress is not None:
        sha_path = path.with_suffix(path.suffix + '.sha256')
        tmp_sha = sha_path.with_suffix(sha_path.suffix + '.tmp')
        tmp_sha.write_text(sha256_file(path) + '\n')
        os.replace(tmp_sha, sha_path)


def load_checkpoint(path, model, optimizer, expected_hash, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("resolved_config_sha256") != expected_hash:
        raise FullTrainError("resume config mismatch")
    model.load_state_dict(p["model"], strict=True)
    optimizer.load_state_dict(p["optimizer"])
    model._resume_partial = p.get("progress") or {}
    rng = p.get("rng", {})
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return int(p["epoch"]), int(p["global_step"]), p.get("best_val")


def validate(model, loader, criterion, device, protocol=None):
    model.eval()
    sums = {"total_loss": 0.0, "bce": 0.0, "dice": 0.0}
    n = 0
    metrics = None
    if protocol is not None and protocol.get("evaluation", {}).get("synthetic_protocol"):
        from src.eval.full_scale import DEVMetricAccumulator
        metrics = DEVMetricAccumulator(protocol["evaluation"]["synthetic_protocol"], per_region=False,disk_backed=True)
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with amp_context(protocol, device):
                logits, _ = model(batch)
                out = criterion(logits, batch["mask"])
            if not bool(torch.isfinite(out["loss"])):
                raise FullTrainError("validation loss NaN/Inf")
            weight = int(logits.shape[0]) if metrics is not None else 1
            sums["total_loss"] += weight * float(out["loss"].detach().cpu())
            sums["bce"] += weight * float(out["bce"].detach().cpu())
            sums["dice"] += weight * float(out["dice"].detach().cpu())
            n += weight
            if metrics is not None:
                scores = logits.float().sigmoid().cpu().numpy()
                masks = batch["mask"].cpu().numpy()
                for i, meta in enumerate(batch["meta"]):
                    metrics.add(scores[i,0], masks[i,0], split=meta["split"],
                                image_id=str(meta.get("native_image_id", i)))
    if n == 0:
        raise FullTrainError("validation loader empty")
    result = {k: v / n for k, v in sums.items()}
    if metrics is not None:
        report = metrics.result()
        if report["dev_aupro"] is None:
            raise FullTrainError("DEV tiny/mixed AU-PRO is undefined; selection cannot proceed")
        result.update(dev_aupro=report["dev_aupro"],
                      dev_tiny_aupro=report["groups"]["dev_tiny"]["aupro_0_05"],
                      dev_mixed_aupro=report["groups"]["dev_mixed"]["aupro_0_05"])
    return result


def save_preview(model, loader, device, path):
    model.eval()
    batch = move_to_device(next(iter(loader)), device)
    with torch.no_grad():
        logits, _ = model(batch)
        score = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    lo, hi = float(score.min()), float(score.max())
    vis = np.zeros_like(score, dtype=np.uint8) if hi <= lo else ((score - lo) / (hi - lo) * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)
    np.save(path.with_suffix(".npy"), score.astype(np.float32))


def lock_payload(resolved):
    keys = [
        "category", "seed", "adapter", "backbone", "input", "projection",
        "fusion", "decoder", "loss", "training", "data_fingerprints",
        "cache_producer_sha256", "git_commit", "runner_sha256", "checkpoint_rule",
        "data_loader", "day05_config_sha256",
    ]
    payload = {k: resolved[k] for k in keys}
    if resolved.get('day05_config', {}).get('schema') == 'msila.full_scale.representation.v2':
        payload['evaluation'] = resolved['evaluation']
    return payload


def enforce_lock(output_root, payload):
    path = Path(output_root) / "protocol_lock.json"
    digest = sha256_json(payload)
    record = {"sha256": digest, "payload": payload}
    if not path.is_file():
        save_json(record, path)
        return digest
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("sha256") != digest:
        raise FullTrainError(
            f"PROTOCOL DRIFT across R0/R1/R2: old={old.get('sha256')} new={digest}"
        )
    return digest


def train_candidate(args):
    candidate_started = time.perf_counter()
    day05 = read_yaml(args.day05_config)
    protocol = read_yaml(args.training_protocol)
    verify_protocol(protocol)
    validate_day05(day05)
    full = day05.get("schema") == "msila.full_scale.representation.v2"
    if getattr(args, "backbone", None) and args.backbone != day05["locked"]["backbone"]["name"]:
        raise FullTrainError("CLI backbone differs from the model config")
    if full and day05["locked"]["training"]["seed"] != args.seed:
        raise FullTrainError("CLI seed differs from resolved representation config")
    if not full and (args.r, args.d, args.seed) != (32, 384, 42):
        raise FullTrainError("Day-05 locked adapter/seed are r32,d384,seed42")
    if not full and not args.adapter_selection_report:
        raise FullTrainError("Provide the real Day-04 selection report")
    if not full:
        validate_selection(args.adapter_selection_report)
    elif protocol["checkpoint"] != {"monitor": "dev_aupro", "mode": "max"}:
        raise FullTrainError("Full-scale checkpoint selection requires locked DEV AU-PRO, mode=max")

    candidate = args.candidate.upper()
    if candidate not in {"R0", "R1", "R2"}:
        raise FullTrainError("candidate must be R0/R1/R2")
    if args.r <= 0 or args.d <= 0:
        raise FullTrainError("r,d must be final locked Day-04 winner >0")
    if not full and day05["locked"]["backbone"]["name"] != "dinov3_vits16":
        raise FullTrainError("Day-05 full runner is locked to dinov3_vits16")
    if day05["locked"]["fusion"]["type"] != "mean" or day05["locked"]["fusion"]["trainable"]:
        raise FullTrainError("Day-05 must use parameter-free MeanFusion")
    if day05["experiment"]["candidate_ids"] != ["R0", "R1", "R2"]:
        raise FullTrainError("candidate config drift")

    train_records = filter_records(load_training_records(args.train_records), args.category, "train", full)
    val_records = filter_records(load_training_records(args.val_records), args.category, "val", full)

    validate_record_sources(train_records, val_records, full_scale=full)
    tp, vp = known_positive_count(train_records), known_positive_count(val_records)
    if tp == 0:
        raise FullTrainError("TRAIN-core explicitly has zero anomaly samples; dense BCE+Dice training is invalid")
    if vp == 0:
        raise FullTrainError("DEV-synthetic explicitly has zero anomaly samples")

    reader, provenance = verify_cache(
        Path(args.cache_dir),
        train_records,
        val_records,
        day05["locked"]["backbone"]["name"],
        allow_unverified=args.allow_unverified_cache_provenance,
    )
    in_channels = infer_in_channels(reader, train_records[0])
    from src.models.backbone_registry import backbone_spec
    spec = backbone_spec(day05["locked"]["backbone"]["name"])
    if in_channels != spec.channels:
        raise FullTrainError(f"Cache channel mismatch: {in_channels} != {spec.channels}")
    if getattr(args, "dino_checkpoint", None):
        validate_signature(provenance["producer_signature"], args.dino_checkpoint,
                           expected_backbone=day05["locked"]["backbone"]["name"])
    if full and provenance["producer_signature"].get("schema") != "msila.full_scale.cache.v2":
        raise FullTrainError("Full-scale requires newly versioned tiny/mixed DEV cache")
    if full and (protocol.get('evaluation', {}).get('synthetic_protocol') !=
                 provenance['producer_signature']['synthetic_protocol']):
        raise FullTrainError('Evaluation protocol differs from the predeclared synthetic producer')
    if provenance["producer_signature"]["logical_layers_1based"] != day05["locked"]["backbone"]["feature_blocks"]:
        raise FullTrainError("Cache physical blocks differ from model config")
    expected_sig = protocol["data"].get("expected_producer_signature")
    if expected_sig is not None and expected_sig != provenance["producer_signature"]:
        raise FullTrainError("cache differs from Day-04 protocol producer signature")

    mask_root = None if args.mask_root is None else Path(args.mask_root)
    train_ds = make_dataset(Path(args.cache_dir), train_records, protocol, mask_root)
    val_ds = make_dataset(Path(args.cache_dir), val_records, protocol, mask_root)
    train_loader = make_loader(train_ds, protocol, True, args.seed)
    val_loader = make_loader(val_ds, protocol, False, args.seed)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise FullTrainError("empty train/val loader")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FullTrainError("BLOCKED_MISSING_GPU: CUDA requested but unavailable")

    runner_path = Path(__file__).resolve()
    resolved = {
        "schema": "msila.day05.full_train.resolved.v1",
        "candidate": candidate,
        "representation": day05["representations"][candidate],
        "day05_config": day05,
        "day05_config_sha256": sha256_json(day05),
        "data_loader": {k: v for k, v in protocol["data"].items()
                        if k not in {"cache_dir", "train_records", "val_records", "mask_root", "expected_producer_signature"}},
        "category": args.category,
        "seed": int(args.seed),
        "adapter": {
            "r": int(args.r),
            "d": int(args.d),
            "kernel_size": int(day05["locked"]["adapter"]["kernel_size"]),
            "gamma_init": float(day05["locked"]["adapter"]["gamma_init"]),
            "selection_report": args.adapter_selection_report,
            "selection_report_sha256": (
                sha256_file(args.adapter_selection_report)
                if args.adapter_selection_report else None
            ),
        },
        "backbone": day05["locked"]["backbone"],
        "input": day05["locked"]["input"],
        "projection": day05["locked"]["projection"],
        "fusion": day05["locked"]["fusion"],
        "decoder": day05["locked"]["decoder"],
        "loss": day05["locked"]["loss"],
        "training": {
            "epochs": int(protocol["training"]["epochs"]),
            "batch_size": int(protocol["training"]["batch_size"]),
            "optimizer": protocol["training"]["optimizer"],
            "scheduler": protocol["training"].get("scheduler"),
            "gradient_clip_norm": protocol["training"].get("gradient_clip_norm"),
            "amp": protocol["training"].get("amp", {}),
            "deterministic_algorithms": protocol["training"].get("deterministic_algorithms", True),
            "deterministic_warn_only": protocol["training"].get("deterministic_warn_only", False),
            "sampler_rule": "separate_generator_seed_plus_epoch_minus_one",
            "updates_per_epoch": len(train_loader),
            "total_update_budget": int(protocol["training"]["epochs"]) * len(train_loader),
        },
        "data_fingerprints": {
            "train_source_sha256": sha256_file(args.train_records),
            "val_source_sha256": sha256_file(args.val_records),
            "train_filtered_sha256": records_fingerprint(train_records),
            "val_filtered_sha256": records_fingerprint(val_records),
            "train_masks_sha256": masks_fingerprint(train_records, mask_root),
            "val_masks_sha256": masks_fingerprint(val_records, mask_root),
        },
        "cache_producer_sha256": reader.manifest.get("producer_sha256"),
        "cache": {
            "dir": str(Path(args.cache_dir).resolve()),
            "manifest_sha256": sha256_file(Path(args.cache_dir) / "manifest.json"),
            "provenance_check": provenance,
            "in_channels": in_channels,
        },
        "checkpoint_rule": dict(protocol["checkpoint"]),
        "evaluation": protocol.get("evaluation", {}),
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "runner_sha256": sha256_file(runner_path),
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None
            if torch.version.cuda is None
            else str(torch.version.cuda)
        ),
    }

    output_root = Path(args.output_root)
    lock_root = output_root/'protocol_locks'/f'seed_{args.seed}'/args.category if full else output_root
    lock_sha = enforce_lock(lock_root, lock_payload(resolved))
    resolved["protocol_lock_sha256"] = lock_sha
    if full:
        resolved['protocol_lock_path'] = str(lock_root/'protocol_lock.json')

    run_dir = output_root / f"seed_{args.seed}" / args.category / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_hash = sha256_json(resolved)
    if resolved_path.is_file():
        old = read_yaml(resolved_path)
        if sha256_json(old) != resolved_hash:
            raise FullTrainError(f"resolved config drift in {run_dir}")
    else:
        save_yaml(resolved, resolved_path)

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "COMPLETE":
            for name in ("best.pt", "last.pt", "training_log.csv", "epoch_log.csv", "preflight_report.json", "selection_record.json", "sample_anomaly_map.png"):
                path = run_dir / name
                if not path.is_file() or old.get("artifact_sha256", {}).get(name) != sha256_file(path):
                    raise FullTrainError(f"COMPLETE artifact missing/modified: {path}")
            if full:
                for name in ('best.pt.sha256','last.pt.sha256','trainable_parameters.json'):
                    if old.get('artifact_sha256',{}).get(name) != sha256_file(run_dir/name):
                        raise FullTrainError(f'COMPLETE artifact missing/modified: {name}')
            print(f"[SKIP] already COMPLETE: {run_dir}")
            return run_dir

    manifest = {
        "schema": "msila.day05.full_train.manifest.v1",
        "status": "RUNNING",
        "candidate": candidate,
        "category": args.category,
        "seed": int(args.seed),
        "adapter": {"r": int(args.r), "d": int(args.d)},
        "sources": list(day05["representations"][candidate]["sources"]),
        "git_branch": resolved["git_branch"],
        "git_commit": resolved["git_commit"],
        "resolved_config_sha256": resolved_hash,
        "protocol_lock_sha256": lock_sha,
        "started_at_unix": time.time(),
    }
    if full:
        manifest.update(backbone=day05['locked']['backbone'],
                        total_update_budget=resolved['training']['total_update_budget'])
    save_json(manifest, manifest_path)

    criterion = make_criterion(day05, protocol).to(device)

    # Disposable exact-final-r,d preflight.
    seed_everything(args.seed, protocol)
    pf_model = Day05RepresentationModel(
        day05_config=day05,
        candidate=candidate,
        in_channels=in_channels,
        adapter_r=args.r,
        adapter_d=args.d,
    ).to(device)
    pf_opt = build_optimizer(pf_model, protocol)
    if full:
        parts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                 for name, module in (("adapter", pf_model.adapters), ("projection", pf_model.projection),
                                       ("decoder", pf_model.decoder), ("fusion", pf_model.fusion))}
        save_json(dict(trainable_parameters=sum(parts.values()), parts=parts,
                       backbone_frozen=True, backbone=args.backbone if getattr(args, "backbone", None)
                       else day05["locked"]["backbone"]["name"], in_channels=in_channels,
                       adapter_r=args.r, adapter_d=args.d,
                       fusion_dim=day05["locked"]["projection"]["fusion_dim"]),run_dir/"trainable_parameters.json")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    pf = run_preflight(
        pf_model, train_loader, criterion, pf_opt, device, args.preflight_steps, protocol
    )
    pf["peak_vram_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda" else 0.0
    )
    save_json(pf, run_dir / "preflight_report.json")
    del pf_model, pf_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if getattr(args, "preflight_only", False):
        manifest["status"] = "PREFLIGHT_PASS"
        save_json(manifest, manifest_path)
        return run_dir

    # Fresh actual model.
    seed_everything(args.seed, protocol)
    model = Day05RepresentationModel(
        day05_config=day05,
        candidate=candidate,
        in_channels=in_channels,
        adapter_r=args.r,
        adapter_d=args.d,
    ).to(device)
    optimizer = build_optimizer(model, protocol)

    train_log = run_dir / "training_log.csv"
    epoch_log = run_dir / "epoch_log.csv"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    start_epoch, global_step, best_val = 1, 0, None
    if args.resume and last_path.is_file():
        if full:
            sidecar = last_path.with_suffix('.pt.sha256')
            if not sidecar.is_file() or sidecar.read_text().strip() != sha256_file(last_path):
                raise FullTrainError("Resume checkpoint checksum missing or invalid; configuration preserved")
        done_epoch, global_step, best_val = load_checkpoint(
            last_path, model, optimizer, resolved_hash, device
        )
        if full:
            progress=getattr(model,'_resume_partial',{})
            step=int(progress.get('step_in_epoch',0))
            count=len(train_loader)
            if (not 0 <= done_epoch <= int(protocol['training']['epochs']) or not 0 <= step <= count
                    or progress.get('epoch') != done_epoch+1 or global_step != done_epoch*count+step
                    or (step and int(progress.get('n',-1)) != step)):
                raise FullTrainError('Resume checkpoint epoch/update progress is inconsistent')
        start_epoch = done_epoch + 1
        print(f"[RESUME] {candidate}: epoch={done_epoch}, step={global_step}, best={best_val}")

    if last_path.is_file() and not args.resume:
        raise FullTrainError("Partial run exists; use --resume")
    # Roll back append-only logs to the last committed epoch before replaying.
    if args.resume:
        for log in (train_log, epoch_log):
            if log.is_file():
                with log.open(newline="", encoding="utf-8") as f:
                    reader_log = csv.DictReader(f)
                    fields = reader_log.fieldnames
                    rows = [r for r in reader_log if (int(r["global_step"]) <= global_step
                            if full and log == train_log else int(r["epoch"]) < start_epoch)]
                with log.open("w", newline="", encoding="utf-8") as f:
                    writer_log = csv.DictWriter(f, fieldnames=fields)
                    writer_log.writeheader()
                    writer_log.writerows(rows)

    epochs = int(protocol["training"]["epochs"])
    grad_clip = protocol["training"].get("gradient_clip_norm")
    t0 = time.perf_counter()
    manifest['preparation_elapsed_sec'] = t0-candidate_started
    partial = getattr(model, '_resume_partial', {}) if full else {}
    previous_elapsed = float(partial.get('elapsed_sec', 0.))
    previous_peak = float(partial.get('peak_vram_mb', 0.))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs + 1):
        # Same sampler order after epoch-boundary resume, independent of preflight.
        train_loader.sampler.generator.manual_seed(args.seed + epoch - 1)
        model.train()
        sums = {"total_loss": 0.0, "bce": 0.0, "dice": 0.0, "grad_norm": 0.0}
        n = 0
        skip_steps = 0
        if full and partial.get('epoch') == epoch:
            skip_steps = int(partial.get('step_in_epoch', 0))
            if skip_steps:
                sums = dict(partial['sums'])
                n = int(partial['n'])
        e0 = time.perf_counter()

        for step_in_epoch, batch in enumerate(train_loader, start=1):
            if step_in_epoch <= skip_steps:
                continue
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(protocol, device):
                logits, _ = model(batch)
                out = criterion(logits, batch["mask"])
            loss = out["loss"]
            if not bool(torch.isfinite(loss)):
                raise FullTrainError(f"NaN/Inf loss at epoch={epoch}, step={step_in_epoch}")
            loss.backward()
            gn = grad_norm(model)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=float(grad_clip),
                )
            optimizer.step()

            global_step += 1
            n += 1
            elapsed = time.perf_counter() - t0
            peak = (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda" else 0.0
            )
            row = {
                "candidate": candidate,
                "category": args.category,
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "total_loss": float(loss.detach().cpu()),
                "bce": float(out["bce"].detach().cpu()),
                "dice": float(out["dice"].detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(gn),
                "elapsed_sec": float(elapsed),
                "peak_vram_mb": peak,
            }
            if full:
                row['elapsed_sec'] += previous_elapsed
                row['peak_vram_mb'] = max(previous_peak,row['peak_vram_mb'])
            append_csv(train_log, row)
            sums["total_loss"] += row["total_loss"]
            sums["bce"] += row["bce"]
            sums["dice"] += row["dice"]
            sums["grad_norm"] += row["grad_norm"]
            if full and step_in_epoch % int(protocol['training'].get('checkpoint_interval_steps', 100)) == 0:
                save_checkpoint(last_path, model, optimizer, epoch-1, global_step, best_val, resolved_hash,
                    progress=dict(epoch=epoch,step_in_epoch=step_in_epoch,sums=sums,n=n,
                                  elapsed_sec=previous_elapsed+time.perf_counter()-t0,
                                  peak_vram_mb=max(previous_peak,peak)))

        if n == 0:
            raise FullTrainError("no training batches")

        val = validate(model, val_loader, criterion, device, protocol)
        train_mean = {k: v / n for k, v in sums.items()}
        epoch_row = {
            "candidate": candidate,
            "category": args.category,
            "epoch": epoch,
            "global_step": global_step,
            "train_total_loss": train_mean["total_loss"],
            "train_bce": train_mean["bce"],
            "train_dice": train_mean["dice"],
            "train_grad_norm": train_mean["grad_norm"],
            "val_total_loss": val["total_loss"],
            "val_bce": val["bce"],
            "val_dice": val["dice"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_elapsed_sec": float(time.perf_counter() - e0),
            "elapsed_sec": float(time.perf_counter() - t0),
            "peak_vram_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda" else 0.0
            ),
        }
        if full:
            epoch_row.update({k:val[k] for k in ('dev_aupro','dev_tiny_aupro','dev_mixed_aupro')})
            epoch_row['elapsed_sec'] += previous_elapsed
            epoch_row['peak_vram_mb'] = max(previous_peak,epoch_row['peak_vram_mb'])
        append_csv(epoch_log, epoch_row)
        monitor = val["dev_aupro"] if full else val["total_loss"]
        progress = dict(epoch=epoch+1,step_in_epoch=0,elapsed_sec=previous_elapsed+time.perf_counter()-t0,
                        peak_vram_mb=max(previous_peak,epoch_row['peak_vram_mb'])) if full else None
        if best_val is None or (monitor > best_val if full else monitor < best_val):
            best_val = float(monitor)
            save_checkpoint(best_path, model, optimizer, epoch, global_step, best_val, resolved_hash, progress)
        save_checkpoint(last_path, model, optimizer, epoch, global_step, best_val, resolved_hash, progress)

        manifest.update({
            "status": "RUNNING",
            "completed_epochs": epoch,
            "global_step": global_step,
            "best_val_total_loss": best_val,
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "elapsed_sec": float(time.perf_counter() - t0),
            "peak_vram_mb": epoch_row["peak_vram_mb"],
        })
        save_json(manifest, manifest_path)

        print(
            f"[{candidate}] epoch {epoch:03d}/{epochs} | "
            f"train={train_mean['total_loss']:.6f} | "
            f"val={val['total_loss']:.6f} | best={best_val:.6f} | "
            f"step={global_step} | VRAM={epoch_row['peak_vram_mb']:.1f}MB"
        )

    if not best_path.is_file():
        raise FullTrainError("best.pt missing")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    save_preview(model, val_loader, device, run_dir / "sample_anomaly_map.png")

    save_json({"rule": dict(protocol["checkpoint"]), "log_monitor": "dev_aupro" if full else "val_total_loss",
               "selected_epoch": best["epoch"], "selected_value": best["best_val"],
               "checkpoint_sha256": sha256_file(best_path), "manual_selection": False},
              run_dir / "selection_record.json")
    manifest.update({
        "status": "COMPLETE",
        "completed_epochs": epochs,
        "global_step": global_step,
        "best_val_total_loss": best_val,
        "finished_at_unix": time.time(),
        "total_elapsed_sec": float(time.perf_counter() - t0),
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda" else 0.0
        ),
        "artifacts": {
            "training_log": str(train_log),
            "epoch_log": str(epoch_log),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "resolved_config": str(resolved_path),
            "preflight_report": str(run_dir / "preflight_report.json"),
            "sample_anomaly_map": str(run_dir / "sample_anomaly_map.png"),
        },
    })
    manifest["artifact_sha256"] = {name: sha256_file(run_dir / name) for name in
        ("best.pt", "last.pt", "training_log.csv", "epoch_log.csv", "preflight_report.json",
         "selection_record.json", "sample_anomaly_map.png", "resolved_config.yaml")}
    if full:
        manifest.pop('best_val_total_loss', None)
        manifest.update(best_dev_aupro=best_val, selection_data='DEV_tiny+DEV_mixed_cached_native_tiles',
                        total_elapsed_sec=previous_elapsed+time.perf_counter()-t0,
                        peak_vram_mb=max(previous_peak,manifest['peak_vram_mb']))
        manifest['artifact_sha256'].update({name:sha256_file(run_dir/name) for name in
                                            ('best.pt.sha256','last.pt.sha256','trainable_parameters.json')})
    save_json(manifest, manifest_path)
    print(f"[PASS] COMPLETE: {run_dir}")
    return run_dir


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, choices=["R0", "R1", "R2"])
    p.add_argument("--category", default="fabric")
    p.add_argument("--r", "--adapter-r", dest="r", required=True, type=int)
    p.add_argument("--d", "--adapter-d", dest="d", required=True, type=int)
    p.add_argument("--backbone", default=None)
    p.add_argument("--dino-checkpoint", default=None)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--train-records", required=True)
    p.add_argument("--val-records", required=True)
    p.add_argument("--mask-root", default=None)
    p.add_argument("--output-root", required=True)
    p.add_argument("--day05-config", default="configs/day05_representation.yaml")
    p.add_argument("--training-protocol", default="configs/day04_train_protocol.yaml")
    p.add_argument("--adapter-selection-report", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-steps", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--allow-unverified-cache-provenance", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        train_candidate(args)
        return 0
    except (FullTrainError, ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        report = dict(status=str(exc).split(':')[0] if str(exc).startswith('BLOCKED_') else 'FAIL',
                      reason=str(exc), candidate=args.candidate,
                      stage='preflight' if args.preflight_only else 'train')
        save_json(report, Path(args.output_root) / 'last_gate_report.json')
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
