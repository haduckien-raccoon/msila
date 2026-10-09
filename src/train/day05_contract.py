"""Shared Day-05 provenance checks. No model defaults or inferred backbone names."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from src.models.backbone_registry import backbone_spec, validate_blocks

BACKBONE = "dinov3_vits16"
SOURCES = {
    "R0": ["local_b12"],
    "R1": ["local_b4", "local_b8", "local_b12"],
    "R2": ["local_b4", "local_b8", "local_b12", "context_b4", "context_b8", "context_b12"],
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_signature(signature, checkpoint=None, expected_backbone=BACKBONE):
    if not isinstance(signature, dict) or "backbone" not in signature:
        raise ValueError("cache producer_signature lacks backbone; rebuild from the verified producer")
    if signature["backbone"] != expected_backbone:
        raise ValueError(f"cache backbone mismatch: {signature['backbone']!r} != {expected_backbone}")
    sha = signature.get("checkpoint_sha256", "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("cache lacks valid checkpoint identity")
    if signature.get("schema") == "msila.full_scale.cache.v2":
        spec = backbone_spec(expected_backbone)
        blocks = validate_blocks(signature.get("logical_layers_1based", ()), spec.depth)
        if (signature.get("architecture") != dict(channels=spec.channels, depth=spec.depth, patch_size=spec.patch_size)
                or signature.get("internal_indices_0based") != [b-1 for b in blocks]
                or signature.get("cache_slot_blocks") != dict(zip(("b4", "b8", "b12"), blocks))):
            raise ValueError("cache architecture/block identity mismatch")
        for k, value in {"smoke_only": False, "full_source_coverage": True,
                         "source_split_version": "train_normal_hash80_10_10_tiny_v2",
                         "local_source_size": [512, 512], "context_source_size": [768, 768],
                         "model_input_size": [512, 512], "normalization": "ImageNet mean/std"}.items():
            if signature.get(k) != value:
                raise ValueError(f"full-scale cache {k} mismatch")
        if not signature.get("synthetic_protocol") or not signature.get("preprocessing") or not signature.get("source_sha256"):
            raise ValueError("full-scale cache missing synthetic/preprocessing/source provenance")
        if checkpoint is not None and file_hash(checkpoint) != sha:
            raise ValueError("DINO checkpoint hash differs from the cache producer")
        return signature
    checks = {
        "logical_layers_1based": [4, 8, 12],
        "local_source_size": [512, 512],
        "context_source_size": [768, 768],
        "model_input_size": [512, 512],
        "normalization": "ImageNet mean/std",
        "source_split_version": "day04_full_hash80_20_v3",
        "train_fraction": 0.8,
        "train_anomaly_types": ["intensity", "color", "noise"],
        "dev_anomaly_types": ["cutpaste"],
    }
    for k, expected in checks.items():
        if signature.get(k) != expected:
            raise ValueError(f"cache producer {k} mismatch: {signature.get(k)!r}")
    if not signature.get("schema") or signature.get("smoke_only") is not False:
        raise ValueError("Day-05 requires a versioned full cache, not smoke/Overfit-16")
    if signature.get("full_source_coverage") is not True:
        raise ValueError("cache does not certify full TRAIN-source coverage")
    if checkpoint is not None and file_hash(checkpoint) != sha:
        raise ValueError("DINO checkpoint hash differs from the cache producer")
    return signature


def validate_selection(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    selection = obj.get("selection", {})
    if (obj.get("analysis_status") != "PASS" or selection.get("status") != "SELECTED"
            or selection.get("selected_candidate") != "adapter_r32_d384"):
        raise ValueError("Day-04 selection must PASS and select adapter_r32_d384")
    return obj


def validate_day05(cfg):
    locked = cfg["locked"]
    full = cfg.get("schema") == "msila.full_scale.representation.v2"
    for k, expected in {"tile_size": 512, "context_size": 768, "overlap": 128}.items():
        if locked["input"].get(k) != expected:
            raise ValueError(f"Day-05 input.{k} drift")
    if full:
        bb = locked["backbone"]
        spec = backbone_spec(bb["name"])
        validate_blocks(bb["feature_blocks"], spec.depth)
        if bb["frozen"] is not True:
            raise ValueError("DINOv3 must stay frozen")
    elif locked["backbone"] != {"name": BACKBONE, "frozen": True, "feature_blocks": [4, 8, 12]}:
        raise ValueError("Day-05 frozen ViT-S/16 backbone drift")
    if locked["projection"]["fusion_dim"] != 64 or not locked["projection"]["share_across_views"]:
        raise ValueError("Day-05 projection must be shared C->64")
    if locked["fusion"]["type"] != "mean" or locked["fusion"]["trainable"]:
        raise ValueError("Day-05 requires parameter-free MeanFusion")
    if locked["adapter"]["kernel_size"] != 3 or locked["adapter"]["gamma_init"] != 0.0:
        raise ValueError("Day-04 Adapter kernel/gamma drift")
    if locked["loss"]["bce_weight"] != 1.0 or locked["loss"]["dice_weight"] != 1.0 or locked["loss"]["illumination_weight"] != 0.0:
        raise ValueError("Locked BCE+Dice loss drift")
    if not full and locked["training"]["seed"] != 42:
        raise ValueError("Phase 1 seed must be 42")
    for c, sources in SOURCES.items():
        if cfg["representations"][c]["sources"] != sources:
            raise ValueError(f"{c}: representation source drift")


def validate_record_sources(train, dev, full_scale=False):
    def identities(rows, split):
        ids, sources = set(), set()
        for r in rows:
            meta = r.get("meta", {})
            role = r.get("split", meta.get("split"))
            if role not in split:
                raise ValueError(f"records must declare {split}, got {role!r}")
            key = (r["category"], r["image_id"])
            if key in ids:
                raise ValueError(f"duplicate record ID: {key}")
            ids.add(key)
            sid = r.get("source_image_id", meta.get("source_identity"))
            if not sid or "synthetic" not in meta:
                raise ValueError(f"{key}: source identity/synthetic provenance missing")
            sources.add((r["category"], sid))
            if meta.get("smoke_only") or meta.get("full_source_coverage") is not True:
                raise ValueError(f"{key}: not a full-cache record")
        return sources
    dev_roles = {"dev_tiny", "dev_mixed"} if full_scale else {"dev_synthetic"}
    if identities(train, {"train_core"}) & identities(dev, dev_roles):
        raise ValueError("TRAIN-core/DEV-synthetic source leakage")
    if full_scale:
        tiny = {r["meta"]["source_identity"] for r in dev if r["meta"]["split"] == "dev_tiny"}
        mixed = {r["meta"]["source_identity"] for r in dev if r["meta"]["split"] == "dev_mixed"}
        if tiny & mixed or not tiny or not mixed:
            raise ValueError("DEV tiny/mixed must be nonempty and source-disjoint")
