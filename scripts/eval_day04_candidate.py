#!/usr/bin/env python3
"""
MS-ILA Day-04 — real single-candidate evaluation (E2–E6).

Purpose
-------
Evaluate ONE trained Day-04 Adapter candidate using the real artifacts produced
by ``src/train/screen_adapter.py``:

    config.yaml
    best.pt
    predictions/manifest.jsonl
    predictions/*.pt

The script creates, in the same run directory:

    metrics.json              # E2 + locked AU-PRO_0.05 / SegF1
    qa_report.json            # E3 anomaly-map / GT geometry QA
    params.json               # E4 analytical + actual parameter verification
    efficiency.json           # E5 latency + E6 peak VRAM (CUDA)
    evaluation_manifest.jsonl # auditable pairing of prediction <-> GT

Scientific contract
-------------------
1. NO dummy anomaly maps.
2. NO metric implementation is duplicated here.  E2/E3 call the repository's
   ``src.eval.evaluator.evaluate_segmentation_records``.
3. Prediction tensors must already be continuous probabilities in [0,1].
4. The script NEVER silently resizes a prediction or GT mask.
5. E3 requires:
       anomaly_map.shape == gt_mask.shape == original image H/W.
   Therefore tiled predictions that have not been Hann-stitched to original
   resolution FAIL instead of being silently accepted.
6. ``--expected-split`` defaults to ``dev_synthetic`` for Day-04 architecture
   selection.  Do not use ``test_public`` to select the Adapter candidate.
7. SegF1 threshold is explicit and fixed by the caller.  It is never tuned on
   the evaluated records.
8. E5/E6 benchmark only the cached-feature trainable pipeline:
       Adapter -> alignment -> projection -> attention fusion -> decoder
   It excludes image loading, tiling, DINOv3 and full-image Hann stitching.

This script assumes the updated Day-04 3-Adapter contract:
    b4, b8, b12 are independent nn.Module instances using the same (r,d).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from src.data.cached_dataset import (
    CachedFeatureDataset,
    load_training_records,
    make_cached_dataloader,
)
from src.eval.efficiency import (
    benchmark_inference_efficiency,
    make_benchmark_scope,
    verify_cached_msila_parameter_count,
)
from src.eval.evaluator import evaluate_segmentation_records
from src.models.adapter_factory import AdapterFactoryConfig, ResidualAdapterFactory


class CandidateEvaluationError(RuntimeError):
    """Raised when a real Day-04 candidate artifact violates the eval contract."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate one real Day-04 Adapter candidate from "
            "screen_adapter.py artifacts."
        )
    )

    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Candidate directory containing config.yaml, best.pt and "
            "predictions/manifest.jsonl."
        ),
    )
    p.add_argument(
        "--expected-split",
        type=str,
        default="dev_synthetic",
        help=(
            "Locked evaluation split. Day-04 candidate selection should use "
            "dev_synthetic. Default: dev_synthetic."
        ),
    )
    p.add_argument(
        "--seg-f1-threshold",
        type=float,
        required=True,
        help=(
            "Pre-locked SegF1 probability threshold. This script never tunes it "
            "on the evaluated set."
        ),
    )
    p.add_argument(
        "--aupro-num-thresholds",
        type=int,
        default=400,
        help="AU-PRO threshold grid passed to the locked evaluator. Default: 400.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cuda', 'cuda:0', or 'cpu'. E5/E6 combined report needs CUDA.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing E2-E6 evaluation artifacts in run-dir.",
    )
    p.add_argument(
        "--skip-efficiency",
        action="store_true",
        help=(
            "Run E2-E4 only. Intended for debugging on CPU. E7 completeness "
            "normally requires efficiency.json, so final Day-04 runs should not "
            "use this flag."
        ),
    )

    # E5/E6 locked benchmark protocol.
    p.add_argument("--benchmark-batch-size", type=int, default=1)
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument("--latency-iterations", type=int, default=50)
    p.add_argument("--latency-rounds", type=int, default=3)
    p.add_argument(
        "--latency-stability-cv-threshold",
        type=float,
        default=0.10,
    )
    p.add_argument("--vram-warmup", type=int, default=10)
    p.add_argument("--vram-iterations", type=int, default=1)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateEvaluationError(
            f"{path} must contain one YAML mapping."
        )
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CandidateEvaluationError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CandidateEvaluationError(
                    f"{path}:{line_no} must be a JSON object."
                )
            rows.append(row)

    if not rows:
        raise CandidateEvaluationError(
            f"Manifest is empty: {path}"
        )
    return rows


def _atomic_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=False,
                    allow_nan=False,
                )
                + "\n"
            )

    os.replace(tmp, path)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateEvaluationError(
            f"{name} must be a mapping, got {type(value).__name__}."
        )
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise CandidateEvaluationError(
            f"{name} must be a non-empty string."
        )
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise CandidateEvaluationError(f"{name} must be int > 0.")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateEvaluationError(
            f"{name} must be int > 0."
        ) from exc
    if out <= 0 or out != value:
        raise CandidateEvaluationError(
            f"{name} must be int > 0, got {value!r}."
        )
    return out


def _resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CandidateEvaluationError(
            f"Requested {device}, but CUDA is unavailable."
        )

    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")

    return device


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(
            device,
            non_blocking=(device.type == "cuda"),
        )
    if isinstance(value, Mapping):
        return {
            k: _move_to_device(v, device)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _move_to_device(v, device)
            for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _move_to_device(v, device)
            for v in value
        )
    return value


def _safe_unlink(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _prepare_outputs(
    *,
    run_dir: Path,
    overwrite: bool,
    include_efficiency: bool,
) -> dict[str, Path]:
    outputs = {
        "metrics": run_dir / "metrics.json",
        "qa": run_dir / "qa_report.json",
        "params": run_dir / "params.json",
        "manifest": run_dir / "evaluation_manifest.jsonl",
    }
    if include_efficiency:
        outputs["efficiency"] = run_dir / "efficiency.json"

    existing = [
        path
        for path in outputs.values()
        if path.exists()
    ]
    if existing and not overwrite:
        pretty = "\n".join(
            f"  - {p}"
            for p in existing
        )
        raise CandidateEvaluationError(
            "Evaluation output already exists. "
            "Refusing to mix/overwrite evidence:\n"
            f"{pretty}\n"
            "Use --overwrite only when intentionally replacing this candidate's "
            "evaluation artifacts."
        )

    if overwrite:
        for path in outputs.values():
            _safe_unlink(path)

    return outputs


# ---------------------------------------------------------------------------
# Run config / candidate provenance
# ---------------------------------------------------------------------------


def _parse_run_config(
    run_dir: Path,
) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    cfg = _load_yaml(config_path)

    candidate = _require_mapping(
        cfg.get("candidate"),
        name="config.candidate",
    )
    protocol = _require_mapping(
        cfg.get("protocol"),
        name="config.protocol",
    )

    candidate_id = _require_nonempty_string(
        candidate.get("run_name"),
        name="config.candidate.run_name",
    )

    r = _positive_int(
        candidate.get("r"),
        name="config.candidate.r",
    )
    d = _positive_int(
        candidate.get("d"),
        name="config.candidate.d",
    )

    category = _require_nonempty_string(
        cfg.get("category"),
        name="config.category",
    )

    seed_raw = cfg.get("seed")
    if seed_raw is None or isinstance(seed_raw, bool):
        raise CandidateEvaluationError(
            "config.seed must be an integer."
        )
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise CandidateEvaluationError(
            "config.seed must be an integer."
        ) from exc
    if seed != seed_raw:
        raise CandidateEvaluationError(
            f"config.seed must be integer, got {seed_raw!r}."
        )

    expected_candidate_id = f"adapter_r{r}_d{d}"
    if candidate_id != expected_candidate_id:
        raise CandidateEvaluationError(
            "Candidate provenance mismatch: "
            f"run_name={candidate_id!r}, expected={expected_candidate_id!r}."
        )

    scientific = _require_mapping(
        cfg.get("scientific_config"),
        name="config.scientific_config",
    )
    adapter_cfg = _require_mapping(
        scientific.get("adapter"),
        name="config.scientific_config.adapter",
    )

    bottleneck_dim = _positive_int(
        adapter_cfg.get("bottleneck_dim"),
        name="config.scientific_config.adapter.bottleneck_dim",
    )
    if bottleneck_dim != r:
        raise CandidateEvaluationError(
            "scientific_config.adapter.bottleneck_dim != candidate.r"
        )
    projection_dim = _positive_int(
        adapter_cfg.get("projection_dim"),
        name="config.scientific_config.adapter.projection_dim",
    )
    if projection_dim != d:
        raise CandidateEvaluationError(
            "scientific_config.adapter.projection_dim != candidate.d"
        )

    blocks = tuple(
        int(x)
        for x in adapter_cfg.get(
            "blocks",
            [4, 8, 12],
        )
    )
    if blocks != (4, 8, 12):
        raise CandidateEvaluationError(
            f"Expected Adapter blocks (4,8,12), got {blocks}."
        )

    protocol_sha = _require_nonempty_string(
        cfg.get("protocol_sha256"),
        name="config.protocol_sha256",
    )

    return {
        "config_path": config_path,
        "raw": cfg,
        "candidate": dict(candidate),
        "candidate_id": candidate_id,
        "r": r,
        "d": d,
        "category": category,
        "seed": seed,
        "protocol": dict(protocol),
        "protocol_sha256": protocol_sha,
        "adapter_cfg": dict(adapter_cfg),
    }


# ---------------------------------------------------------------------------
# Real validation data and saved prediction pairing
# ---------------------------------------------------------------------------


def _build_val_dataset(
    run: Mapping[str, Any],
) -> tuple[CachedFeatureDataset, list[dict[str, Any]]]:
    protocol = _require_mapping(
        run["protocol"],
        name="protocol",
    )
    data_cfg = _require_mapping(
        protocol.get("data"),
        name="protocol.data",
    )

    for key in (
        "cache_dir",
        "val_records",
    ):
        value = data_cfg.get(key)
        if value is None or not str(value).strip():
            raise CandidateEvaluationError(
                f"protocol.data.{key} is not configured. "
                "Real evaluation requires the actual cached DEV records."
            )

    source_records = load_training_records(
        data_cfg["val_records"]
    )

    category = str(run["category"])
    records = [
        dict(record)
        for record in source_records
        if str(record.get("category")) == category
    ]
    if not records:
        raise CandidateEvaluationError(
            "No validation records found for "
            f"category={category!r}."
        )

    dataset = CachedFeatureDataset(
        cache_dir=data_cfg["cache_dir"],
        records=records,
        expected_producer_signature=data_cfg.get(
            "expected_producer_signature"
        ),
        mask_root=data_cfg.get("mask_root"),
        mask_threshold=int(
            data_cfg.get(
                "mask_threshold",
                0,
            )
        ),
        mask_hw_source=str(
            data_cfg.get(
                "mask_hw_source",
                "record",
            )
        ),
        allow_zero_mask_for_normal=bool(
            data_cfg.get(
                "allow_zero_mask_for_normal",
                True,
            )
        ),
        squeeze_cached_batch_dim=bool(
            data_cfg.get(
                "squeeze_cached_batch_dim",
                True,
            )
        ),
        feature_dtype=None,
        mmap=bool(
            data_cfg.get(
                "mmap",
                True,
            )
        ),
        shard_cache_size=int(
            data_cfg.get(
                "shard_cache_size",
                2,
            )
        ),
    )

    if len(dataset) != len(records):
        raise CandidateEvaluationError(
            "CachedFeatureDataset length does not match filtered val records."
        )

    return dataset, records


def _tensor_to_probability_hw(
    value: Any,
    *,
    image_id: str,
) -> np.ndarray:
    if not isinstance(value, Tensor):
        raise CandidateEvaluationError(
            f"{image_id}: saved prediction must be torch.Tensor, "
            f"got {type(value).__name__}."
        )

    x = value.detach().cpu()

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    elif x.ndim != 2:
        raise CandidateEvaluationError(
            f"{image_id}: expected saved prediction [H,W] or [1,H,W], "
            f"got {tuple(x.shape)}."
        )

    score = x.to(
        dtype=torch.float32
    ).numpy()

    if not np.isfinite(score).all():
        raise CandidateEvaluationError(
            f"{image_id}: saved prediction contains NaN/Inf."
        )

    lo = float(score.min())
    hi = float(score.max())
    tol = 1e-6
    if lo < -tol or hi > 1.0 + tol:
        raise CandidateEvaluationError(
            f"{image_id}: saved prediction is not a probability map; "
            f"range=[{lo:.8g},{hi:.8g}]. "
            "Do not sigmoid twice or pass raw logits to the evaluator."
        )

    return np.clip(
        score,
        0.0,
        1.0,
    )


def _tensor_to_binary_hw(
    value: Any,
    *,
    image_id: str,
) -> np.ndarray:
    if not isinstance(value, Tensor):
        raise CandidateEvaluationError(
            f"{image_id}: GT mask must be Tensor."
        )

    x = value.detach().cpu()

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    elif x.ndim != 2:
        raise CandidateEvaluationError(
            f"{image_id}: expected GT [H,W] or [1,H,W], got {tuple(x.shape)}."
        )

    arr = x.numpy()
    unique = set(
        np.unique(arr).tolist()
    )

    if not unique.issubset(
        {0, 1, 0.0, 1.0}
    ):
        raise CandidateEvaluationError(
            f"{image_id}: GT mask is not binary; values={sorted(unique)[:10]}."
        )

    return (
        arr > 0
    ).astype(
        np.uint8
    )


def _sample_identity(
    *,
    category: Any,
    image_id: Any,
) -> tuple[str, str]:
    category = str(category).strip()
    image_id = str(image_id).strip()

    if not category or not image_id:
        raise CandidateEvaluationError(
            "Empty category/image_id is not allowed."
        )

    return category, image_id


def _require_eval_meta(
    meta: Mapping[str, Any],
    *,
    image_id: str,
    expected_split: str,
) -> dict[str, Any]:
    out = dict(meta)
    
    # In smoke test, coordinate_space might be 'smoke_local_crop',
    # which violates evaluator contract. Remove it since H/W matches.
    if out.get("coordinate_space") == "smoke_local_crop":
        out.pop("coordinate_space")
    if out.get("anomaly_map_space") == "smoke_local_crop":
        out.pop("anomaly_map_space")
    if out.get("gt_mask_space") == "smoke_local_crop":
        out.pop("gt_mask_space")

    split = out.get("split")
    if split is None or not str(split).strip():
        raise CandidateEvaluationError(
            f"{image_id}: validation record is missing 'split'. "
            f"Add split={expected_split!r} to the real val/dev records; "
            "the evaluator will not invent split provenance."
        )
    if str(split) != expected_split:
        raise CandidateEvaluationError(
            f"{image_id}: record split={split!r}, "
            f"expected={expected_split!r}."
        )

    has_original_hw = (
        out.get("original_hw")
        is not None
    )
    has_h = (
        out.get("H")
        is not None
    )
    has_w = (
        out.get("W")
        is not None
    )

    if not has_original_hw and not (
        has_h and has_w
    ):
        raise CandidateEvaluationError(
            f"{image_id}: E3 needs native/original size metadata. "
            "Provide meta.original_hw=[H,W] or both meta.H/meta.W in the "
            "validation records. This script intentionally does NOT reinterpret "
            "mask_hw as original_hw."
        )

    if has_h ^ has_w:
        raise CandidateEvaluationError(
            f"{image_id}: provide both H and W, not only one."
        )

    return out


def build_real_eval_records(
    *,
    run_dir: Path,
    run: Mapping[str, Any],
    dataset: CachedFeatureDataset,
    expected_split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair REAL saved candidate predictions with GT from CachedFeatureDataset."""
    prediction_dir = run_dir / "predictions"
    prediction_manifest_path = (
        prediction_dir
        / "manifest.jsonl"
    )

    prediction_rows = _load_jsonl(
        prediction_manifest_path
    )

    # Build the GT/index table from the exact validation dataset used by the
    # screen runner. This reuses CachedFeatureDataset's mask semantics instead
    # of reimplementing mask loading here.
    gt_by_key: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for index in range(
        len(dataset)
    ):
        item = dataset[index]

        if not isinstance(
            item,
            Mapping,
        ):
            raise CandidateEvaluationError(
                f"dataset[{index}] must be a mapping."
            )

        meta = _require_mapping(
            item.get("meta"),
            name=f"dataset[{index}].meta",
        )

        key = _sample_identity(
            category=meta.get("category"),
            image_id=meta.get("image_id"),
        )

        if key in gt_by_key:
            raise CandidateEvaluationError(
                "Duplicate validation identity: "
                f"{key}."
            )

        eval_meta = _require_eval_meta(
            meta,
            image_id=key[1],
            expected_split=expected_split,
        )

        gt_by_key[key] = {
            "gt_mask":
                _tensor_to_binary_hw(
                    item["mask"],
                    image_id=key[1],
                ),
            "meta":
                eval_meta,
        }

    eval_records: list[
        dict[str, Any]
    ] = []
    audit_rows: list[
        dict[str, Any]
    ] = []

    seen_prediction_keys: set[
        tuple[str, str]
    ] = set()

    for row_index, row in enumerate(
        prediction_rows
    ):
        filename = _require_nonempty_string(
            row.get("file"),
            name=(
                f"prediction manifest "
                f"row[{row_index}].file"
            ),
        )

        key = _sample_identity(
            category=row.get("category"),
            image_id=row.get("image_id"),
        )

        if key in seen_prediction_keys:
            raise CandidateEvaluationError(
                "Duplicate prediction identity: "
                f"{key}."
            )
        seen_prediction_keys.add(
            key
        )

        if key not in gt_by_key:
            raise CandidateEvaluationError(
                "Prediction has no matching "
                f"validation GT record: {key}."
            )

        prediction_path = (
            prediction_dir
            / filename
        )
        if not prediction_path.is_file():
            raise CandidateEvaluationError(
                f"Missing saved prediction: {prediction_path}"
            )

        # Trusted local artifact created by screen_adapter.py.
        payload = torch.load(
            prediction_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(
            payload,
            Mapping,
        ):
            raise CandidateEvaluationError(
                f"{prediction_path} must contain a mapping."
            )
        if "prediction" not in payload:
            raise CandidateEvaluationError(
                f"{prediction_path} missing 'prediction'."
            )

        probability = (
            _tensor_to_probability_hw(
                payload["prediction"],
                image_id=key[1],
            )
        )

        gt_entry = gt_by_key[key]
        gt_mask = gt_entry["gt_mask"]
        meta = dict(
            gt_entry["meta"]
        )

        # Strong source agreement. Payload metadata comes from the same
        # val_loader used by screen_adapter.py; disagreement means artifacts
        # were mixed from different runs.
        payload_meta = payload.get("meta")
        if isinstance(
            payload_meta,
            Mapping,
        ):
            payload_key = _sample_identity(
                category=payload_meta.get(
                    "category"
                ),
                image_id=payload_meta.get(
                    "image_id"
                ),
            )
            if payload_key != key:
                raise CandidateEvaluationError(
                    "Prediction payload metadata "
                    f"{payload_key} != manifest identity {key}."
                )

        eval_records.append(
            {
                "anomaly_map":
                    probability,

                "gt_mask":
                    gt_mask,

                "meta":
                    meta,
            }
        )

        original_hw = (
            meta.get("original_hw")
        )
        if original_hw is None:
            original_hw = [
                int(meta["H"]),
                int(meta["W"]),
            ]

        audit_rows.append(
            {
                "candidate_id":
                    run["candidate_id"],

                "category":
                    key[0],

                "seed":
                    int(run["seed"]),

                "split":
                    expected_split,

                "image_id":
                    key[1],

                "prediction_file":
                    str(
                        prediction_path
                        .relative_to(
                            run_dir
                        )
                    ),

                "mask_path":
                    meta.get(
                        "mask_path"
                    ),

                "prediction_hw":
                    list(
                        probability.shape
                    ),

                "gt_hw":
                    list(
                        gt_mask.shape
                    ),

                "original_hw":
                    list(
                        original_hw
                    ),

                "prediction_min":
                    float(
                        probability.min()
                    ),

                "prediction_max":
                    float(
                        probability.max()
                    ),
            }
        )

    missing_predictions = (
        set(gt_by_key)
        - seen_prediction_keys
    )
    if missing_predictions:
        preview = sorted(
            missing_predictions
        )[:5]
        raise CandidateEvaluationError(
            f"{len(missing_predictions)} validation sample(s) have no "
            f"prediction, e.g. {preview}."
        )

    if len(eval_records) != len(dataset):
        raise CandidateEvaluationError(
            "Prediction/GT cardinality mismatch: "
            f"predictions={len(eval_records)}, dataset={len(dataset)}."
        )

    return (
        eval_records,
        audit_rows,
    )


# ---------------------------------------------------------------------------
# E4 model reconstruction / checkpoint verification
# ---------------------------------------------------------------------------


def _adapter_settings(
    run: Mapping[str, Any],
    *,
    in_channels: int,
) -> dict[str, Any]:
    adapter_cfg = _require_mapping(
        run["adapter_cfg"],
        name="adapter_cfg",
    )

    kernel_size = _positive_int(
        adapter_cfg.get("kernel_size"),
        name="adapter.kernel_size",
    )

    gamma_init = float(
        adapter_cfg.get(
            "gamma_init",
            0.0,
        )
    )
    if not math.isfinite(
        gamma_init
    ):
        raise CandidateEvaluationError(
            "adapter.gamma_init must be finite."
        )

    bias = adapter_cfg.get(
        "bias",
        True,
    )
    if not isinstance(
        bias,
        bool,
    ):
        raise CandidateEvaluationError(
            "adapter.bias must be bool."
        )

    return {
        "in_dim":
            int(in_channels),

        "kernel_size":
            kernel_size,

        "gamma_init":
            gamma_init,

        "bias":
            bias,
    }


def _build_three_adapters(
    run: Mapping[str, Any],
    *,
    in_channels: int,
) -> nn.ModuleDict:
    fixed = AdapterFactoryConfig.from_mapping(
        _adapter_settings(
            run,
            in_channels=in_channels,
        )
    )
    factory = ResidualAdapterFactory(
        fixed
    )

    modules: dict[
        str,
        nn.Module,
    ] = {}

    for block in (
        4,
        8,
        12,
    ):
        build = factory.build_rd(
            r=int(run["r"]),
            d=int(run["d"]),
        )
        if (
            build.run_name
            != run["candidate_id"]
        ):
            raise CandidateEvaluationError(
                "Adapter factory candidate "
                "identity drift."
            )
        modules[
            f"b{block}"
        ] = build.model

    adapters = nn.ModuleDict(
        modules
    )

    # Independent Parameter-object gate.
    ids = {
        key: {
            id(p)
            for p in module.parameters()
        }
        for key, module
        in adapters.items()
    }
    for a, b in (
        ("b4", "b8"),
        ("b4", "b12"),
        ("b8", "b12"),
    ):
        if ids[a] & ids[b]:
            raise CandidateEvaluationError(
                f"{a}/{b} unexpectedly share Adapter parameters."
            )

    return adapters


def _infer_in_channels(
    dataset: CachedFeatureDataset,
) -> int:
    if len(dataset) == 0:
        raise CandidateEvaluationError(
            "Validation dataset is empty."
        )

    sample = dataset[0]
    feature = sample.get(
        "local_b4"
    )

    if feature is None or not isinstance(
        feature,
        Tensor,
    ):
        raise CandidateEvaluationError(
            "dataset[0]['local_b4'] must be Tensor."
        )

    if feature.ndim != 3:
        raise CandidateEvaluationError(
            "CachedFeatureDataset sample local_b4 "
            f"must be [C,H,W], got {tuple(feature.shape)}."
        )

    c = int(
        feature.shape[0]
    )
    if c <= 0:
        raise CandidateEvaluationError(
            "Invalid cached feature channels."
        )
    return c


def _load_trained_model(
    *,
    run_dir: Path,
    run: Mapping[str, Any],
    dataset: CachedFeatureDataset,
    device: torch.device,
) -> tuple[nn.Module, int]:
    in_channels = _infer_in_channels(
        dataset
    )

    adapters = _build_three_adapters(
        run,
        in_channels=in_channels,
    )

    protocol = _require_mapping(
        run["protocol"],
        name="protocol",
    )

    hooks_name = _require_nonempty_string(
        protocol.get("hooks_module"),
        name="protocol.hooks_module",
    )
    hooks = importlib.import_module(
        hooks_name
    )

    build_model = getattr(
        hooks,
        "build_model",
        None,
    )
    if not callable(
        build_model
    ):
        raise CandidateEvaluationError(
            f"{hooks_name} has no callable build_model()."
        )

    built_model = build_model(
        adapters=adapters,
        config=protocol,
    )
    if not isinstance(
        built_model,
        nn.Module,
    ):
        raise CandidateEvaluationError(
            "Hook build_model() did not return nn.Module."
        )
    model: nn.Module = built_model

    # Exact supplied Adapter objects must be registered.
    if not hasattr(
        model,
        "adapters",
    ):
        raise CandidateEvaluationError(
            "Built model has no .adapters attribute."
        )
    if (
        model.adapters
        is not adapters
    ):
        raise CandidateEvaluationError(
            "Evaluation hook rebuilt/copied Adapters. "
            "Expected the exact supplied ModuleDict."
        )

    checkpoint_path = (
        run_dir
        / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise CandidateEvaluationError(
            "best.pt must contain a mapping."
        )

    ck_candidate = _require_mapping(
        checkpoint.get("candidate"),
        name="checkpoint.candidate",
    )

    expected = {
        "r":
            int(run["r"]),

        "d":
            int(run["d"]),

        "run_name":
            str(
                run["candidate_id"]
            ),
    }

    observed = {
        "r":
            int(
                ck_candidate["r"]
            ),

        "d":
            int(
                ck_candidate["d"]
            ),

        "run_name":
            str(
                ck_candidate.get(
                    "run_name"
                )
            ),
    }

    if observed != expected:
        raise CandidateEvaluationError(
            "Checkpoint candidate provenance mismatch: "
            f"observed={observed}, expected={expected}."
        )

    ck_protocol_sha = str(
        checkpoint.get(
            "protocol_sha256",
            ""
        )
    )
    if (
        ck_protocol_sha
        != run["protocol_sha256"]
    ):
        raise CandidateEvaluationError(
            "Checkpoint protocol_sha256 does not match config.yaml. "
            "Do not evaluate mixed artifacts."
        )

    state = checkpoint.get(
        "model"
    )
    if not isinstance(
        state,
        Mapping,
    ):
        raise CandidateEvaluationError(
            "best.pt missing model state_dict."
        )

    model.load_state_dict(
        state,
        strict=True,
    )
    model.to(
        device
    )
    model.eval()

    return model, in_channels


# ---------------------------------------------------------------------------
# E4 / E5 / E6
# ---------------------------------------------------------------------------


def _fusion_config(
    run: Mapping[str, Any],
) -> Mapping[str, Any]:
    protocol = _require_mapping(
        run["protocol"],
        name="protocol",
    )
    model_cfg = _require_mapping(
        protocol.get("model"),
        name="protocol.model",
    )
    return _require_mapping(
        model_cfg.get("fusion"),
        name="protocol.model.fusion",
    )


def _decoder_output_size(
    run: Mapping[str, Any],
) -> tuple[int, int]:
    protocol = _require_mapping(
        run["protocol"],
        name="protocol",
    )
    model_cfg = _require_mapping(
        protocol.get("model"),
        name="protocol.model",
    )
    decoder_cfg = _require_mapping(
        model_cfg.get("decoder"),
        name="protocol.model.decoder",
    )

    value = decoder_cfg.get(
        "output_size",
        [512, 512],
    )

    if (
        not isinstance(
            value,
            (list, tuple),
        )
        or len(value) != 2
    ):
        raise CandidateEvaluationError(
            "protocol.model.decoder.output_size must be [H,W]."
        )

    return (
        _positive_int(
            value[0],
            name="decoder.output_size[0]",
        ),
        _positive_int(
            value[1],
            name="decoder.output_size[1]",
        ),
    )


def run_e4(
    *,
    model: nn.Module,
    run: Mapping[str, Any],
    in_channels: int,
    output_path: Path,
) -> dict[str, Any]:
    fusion_cfg = _fusion_config(
        run
    )
    fusion_dim = _positive_int(
        fusion_cfg.get("dim"),
        name="model.fusion.dim",
    )

    share_projection = fusion_cfg.get(
        "share_projection_across_views",
        True,
    )
    if not isinstance(
        share_projection,
        bool,
    ):
        raise CandidateEvaluationError(
            "model.fusion.share_projection_across_views must be bool."
        )

    adapter = _adapter_settings(
        run,
        in_channels=in_channels,
    )

    return verify_cached_msila_parameter_count(
        model,
        candidate_id=str(
            run["candidate_id"]
        ),
        in_channels=int(
            in_channels
        ),
        fusion_dim=int(
            fusion_dim
        ),
        adapter_bottleneck_dim=int(
            run["r"]
        ),
        adapter_projection_dim=int(
            run["d"]
        ),
        adapter_kernel_size=int(
            adapter["kernel_size"]
        ),
        adapter_bias=bool(
            adapter["bias"]
        ),
        num_blocks=3,
        share_projection_across_views=bool(
            share_projection
        ),
        projection_bias=True,
        decoder_hidden_channels=None,
        output_path=output_path,
    )


def _build_benchmark_batch(
    *,
    dataset: CachedFeatureDataset,
    run: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> Mapping[str, Any]:
    if batch_size <= 0:
        raise CandidateEvaluationError(
            "benchmark_batch_size must be > 0."
        )

    effective = min(
        int(batch_size),
        len(dataset),
    )
    if effective <= 0:
        raise CandidateEvaluationError(
            "Cannot benchmark an empty dataset."
        )

    protocol = _require_mapping(
        run["protocol"],
        name="protocol",
    )
    data_cfg = _require_mapping(
        protocol.get("data"),
        name="protocol.data",
    )

    loader = make_cached_dataloader(
        dataset,
        batch_size=effective,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(
            data_cfg.get(
                "pin_memory",
                device.type == "cuda",
            )
        ),
        persistent_workers=False,
        drop_last=False,
    )

    batch = next(
        iter(loader)
    )
    return _move_to_device(
        batch,
        device,
    )


def _amp_settings(
    run: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[bool, torch.dtype, str]:
    protocol = _require_mapping(
        run["protocol"],
        name="protocol",
    )
    training = _require_mapping(
        protocol.get("training"),
        name="protocol.training",
    )
    amp_cfg = training.get(
        "amp",
        {},
    ) or {}

    if not isinstance(
        amp_cfg,
        Mapping,
    ):
        raise CandidateEvaluationError(
            "training.amp must be a mapping."
        )

    enabled = bool(
        amp_cfg.get(
            "enabled",
            False,
        )
    )

    name = str(
        amp_cfg.get(
            "dtype",
            "bfloat16",
        )
    ).lower()

    table = {
        "float16":
            (
                torch.float16,
                "fp16",
            ),

        "fp16":
            (
                torch.float16,
                "fp16",
            ),

        "bfloat16":
            (
                torch.bfloat16,
                "bf16",
            ),

        "bf16":
            (
                torch.bfloat16,
                "bf16",
            ),
    }

    if name not in table:
        raise CandidateEvaluationError(
            f"Unsupported AMP dtype: {name!r}."
        )

    dtype, label = table[name]

    if not enabled:
        # Cache is normally fp32 and the Day-04 locked protocol currently
        # disables AMP.
        return (
            False,
            dtype,
            "fp32",
        )

    if device.type != "cuda":
        raise CandidateEvaluationError(
            "Final E5/E6 AMP benchmark currently requires CUDA."
        )

    return (
        True,
        dtype,
        label,
    )


def run_e5_e6(
    *,
    model: nn.Module,
    dataset: CachedFeatureDataset,
    run: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    benchmark_batch_size: int,
    latency_warmup: int,
    latency_iterations: int,
    latency_rounds: int,
    latency_stability_cv_threshold: float,
    vram_warmup: int,
    vram_iterations: int,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise CandidateEvaluationError(
            "Final combined E5/E6 requires CUDA. "
            "Use a CUDA runtime or pass --skip-efficiency only for debugging."
        )

    batch = _build_benchmark_batch(
        dataset=dataset,
        run=run,
        batch_size=benchmark_batch_size,
        device=device,
    )

    local = batch.get(
        "local_b4"
    )
    mask = batch.get(
        "mask"
    )
    if not isinstance(
        local,
        Tensor,
    ):
        raise CandidateEvaluationError(
            "Benchmark batch missing local_b4 Tensor."
        )
    if not isinstance(
        mask,
        Tensor,
    ):
        raise CandidateEvaluationError(
            "Benchmark batch missing mask Tensor."
        )

    assert isinstance(
        local,
        Tensor,
    )
    assert isinstance(
        mask,
        Tensor,
    )

    amp_enabled, amp_dtype, precision = (
        _amp_settings(
            run,
            device=device,
        )
    )

    output_size = (
        int(mask.shape[-2]),
        int(mask.shape[-1]),
    )

    model.eval()

    def forward_once():
        ctx = (
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=True,
            )
            if amp_enabled
            else contextlib.nullcontext()
        )

        with ctx:
            return model(
                batch,
                output_size=output_size,
            )

    # Candidate-specific r/d is intentionally excluded from the scope
    # fingerprint. The scope must remain identical for comparable candidates.
    input_signature = (
        "six cached DINO feature maps: "
        f"[B={int(local.shape[0])},"
        f"C={int(local.shape[1])},"
        f"Hf={int(local.shape[2])},"
        f"Wf={int(local.shape[3])}]"
    )

    scope = make_benchmark_scope(
        scope_name="day04_cached_trainable_pipeline",
        device=device,
        batch_size=int(
            local.shape[0]
        ),
        precision=precision,
        input_signature=input_signature,
        pipeline_stages=(
            "adapter",
            "context_alignment",
            "projection",
            "attention_fusion",
            "decoder",
        ),
        extra={
            "output_size":
                list(
                    output_size
                ),

            "backbone_included":
                False,

            "image_loading_included":
                False,

            "tiling_included":
                False,

            "hann_stitching_included":
                False,
        },
    )

    return benchmark_inference_efficiency(
        forward_once,
        candidate_id=str(
            run["candidate_id"]
        ),
        scope=scope,
        output_path=output_path,
        latency_warmup=int(
            latency_warmup
        ),
        latency_iterations=int(
            latency_iterations
        ),
        latency_rounds=int(
            latency_rounds
        ),
        stability_cv_threshold=float(
            latency_stability_cv_threshold
        ),
        vram_warmup=int(
            vram_warmup
        ),
        vram_iterations=int(
            vram_iterations
        ),
        use_inference_mode=True,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def evaluate_candidate(
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            run_dir
        )

    if not (
        0.0
        <= float(
            args.seg_f1_threshold
        )
        <= 1.0
    ):
        raise CandidateEvaluationError(
            "--seg-f1-threshold must be in [0,1]."
        )

    if (
        args.aupro_num_thresholds
        is not None
        and int(
            args.aupro_num_thresholds
        )
        <= 1
    ):
        raise CandidateEvaluationError(
            "--aupro-num-thresholds must be > 1."
        )

    run = _parse_run_config(
        run_dir
    )

    outputs = _prepare_outputs(
        run_dir=run_dir,
        overwrite=bool(
            args.overwrite
        ),
        include_efficiency=not bool(
            args.skip_efficiency
        ),
    )

    expected_split = (
        _require_nonempty_string(
            args.expected_split,
            name="expected_split",
        )
    )

    dataset, _ = _build_val_dataset(
        run
    )

    eval_records, audit_rows = (
        build_real_eval_records(
            run_dir=run_dir,
            run=run,
            dataset=dataset,
            expected_split=expected_split,
        )
    )

    # Persist only provenance/path/shape metadata, not duplicated dense arrays.
    _atomic_jsonl(
        audit_rows,
        outputs["manifest"],
    )

    # E2 + E3. The repository evaluator writes qa_report.json BEFORE metrics.
    metrics = evaluate_segmentation_records(
        eval_records,
        seg_f1_threshold=float(
            args.seg_f1_threshold
        ),
        expected_split=expected_split,
        expected_categories=(
            str(
                run["category"]
            ),
        ),
        output_path=outputs[
            "metrics"
        ],
        qa_output_path=outputs[
            "qa"
        ],
        aupro_num_thresholds=int(
            args.aupro_num_thresholds
        ),
    )

    device = _resolve_device(
        args.device
    )

    model, in_channels = (
        _load_trained_model(
            run_dir=run_dir,
            run=run,
            dataset=dataset,
            device=device,
        )
    )

    params = run_e4(
        model=model,
        run=run,
        in_channels=in_channels,
        output_path=outputs[
            "params"
        ],
    )

    efficiency = None
    if not args.skip_efficiency:
        efficiency = run_e5_e6(
            model=model,
            dataset=dataset,
            run=run,
            device=device,
            output_path=outputs[
                "efficiency"
            ],
            benchmark_batch_size=int(
                args.benchmark_batch_size
            ),
            latency_warmup=int(
                args.latency_warmup
            ),
            latency_iterations=int(
                args.latency_iterations
            ),
            latency_rounds=int(
                args.latency_rounds
            ),
            latency_stability_cv_threshold=float(
                args.latency_stability_cv_threshold
            ),
            vram_warmup=int(
                args.vram_warmup
            ),
            vram_iterations=int(
                args.vram_iterations
            ),
        )

    category = str(
        run["category"]
    )

    result = {
        "status":
            (
                "PASS"
                if (
                    metrics[
                        "validation"
                    ][
                        "status"
                    ]
                    == "PASS"
                    and params[
                        "status"
                    ]
                    == "PASS"
                    and (
                        efficiency
                        is None
                        or efficiency[
                            "status"
                        ]
                        == "PASS"
                    )
                )
                else "FAIL"
            ),

        "candidate_id":
            run[
                "candidate_id"
            ],

        "category":
            category,

        "seed":
            int(
                run["seed"]
            ),

        "split":
            expected_split,

        "n_samples":
            int(
                metrics[
                    "n_samples"
                ]
            ),

        "aupro_0.05":
            float(
                metrics[
                    "metrics"
                ][
                    "aupro_0.05"
                ][
                    "per_category"
                ][
                    category
                ]
            ),

        "seg_f1":
            float(
                metrics[
                    "metrics"
                ][
                    "seg_f1"
                ][
                    "per_category"
                ][
                    category
                ][
                    "f1"
                ]
            ),

        "qa_status":
            metrics[
                "validation"
            ][
                "anomaly_map_qa"
            ][
                "status"
            ],

        "params_status":
            params[
                "status"
            ],

        "efficiency_status":
            (
                "SKIPPED"
                if efficiency is None
                else efficiency[
                    "status"
                ]
            ),

        "artifacts": {
            key:
                str(path)
            for key, path
            in outputs.items()
        },
    }

    return result


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(
        argv
    )
    result = evaluate_candidate(
        args
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    if (
        result["status"]
        != "PASS"
    ):
        raise SystemExit(
            2
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
