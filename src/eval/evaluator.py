from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.metrics.aupro import aggregate_aupro
from src.metrics.segf1 import aggregate_seg_f1


AUPRO_MAX_FPR = 0.05
METRIC_PROTOCOL_VERSION = "1.0"
EVALUATOR_SCHEMA_VERSION = "msila-evaluator-v1"
QA_REPORT_SCHEMA_VERSION = "msila-anomaly-map-qa-v1"
_RANGE_TOL = 1e-6

# Canonical MVTec AD 2 category tokens used by this repository.
MVTEC_AD2_CATEGORIES: tuple[str, ...] = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)

# Official dataset splits plus the explicit development split used by the
# MS-ILA research protocol for synthetic supervision/model selection.
MVTEC_AD2_SPLITS: tuple[str, ...] = (
    "train",
    "validation",
    "test_public",
    "test_private",
    "test_private_mixed",
    "dev_synthetic",
)

_HIDDEN_GT_SPLITS = {"test_private", "test_private_mixed"}

# Optional provenance declarations accepted by E3.  These declarations are
# not required because existing dataset records already carry native H/W, but
# if they are present they must state that both prediction and GT live in the
# original-image coordinate system.
_ORIGINAL_SPACE_ALIASES = {
    "original": "original_image",
    "original_image": "original_image",
    "native": "original_image",
    "native_image": "original_image",
    "image": "original_image",
}


class EvaluationContractError(ValueError):
    """Raised when an evaluator input violates the MS-ILA evaluation contract."""


def _norm_token(value: Any) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[\s\-]+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token


def _canonical_category(value: Any) -> str:
    token = _norm_token(value)
    aliases = {
        "can": "can",
        "fabric": "fabric",
        "fruit_jelly": "fruit_jelly",
        "fruitjelly": "fruit_jelly",
        "rice": "rice",
        "sheet_metal": "sheet_metal",
        "sheetmetal": "sheet_metal",
        "vial": "vial",
        "wallplugs": "wallplugs",
        "wall_plugs": "wallplugs",
        "wall_plug": "wallplugs",
        "walnuts": "walnuts",
        "walnut": "walnuts",
    }
    if token not in aliases:
        raise EvaluationContractError(
            f"Unknown MVTec AD 2 category {value!r}. "
            f"Expected one of {MVTEC_AD2_CATEGORIES}."
        )
    return aliases[token]


def _canonical_split(value: Any) -> str:
    token = _norm_token(value)
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test_public": "test_public",
        "testpublic": "test_public",
        "public_test": "test_public",
        "test_private": "test_private",
        "testprivate": "test_private",
        "private_test": "test_private",
        "test_private_mixed": "test_private_mixed",
        "testprivatemixed": "test_private_mixed",
        "private_mixed": "test_private_mixed",
        "dev_synthetic": "dev_synthetic",
        "synthetic_dev": "dev_synthetic",
        "devsynthetic": "dev_synthetic",
    }
    if token not in aliases:
        raise EvaluationContractError(
            f"Unknown split {value!r}. Expected one of {MVTEC_AD2_SPLITS}."
        )
    return aliases[token]


def _meta_mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = record.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _meta_value(record: Mapping[str, Any], key: str) -> Any:
    meta = _meta_mapping(record)
    if key in meta:
        return meta[key]
    if key in record:
        return record[key]
    raise EvaluationContractError(f"Missing metadata field {key!r}")


def _optional_meta_value(record: Mapping[str, Any], key: str) -> Any | None:
    meta = _meta_mapping(record)
    if key in meta:
        return meta[key]
    return record.get(key)


def _image_id(record: Mapping[str, Any]) -> str:
    meta = _meta_mapping(record)
    for key in ("image_id", "path"):
        if key in meta and str(meta[key]).strip():
            return str(meta[key]).strip()
    for key in ("image_id", "path"):
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    raise EvaluationContractError(
        "Missing sample identity: provide meta.image_id or meta.path"
    )


def _validate_probability_map(value: Any, *, image_id: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise EvaluationContractError(
            f"{image_id}: anomaly_map must be 2-D [H,W], got shape={arr.shape}"
        )
    if arr.size == 0:
        raise EvaluationContractError(f"{image_id}: anomaly_map is empty")
    if not np.issubdtype(arr.dtype, np.number):
        raise EvaluationContractError(
            f"{image_id}: anomaly_map must be numeric, got dtype={arr.dtype}"
        )

    score = arr.astype(np.float64, copy=False)
    if not np.isfinite(score).all():
        raise EvaluationContractError(
            f"{image_id}: anomaly_map contains NaN or Inf"
        )

    lo = float(score.min())
    hi = float(score.max())
    if lo < -_RANGE_TOL or hi > 1.0 + _RANGE_TOL:
        raise EvaluationContractError(
            f"{image_id}: anomaly_map must be a probability map in [0,1]; "
            f"observed range=[{lo:.8g}, {hi:.8g}]"
        )

    # Tiny interpolation round-off is not a semantic range violation.
    return np.clip(score, 0.0, 1.0)


def _validate_binary_mask(value: Any, *, image_id: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise EvaluationContractError(
            f"{image_id}: gt_mask must be 2-D [H,W], got shape={arr.shape}"
        )
    if arr.size == 0:
        raise EvaluationContractError(f"{image_id}: gt_mask is empty")

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8, copy=False)

    if not np.issubdtype(arr.dtype, np.number):
        raise EvaluationContractError(
            f"{image_id}: gt_mask must be numeric/bool, got dtype={arr.dtype}"
        )

    numeric = arr.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise EvaluationContractError(f"{image_id}: gt_mask contains NaN or Inf")

    unique = set(np.unique(numeric).tolist())
    if unique.issubset({0.0, 1.0}):
        return (numeric > 0.0).astype(np.uint8)
    if unique.issubset({0.0, 255.0}):
        return (numeric > 0.0).astype(np.uint8)

    preview = sorted(unique)[:10]
    raise EvaluationContractError(
        f"{image_id}: gt_mask must be binary encoded as {{0,1}} or {{0,255}}; "
        f"observed values={preview}"
    )


def _coerce_positive_int(value: Any, *, field: str, image_id: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise EvaluationContractError(
            f"{image_id}: {field} must be a positive integer, got {value!r}"
        )
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(
            f"{image_id}: {field} must be a positive integer, got {value!r}"
        ) from exc

    if not math.isfinite(as_float) or not as_float.is_integer() or as_float <= 0:
        raise EvaluationContractError(
            f"{image_id}: {field} must be a positive integer, got {value!r}"
        )
    return int(as_float)


def _parse_hw(value: Any, *, field: str, image_id: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        if "H" in value and "W" in value:
            raw_h, raw_w = value["H"], value["W"]
        elif "h" in value and "w" in value:
            raw_h, raw_w = value["h"], value["w"]
        else:
            raise EvaluationContractError(
                f"{image_id}: {field} mapping must contain H/W or h/w"
            )
    else:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise EvaluationContractError(
                f"{image_id}: {field} must be a 2-item (H,W) sequence"
            )
        raw_h, raw_w = value[0], value[1]

    h = _coerce_positive_int(raw_h, field=f"{field}.H", image_id=image_id)
    w = _coerce_positive_int(raw_w, field=f"{field}.W", image_id=image_id)
    return h, w


def _resolve_original_hw(
    record: Mapping[str, Any],
    *,
    image_id: str,
) -> tuple[int, int]:
    """
    Resolve native image size without guessing.

    Preferred metadata is ``original_hw=(H,W)``.  The existing project loader
    stores native dimensions as ``meta.H`` and ``meta.W``; that pair is accepted
    as an equivalent source.  If both forms are present, they must agree.
    """
    candidates: list[tuple[str, tuple[int, int]]] = []

    original_hw = _optional_meta_value(record, "original_hw")
    if original_hw is not None:
        candidates.append(
            (
                "original_hw",
                _parse_hw(original_hw, field="original_hw", image_id=image_id),
            )
        )

    raw_h = _optional_meta_value(record, "H")
    raw_w = _optional_meta_value(record, "W")
    if (raw_h is None) ^ (raw_w is None):
        raise EvaluationContractError(
            f"{image_id}: original-size metadata must provide both H and W"
        )
    if raw_h is not None and raw_w is not None:
        hw = (
            _coerce_positive_int(raw_h, field="H", image_id=image_id),
            _coerce_positive_int(raw_w, field="W", image_id=image_id),
        )
        candidates.append(("H/W", hw))

    if not candidates:
        raise EvaluationContractError(
            f"{image_id}: missing original image size; provide meta.original_hw=(H,W) "
            "or the loader-native meta.H and meta.W"
        )

    reference = candidates[0][1]
    conflicts = [(name, hw) for name, hw in candidates if hw != reference]
    if conflicts:
        detail = ", ".join(f"{name}={hw}" for name, hw in candidates)
        raise EvaluationContractError(
            f"{image_id}: conflicting original-size metadata: {detail}"
        )
    return reference


def _canonical_coordinate_space(value: Any, *, field: str, image_id: str) -> str:
    token = _norm_token(value)
    if token not in _ORIGINAL_SPACE_ALIASES:
        raise EvaluationContractError(
            f"{image_id}: {field}={value!r} is not the original-image coordinate space"
        )
    return _ORIGINAL_SPACE_ALIASES[token]


def _check_declared_coordinate_space(
    record: Mapping[str, Any],
    *,
    image_id: str,
) -> dict[str, Any]:
    """
    Validate optional spatial-provenance metadata.

    A common ``coordinate_space`` applies to both maps.  Alternatively callers
    may declare ``anomaly_map_space`` and ``gt_mask_space`` separately.  If any
    declaration is provided, it must resolve both arrays to ``original_image``.
    """
    common = _optional_meta_value(record, "coordinate_space")
    score_space = _optional_meta_value(record, "anomaly_map_space")
    mask_space = _optional_meta_value(record, "gt_mask_space")

    declared = any(value is not None for value in (common, score_space, mask_space))
    if not declared:
        return {
            "declared": False,
            "anomaly_map_space": None,
            "gt_mask_space": None,
            "valid": True,
        }

    if common is not None:
        canonical_common = _canonical_coordinate_space(
            common,
            field="coordinate_space",
            image_id=image_id,
        )
        if score_space is None:
            score_space = canonical_common
        if mask_space is None:
            mask_space = canonical_common

    if score_space is None or mask_space is None:
        raise EvaluationContractError(
            f"{image_id}: partial coordinate-space metadata is ambiguous; declare "
            "both anomaly_map_space and gt_mask_space, or one common coordinate_space"
        )

    score_space = _canonical_coordinate_space(
        score_space,
        field="anomaly_map_space",
        image_id=image_id,
    )
    mask_space = _canonical_coordinate_space(
        mask_space,
        field="gt_mask_space",
        image_id=image_id,
    )

    if score_space != mask_space:
        raise EvaluationContractError(
            f"{image_id}: anomaly_map_space={score_space!r} != gt_mask_space={mask_space!r}"
        )

    return {
        "declared": True,
        "anomaly_map_space": score_space,
        "gt_mask_space": mask_space,
        "valid": True,
    }


def _safe_shape(value: Any) -> list[int] | None:
    try:
        return [int(x) for x in np.asarray(value).shape]
    except Exception:
        return None


def _sample_qa(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    """Run E3 QA on one sample and collect all detectable issues."""
    issues: list[str] = []

    try:
        image_id = _image_id(record)
    except EvaluationContractError as exc:
        image_id = f"record[{index}]"
        issues.append(str(exc))

    category = _optional_meta_value(record, "category")
    split = _optional_meta_value(record, "split")

    raw_score = record.get("anomaly_map") if isinstance(record, Mapping) else None
    raw_gt = record.get("gt_mask") if isinstance(record, Mapping) else None

    score_shape = _safe_shape(raw_score) if raw_score is not None else None
    gt_shape = _safe_shape(raw_gt) if raw_gt is not None else None

    score = None
    gt = None
    score_min = None
    score_max = None

    if raw_score is None:
        issues.append(f"{image_id}: missing 'anomaly_map'")
    else:
        try:
            score = _validate_probability_map(raw_score, image_id=image_id)
            score_min = float(score.min())
            score_max = float(score.max())
        except EvaluationContractError as exc:
            issues.append(str(exc))

    if raw_gt is None:
        issues.append(f"{image_id}: missing 'gt_mask'")
    else:
        try:
            gt = _validate_binary_mask(raw_gt, image_id=image_id)
        except EvaluationContractError as exc:
            issues.append(str(exc))

    same_hw = False
    if score is not None and gt is not None:
        same_hw = score.shape == gt.shape
        if not same_hw:
            issues.append(
                f"{image_id}: geometric alignment failed: "
                f"anomaly_map={score.shape}, gt_mask={gt.shape}"
            )

    original_hw: tuple[int, int] | None = None
    try:
        original_hw = _resolve_original_hw(record, image_id=image_id)
    except EvaluationContractError as exc:
        issues.append(str(exc))

    score_matches_original = False
    mask_matches_original = False
    if original_hw is not None:
        if score is not None:
            score_matches_original = tuple(score.shape) == original_hw
            if not score_matches_original:
                issues.append(
                    f"{image_id}: anomaly_map is not at original resolution: "
                    f"map={tuple(score.shape)}, original_hw={original_hw}"
                )
        if gt is not None:
            mask_matches_original = tuple(gt.shape) == original_hw
            if not mask_matches_original:
                issues.append(
                    f"{image_id}: gt_mask is not at original resolution: "
                    f"mask={tuple(gt.shape)}, original_hw={original_hw}"
                )

    try:
        coordinate_space = _check_declared_coordinate_space(
            record,
            image_id=image_id,
        )
    except EvaluationContractError as exc:
        coordinate_space = {
            "declared": True,
            "anomaly_map_space": None,
            "gt_mask_space": None,
            "valid": False,
        }
        issues.append(str(exc))

    status = "PASS" if not issues else "FAIL"
    return {
        "image_id": image_id,
        "category": None if category is None else str(category),
        "split": None if split is None else str(split),
        "status": status,
        "anomaly_map_hw": score_shape,
        "gt_mask_hw": gt_shape,
        "original_hw": None if original_hw is None else list(original_hw),
        "score_min": score_min,
        "score_max": score_max,
        "checks": {
            "probability_map_valid": score is not None,
            "binary_gt_valid": gt is not None,
            "map_mask_same_hw": same_hw,
            "map_matches_original_hw": score_matches_original,
            "mask_matches_original_hw": mask_matches_original,
            "declared_coordinate_space_valid": bool(coordinate_space["valid"]),
        },
        "coordinate_space": coordinate_space,
        "issues": issues,
    }


def build_anomaly_map_qa_report(
    records: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate anomaly-map geometry/integrity before metric computation.

    E3 hard requirements per sample:
      1. anomaly map is 2-D, finite and a probability map in [0,1];
      2. GT mask is 2-D and binary;
      3. anomaly map and GT have identical H/W;
      4. both H/W equal the native/original image H/W carried by metadata.

    Optional coordinate-space declarations, when present, must state that both
    arrays are in the original-image coordinate system.

    Important: equality of H/W proves geometric compatibility, not semantic
    registration. A left/right flip or spatial translation can still have the
    same shape; detecting that requires transformation provenance or qualitative
    image/GT/heatmap inspection.
    """
    if not records:
        raise EvaluationContractError("records must be non-empty")

    per_sample: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            per_sample.append(
                {
                    "image_id": f"record[{index}]",
                    "category": None,
                    "split": None,
                    "status": "FAIL",
                    "anomaly_map_hw": None,
                    "gt_mask_hw": None,
                    "original_hw": None,
                    "score_min": None,
                    "score_max": None,
                    "checks": {
                        "probability_map_valid": False,
                        "binary_gt_valid": False,
                        "map_mask_same_hw": False,
                        "map_matches_original_hw": False,
                        "mask_matches_original_hw": False,
                        "declared_coordinate_space_valid": False,
                    },
                    "coordinate_space": {
                        "declared": False,
                        "anomaly_map_space": None,
                        "gt_mask_space": None,
                        "valid": False,
                    },
                    "issues": [
                        f"record[{index}] must be a mapping, got {type(record)!r}"
                    ],
                }
            )
            continue
        per_sample.append(_sample_qa(record, index=index))

    n_samples = len(per_sample)
    n_pass = sum(item["status"] == "PASS" for item in per_sample)
    n_fail = n_samples - n_pass
    n_declared = sum(bool(item["coordinate_space"]["declared"]) for item in per_sample)

    check_names = (
        "probability_map_valid",
        "binary_gt_valid",
        "map_mask_same_hw",
        "map_matches_original_hw",
        "mask_matches_original_hw",
        "declared_coordinate_space_valid",
    )
    check_pass_counts = {
        name: sum(bool(item["checks"][name]) for item in per_sample)
        for name in check_names
    }

    report: dict[str, Any] = {
        "schema_version": QA_REPORT_SCHEMA_VERSION,
        "dataset": "mvtec_ad2",
        "requirements": {
            "anomaly_map": "2-D finite probability map in [0,1]",
            "gt_mask": "2-D binary mask encoded as {0,1} or {0,255}",
            "geometric_alignment": "anomaly_map_hw == gt_mask_hw == original_hw",
            "original_hw_source": "meta.original_hw=(H,W), or loader-native meta.H/meta.W",
            "coordinate_space": (
                "optional declaration; if present, anomaly map and GT must both be original_image"
            ),
        },
        "summary": {
            "status": "PASS" if n_fail == 0 else "FAIL",
            "n_samples": n_samples,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "valid_fraction": float(n_pass / n_samples),
            "n_coordinate_space_declared": n_declared,
            "n_coordinate_space_undeclared": n_samples - n_declared,
            "check_pass_counts": check_pass_counts,
        },
        "per_sample": per_sample,
        "limitations": [
            "Matching H/W and original_hw proves geometric compatibility, not semantic pixel registration.",
            "A same-size flip/translation can only be ruled out by spatial-transform provenance or qualitative overlay inspection.",
        ],
    }

    if output_path is not None:
        write_qa_report_json(report, output_path)
    return report


def _canonical_expected_categories(
    expected_categories: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if expected_categories is None:
        return None
    canonical = tuple(_canonical_category(x) for x in expected_categories)
    if not canonical:
        raise EvaluationContractError("expected_categories must be non-empty")
    if len(set(canonical)) != len(canonical):
        raise EvaluationContractError("expected_categories contains duplicates")
    return canonical


def _validate_and_normalize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_split: str | None,
    expected_categories: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], str, tuple[str, ...], dict[str, dict[str, int]]]:
    if not records:
        raise EvaluationContractError("records must be non-empty")

    canonical_expected_split = (
        _canonical_split(expected_split) if expected_split is not None else None
    )
    canonical_expected_categories = _canonical_expected_categories(
        expected_categories
    )

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    observed_splits: set[str] = set()
    observed_categories: set[str] = set()
    counts: dict[str, dict[str, int]] = {}

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EvaluationContractError(
                f"record[{index}] must be a mapping, got {type(record)!r}"
            )
        if "anomaly_map" not in record:
            raise EvaluationContractError(f"record[{index}] missing 'anomaly_map'")
        if "gt_mask" not in record:
            raise EvaluationContractError(f"record[{index}] missing 'gt_mask'")

        image_id = _image_id(record)
        if image_id in seen_ids:
            raise EvaluationContractError(f"Duplicate image_id/path: {image_id!r}")
        seen_ids.add(image_id)

        category = _canonical_category(_meta_value(record, "category"))
        split = _canonical_split(_meta_value(record, "split"))

        if split in _HIDDEN_GT_SPLITS:
            raise EvaluationContractError(
                f"{image_id}: split={split!r} has hidden ground truth in MVTec AD 2; "
                "local pixel-level AU-PRO/SegF1 evaluation is not an official protocol."
            )

        if canonical_expected_split is not None and split != canonical_expected_split:
            raise EvaluationContractError(
                f"{image_id}: split={split!r}, expected {canonical_expected_split!r}"
            )

        score = _validate_probability_map(record["anomaly_map"], image_id=image_id)
        gt = _validate_binary_mask(record["gt_mask"], image_id=image_id)

        if score.shape != gt.shape:
            raise EvaluationContractError(
                f"{image_id}: shape mismatch: anomaly_map={score.shape}, gt_mask={gt.shape}"
            )

        # E3 hard contract: both arrays must already be restored to native image
        # resolution before any pixel-level metric is computed.
        original_hw = _resolve_original_hw(record, image_id=image_id)
        if tuple(score.shape) != original_hw or tuple(gt.shape) != original_hw:
            raise EvaluationContractError(
                f"{image_id}: E3 original-size contract failed: "
                f"anomaly_map={score.shape}, gt_mask={gt.shape}, original_hw={original_hw}"
            )
        _check_declared_coordinate_space(record, image_id=image_id)

        is_anomalous = bool(gt.any())
        stats = counts.setdefault(
            category,
            {"samples": 0, "anomalous_samples": 0, "normal_samples": 0},
        )
        stats["samples"] += 1
        stats["anomalous_samples" if is_anomalous else "normal_samples"] += 1

        observed_splits.add(split)
        observed_categories.add(category)
        normalized.append(
            {
                "image_id": image_id,
                "category": category,
                "split": split,
                "original_hw": original_hw,
                "anomaly_map": score,
                "gt_mask": gt,
            }
        )

    if len(observed_splits) != 1:
        raise EvaluationContractError(
            "A single evaluator run must contain exactly one split; "
            f"observed={sorted(observed_splits)}"
        )
    split = next(iter(observed_splits))

    if canonical_expected_categories is not None:
        expected_set = set(canonical_expected_categories)
        missing = expected_set - observed_categories
        unexpected = observed_categories - expected_set
        if missing or unexpected:
            raise EvaluationContractError(
                "Category set mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        category_order = canonical_expected_categories
    else:
        category_order = tuple(
            cat for cat in MVTEC_AD2_CATEGORIES if cat in observed_categories
        )

    # AU-PRO is defined from anomalous connected regions and normal pixels.
    # Catch invalid category subsets here instead of allowing NaN/silent skips.
    for category in category_order:
        subset = [r for r in normalized if r["category"] == category]
        if not any(bool(r["gt_mask"].any()) for r in subset):
            raise EvaluationContractError(
                f"category={category!r}: AU-PRO requires at least one anomalous region"
            )
        if not any(bool((r["gt_mask"] == 0).any()) for r in subset):
            raise EvaluationContractError(
                f"category={category!r}: AU-PRO requires at least one normal pixel"
            )

    return normalized, split, category_order, counts


def _ordered_mapping(
    mapping: Mapping[str, Any],
    category_order: Sequence[str],
) -> dict[str, Any]:
    return {category: mapping[category] for category in category_order}


def _assert_finite_metric_tree(value: Any, *, path: str = "metrics") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_metric_tree(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_metric_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise EvaluationContractError(f"Non-finite metric produced at {path}: {value}")


def _default_qa_path(output_path: str | Path | None) -> Path | None:
    if output_path is None:
        return None
    path = Path(output_path)
    return path.with_name("qa_report.json")


def evaluate_segmentation_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seg_f1_threshold: float,
    expected_split: str | None = None,
    expected_categories: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    qa_output_path: str | Path | None = None,
    aupro_num_thresholds: int | None = 400,
) -> dict[str, Any]:
    """
    Validate one split and compute the locked MS-ILA segmentation metrics.

    E3 runs before metric computation. Every anomaly map and GT mask must be
    valid, share H/W, and already be restored to the original image H/W from
    metadata. If ``output_path`` is given and ``qa_output_path`` is omitted,
    the QA artifact is written beside metrics.json as ``qa_report.json``.
    """
    threshold = float(seg_f1_threshold)
    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise EvaluationContractError("seg_f1_threshold must be finite in [0,1]")

    resolved_qa_path = Path(qa_output_path) if qa_output_path is not None else _default_qa_path(output_path)
    qa_report = build_anomaly_map_qa_report(
        records,
        output_path=resolved_qa_path,
    )
    if qa_report["summary"]["status"] != "PASS":
        first_failed = next(
            item for item in qa_report["per_sample"] if item["status"] == "FAIL"
        )
        suffix = f" See {resolved_qa_path}." if resolved_qa_path is not None else ""
        raise EvaluationContractError(
            "Anomaly-map QA failed before metric computation: "
            f"{first_failed['image_id']}: {first_failed['issues'][0]}.{suffix}"
        )

    normalized, split, category_order, counts = _validate_and_normalize_records(
        records,
        expected_split=expected_split,
        expected_categories=expected_categories,
    )

    # Reuse the locked metric implementations; do not duplicate metric logic.
    try:
        au = aggregate_aupro(
            normalized,
            max_fpr=AUPRO_MAX_FPR,
            num_thresholds=aupro_num_thresholds,
        )
        sf1 = aggregate_seg_f1(
            normalized,
            threshold=threshold,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(
            f"Metric computation failed after input validation: {exc}"
        ) from exc

    au_per_category = _ordered_mapping(au["per_category"], category_order)
    sf1_per_category = _ordered_mapping(sf1["per_category"], category_order)
    ordered_counts = _ordered_mapping(counts, category_order)

    result: dict[str, Any] = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "dataset": "mvtec_ad2",
        "split": split,
        "n_samples": len(normalized),
        "n_categories": len(category_order),
        "categories": list(category_order),
        "settings": {
            "aupro_max_fpr": AUPRO_MAX_FPR,
            "seg_f1_threshold": threshold,
        },
        "metrics": {
            "aupro_0.05": {
                "per_category": au_per_category,
                "macro": float(au["macro_aupro"]),
            },
            "seg_f1": {
                "per_category": sf1_per_category,
                "macro": float(sf1["macro_f1"]),
            },
        },
        "counts": {
            "per_category": ordered_counts,
        },
        "validation": {
            "status": "PASS",
            "anomaly_map_contract": "2-D finite probability map in [0,1]",
            "gt_mask_contract": "2-D binary mask; accepted encodings {0,1} or {0,255}",
            "single_split": True,
            "unique_sample_ids": True,
            "anomaly_map_qa": {
                "schema_version": QA_REPORT_SCHEMA_VERSION,
                "status": qa_report["summary"]["status"],
                "valid_samples": qa_report["summary"]["n_pass"],
                "total_samples": qa_report["summary"]["n_samples"],
                "valid_fraction": qa_report["summary"]["valid_fraction"],
                "original_size_checked": True,
            },
        },
    }

    _assert_finite_metric_tree(result["metrics"])

    if output_path is not None:
        write_metrics_json(result, output_path)

    return result


def _write_json_atomic(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
            allow_nan=False,
        )
        handle.write("\n")

    tmp.replace(path)
    return path


def write_metrics_json(metrics: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write metrics JSON atomically and reject non-standard NaN/Inf JSON."""
    return _write_json_atomic(metrics, output_path)


def write_qa_report_json(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write E3 QA JSON atomically and reject non-standard NaN/Inf JSON."""
    return _write_json_atomic(report, output_path)
