
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence
import numpy as np


def _as_array(x):
    return np.asarray(x)


def validate_segmentation_pair(
    anomaly_map,
    gt_mask,
    require_unit_range: bool = True,
):
    score = _as_array(anomaly_map).astype(np.float64)
    gt = _as_array(gt_mask)

    if score.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: anomaly_map={score.shape}, gt={gt.shape}"
        )

    if score.ndim != 2:
        raise ValueError(f"Expected HW arrays, got ndim={score.ndim}")

    if not np.isfinite(score).all():
        raise ValueError("anomaly_map contains NaN/Inf")

    if require_unit_range:
        if score.min(initial=0.0) < 0.0 or score.max(initial=0.0) > 1.0:
            raise ValueError(
                f"anomaly_map must be in [0,1], got [{score.min()}, {score.max()}]"
            )

    gt = gt > 0
    return score, gt


def segf1_stats(
    anomaly_map,
    gt_mask,
    threshold: float,
    require_unit_range: bool = True,
) -> Dict[str, float]:
    score, gt = validate_segmentation_pair(
        anomaly_map,
        gt_mask,
        require_unit_range=require_unit_range,
    )

    pred = score >= float(threshold)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 1.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def best_threshold_f1(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    thresholds: Sequence[float] | None = None,
) -> Dict[str, float]:
    if len(anomaly_maps) != len(gt_masks):
        raise ValueError("anomaly_maps and gt_masks length mismatch")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 501)

    best = None

    for t in thresholds:
        total_tp = total_fp = total_fn = 0

        for score, gt in zip(anomaly_maps, gt_masks):
            s = segf1_stats(score, gt, threshold=float(t))
            total_tp += s["tp"]
            total_fp += s["fp"]
            total_fn += s["fn"]

        denom = 2 * total_tp + total_fp + total_fn
        f1 = (2 * total_tp / denom) if denom else 1.0

        row = {
            "threshold": float(t),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "f1": float(f1),
        }

        if best is None or row["f1"] > best["f1"]:
            best = row

    return best


def aggregate_seg_f1(
    samples: Sequence[Mapping],
    threshold: float,
) -> Dict:
    # Each sample:
    # {"category": str, "anomaly_map": HW, "gt_mask": HW}
    by_cat = {}

    categories = sorted({str(s["category"]) for s in samples})

    for category in categories:
        subset = [s for s in samples if str(s["category"]) == category]

        tp = fp = fn = 0

        for item in subset:
            stats = segf1_stats(
                item["anomaly_map"],
                item["gt_mask"],
                threshold=threshold,
            )
            tp += stats["tp"]
            fp += stats["fp"]
            fn += stats["fn"]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 1.0

        by_cat[category] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    macro_f1 = float(np.mean([v["f1"] for v in by_cat.values()])) if by_cat else np.nan

    return {
        "threshold": float(threshold),
        "per_category": by_cat,
        "macro_f1": macro_f1,
    }
