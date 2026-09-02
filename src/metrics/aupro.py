
from __future__ import annotations

from typing import Dict, Mapping, Sequence
import numpy as np
from scipy.ndimage import label as cc_label

from .segf1 import validate_segmentation_pair


def _sorted_component_scores(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
):
    normal_scores = []
    components = []

    score_min = np.inf
    score_max = -np.inf

    for score, gt in zip(anomaly_maps, gt_masks):
        score, gt = validate_segmentation_pair(
            score,
            gt,
            require_unit_range=True,
        )

        score_min = min(score_min, float(score.min()))
        score_max = max(score_max, float(score.max()))

        normal_scores.append(score[~gt].ravel())

        labels, ncc = cc_label(gt.astype(np.uint8))

        for cid in range(1, ncc + 1):
            vals = score[labels == cid].ravel()

            if vals.size:
                components.append(np.sort(vals))

    if not normal_scores:
        raise ValueError("No images")

    normal_scores = np.sort(np.concatenate(normal_scores))

    if len(components) == 0:
        raise ValueError("AU-PRO requires at least one anomalous connected component")

    return normal_scores, components, score_min, score_max


def _curve(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    num_thresholds: int = 400,
):
    normal_scores, components, smin, smax = _sorted_component_scores(
        anomaly_maps,
        gt_masks,
    )

    span = max(smax - smin, 1e-12)

    thresholds = np.concatenate([
        [smax + span * 1e-6 + 1e-12],
        np.linspace(smax, smin, int(num_thresholds)),
        [smin - span * 1e-6 - 1e-12],
    ])

    n_normal = normal_scores.size

    fprs = []
    pros = []

    for t in thresholds:
        # Prediction: score >= threshold.
        idx = np.searchsorted(normal_scores, t, side="left")
        fp = n_normal - idx
        fpr = fp / max(n_normal, 1)

        component_recalls = []

        for vals in components:
            j = np.searchsorted(vals, t, side="left")
            predicted = vals.size - j
            component_recalls.append(predicted / vals.size)

        pro = float(np.mean(component_recalls))

        fprs.append(float(fpr))
        pros.append(pro)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)

    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    # Multiple thresholds may give same FPR. Keep maximum PRO
    # because vertical segments have zero width in integration.
    unique_fpr = []
    unique_pro = []

    for f in np.unique(fprs):
        unique_fpr.append(f)
        unique_pro.append(pros[fprs == f].max())

    return np.asarray(unique_fpr), np.asarray(unique_pro)


def aupro(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    max_fpr: float = 0.05,
    num_thresholds: int = 400,
    return_curve: bool = False,
):
    if len(anomaly_maps) != len(gt_masks):
        raise ValueError("anomaly_maps and gt_masks length mismatch")

    if not (0.0 < max_fpr <= 1.0):
        raise ValueError("max_fpr must be in (0,1]")

    fpr, pro = _curve(
        anomaly_maps,
        gt_masks,
        num_thresholds=num_thresholds,
    )

    # Ensure FPR=0 exists.
    if fpr[0] > 0:
        fpr = np.concatenate([[0.0], fpr])
        pro = np.concatenate([[0.0], pro])

    # Interpolate exact endpoint max_fpr.
    pro_at_max = float(np.interp(max_fpr, fpr, pro))

    keep = fpr < max_fpr
    x = np.concatenate([fpr[keep], [max_fpr]])
    y = np.concatenate([pro[keep], [pro_at_max]])

    # Ensure sorted and unique after endpoint insertion.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    area = float(np.trapezoid(y, x))
    normalized = area / max_fpr

    out = {
        "aupro": float(normalized),
        "max_fpr": float(max_fpr),
    }

    if return_curve:
        out["fpr"] = fpr
        out["pro"] = pro

    return out


def aggregate_aupro(
    samples: Sequence[Mapping],
    max_fpr: float = 0.05,
    num_thresholds: int = 400,
) -> Dict:
    categories = sorted({str(s["category"]) for s in samples})
    per_category = {}

    for category in categories:
        subset = [s for s in samples if str(s["category"]) == category]

        maps = [np.asarray(s["anomaly_map"]) for s in subset]
        masks = [np.asarray(s["gt_mask"]) for s in subset]

        try:
            value = aupro(
                maps,
                masks,
                max_fpr=max_fpr,
                num_thresholds=num_thresholds,
            )["aupro"]
        except ValueError as e:
            value = np.nan

        per_category[category] = float(value) if np.isfinite(value) else np.nan

    valid = [v for v in per_category.values() if np.isfinite(v)]

    return {
        "per_category": per_category,
        "macro_aupro": float(np.mean(valid)) if valid else np.nan,
        "max_fpr": float(max_fpr),
    }
