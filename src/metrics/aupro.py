from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import label as cc_label

from .segf1 import validate_segmentation_pair


# 8-connected components in 2-D. This matches the full-connectivity convention
# commonly used by MVTec-style PRO implementations based on skimage.measure.label.
_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def _collect_scores(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Collect normal-pixel scores and per-region anomalous-pixel scores."""
    if len(anomaly_maps) != len(gt_masks):
        raise ValueError("anomaly_maps and gt_masks length mismatch")
    if len(anomaly_maps) == 0:
        raise ValueError("AU-PRO requires at least one image")

    normal_parts: list[np.ndarray] = []
    components: list[np.ndarray] = []

    for sample_idx, (score, gt) in enumerate(zip(anomaly_maps, gt_masks)):
        # AU-PRO is threshold/ranking based, so finite logits or probabilities are
        # both valid. Do not impose an artificial [0,1] restriction here.
        score, gt = validate_segmentation_pair(
            score,
            gt,
            require_unit_range=False,
        )

        normal_parts.append(score[~gt].ravel())

        labels, n_components = cc_label(
            gt.astype(np.uint8),
            structure=_CONNECTIVITY_8,
        )
        n_components = int(n_components)
        for component_id in range(1, n_components + 1):
            values = score[labels == component_id].ravel()
            if values.size:
                components.append(values)

    normal_scores = np.concatenate(normal_parts)

    if normal_scores.size == 0:
        raise ValueError("AU-PRO requires at least one normal pixel to define FPR")
    if not components:
        raise ValueError(
            "AU-PRO requires at least one anomalous connected component"
        )

    return normal_scores, components


def _aggregate_component_events(
    components: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return anomaly-score event values and their exact PRO increments.

    Each connected component contributes total mass 1/K, independent of region
    area. A pixel in component C_k therefore contributes 1/(K*|C_k|).
    """
    n_regions = len(components)
    value_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    for values in components:
        unique_values, counts = np.unique(values, return_counts=True)
        value_parts.append(unique_values)
        weight_parts.append(
            counts.astype(np.float64) / (n_regions * values.size)
        )

    values = np.concatenate(value_parts)
    weights = np.concatenate(weight_parts)

    unique_values, inverse = np.unique(values, return_inverse=True)
    weight_by_value = np.bincount(
        inverse,
        weights=weights,
        minlength=unique_values.size,
    ).astype(np.float64, copy=False)

    return unique_values, weight_by_value


def _curve_exact(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the exact empirical PRO-vs-FPR curve.

    The sweep changes threshold only at observed score values. Equal scores are
    processed as one event, which gives deterministic tie handling and removes
    dependence on an arbitrary number of sampled thresholds.
    """
    normal_scores, components = _collect_scores(anomaly_maps, gt_masks)

    normal_values, normal_counts = np.unique(
        normal_scores,
        return_counts=True,
    )
    anomaly_values, pro_increments = _aggregate_component_events(components)

    event_values = np.union1d(normal_values, anomaly_values)  # ascending
    normal_increments = np.zeros(event_values.size, dtype=np.float64)
    pro_event_increments = np.zeros(event_values.size, dtype=np.float64)

    normal_positions = np.searchsorted(event_values, normal_values)
    anomaly_positions = np.searchsorted(event_values, anomaly_values)

    normal_increments[normal_positions] = (
        normal_counts.astype(np.float64) / normal_scores.size
    )
    pro_event_increments[anomaly_positions] = pro_increments

    # Lowering the threshold includes score groups from high to low.
    fpr = np.concatenate([
        np.array([0.0], dtype=np.float64),
        np.cumsum(normal_increments[::-1]),
    ])
    pro = np.concatenate([
        np.array([0.0], dtype=np.float64),
        np.cumsum(pro_event_increments[::-1]),
    ])

    # Eliminate only round-off at the mathematically exact endpoint.
    fpr[-1] = 1.0
    pro[-1] = 1.0

    return fpr, pro


def _normalized_partial_auc(
    fpr: np.ndarray,
    pro: np.ndarray,
    max_fpr: float,
) -> float:
    """Integrate PRO over FPR in [0, max_fpr] and normalize by max_fpr."""
    # Keep all vertical segments at the endpoint; they have zero integration width.
    right = int(np.searchsorted(fpr, max_fpr, side="right"))
    x = fpr[:right].copy()
    y = pro[:right].copy()

    if x.size == 0:
        raise RuntimeError("Internal AU-PRO curve does not start at FPR=0")

    if x[-1] < max_fpr:
        if right >= fpr.size:
            raise RuntimeError("Internal AU-PRO curve does not reach max_fpr")

        x0, y0 = float(x[-1]), float(y[-1])
        x1, y1 = float(fpr[right]), float(pro[right])
        if not x1 > x0:
            raise RuntimeError("Internal AU-PRO curve is not monotonic in FPR")

        alpha = (max_fpr - x0) / (x1 - x0)
        y_at_max = y0 + alpha * (y1 - y0)
        x = np.concatenate([x, [max_fpr]])
        y = np.concatenate([y, [y_at_max]])

    area = float(np.trapz(y, x))
    normalized = area / max_fpr

    # Numerical round-off only; a valid normalized AU-PRO is in [0, 1].
    return float(np.clip(normalized, 0.0, 1.0))


def aupro(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    max_fpr: float = 0.05,
    num_thresholds: int | None = None,
    return_curve: bool = False,
):
    """
    Compute normalized AU-PRO up to ``max_fpr``.

    ``num_thresholds`` is retained only for backward compatibility with the
    previous sampled implementation. The metric is now exact on the empirical
    score values, so changing this argument does not change the result.
    """
    if not (0.0 < float(max_fpr) <= 1.0):
        raise ValueError("max_fpr must be in (0,1]")

    if num_thresholds is not None:
        if not isinstance(num_thresholds, (int, np.integer)):
            raise TypeError("num_thresholds must be an integer or None")
        if int(num_thresholds) <= 0:
            raise ValueError("num_thresholds must be > 0")

    fpr, pro = _curve_exact(anomaly_maps, gt_masks)
    value = _normalized_partial_auc(fpr, pro, float(max_fpr))

    out: dict[str, float | np.ndarray] = {
        "aupro": value,
        "max_fpr": float(max_fpr),
    }

    if return_curve:
        out["fpr"] = fpr
        out["pro"] = pro

    return out


def aggregate_aupro(
    samples: Sequence[Mapping],
    max_fpr: float = 0.05,
    num_thresholds: int | None = None,
) -> Dict:
    """Compute AU-PRO per category and the unweighted macro mean."""
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")

    categories = sorted({str(sample["category"]) for sample in samples})
    per_category: Dict[str, float] = {}

    for category in categories:
        subset = [
            sample
            for sample in samples
            if str(sample["category"]) == category
        ]
        maps = [np.asarray(sample["anomaly_map"]) for sample in subset]
        masks = [np.asarray(sample["gt_mask"]) for sample in subset]

        per_category[category] = float(
            aupro(
                maps,
                masks,
                max_fpr=max_fpr,
                num_thresholds=num_thresholds,
            )["aupro"]
        )

    return {
        "per_category": per_category,
        "macro_aupro": float(np.mean(list(per_category.values()))),
        "max_fpr": float(max_fpr),
    }
