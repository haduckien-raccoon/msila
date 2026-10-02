import numpy as np
import pytest
from scipy.ndimage import label as cc_label

from src.metrics.aupro import aupro, aggregate_aupro


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def _reference_partial_auc(fpr, pro, max_fpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    pro = np.asarray(pro, dtype=np.float64)

    right = int(np.searchsorted(fpr, max_fpr, side="right"))
    x = fpr[:right].copy()
    y = pro[:right].copy()

    if x[-1] < max_fpr:
        x0, y0 = x[-1], y[-1]
        x1, y1 = fpr[right], pro[right]
        alpha = (max_fpr - x0) / (x1 - x0)
        x = np.append(x, max_fpr)
        y = np.append(y, y0 + alpha * (y1 - y0))

    return float(np.trapz(y, x) / max_fpr)


def _reference_aupro(anomaly_maps, gt_masks, max_fpr=0.05):
    """Slow, direct implementation used only as an independent test oracle."""
    if len(anomaly_maps) != len(gt_masks) or not anomaly_maps:
        raise ValueError

    maps = [np.asarray(x, dtype=np.float64) for x in anomaly_maps]
    masks = [np.asarray(x) > 0 for x in gt_masks]

    components = []
    normal_pixels = 0
    for mask in masks:
        normal_pixels += int((~mask).sum())
        labels, n_components = cc_label(mask, structure=CONNECTIVITY_8)
        components.extend([
            labels == component_id
            for component_id in range(1, n_components + 1)
        ])

    if normal_pixels == 0 or not components:
        raise ValueError

    thresholds = np.unique(np.concatenate([m.ravel() for m in maps]))[::-1]
    fprs = [0.0]
    pros = [0.0]

    for threshold in thresholds:
        predictions = [score >= threshold for score in maps]

        fp = sum(
            int(np.logical_and(pred, ~mask).sum())
            for pred, mask in zip(predictions, masks)
        )
        fprs.append(fp / normal_pixels)

        region_overlaps = []
        component_idx = 0
        for pred, mask in zip(predictions, masks):
            labels, n_components = cc_label(mask, structure=CONNECTIVITY_8)
            for component_id in range(1, n_components + 1):
                region = labels == component_id
                region_overlaps.append(float(pred[region].mean()))
                component_idx += 1

        pros.append(float(np.mean(region_overlaps)))

    return _reference_partial_auc(fprs, pros, max_fpr)


def test_aupro_matches_independent_reference():
    rng = np.random.default_rng(2026)

    for _ in range(8):
        gt = np.zeros((12, 13), dtype=np.uint8)
        gt[1:3, 1:4] = 1
        gt[7:10, 8:12] = 1
        score = rng.normal(size=gt.shape)

        expected = _reference_aupro([score], [gt], max_fpr=0.05)
        actual = aupro([score], [gt], max_fpr=0.05)["aupro"]

        assert actual == pytest.approx(expected, abs=1e-12)


def test_perfect_localization_is_one():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[4:8, 5:9] = 1
    gt[20:25, 22:27] = 1

    score = np.zeros_like(gt, dtype=np.float64)
    score[gt > 0] = 1.0

    assert aupro([score], [gt])["aupro"] == pytest.approx(1.0, abs=1e-12)


def test_reversed_localization_is_zero_at_low_fpr():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:16, 8:16] = 1

    score = np.ones_like(gt, dtype=np.float64)
    score[gt > 0] = 0.0

    assert aupro([score], [gt])["aupro"] == pytest.approx(0.0, abs=1e-12)


def test_regions_are_weighted_equally_not_by_area():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[1, 1] = 1                 # region 1: 1 pixel
    gt[10:14, 10:14] = 1         # region 2: 16 pixels

    score = np.full(gt.shape, 0.2, dtype=np.float64)
    score[1, 1] = 0.9             # detect small region perfectly at FPR=0
    score[10:14, 10:14] = 0.1     # miss large region before false positives

    # PRO at low FPR is (1 + 0) / 2 = 0.5, not 1/17.
    assert aupro([score], [gt])["aupro"] == pytest.approx(0.5, abs=1e-12)


def test_connectivity_is_explicitly_8_connected():
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[1, 1] = 1
    gt[1, 2] = 1
    gt[2, 2] = 1
    gt[3, 3] = 1  # diagonal contact -> same region under 8-connectivity

    score = np.full(gt.shape, 0.2, dtype=np.float64)
    score[1, 1] = score[1, 2] = score[2, 2] = 0.9
    score[3, 3] = 0.1

    # One 4-pixel region: 3/4 detected at FPR=0.
    assert aupro([score], [gt])["aupro"] == pytest.approx(0.75, abs=1e-12)


def test_strictly_monotonic_score_transform_does_not_change_aupro():
    rng = np.random.default_rng(42)
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[3:6, 4:8] = 1
    gt[11:14, 12:15] = 1
    score = rng.normal(size=gt.shape)

    base = aupro([score], [gt])["aupro"]
    transformed = aupro([3.7 * score - 2.1], [gt])["aupro"]

    assert transformed == pytest.approx(base, abs=1e-12)


def test_legacy_num_thresholds_no_longer_changes_result():
    rng = np.random.default_rng(7)
    gt = np.zeros((14, 14), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    score = rng.random(gt.shape)

    coarse = aupro([score], [gt], num_thresholds=20)["aupro"]
    dense = aupro([score], [gt], num_thresholds=5000)["aupro"]

    assert coarse == pytest.approx(dense, abs=0.0)


def test_category_and_macro_aggregation():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:10, 5:10] = 1

    perfect = np.zeros_like(gt, dtype=np.float64)
    perfect[gt > 0] = 1.0

    reversed_score = np.ones_like(gt, dtype=np.float64)
    reversed_score[gt > 0] = 0.0

    out = aggregate_aupro([
        {"category": "A", "anomaly_map": perfect, "gt_mask": gt},
        {"category": "B", "anomaly_map": reversed_score, "gt_mask": gt},
    ])

    assert out["per_category"]["A"] == pytest.approx(1.0, abs=1e-12)
    assert out["per_category"]["B"] == pytest.approx(0.0, abs=1e-12)
    assert out["macro_aupro"] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("bad_max_fpr", [0.0, -0.1, 1.01])
def test_invalid_max_fpr_is_rejected(bad_max_fpr):
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:4, 2:4] = 1
    score = np.zeros_like(gt, dtype=np.float64)

    with pytest.raises(ValueError, match="max_fpr"):
        aupro([score], [gt], max_fpr=bad_max_fpr)


def test_invalid_inputs_fail_loudly():
    valid_gt = np.zeros((8, 8), dtype=np.uint8)
    valid_gt[2:4, 2:4] = 1
    valid_score = np.zeros_like(valid_gt, dtype=np.float64)

    with pytest.raises(ValueError, match="length mismatch"):
        aupro([valid_score], [])

    with pytest.raises(ValueError, match="at least one image"):
        aupro([], [])

    no_anomaly = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="anomalous connected component"):
        aupro([valid_score], [no_anomaly])

    all_anomaly = np.ones((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="normal pixel"):
        aupro([valid_score], [all_anomaly])

    bad_score = valid_score.copy()
    bad_score[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        aupro([bad_score], [valid_gt])
