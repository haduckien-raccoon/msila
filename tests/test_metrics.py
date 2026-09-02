
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.segf1 import (
    segf1_stats,
    best_threshold_f1,
    aggregate_seg_f1,
)
from src.metrics.aupro import (
    aupro,
    aggregate_aupro,
)


def test_segf1_tp_fp_fn():
    gt = np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.uint8)

    score = np.array([
        [0.1, 0.9],
        [0.8, 0.7],
    ], dtype=np.float32)

    s = segf1_stats(score, gt, threshold=0.75)

    assert s["tp"] == 2
    assert s["fp"] == 0
    assert s["fn"] == 0
    assert s["f1"] == 1.0


def test_shape_check():
    gt = np.zeros((4, 4), dtype=np.uint8)
    score = np.zeros((5, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        segf1_stats(score, gt, threshold=0.5)


def test_nan_inf_check():
    gt = np.zeros((4, 4), dtype=np.uint8)

    score_nan = np.zeros((4, 4), dtype=np.float32)
    score_nan[0, 0] = np.nan

    with pytest.raises(ValueError):
        segf1_stats(score_nan, gt, threshold=0.5)

    score_inf = np.zeros((4, 4), dtype=np.float32)
    score_inf[0, 0] = np.inf

    with pytest.raises(ValueError):
        segf1_stats(score_inf, gt, threshold=0.5)


def test_range_check():
    gt = np.zeros((4, 4), dtype=np.uint8)
    score = np.ones((4, 4), dtype=np.float32) * 1.2

    with pytest.raises(ValueError):
        segf1_stats(score, gt, threshold=0.5)


def test_best_threshold_validation_interface():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[3:7, 3:7] = 1

    score = np.zeros((10, 10), dtype=np.float32)
    score[3:7, 3:7] = 0.9
    score[0:2, 0:2] = 0.2

    out = best_threshold_f1(
        [score],
        [gt],
        thresholds=[0.1, 0.5, 0.95],
    )

    assert out["threshold"] == 0.5
    assert out["f1"] == 1.0


def test_perfect_aupro_is_one():
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[10:20, 10:20] = 1
    gt[40:50, 42:55] = 1

    score = np.zeros((64, 64), dtype=np.float32)
    score[gt > 0] = 1.0

    out = aupro(
        [score],
        [gt],
        max_fpr=0.05,
        num_thresholds=100,
    )

    assert abs(out["aupro"] - 1.0) < 1e-6


def test_macro_aggregation():
    samples = []

    for cat in ["a", "b"]:
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:16, 8:16] = 1

        score = np.zeros((32, 32), dtype=np.float32)
        score[gt > 0] = 1.0

        samples.append({
            "category": cat,
            "anomaly_map": score,
            "gt_mask": gt,
        })

    f1 = aggregate_seg_f1(samples, threshold=0.5)
    ap = aggregate_aupro(samples, max_fpr=0.05, num_thresholds=100)

    assert abs(f1["macro_f1"] - 1.0) < 1e-8
    assert abs(ap["macro_aupro"] - 1.0) < 1e-6
