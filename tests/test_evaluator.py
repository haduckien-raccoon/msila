from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import evaluator as evaluator


def _record(
    *,
    image_id: str = "fabric/sample_001.png",
    category: str = "fabric",
    split: str = "dev_synthetic",
    score: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    hw: tuple[int, int] = (4, 4),
    coordinate_space: str | None = "original_image",
):
    h, w = hw

    if score is None:
        score = np.full((h, w), 0.05, dtype=np.float32)
        score[1:3, 1:3] = 0.95

    if gt is None:
        gt = np.zeros((h, w), dtype=np.uint8)
        gt[1:3, 1:3] = 1

    meta = {
        "image_id": image_id,
        "category": category,
        "split": split,
        "H": h,
        "W": w,
    }
    if coordinate_space is not None:
        meta["coordinate_space"] = coordinate_space

    return {
        "anomaly_map": score,
        "gt_mask": gt,
        "meta": meta,
    }


def test_perfect_predictions_give_one_for_both_metrics():
    records = [
        _record(
            image_id="fabric/a.png",
            category="fabric",
        ),
        _record(
            image_id="vial/a.png",
            category="vial",
        ),
    ]

    result = evaluator.evaluate_segmentation_records(
        records,
        seg_f1_threshold=0.5,
        expected_split="dev_synthetic",
        expected_categories=("fabric", "vial"),
    )

    assert result["validation"]["status"] == "PASS"
    assert result["validation"]["anomaly_map_qa"]["valid_fraction"] == 1.0

    assert result["metrics"]["aupro_0.05"]["per_category"]["fabric"] == pytest.approx(1.0)
    assert result["metrics"]["aupro_0.05"]["per_category"]["vial"] == pytest.approx(1.0)
    assert result["metrics"]["aupro_0.05"]["macro"] == pytest.approx(1.0)

    assert result["metrics"]["seg_f1"]["per_category"]["fabric"]["f1"] == pytest.approx(1.0)
    assert result["metrics"]["seg_f1"]["per_category"]["vial"]["f1"] == pytest.approx(1.0)
    assert result["metrics"]["seg_f1"]["macro"] == pytest.approx(1.0)


def test_segf1_matches_hand_counted_confusion_matrix():
    # threshold=0.5:
    # TP=1 (0.9), FP=1 (0.8), FN=0 -> F1 = 2/(2+1)=2/3.
    score = np.array(
        [
            [0.9, 0.8],
            [0.2, 0.1],
        ],
        dtype=np.float64,
    )
    gt = np.array(
        [
            [1, 0],
            [0, 0],
        ],
        dtype=np.uint8,
    )

    result = evaluator.evaluate_segmentation_records(
        [
            _record(
                image_id="fabric/hand_count.png",
                score=score,
                gt=gt,
                hw=(2, 2),
            )
        ],
        seg_f1_threshold=0.5,
        expected_split="dev_synthetic",
        expected_categories=("fabric",),
    )

    stats = result["metrics"]["seg_f1"]["per_category"]["fabric"]
    assert stats["tp"] == 1
    assert stats["fp"] == 1
    assert stats["fn"] == 0
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(1.0)
    assert stats["f1"] == pytest.approx(2.0 / 3.0)

    # The anomalous pixel is ranked above all normal pixels, so low-FPR PRO is 1.
    assert result["metrics"]["aupro_0.05"]["macro"] == pytest.approx(1.0)


def test_expected_category_order_is_preserved():
    records = [
        _record(image_id="fabric/a.png", category="fabric"),
        _record(image_id="vial/a.png", category="vial"),
    ]

    result = evaluator.evaluate_segmentation_records(
        records,
        seg_f1_threshold=0.5,
        expected_categories=("vial", "fabric"),
    )

    assert result["categories"] == ["vial", "fabric"]
    assert list(result["metrics"]["aupro_0.05"]["per_category"]) == ["vial", "fabric"]
    assert list(result["metrics"]["seg_f1"]["per_category"]) == ["vial", "fabric"]


def test_metric_and_qa_json_are_written_atomically(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"

    result = evaluator.evaluate_segmentation_records(
        [_record()],
        seg_f1_threshold=0.5,
        output_path=metrics_path,
    )

    qa_path = tmp_path / "qa_report.json"

    assert metrics_path.is_file()
    assert qa_path.is_file()
    assert not (tmp_path / "metrics.json.tmp").exists()
    assert not (tmp_path / "qa_report.json.tmp").exists()

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved_qa = json.loads(qa_path.read_text(encoding="utf-8"))

    assert saved_metrics == result
    assert saved_qa["summary"]["status"] == "PASS"
    assert saved_qa["summary"]["valid_fraction"] == 1.0


@pytest.mark.parametrize(
    ("field", "mutator", "match"),
    [
        (
            "nan",
            lambda r: r["anomaly_map"].__setitem__((0, 0), np.nan),
            "NaN|QA failed",
        ),
        (
            "range",
            lambda r: r["anomaly_map"].__setitem__((0, 0), 1.2),
            r"\[0,1\]|QA failed",
        ),
    ],
)
def test_invalid_probability_map_is_rejected(field, mutator, match):
    record = _record()
    mutator(record)

    with pytest.raises(evaluator.EvaluationContractError, match=match):
        evaluator.evaluate_segmentation_records(
            [record],
            seg_f1_threshold=0.5,
        )


def test_nonbinary_gt_is_rejected():
    record = _record()
    record["gt_mask"][0, 0] = 2

    with pytest.raises(evaluator.EvaluationContractError, match="binary|QA failed"):
        evaluator.evaluate_segmentation_records(
            [record],
            seg_f1_threshold=0.5,
        )


def test_map_mask_shape_mismatch_is_rejected():
    record = _record()
    record["anomaly_map"] = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(evaluator.EvaluationContractError, match="QA failed|shape"):
        evaluator.evaluate_segmentation_records(
            [record],
            seg_f1_threshold=0.5,
        )


def test_matching_map_mask_but_wrong_original_size_is_rejected():
    record = _record()
    record["meta"]["H"] = 8
    record["meta"]["W"] = 8

    with pytest.raises(
        evaluator.EvaluationContractError,
        match="QA failed|original",
    ):
        evaluator.evaluate_segmentation_records(
            [record],
            seg_f1_threshold=0.5,
        )


def test_qa_fails_before_metrics_are_called(monkeypatch):
    calls = {"aupro": 0, "segf1": 0}

    def bomb_aupro(*args, **kwargs):
        calls["aupro"] += 1
        raise AssertionError("AU-PRO must not run after E3 QA failure")

    def bomb_segf1(*args, **kwargs):
        calls["segf1"] += 1
        raise AssertionError("SegF1 must not run after E3 QA failure")

    monkeypatch.setattr(evaluator, "aggregate_aupro", bomb_aupro)
    monkeypatch.setattr(evaluator, "aggregate_seg_f1", bomb_segf1)

    record = _record()
    record["anomaly_map"] = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(evaluator.EvaluationContractError, match="QA failed"):
        evaluator.evaluate_segmentation_records(
            [record],
            seg_f1_threshold=0.5,
        )

    assert calls == {"aupro": 0, "segf1": 0}


def test_duplicate_sample_identity_is_rejected():
    records = [
        _record(image_id="fabric/duplicate.png"),
        _record(image_id="fabric/duplicate.png"),
    ]

    with pytest.raises(evaluator.EvaluationContractError, match="Duplicate"):
        evaluator.evaluate_segmentation_records(
            records,
            seg_f1_threshold=0.5,
        )


def test_mixed_split_is_rejected():
    records = [
        _record(image_id="fabric/a.png", split="dev_synthetic"),
        _record(image_id="fabric/b.png", split="test_public"),
    ]

    with pytest.raises(evaluator.EvaluationContractError, match="one split"):
        evaluator.evaluate_segmentation_records(
            records,
            seg_f1_threshold=0.5,
        )


@pytest.mark.parametrize("split", ["test_private", "test_private_mixed"])
def test_hidden_private_ground_truth_split_is_rejected(split):
    with pytest.raises(
        evaluator.EvaluationContractError,
        match="hidden ground truth",
    ):
        evaluator.evaluate_segmentation_records(
            [_record(split=split)],
            seg_f1_threshold=0.5,
        )


def test_expected_category_set_mismatch_is_rejected():
    with pytest.raises(evaluator.EvaluationContractError, match="Category set mismatch"):
        evaluator.evaluate_segmentation_records(
            [_record(category="fabric")],
            seg_f1_threshold=0.5,
            expected_categories=("fabric", "vial"),
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_segf1_threshold_is_rejected(threshold):
    with pytest.raises(
        evaluator.EvaluationContractError,
        match="seg_f1_threshold",
    ):
        evaluator.evaluate_segmentation_records(
            [_record()],
            seg_f1_threshold=threshold,
        )


def test_optional_coordinate_space_must_be_original_image():
    record = _record()
    record["meta"]["coordinate_space"] = "tile"

    with pytest.raises(
        evaluator.EvaluationContractError,
        match="QA failed|coordinate",
    ):
        evaluator.evaluate_segmentation_records(
            [record],
            seg_f1_threshold=0.5,
        )


def test_loader_native_hw_without_coordinate_declaration_is_valid():
    record = _record(coordinate_space=None)

    qa = evaluator.build_anomaly_map_qa_report([record])

    assert qa["summary"]["status"] == "PASS"
    assert qa["summary"]["n_coordinate_space_declared"] == 0
    assert qa["summary"]["n_coordinate_space_undeclared"] == 1
