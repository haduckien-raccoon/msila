from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.visualize import (
    VisualizationContractError,
    prepare_training_visualization,
    save_training_curves,
    save_training_sample,
)


def _sample_tensors():
    image = torch.zeros(2, 3, 8, 8, dtype=torch.float32)
    image[:, 0] = 0.25
    image[:, 1] = 0.50
    image[:, 2] = 0.75

    mask = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
    mask[1, 0, 2:6, 3:7] = 1.0

    logits = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
    logits[1, 0, 2:6, 3:7] = 4.0
    logits[1, 0, :2, :2] = -4.0

    return image, mask, logits


def test_probability_is_exact_sigmoid_without_minmax_normalization():
    image = torch.zeros(1, 3, 1, 3)
    mask = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    logits = torch.tensor([[[[-4.0, 0.0, 4.0]]]])

    vis = prepare_training_visualization(
        image=image,
        mask=mask,
        logits=logits,
    )

    expected = torch.sigmoid(logits[0])
    assert torch.allclose(vis.probability, expected)
    assert float(vis.probability[0, 0, 0]) > 0.0
    assert float(vis.probability[0, 0, 2]) < 1.0
    # A per-image min-max transform would have forced the extrema to exactly 0/1.
    assert float(vis.probability.min()) != 0.0
    assert float(vis.probability.max()) != 1.0


def test_signed_error_semantics_are_correct():
    image = torch.zeros(1, 3, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    # prediction at 0.5 -> [[1,1],[0,0]]
    logits = torch.tensor([[[[4.0, 4.0], [-4.0, -4.0]]]])

    vis = prepare_training_visualization(
        image=image,
        mask=mask,
        logits=logits,
        threshold=0.5,
    )

    expected = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    assert torch.equal(vis.signed_error, expected)
    assert int(vis.false_positive.sum()) == 1
    assert int(vis.false_negative.sum()) == 1


def test_shape_mismatch_is_rejected():
    image = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    logits = torch.zeros(1, 1, 4, 4)

    with pytest.raises(VisualizationContractError, match="share H/W"):
        prepare_training_visualization(
            image=image,
            mask=mask,
            logits=logits,
        )


def test_nonbinary_mask_is_rejected():
    image = torch.zeros(1, 3, 8, 8)
    mask = torch.full((1, 1, 8, 8), 0.25)
    logits = torch.zeros(1, 1, 8, 8)

    with pytest.raises(VisualizationContractError, match="exactly"):
        prepare_training_visualization(
            image=image,
            mask=mask,
            logits=logits,
        )


def test_dino_normalized_display_tensor_is_rejected():
    image = torch.full((1, 3, 8, 8), -1.0)
    mask = torch.zeros(1, 1, 8, 8)
    logits = torch.zeros(1, 1, 8, 8)

    with pytest.raises(VisualizationContractError, match="display-ready"):
        prepare_training_visualization(
            image=image,
            mask=mask,
            logits=logits,
        )


def test_save_training_sample_creates_nonempty_png(tmp_path):
    image, mask, logits = _sample_tensors()

    path, vis = save_training_sample(
        image=image,
        mask=mask,
        logits=logits,
        index=1,
        sample_id="ov16_08_anomaly",
        epoch=3,
        step=12,
        output_path=tmp_path / "qualitative.png",
    )

    assert path.is_file()
    assert path.stat().st_size > 0
    assert vis.sample_id == "ov16_08_anomaly"
    assert vis.epoch == 3
    assert vis.step == 12


@dataclass
class HistoryItem:
    step: int
    loss: float
    pixel_dice: float
    iou: float
    normal_fpr: float
    grad_norm: float

    def to_dict(self):
        return self.__dict__.copy()


def test_save_training_curves_writes_one_figure_per_metric(tmp_path):
    history = [
        HistoryItem(1, 1.0, 0.1, 0.05, 0.2, 2.0),
        HistoryItem(2, 0.7, 0.4, 0.25, 0.1, 1.5),
        HistoryItem(3, 0.3, 0.8, 0.7, 0.02, 0.8),
    ]

    saved = save_training_curves(history, tmp_path / "curves")

    assert set(saved) == {
        "loss",
        "pixel_dice",
        "iou",
        "normal_fpr",
        "grad_norm",
    }
    assert len(saved) == 5
    for path in saved.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_invalid_threshold_is_rejected():
    image, mask, logits = _sample_tensors()

    with pytest.raises(ValueError, match="threshold"):
        prepare_training_visualization(
            image=image,
            mask=mask,
            logits=logits,
            threshold=1.0,
        )
