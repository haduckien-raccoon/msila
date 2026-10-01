from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.anomaly_loss import AnomalySegmentationLoss, dice_loss_from_logits


def _square_mask(batch: int = 2, h: int = 16, w: int = 16) -> torch.Tensor:
    y = torch.zeros(batch, 1, h, w, dtype=torch.float32)
    y[:, :, 4:12, 5:11] = 1.0
    return y


def test_loss_is_finite_and_backward_produces_finite_nonzero_gradient():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = _square_mask()

    criterion = AnomalySegmentationLoss()
    out = criterion(logits, target)

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["bce"])
    assert torch.isfinite(out["dice"])
    assert int(out["positive_samples"].item()) == 2

    out["loss"].backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_good_prediction_has_lower_loss_than_inverted_prediction():
    target = _square_mask(batch=1, h=12, w=12)

    good = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    bad = -good

    criterion = AnomalySegmentationLoss()
    good_out = criterion(good, target)
    bad_out = criterion(bad, target)

    assert float(good_out["loss"]) < float(bad_out["loss"])
    assert float(good_out["bce"]) < float(bad_out["bce"])
    assert float(good_out["dice"]) < float(bad_out["dice"])


def test_bce_component_matches_pytorch_bce_with_logits_exactly():
    torch.manual_seed(4)
    logits = torch.randn(2, 1, 8, 8)
    target = _square_mask(batch=2, h=8, w=8)

    out = AnomalySegmentationLoss()(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits, target)

    assert torch.allclose(out["bce"], expected, atol=0.0, rtol=0.0)


def test_normal_only_batch_uses_zero_dice_and_finite_bce():
    logits = torch.zeros(3, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)

    out = AnomalySegmentationLoss()(logits, target)

    assert int(out["positive_samples"].item()) == 0
    assert float(out["dice"].item()) == 0.0
    assert torch.isfinite(out["bce"])
    assert torch.allclose(out["loss"], out["bce"])

    out["loss"].backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_mixed_batch_dice_counts_only_positive_mask_samples():
    logits = torch.zeros(3, 1, 8, 8)
    target = torch.zeros_like(logits)
    target[1, :, 2:6, 2:6] = 1.0

    dice, n_pos = dice_loss_from_logits(logits, target)

    assert int(n_pos.item()) == 1
    assert 0.0 <= float(dice.item()) <= 1.0


def test_rejects_prediction_target_shape_mismatch():
    logits = torch.randn(2, 1, 16, 16)
    target = torch.zeros(2, 1, 15, 16)

    with pytest.raises(ValueError, match="shape mismatch"):
        AnomalySegmentationLoss()(logits, target)


def test_rejects_non_binary_target():
    logits = torch.randn(1, 1, 8, 8)
    target = torch.zeros_like(logits)
    target[0, 0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="binary"):
        AnomalySegmentationLoss()(logits, target)


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        AnomalySegmentationLoss(bce_weight=0.0, dice_weight=0.0)
