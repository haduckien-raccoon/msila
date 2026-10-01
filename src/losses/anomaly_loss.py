"""Loss for Day-3 MS-ILA Overfit-16 architecture QA.

The criterion is intentionally minimal:

    L = lambda_bce * BCEWithLogits(logits, target)
        + lambda_dice * DiceLoss(sigmoid(logits), target)

Design choices
--------------
1. BCE is computed directly from raw logits via ``binary_cross_entropy_with_logits``.
   Do NOT apply sigmoid before BCEWithLogits.
2. Dice is computed only for samples whose ground-truth mask contains at least
   one anomalous pixel. For normal samples with an empty target mask, BCE alone
   provides the background supervision. This avoids assigning an arbitrary
   overlap score to an empty target.
3. No resizing is performed inside the loss. Prediction and target must already
   have identical ``[B,1,H,W]`` shape; a mismatch is treated as a pipeline bug.
4. This module is for Day-3 architecture QA, not a final scientific claim about
   the optimal training objective for MVTec AD 2.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AnomalyLossOutput",
    "AnomalySegmentationLoss",
    "dice_loss_from_logits",
]


class AnomalyLossOutput(TypedDict):
    """Scalar loss terms returned for training/logging."""

    loss: Tensor
    bce: Tensor
    dice: Tensor
    positive_samples: Tensor


def _validate_binary_segmentation_inputs(logits: Tensor, target: Tensor) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be torch.Tensor, got {type(logits)!r}")
    if not isinstance(target, Tensor):
        raise TypeError(f"target must be torch.Tensor, got {type(target)!r}")

    if logits.ndim != 4:
        raise ValueError(
            f"logits must have shape [B,1,H,W], got {tuple(logits.shape)}"
        )
    if target.ndim != 4:
        raise ValueError(
            f"target must have shape [B,1,H,W], got {tuple(target.shape)}"
        )
    if logits.shape[1] != 1:
        raise ValueError(
            f"logits must have one anomaly channel, got C={logits.shape[1]}"
        )
    if target.shape[1] != 1:
        raise ValueError(
            f"target must have one anomaly channel, got C={target.shape[1]}"
        )
    if tuple(logits.shape) != tuple(target.shape):
        raise ValueError(
            "logits/target shape mismatch: "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    if not target.is_floating_point() and target.dtype is not torch.bool:
        raise TypeError(
            "target must be floating point or bool with binary values {0,1}; "
            f"got {target.dtype}"
        )
    if logits.device != target.device:
        raise ValueError(
            f"logits/target device mismatch: {logits.device} vs {target.device}"
        )

    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain NaN/Inf")

    target_float = target.to(dtype=torch.float32)
    if not bool(torch.isfinite(target_float).all()):
        raise ValueError("target contains NaN/Inf")

    is_binary = torch.logical_or(target_float == 0.0, target_float == 1.0)
    if not bool(is_binary.all()):
        bad = target_float[~is_binary]
        example = float(bad.flatten()[0].item())
        raise ValueError(
            "target must be binary with values exactly {0,1}; "
            f"found example value {example}"
        )


def dice_loss_from_logits(
    logits: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Compute mean Dice loss over *positive-mask samples only*.

    Parameters
    ----------
    logits:
        Raw anomaly logits with shape ``[B,1,H,W]``.
    target:
        Binary mask with the same shape, values in ``{0,1}``.
    eps:
        Numerical stabilizer.

    Returns
    -------
    dice_loss:
        Scalar. If the batch contains no positive target masks, this is an
        exact differentiable zero (``logits.sum() * 0``).
    positive_samples:
        Scalar int64 tensor containing the number of samples included in Dice.
    """
    _validate_binary_segmentation_inputs(logits, target)

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    target_f = target.to(dtype=logits.dtype)
    probs = torch.sigmoid(logits)

    probs_flat = probs.flatten(start_dim=1)
    target_flat = target_f.flatten(start_dim=1)

    positive_mask = target_flat.sum(dim=1) > 0
    positive_samples = positive_mask.sum().to(dtype=torch.int64)

    if not bool(positive_mask.any()):
        # Keep the return connected to the graph/device/dtype while contributing
        # exactly zero gradient to a normal-only batch.
        return logits.sum() * 0.0, positive_samples

    p = probs_flat[positive_mask]
    y = target_flat[positive_mask]

    intersection = (p * y).sum(dim=1)
    denominator = p.sum(dim=1) + y.sum(dim=1)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = (1.0 - dice_score).mean()

    return dice_loss, positive_samples


class AnomalySegmentationLoss(nn.Module):
    """BCEWithLogits + positive-mask Dice loss for Day-3 Overfit-16.

    Notes
    -----
    - ``logits`` must be the raw decoder output, e.g. ``[B,1,512,512]``.
    - ``target`` must be the exact binary mask with identical shape.
    - The loss performs no interpolation or thresholding.
    """

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_eps = float(dice_eps)

        if self.bce_weight < 0:
            raise ValueError("bce_weight must be >= 0")
        if self.dice_weight < 0:
            raise ValueError("dice_weight must be >= 0")
        if self.bce_weight == 0 and self.dice_weight == 0:
            raise ValueError("At least one loss weight must be > 0")
        if self.dice_eps <= 0:
            raise ValueError("dice_eps must be > 0")

    def forward(self, logits: Tensor, target: Tensor) -> AnomalyLossOutput:
        _validate_binary_segmentation_inputs(logits, target)

        target_f = target.to(dtype=logits.dtype)

        # Numerically stable BCE directly on raw logits.
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target_f,
            reduction="mean",
        )

        dice, positive_samples = dice_loss_from_logits(
            logits,
            target_f,
            eps=self.dice_eps,
        )

        total = self.bce_weight * bce + self.dice_weight * dice

        if not bool(torch.isfinite(total)):
            raise RuntimeError("combined anomaly loss became NaN/Inf")

        return {
            "loss": total,
            "bce": bce,
            "dice": dice,
            "positive_samples": positive_samples,
        }
