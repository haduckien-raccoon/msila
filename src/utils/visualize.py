"""
MS-ILA Day-3 — Task 11: training visualization.

Purpose
-------
Turn raw training tensors into reproducible qualitative diagnostics without
changing the model output numerically.

Locked visualization contract
-----------------------------
Input:
    image  : [B,3,H,W] or [3,H,W], RGB float in [0,1]
    mask   : [B,1,H,W] or [1,H,W], binary {0,1}
    logits : [B,1,H,W] or [1,H,W], raw decoder logits

Derived maps:
    probability = sigmoid(logits)          in [0,1]
    prediction  = probability >= threshold
    signed_error = prediction - mask       in {-1,0,+1}

Important:
- Probability maps are NEVER min-max normalized per image.
- Score visualizations use a fixed [0,1] scale.
- Signed error uses a fixed [-1,1] scale.
- The module does not modify the trainer, model, loss, checkpoint, or resume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

__all__ = [
    "TrainingVisualization",
    "VisualizationContractError",
    "prepare_training_visualization",
    "plot_training_sample",
    "save_training_sample",
    "save_training_curves",
]


class VisualizationContractError(ValueError):
    """Raised when tensors violate the Day-3 visualization contract."""


@dataclass(frozen=True)
class TrainingVisualization:
    """CPU tensors/metadata for one qualitative training sample."""

    image: Tensor
    target: Tensor
    probability: Tensor
    prediction: Tensor
    signed_error: Tensor
    false_positive: Tensor
    false_negative: Tensor
    threshold: float
    sample_id: str | None
    epoch: int | None
    step: int | None
    dice: float
    iou: float
    precision: float
    recall: float

    def metrics_dict(self) -> dict[str, float | int | str | None]:
        return {
            "sample_id": self.sample_id,
            "epoch": self.epoch,
            "step": self.step,
            "threshold": self.threshold,
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
        }


def _select_sample(
    tensor: Tensor,
    *,
    index: int,
    expected_channels: int,
    name: str,
) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)!r}")

    if tensor.ndim == 4:
        if tensor.shape[0] <= index or index < 0:
            raise IndexError(
                f"{name}: index={index} invalid for batch size {tensor.shape[0]}"
            )
        sample = tensor[index]
    elif tensor.ndim == 3:
        if index != 0:
            raise IndexError(
                f"{name} is unbatched; only index=0 is valid, got {index}"
            )
        sample = tensor
    else:
        raise VisualizationContractError(
            f"{name} must be [B,C,H,W] or [C,H,W], got {tuple(tensor.shape)}"
        )

    if sample.shape[0] != expected_channels:
        raise VisualizationContractError(
            f"{name}: expected C={expected_channels}, got C={sample.shape[0]}"
        )
    return sample


def _validate_image(image: Tensor) -> None:
    if not image.is_floating_point():
        raise VisualizationContractError(
            f"image must be floating point RGB in [0,1], got {image.dtype}"
        )
    if not bool(torch.isfinite(image).all()):
        raise VisualizationContractError("image contains NaN/Inf")
    lo = float(image.min().item())
    hi = float(image.max().item())
    if lo < 0.0 or hi > 1.0:
        raise VisualizationContractError(
            "image must already be display-ready RGB in [0,1]. "
            f"Observed range [{lo:.6g}, {hi:.6g}]. "
            "Do not pass DINO-normalized tensors directly."
        )


def _validate_target(mask: Tensor) -> None:
    target = mask.to(torch.float32)
    if not bool(torch.isfinite(target).all()):
        raise VisualizationContractError("mask contains NaN/Inf")
    binary = torch.logical_or(target == 0.0, target == 1.0)
    if not bool(binary.all()):
        raise VisualizationContractError("mask must contain exactly {0,1}")


def _validate_logits(logits: Tensor) -> None:
    if not logits.is_floating_point():
        raise VisualizationContractError(
            f"logits must be floating point, got {logits.dtype}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise VisualizationContractError("logits contain NaN/Inf")


def _sample_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    pred = prediction.bool()
    truth = target.bool()

    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())

    if tp + fp + fn == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn)
        iou = tp / (tp + fp + fn)

    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


@torch.no_grad()
def prepare_training_visualization(
    *,
    image: Tensor,
    mask: Tensor,
    logits: Tensor,
    index: int = 0,
    threshold: float = 0.5,
    sample_id: str | None = None,
    epoch: int | None = None,
    step: int | None = None,
) -> TrainingVisualization:
    """Prepare one sample for qualitative training visualization.

    No min-max normalization is performed. ``probability`` is exactly
    ``sigmoid(raw_logits)``.
    """
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError(f"threshold must be in (0,1), got {threshold}")

    rgb = _select_sample(
        image,
        index=index,
        expected_channels=3,
        name="image",
    )
    target = _select_sample(
        mask,
        index=index,
        expected_channels=1,
        name="mask",
    )
    raw_logits = _select_sample(
        logits,
        index=index,
        expected_channels=1,
        name="logits",
    )

    if rgb.shape[-2:] != target.shape[-2:] or target.shape != raw_logits.shape:
        raise VisualizationContractError(
            "image/mask/logits must share H/W and mask/logits shape. "
            f"image={tuple(rgb.shape)}, mask={tuple(target.shape)}, "
            f"logits={tuple(raw_logits.shape)}"
        )

    _validate_image(rgb)
    _validate_target(target)
    _validate_logits(raw_logits)

    rgb = rgb.detach().to(device="cpu", dtype=torch.float32).contiguous()
    target = target.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw_logits = raw_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()

    probability = torch.sigmoid(raw_logits)
    prediction = probability >= float(threshold)

    target_bool = target >= 0.5
    false_positive = prediction & ~target_bool
    false_negative = ~prediction & target_bool

    # +1 = false positive, -1 = false negative, 0 = correct.
    signed_error = (
        prediction.to(torch.float32)
        - target_bool.to(torch.float32)
    )

    metrics = _sample_metrics(prediction, target_bool)

    return TrainingVisualization(
        image=rgb,
        target=target,
        probability=probability,
        prediction=prediction,
        signed_error=signed_error,
        false_positive=false_positive,
        false_negative=false_negative,
        threshold=float(threshold),
        sample_id=sample_id,
        epoch=None if epoch is None else int(epoch),
        step=None if step is None else int(step),
        dice=metrics["dice"],
        iou=metrics["iou"],
        precision=metrics["precision"],
        recall=metrics["recall"],
    )


def _to_hwc_rgb(image: Tensor) -> np.ndarray:
    return image.permute(1, 2, 0).numpy()


def _to_hw(single_channel: Tensor) -> np.ndarray:
    return single_channel.squeeze(0).numpy()


def plot_training_sample(
    vis: TrainingVisualization,
    *,
    overlay_alpha: float = 0.45,
    figsize: tuple[float, float] = (15.0, 8.0),
):
    """Create a six-panel qualitative QA figure.

    Panels:
        1. RGB input
        2. exact GT mask
        3. sigmoid probability map, fixed [0,1]
        4. thresholded prediction
        5. probability overlay on RGB
        6. signed error: -1 FN, 0 correct, +1 FP
    """
    if not (0.0 <= float(overlay_alpha) <= 1.0):
        raise ValueError("overlay_alpha must be in [0,1]")

    rgb = _to_hwc_rgb(vis.image)
    gt = _to_hw(vis.target)
    prob = _to_hw(vis.probability)
    pred = _to_hw(vis.prediction.to(torch.float32))
    error = _to_hw(vis.signed_error)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB")

    axes[0, 1].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("GT mask")

    score_artist = axes[0, 2].imshow(
        prob,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 2].set_title("Anomaly probability = sigmoid(logit)")
    fig.colorbar(score_artist, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(pred, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(f"Prediction @ {vis.threshold:.2f}")

    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(
        prob,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        alpha=float(overlay_alpha),
    )
    axes[1, 1].set_title("Probability overlay")

    error_artist = axes[1, 2].imshow(
        error,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[1, 2].set_title("Signed error: -1 FN, +1 FP")
    fig.colorbar(error_artist, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis("off")

    identity = "" if vis.sample_id is None else f" | {vis.sample_id}"
    epoch_text = "" if vis.epoch is None else f" | epoch={vis.epoch}"
    step_text = "" if vis.step is None else f" | step={vis.step}"
    fig.suptitle(
        "MS-ILA training QA"
        f"{identity}{epoch_text}{step_text}"
        f" | Dice={vis.dice:.4f}"
        f" | IoU={vis.iou:.4f}"
        f" | P={vis.precision:.4f}"
        f" | R={vis.recall:.4f}"
    )
    fig.tight_layout()
    return fig


def save_training_sample(
    *,
    image: Tensor,
    mask: Tensor,
    logits: Tensor,
    output_path: str | Path,
    index: int = 0,
    threshold: float = 0.5,
    sample_id: str | None = None,
    epoch: int | None = None,
    step: int | None = None,
    overlay_alpha: float = 0.45,
    dpi: int = 160,
) -> tuple[Path, TrainingVisualization]:
    """Prepare, plot, save, then close one qualitative training figure."""
    if dpi <= 0:
        raise ValueError("dpi must be > 0")

    vis = prepare_training_visualization(
        image=image,
        mask=mask,
        logits=logits,
        index=index,
        threshold=threshold,
        sample_id=sample_id,
        epoch=epoch,
        step=step,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_training_sample(
        vis,
        overlay_alpha=overlay_alpha,
    )
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    return path, vis


def _record_to_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "to_dict") and callable(record.to_dict):
        value = record.to_dict()
        if isinstance(value, Mapping):
            return value
    if hasattr(record, "__dict__"):
        return vars(record)
    raise TypeError(
        "history items must be mappings or objects exposing to_dict()/__dict__"
    )


def save_training_curves(
    history: Sequence[Any],
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = (
        "loss",
        "pixel_dice",
        "iou",
        "normal_fpr",
        "grad_norm",
    ),
    x_key: str = "step",
    dpi: int = 160,
) -> dict[str, Path]:
    """Save one independent figure per training diagnostic.

    The function intentionally does not combine metrics with very different
    units/scales into one chart.
    """
    if not history:
        raise ValueError("history must not be empty")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")

    records = [_record_to_mapping(item) for item in history]
    if any(x_key not in record for record in records):
        raise KeyError(f"Every history record must contain x_key={x_key!r}")

    x = np.asarray([float(record[x_key]) for record in records], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{x_key} contains NaN/Inf")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    for metric in metrics:
        if any(metric not in record for record in records):
            raise KeyError(f"Every history record must contain metric={metric!r}")

        y = np.asarray(
            [float(record[metric]) for record in records],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{metric} contains NaN/Inf")

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(x, y)
        ax.set_xlabel(x_key)
        ax.set_ylabel(metric)
        ax.set_title(f"Training {metric}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = out_dir / f"{metric}.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        saved[metric] = path

    return saved
