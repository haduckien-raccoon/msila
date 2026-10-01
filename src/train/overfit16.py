"""
MS-ILA Day-3 — Task 10: Overfit-16 trainer.

Scope
-----
This module implements ONLY the training loop needed to test whether the current
architecture can memorize the fixed Task-7 Overfit-16 set.

It deliberately does NOT implement:
- visualization (Task 11),
- checkpointing (Task 12),
- resume (Task 13),
- full integration QA (Task 14),
- Day-03 report generation (Task 15).

Scientific role
---------------
Overfit-16 is an architecture/debugging experiment, not a generalization
benchmark. A successful run only supports the statement that the current
trainable graph can fit a controlled 16-sample problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


OVERFIT16_SCHEMA_NAME = "msila_overfit16"
OVERFIT16_SCHEMA_VERSION = 1
OVERFIT16_SIZE = 16


class OverfitTrainerError(RuntimeError):
    """Base error for Task-10 trainer contract violations."""


class Overfit16DatasetError(OverfitTrainerError):
    """Raised when the Task-7 Overfit-16 dataset is malformed."""


@dataclass(frozen=True)
class StepLog:
    step: int
    epoch: int
    loss: float
    bce: float
    dice_loss: float
    pixel_dice: float
    iou: float
    precision: float
    recall: float
    normal_fpr: float
    grad_norm: float
    learning_rate: float
    positive_samples: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    loss: float
    bce: float
    dice_loss: float
    pixel_dice: float
    iou: float
    precision: float
    recall: float
    normal_fpr: float
    n_samples: int
    n_positive_samples: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class OverfitTrainResult:
    initial: EvaluationSummary
    final: EvaluationSummary
    loss_ratio: float
    steps: int
    epochs_completed: int
    history: tuple[StepLog, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "initial": self.initial.to_dict(),
            "final": self.final.to_dict(),
            "loss_ratio": self.loss_ratio,
            "steps": self.steps,
            "epochs_completed": self.epochs_completed,
            "history": [item.to_dict() for item in self.history],
        }


SampleTransform = Callable[
    [Tensor, Tensor, Mapping[str, Any]],
    tuple[Tensor, Tensor],
]
ModelForward = Callable[[nn.Module, Mapping[str, Any]], Any]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for a reproducible QA run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .to(torch.float32)
        .div_(255.0)
        .contiguous()
    )


def _load_mask(path: Path) -> Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    mask = torch.from_numpy(arr).gt(0).to(torch.float32).unsqueeze(0)
    return mask.contiguous()


def _validate_image_mask(image: Tensor, mask: Tensor, sample_id: str) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise Overfit16DatasetError(
            f"{sample_id}: image must have shape [3,H,W], got {tuple(image.shape)}"
        )
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise Overfit16DatasetError(
            f"{sample_id}: mask must have shape [1,H,W], got {tuple(mask.shape)}"
        )
    if image.shape[-2:] != mask.shape[-2:]:
        raise Overfit16DatasetError(
            f"{sample_id}: image/mask spatial mismatch: "
            f"{tuple(image.shape[-2:])} vs {tuple(mask.shape[-2:])}"
        )
    if not image.is_floating_point() or not mask.is_floating_point():
        raise Overfit16DatasetError(f"{sample_id}: image/mask must be floating point")
    if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(mask).all()):
        raise Overfit16DatasetError(f"{sample_id}: image/mask contains NaN/Inf")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise Overfit16DatasetError(f"{sample_id}: RGB image must be in [0,1]")
    if not bool(torch.logical_or(mask == 0.0, mask == 1.0).all()):
        raise Overfit16DatasetError(f"{sample_id}: mask must be binary {{0,1}}")


class Overfit16Dataset(Dataset):
    """Read the exact dataset emitted by Task 7.

    The dataset does not resize, normalize for DINO, create Local/Context views,
    or build cached features. Those operations belong to the project's existing
    feature pipeline and may be injected through ``sample_transform`` or
    ``model_forward`` in the trainer.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sample_transform: SampleTransform | None = None,
        verify_contract: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_name") != OVERFIT16_SCHEMA_NAME:
            raise Overfit16DatasetError(
                f"Expected schema_name={OVERFIT16_SCHEMA_NAME!r}, "
                f"got {manifest.get('schema_name')!r}"
            )
        if manifest.get("schema_version") != OVERFIT16_SCHEMA_VERSION:
            raise Overfit16DatasetError(
                f"Expected schema_version={OVERFIT16_SCHEMA_VERSION}, "
                f"got {manifest.get('schema_version')!r}"
            )
        if manifest.get("purpose") != "architecture_qa_only":
            raise Overfit16DatasetError(
                "Overfit-16 manifest must declare purpose='architecture_qa_only'"
            )

        records = manifest.get("samples")
        if not isinstance(records, list) or len(records) != OVERFIT16_SIZE:
            raise Overfit16DatasetError(
                f"Overfit-16 must contain exactly {OVERFIT16_SIZE} records"
            )

        self.manifest = manifest
        self.records = tuple(dict(r) for r in records)
        self.sample_transform = sample_transform
        self.verify_contract = bool(verify_contract)

        ids = [str(r.get("sample_id", "")) for r in self.records]
        if any(not value for value in ids) or len(set(ids)) != OVERFIT16_SIZE:
            raise Overfit16DatasetError("sample_id values must be non-empty and unique")

    def __len__(self) -> int:
        return OVERFIT16_SIZE

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        sample_id = str(record["sample_id"])

        image_path = self.root / str(record["image_path"])
        mask_path = self.root / str(record["mask_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        image = _load_rgb(image_path)
        mask = _load_mask(mask_path)

        if self.sample_transform is not None:
            image, mask = self.sample_transform(image, mask, record)

        if self.verify_contract:
            _validate_image_mask(image, mask, sample_id)

            expected_is_anomaly = bool(record["is_anomaly"])
            observed_is_anomaly = bool(mask.any().item())
            if observed_is_anomaly != expected_is_anomaly:
                raise Overfit16DatasetError(
                    f"{sample_id}: mask/manifest anomaly label mismatch"
                )

        return {
            "image": image,
            "mask": mask,
            "sample_id": sample_id,
            "index": int(record["index"]),
            "is_anomaly": bool(record["is_anomaly"]),
        }


def make_overfit16_loader(
    dataset: Dataset,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int = 2026,
    num_workers: int = 0,
) -> DataLoader:
    """Create a deterministic small-data loader.

    Note: default collation requires samples in a batch to have identical image
    size. If the Task-7 sources have heterogeneous H/W, use ``batch_size=1`` or
    provide a geometry-preserving ``sample_transform`` that transforms image and
    mask together.
    """
    if len(dataset) != OVERFIT16_SIZE:
        raise OverfitTrainerError(
            f"Task-10 expects exactly {OVERFIT16_SIZE} samples, got {len(dataset)}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=generator,
        drop_last=False,
        pin_memory=False,
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    return value


def _default_model_forward(model: nn.Module, batch: Mapping[str, Any]) -> Any:
    if "image" not in batch:
        raise OverfitTrainerError(
            "Default model_forward expects batch['image']; provide a custom "
            "model_forward for Local/Context or cached-feature training."
        )
    return model(batch["image"])


def _extract_logits(model_output: Any) -> Tensor:
    """Normalize common model return styles to raw anomaly logits."""
    if isinstance(model_output, Tensor):
        logits = model_output
    elif (
        isinstance(model_output, tuple)
        and len(model_output) >= 1
        and isinstance(model_output[0], Tensor)
    ):
        logits = model_output[0]
    elif isinstance(model_output, Mapping) and isinstance(model_output.get("logits"), Tensor):
        logits = model_output["logits"]
    elif isinstance(model_output, Mapping) and isinstance(model_output.get("anomaly_logits"), Tensor):
        logits = model_output["anomaly_logits"]
    else:
        raise OverfitTrainerError(
            "Model output must be logits Tensor, (logits, ...), or mapping with "
            "'logits'/'anomaly_logits'."
        )

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise OverfitTrainerError(
            f"Expected anomaly logits [B,1,H,W], got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise OverfitTrainerError("Anomaly logits must be finite floating point")
    return logits


def _loss_terms(criterion_output: Any) -> tuple[Tensor, Tensor, Tensor, int]:
    """Normalize the locked Task-8 loss output."""
    if not isinstance(criterion_output, Mapping):
        raise OverfitTrainerError(
            "Task-10 expects Task-8 criterion output to be a mapping."
        )
    required = {"loss", "bce", "dice", "positive_samples"}
    missing = required - set(criterion_output.keys())
    if missing:
        raise OverfitTrainerError(
            f"Criterion output missing required keys: {sorted(missing)}"
        )

    loss = criterion_output["loss"]
    bce = criterion_output["bce"]
    dice = criterion_output["dice"]
    positive_samples = criterion_output["positive_samples"]

    for name, value in (("loss", loss), ("bce", bce), ("dice", dice)):
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise OverfitTrainerError(f"{name} must be a scalar Tensor")
        if not bool(torch.isfinite(value)):
            raise OverfitTrainerError(f"{name} became NaN/Inf")

    if isinstance(positive_samples, Tensor):
        positive_count = int(positive_samples.detach().cpu().item())
    else:
        positive_count = int(positive_samples)

    return loss, bce, dice, positive_count


def _confusion_counts(
    logits: Tensor,
    target: Tensor,
    *,
    threshold: float,
) -> dict[str, int]:
    if tuple(logits.shape) != tuple(target.shape):
        raise OverfitTrainerError(
            f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    probs = torch.sigmoid(logits)
    pred = probs >= float(threshold)
    truth = target >= 0.5

    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())

    # Normal-sample FPR is computed only on samples whose GT mask is empty.
    per_sample_positive = truth.flatten(start_dim=1).any(dim=1)
    normal_selector = ~per_sample_positive
    if bool(normal_selector.any()):
        normal_pred = pred[normal_selector]
        normal_fp = int(normal_pred.sum().item())
        normal_pixels = int(normal_pred.numel())
    else:
        normal_fp = 0
        normal_pixels = 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "normal_fp": normal_fp,
        "normal_pixels": normal_pixels,
    }


def _metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    normal_fp = int(counts["normal_fp"])
    normal_pixels = int(counts["normal_pixels"])

    eps = 1e-12
    # For an all-empty GT and all-empty prediction, pixel Dice/IoU are defined
    # as 1 for this QA diagnostic.
    if tp + fp + fn == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)

    precision = 1.0 if tp + fp == 0 else tp / (tp + fp + eps)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn + eps)
    normal_fpr = 0.0 if normal_pixels == 0 else normal_fp / normal_pixels

    return {
        "pixel_dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "normal_fpr": float(normal_fpr),
    }


def _gradient_l2_norm(optimizer: Optimizer) -> float:
    total_sq = 0.0
    seen: set[int] = set()
    found = False

    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if not bool(torch.isfinite(grad).all()):
                raise OverfitTrainerError("A trainable gradient contains NaN/Inf")
            total_sq += float(grad.float().square().sum().item())
            found = True

    if not found:
        raise OverfitTrainerError(
            "No optimizer parameter received a gradient; training graph is disconnected."
        )

    return float(math.sqrt(total_sq))


def _current_lr(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        raise OverfitTrainerError("Optimizer has no parameter groups")
    return float(optimizer.param_groups[0]["lr"])


class Overfit16Trainer:
    """Minimal trainer for memorizing the Task-7 Overfit-16 set."""

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: str | torch.device | None = None,
        model_forward: ModelForward | None = None,
        frozen_modules: Mapping[str, nn.Module] | None = None,
        threshold: float = 0.5,
        max_grad_norm: float | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")
        if not isinstance(criterion, nn.Module):
            raise TypeError("criterion must be nn.Module")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be torch.optim.Optimizer")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in (0,1)")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when provided")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.model_forward = model_forward or _default_model_forward
        self.frozen_modules = dict(frozen_modules or {})
        self.threshold = float(threshold)
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)
        self.criterion.to(self.device)
        for name, module in self.frozen_modules.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"frozen_modules[{name!r}] must be nn.Module")
            bad = [n for n, p in module.named_parameters() if p.requires_grad]
            if bad:
                raise OverfitTrainerError(
                    f"Frozen module {name!r} still has trainable parameters: {bad}"
                )
            module.eval()

    def _enter_train_mode(self) -> None:
        self.model.train()
        # Re-assert frozen eval state after parent .train() calls. The project's
        # DINOv3 extractor already protects itself, but this makes Task-10 robust
        # to other frozen modules as well.
        for module in self.frozen_modules.values():
            module.eval()

    def train_step(
        self,
        batch: Mapping[str, Any],
        *,
        step: int,
        epoch: int,
    ) -> StepLog:
        self._enter_train_mode()
        batch = _move_to_device(batch, self.device)

        if "mask" not in batch or not isinstance(batch["mask"], Tensor):
            raise OverfitTrainerError("Batch must contain Tensor batch['mask']")
        target = batch["mask"]

        self.optimizer.zero_grad(set_to_none=True)

        output = self.model_forward(self.model, batch)
        logits = _extract_logits(output)

        if tuple(logits.shape) != tuple(target.shape):
            raise OverfitTrainerError(
                "Decoder output and GT mask must have identical shape; "
                f"logits={tuple(logits.shape)}, mask={tuple(target.shape)}"
            )

        criterion_output = self.criterion(logits, target)
        loss, bce, dice_loss, positive_samples = _loss_terms(criterion_output)

        loss.backward()

        grad_norm_before_clip = _gradient_l2_norm(self.optimizer)

        if self.max_grad_norm is not None:
            params = [
                p
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(params, float(self.max_grad_norm))

        self.optimizer.step()

        counts = _confusion_counts(
            logits.detach(),
            target.detach(),
            threshold=self.threshold,
        )
        metrics = _metrics_from_counts(counts)

        return StepLog(
            step=int(step),
            epoch=int(epoch),
            loss=float(loss.detach().cpu().item()),
            bce=float(bce.detach().cpu().item()),
            dice_loss=float(dice_loss.detach().cpu().item()),
            pixel_dice=metrics["pixel_dice"],
            iou=metrics["iou"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            normal_fpr=metrics["normal_fpr"],
            grad_norm=float(grad_norm_before_clip),
            learning_rate=_current_lr(self.optimizer),
            positive_samples=int(positive_samples),
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvaluationSummary:
        self.model.eval()
        for module in self.frozen_modules.values():
            module.eval()

        total_loss = 0.0
        total_bce = 0.0
        total_dice_loss = 0.0
        total_samples = 0
        positive_samples = 0
        counts = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "normal_fp": 0,
            "normal_pixels": 0,
        }

        for batch in loader:
            batch = _move_to_device(batch, self.device)
            target = batch["mask"]
            logits = _extract_logits(self.model_forward(self.model, batch))

            if tuple(logits.shape) != tuple(target.shape):
                raise OverfitTrainerError(
                    "Decoder output and GT mask must have identical shape during evaluation"
                )

            criterion_output = self.criterion(logits, target)
            loss, bce, dice_loss, positive_count = _loss_terms(criterion_output)

            batch_size = int(target.shape[0])
            total_samples += batch_size
            positive_samples += positive_count
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice_loss += float(dice_loss.item()) * batch_size

            batch_counts = _confusion_counts(
                logits,
                target,
                threshold=self.threshold,
            )
            for key in counts:
                counts[key] += batch_counts[key]

        if total_samples == 0:
            raise OverfitTrainerError("Cannot evaluate an empty loader")

        metrics = _metrics_from_counts(counts)
        return EvaluationSummary(
            loss=total_loss / total_samples,
            bce=total_bce / total_samples,
            dice_loss=total_dice_loss / total_samples,
            pixel_dice=metrics["pixel_dice"],
            iou=metrics["iou"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            normal_fpr=metrics["normal_fpr"],
            n_samples=total_samples,
            n_positive_samples=positive_samples,
        )

    def fit(
        self,
        loader: DataLoader,
        *,
        epochs: int,
        log_every: int = 1,
    ) -> OverfitTrainResult:
        """Train on the same 16 samples and return in-memory diagnostics.

        This method does not save checkpoints, images, CSV/JSON reports, or
        resume state. Those belong to later Day-3 tasks.
        """
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if log_every <= 0:
            raise ValueError("log_every must be > 0")
        if len(loader.dataset) != OVERFIT16_SIZE:
            raise OverfitTrainerError(
                f"Overfit trainer requires exactly {OVERFIT16_SIZE} samples"
            )

        initial = self.evaluate(loader)

        history: list[StepLog] = []
        global_step = 0

        for epoch in range(1, int(epochs) + 1):
            for batch in loader:
                global_step += 1
                item = self.train_step(
                    batch,
                    step=global_step,
                    epoch=epoch,
                )
                if global_step % int(log_every) == 0:
                    history.append(item)

        final = self.evaluate(loader)
        loss_ratio = (
            final.loss / initial.loss
            if initial.loss > 0.0
            else 0.0
        )

        return OverfitTrainResult(
            initial=initial,
            final=final,
            loss_ratio=float(loss_ratio),
            steps=global_step,
            epochs_completed=int(epochs),
            history=tuple(history),
        )
