"""
Day-04 project hooks for the controlled Adapter r×d screen.

This module connects ``src.train.screen_adapter`` to the current cached-feature
MS-ILA architecture without creating a second training pipeline.

Architecture contract
---------------------
Input:
    six frozen DINOv3 cached feature maps
    local_b4/local_b8/local_b12
    context_b4/context_b8/context_b12

Trainable path:
    three independent block-specific Adapters
        b4, b8, b12
    -> Context-to-Local alignment
    -> SixFeatureProjection
    -> MSILADay2Head (AttentionFusion + BasicDecoder)
    -> anomaly logits

Loss:
    BCEWithLogits + positive-mask Dice

Important
---------
- DINOv3 is NOT instantiated here.
- The exact ``nn.ModuleDict`` supplied by ``screen_adapter.py`` is registered
  in the returned model. The hook does not rebuild or copy candidate Adapters.
- Adapter ``d = projection_dim`` is independent from downstream ``fusion_dim``.
- ``predict`` returns sigmoid probabilities; training/validation use raw logits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.cached_training import CachedFeatureTrainingModel
from src.models.context_alignment import ContextToLocalAligner
from src.models.feature_projection import SixFeatureProjection
from src.models.msila import MSILADay2Head


_REQUIRED_ADAPTER_KEYS = {"b4", "b8", "b12"}


class Day04HookConfigError(ValueError):
    """Raised when the Day-04 hook configuration violates the locked contract."""


def _require_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Day04HookConfigError(
            f"{name} must be a mapping, got {type(value).__name__}"
        )
    return value


def _require_positive_int(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Day04HookConfigError(
            f"{name} must be an int, got {type(value).__name__}"
        )
    if value <= 0:
        raise Day04HookConfigError(
            f"{name} must be > 0, got {value}"
        )
    return int(value)


def _require_finite_nonnegative_float(
    value: Any,
    *,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise Day04HookConfigError(
            f"{name} must be numeric"
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise Day04HookConfigError(
            f"{name} must be numeric"
        ) from exc

    if not torch.isfinite(torch.tensor(out)):
        raise Day04HookConfigError(
            f"{name} must be finite"
        )
    if out < 0.0:
        raise Day04HookConfigError(
            f"{name} must be >= 0"
        )
    return out


def _parse_output_size(
    value: Any,
) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise Day04HookConfigError(
            "model.decoder.output_size must be [H, W]"
        )

    h = _require_positive_int(
        value[0],
        name="model.decoder.output_size[0]",
    )
    w = _require_positive_int(
        value[1],
        name="model.decoder.output_size[1]",
    )
    return h, w


def _validate_adapters(
    adapters: nn.ModuleDict,
) -> dict[str, Any]:
    """Validate the exact 3-Adapter contract and return common metadata."""
    if not isinstance(adapters, nn.ModuleDict):
        raise Day04HookConfigError(
            "adapters must be torch.nn.ModuleDict"
        )

    if set(adapters.keys()) != _REQUIRED_ADAPTER_KEYS:
        raise Day04HookConfigError(
            "adapters must contain exactly b4, b8, b12"
        )

    # Required public attributes of the current ResidualAdapter2d.
    attrs = (
        "in_dim",
        "bottleneck_dim",
        "projection_dim",
        "kernel_size",
        "bias",
        "gamma_init",
    )

    rows: dict[str, dict[str, Any]] = {}

    for key in ("b4", "b8", "b12"):
        adapter = adapters[key]

        missing = [
            name
            for name in attrs
            if not hasattr(adapter, name)
        ]
        if missing:
            raise Day04HookConfigError(
                f"{key} Adapter missing attributes: {missing}"
            )

        rows[key] = {
            "in_dim":
                int(adapter.in_dim),

            "bottleneck_dim":
                int(adapter.bottleneck_dim),

            "projection_dim":
                int(adapter.projection_dim),

            "kernel_size":
                int(adapter.kernel_size),

            "bias":
                bool(adapter.bias),

            "gamma_init":
                float(adapter.gamma_init),
        }

    reference = rows["b4"]

    # The candidate architecture must be identical across blocks.
    for key in ("b8", "b12"):
        if rows[key] != reference:
            raise Day04HookConfigError(
                "b4/b8/b12 must use identical Adapter architecture "
                f"hyperparameters; b4={reference}, {key}={rows[key]}"
            )

    # The modules and their Parameter objects must remain independent.
    if (
        adapters["b4"] is adapters["b8"]
        or adapters["b4"] is adapters["b12"]
        or adapters["b8"] is adapters["b12"]
    ):
        raise Day04HookConfigError(
            "b4/b8/b12 must be independent Adapter module instances"
        )

    param_ids = {
        key: {id(p) for p in adapters[key].parameters()}
        for key in ("b4", "b8", "b12")
    }

    for left, right in (
        ("b4", "b8"),
        ("b4", "b12"),
        ("b8", "b12"),
    ):
        if param_ids[left] & param_ids[right]:
            raise Day04HookConfigError(
                f"{left}/{right} share trainable Parameter objects"
            )

    return reference


def _parse_model_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_cfg = _require_mapping(
        config.get("model"),
        name="model",
    )

    fusion_cfg = _require_mapping(
        model_cfg.get("fusion"),
        name="model.fusion",
    )
    decoder_cfg = _require_mapping(
        model_cfg.get("decoder"),
        name="model.decoder",
    )
    loss_cfg = _require_mapping(
        model_cfg.get("loss"),
        name="model.loss",
    )

    fusion_type = str(
        fusion_cfg.get(
            "type",
            "attention_fusion_v0",
        )
    )
    if fusion_type != "attention_fusion_v0":
        raise Day04HookConfigError(
            "Day-04 hook supports only "
            "model.fusion.type='attention_fusion_v0'"
        )

    decoder_type = str(
        decoder_cfg.get(
            "type",
            "basic_decoder",
        )
    )
    if decoder_type != "basic_decoder":
        raise Day04HookConfigError(
            "Day-04 hook supports only "
            "model.decoder.type='basic_decoder'"
        )

    loss_type = str(
        loss_cfg.get(
            "type",
            "bce_plus_dice",
        )
    )
    if loss_type != "bce_plus_dice":
        raise Day04HookConfigError(
            "Day-04 hook supports only "
            "model.loss.type='bce_plus_dice'"
        )

    fusion_dim = _require_positive_int(
        fusion_cfg.get("dim"),
        name="model.fusion.dim",
    )

    share_projection = fusion_cfg.get(
        "share_projection_across_views",
        True,
    )
    if not isinstance(
        share_projection,
        bool,
    ):
        raise Day04HookConfigError(
            "model.fusion.share_projection_across_views "
            "must be bool"
        )

    output_size = _parse_output_size(
        decoder_cfg.get(
            "output_size",
            [512, 512],
        )
    )

    bce_weight = (
        _require_finite_nonnegative_float(
            loss_cfg.get(
                "bce_weight",
                1.0,
            ),
            name="model.loss.bce_weight",
        )
    )

    dice_weight = (
        _require_finite_nonnegative_float(
            loss_cfg.get(
                "dice_weight",
                1.0,
            ),
            name="model.loss.dice_weight",
        )
    )

    if (
        bce_weight == 0.0
        and dice_weight == 0.0
    ):
        raise Day04HookConfigError(
            "At least one loss weight must be > 0"
        )

    dice_eps = float(
        loss_cfg.get(
            "dice_eps",
            1e-6,
        )
    )
    if (
        not torch.isfinite(
            torch.tensor(dice_eps)
        )
        or dice_eps <= 0.0
    ):
        raise Day04HookConfigError(
            "model.loss.dice_eps must be finite and > 0"
        )

    return {
        "fusion_dim":
            fusion_dim,

        "share_projection_across_views":
            share_projection,

        "output_size":
            output_size,

        "bce_weight":
            bce_weight,

        "dice_weight":
            dice_weight,

        "dice_eps":
            dice_eps,
    }


def build_model(
    *,
    adapters: nn.ModuleDict,
    config: Mapping[str, Any],
) -> nn.Module:
    """Build the fixed cached-feature pipeline around the supplied candidate.

    ``screen_adapter.py`` owns candidate construction. This function MUST use
    the exact supplied Adapter objects rather than rebuilding them.
    """
    adapter_meta = _validate_adapters(
        adapters
    )
    model_cfg = _parse_model_config(
        config
    )

    in_channels = int(
        adapter_meta["in_dim"]
    )
    fusion_dim = int(
        model_cfg["fusion_dim"]
    )

    aligner = ContextToLocalAligner(
        check_finite=True,
        check_bounds=True,
    )

    projection = SixFeatureProjection(
        in_channels=in_channels,
        fusion_dim=fusion_dim,
        blocks=(4, 8, 12),
        share_across_views=bool(
            model_cfg[
                "share_projection_across_views"
            ]
        ),
        check_finite=True,
    )

    head = MSILADay2Head(
        fusion_dim=fusion_dim,
        output_size=tuple(
            model_cfg["output_size"]
        ),
        validate=True,
    )

    model = CachedFeatureTrainingModel(
        in_channels=in_channels,
        adapters=adapters,
        aligner=aligner,
        projection=projection,
        head=head,
        blocks=(4, 8, 12),
        validate=True,
    )

    # Loss contains no learned parameters, so registering it does not alter the
    # trainable architecture or optimizer count. Keeping it on the model makes
    # train/val step behavior deterministic and avoids reconstructing it each
    # batch.
    model.day04_loss = (
        AnomalySegmentationLoss(
            bce_weight=float(
                model_cfg["bce_weight"]
            ),
            dice_weight=float(
                model_cfg["dice_weight"]
            ),
            dice_eps=float(
                model_cfg["dice_eps"]
            ),
        )
    )

    # Hard identity gate: the hook must register the exact candidate objects.
    if model.adapters is not adapters:
        raise RuntimeError(
            "Hook rebuilt/copied the Adapter ModuleDict; "
            "the exact supplied object must be registered."
        )

    return model


def _require_mask(
    batch: Mapping[str, Any],
) -> Tensor:
    mask = batch.get("mask")

    if not isinstance(mask, Tensor):
        raise Day04HookConfigError(
            "batch['mask'] must be torch.Tensor"
        )

    if (
        mask.ndim != 4
        or mask.shape[1] != 1
    ):
        raise Day04HookConfigError(
            "batch['mask'] must have shape [B,1,H,W]"
        )

    return mask


def step(
    *,
    model: nn.Module,
    batch: Mapping[str, Any],
    stage: str,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run one train/val/predict forward under the screen-runner contract."""
    del config  # architecture/loss were locked at build_model() time.

    if not isinstance(
        model,
        CachedFeatureTrainingModel,
    ):
        raise TypeError(
            "Day-04 hook expects CachedFeatureTrainingModel"
        )

    if stage not in {
        "train",
        "val",
        "predict",
    }:
        raise ValueError(
            f"Unsupported stage={stage!r}"
        )

    if stage in {
        "train",
        "val",
    }:
        mask = _require_mask(
            batch
        )

        # Raw logits are required by BCEWithLogits.
        logits = model(
            batch,
            output_size=(
                int(mask.shape[-2]),
                int(mask.shape[-1]),
            ),
        )

        loss_out = model.day04_loss(
            logits,
            mask,
        )

        return {
            "loss":
                loss_out["loss"],

            "metrics": {
                "bce":
                    loss_out["bce"],

                "dice_loss":
                    loss_out["dice"],

                "positive_samples":
                    loss_out[
                        "positive_samples"
                    ],
            },
        }

    # Prediction artifact is a probability anomaly map. Do NOT threshold here:
    # AU-PRO_0.05 needs continuous scores.
    mask = batch.get("mask")
    runtime_size = None

    if isinstance(mask, Tensor):
        if (
            mask.ndim != 4
            or mask.shape[1] != 1
        ):
            raise Day04HookConfigError(
                "batch['mask'] must have shape [B,1,H,W]"
            )
        runtime_size = (
            int(mask.shape[-2]),
            int(mask.shape[-1]),
        )

    logits = model(
        batch,
        output_size=runtime_size,
    )

    probability = torch.sigmoid(
        logits
    )

    if not bool(
        torch.isfinite(
            probability
        ).all()
    ):
        raise RuntimeError(
            "Prediction contains NaN/Inf"
        )

    return {
        "prediction":
            probability,
    }
