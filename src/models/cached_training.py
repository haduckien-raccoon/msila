"""
MS-ILA Day-3 — Task 14: cached-feature integration glue.

This module does not introduce a new research component. It only bridges the
existing feature-cache schema to the already implemented trainable pipeline:

    cached frozen DINO features
      -> shared per-block ResidualAdapter2d
      -> Context->Local geometric alignment
      -> SixFeatureProjection
      -> MSILADay2Head (AttentionFusion + BasicDecoder)
      -> anomaly logits

Why this glue is necessary
--------------------------
The cache schema uses keys

    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

for *raw frozen-backbone features*.

The Day-2 head uses the same six names for *aligned + projected* features.
Those two contracts must not be confused. This module explicitly converts
between them instead of feeding raw cached tensors directly into the Day-2 head.

No DINOv3 module is instantiated in cached-feature training.

Day-04 adapter contract
-----------------------
The current ResidualAdapter2d is explicitly parameterized by two independent
screening dimensions:

    r = bottleneck_dim
    d = projection_dim

and implements

    C -> r -> DWConv -> d -> C.

These adapter dimensions are NOT the same quantity as ``fusion_dim`` used by
``SixFeatureProjection`` and ``MSILADay2Head``.  The builder below therefore
keeps the names separate:

    adapter_bottleneck_dim  -> Adapter r
    adapter_projection_dim  -> Adapter d
    fusion_dim              -> downstream projected/fused feature width

The legacy ``adapter_reduction`` API is intentionally not supported here,
because it cannot represent the new two-dimensional Day-04 adapter screen
without ambiguity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

CACHE_FEATURE_KEYS: tuple[str, ...] = (
    "local_b4",
    "local_b8",
    "local_b12",
    "context_b4",
    "context_b8",
    "context_b12",
)


class CachedTrainingContractError(ValueError):
    """Raised when cache/model integration contracts are incompatible."""


def _as_matrix_3x3(value: Any, *, name: str) -> Tensor:
    matrix = torch.as_tensor(value, dtype=torch.float64)
    if matrix.shape != (3, 3):
        raise CachedTrainingContractError(
            f"{name} must have shape [3,3], got {tuple(matrix.shape)}"
        )
    if not bool(torch.isfinite(matrix).all()):
        raise CachedTrainingContractError(f"{name} contains NaN/Inf")
    return matrix


def _as_hw(value: Any, *, name: str) -> Tensor:
    hw = torch.as_tensor(value, dtype=torch.int64).flatten()
    if hw.numel() != 2:
        raise CachedTrainingContractError(
            f"{name} must contain [H,W], got shape={tuple(hw.shape)}"
        )
    if bool((hw <= 0).any()):
        raise CachedTrainingContractError(f"{name} must contain positive H,W")
    return hw


def _geometry_item_for_aligner(geometry: Mapping[str, Any]) -> dict[str, Tensor]:
    """
    Convert one cache geometry record to ContextToLocalAligner geometry.

    Preferred keys:
        local_to_context
        local_input_hw
        context_input_hw

    Backward-compatible cache-v1 fallback:
        local_to_context = inverse(context_to_local)
        local_input_hw   = local_hw
        context_input_hw = context_hw
    """
    if not isinstance(geometry, Mapping):
        raise CachedTrainingContractError(
            f"geometry must be a mapping, got {type(geometry)!r}"
        )

    if "local_to_context" in geometry:
        local_to_context = _as_matrix_3x3(
            geometry["local_to_context"],
            name="geometry.local_to_context",
        )
    elif "context_to_local" in geometry:
        context_to_local = _as_matrix_3x3(
            geometry["context_to_local"],
            name="geometry.context_to_local",
        )
        try:
            local_to_context = torch.linalg.inv(context_to_local)
        except RuntimeError as exc:
            raise CachedTrainingContractError(
                "geometry.context_to_local is singular and cannot be inverted"
            ) from exc
    else:
        raise CachedTrainingContractError(
            "geometry must contain local_to_context or context_to_local"
        )

    if "local_input_hw" in geometry:
        local_hw = _as_hw(
            geometry["local_input_hw"],
            name="geometry.local_input_hw",
        )
    elif "local_hw" in geometry:
        local_hw = _as_hw(
            geometry["local_hw"],
            name="geometry.local_hw",
        )
    else:
        raise CachedTrainingContractError(
            "geometry must contain local_input_hw or local_hw"
        )

    if "context_input_hw" in geometry:
        context_hw = _as_hw(
            geometry["context_input_hw"],
            name="geometry.context_input_hw",
        )
    elif "context_hw" in geometry:
        context_hw = _as_hw(
            geometry["context_hw"],
            name="geometry.context_hw",
        )
    else:
        raise CachedTrainingContractError(
            "geometry must contain context_input_hw or context_hw"
        )

    return {
        "local_to_context": local_to_context,
        "local_input_hw": local_hw,
        "context_input_hw": context_hw,
    }


def cached_meta_to_aligner_geometry(
    meta: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str,
    matrix_dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Batch geometry emitted by ``cached_collate_fn`` for the aligner."""
    if not isinstance(meta, Sequence) or isinstance(meta, (str, bytes)):
        raise CachedTrainingContractError(
            "batch['meta'] must be the list emitted by cached_collate_fn"
        )
    if len(meta) == 0:
        raise CachedTrainingContractError("batch['meta'] must not be empty")

    items: list[dict[str, Tensor]] = []
    for i, record in enumerate(meta):
        if not isinstance(record, Mapping):
            raise CachedTrainingContractError(
                f"meta[{i}] must be a mapping"
            )
        geometry = record.get("geometry")
        if not isinstance(geometry, Mapping):
            raise CachedTrainingContractError(
                f"meta[{i}] is missing mapping field 'geometry'"
            )
        items.append(_geometry_item_for_aligner(geometry))

    local_to_context = torch.stack(
        [item["local_to_context"] for item in items],
        dim=0,
    ).to(device=device, dtype=matrix_dtype)

    local_input_hw = torch.stack(
        [item["local_input_hw"] for item in items],
        dim=0,
    ).to(device=device)

    context_input_hw = torch.stack(
        [item["context_input_hw"] for item in items],
        dim=0,
    ).to(device=device)

    return {
        "local_to_context": local_to_context,
        "local_input_hw": local_input_hw,
        "context_input_hw": context_input_hw,
    }


def _validate_cached_features(
    batch: Mapping[str, Any],
    *,
    in_channels: int,
) -> tuple[int, int, int]:
    missing = [key for key in CACHE_FEATURE_KEYS if key not in batch]
    if missing:
        raise CachedTrainingContractError(
            f"cached batch missing feature keys: {missing}"
        )

    reference: Tensor | None = None
    for key in CACHE_FEATURE_KEYS:
        x = batch[key]
        if not isinstance(x, Tensor):
            raise CachedTrainingContractError(
                f"{key} must be torch.Tensor, got {type(x)!r}"
            )
        if x.ndim != 4:
            raise CachedTrainingContractError(
                f"{key} must be [B,C,H,W], got {tuple(x.shape)}"
            )
        if x.shape[1] != int(in_channels):
            raise CachedTrainingContractError(
                f"{key}: expected C={in_channels}, got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise CachedTrainingContractError(
                f"{key} must be floating point"
            )
        if not bool(torch.isfinite(x).all()):
            raise CachedTrainingContractError(
                f"{key} contains NaN/Inf"
            )

        if reference is None:
            reference = x
        else:
            if x.shape[0] != reference.shape[0]:
                raise CachedTrainingContractError(
                    "all cached sources must share batch dimension"
                )
            if x.shape[-2:] != reference.shape[-2:]:
                raise CachedTrainingContractError(
                    "all cached sources must share feature H/W"
                )
            if x.device != reference.device or x.dtype != reference.dtype:
                raise CachedTrainingContractError(
                    "all cached sources must share device and dtype"
                )

    assert reference is not None
    return (
        int(reference.shape[0]),
        int(reference.shape[-2]),
        int(reference.shape[-1]),
    )


class CachedFeatureTrainingModel(nn.Module):
    """
    Integration wrapper for training directly from frozen cached DINO features.

    The wrapper is intentionally composition-only. It reuses existing project
    modules and adds no new anomaly-detection operation.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        adapters: nn.Module,
        aligner: nn.Module,
        projection: nn.Module,
        head: nn.Module,
        blocks: Sequence[int] = (4, 8, 12),
        validate: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.blocks = tuple(int(b) for b in blocks)
        self.validate = bool(validate)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if self.blocks != (4, 8, 12):
            raise ValueError(
                "Current cache contract is locked to blocks (4,8,12)"
            )

        self.adapters = adapters
        self.aligner = aligner
        self.projection = projection
        self.head = head

        for name, module in (
            ("adapters", adapters),
            ("aligner", aligner),
            ("projection", projection),
            ("head", head),
        ):
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be nn.Module")

        required_adapter_keys = {f"b{b}" for b in self.blocks}
        if not isinstance(self.adapters, nn.ModuleDict):
            raise TypeError(
                "adapters must be nn.ModuleDict keyed by b4/b8/b12"
            )
        if set(self.adapters.keys()) != required_adapter_keys:
            raise ValueError(
                "adapters must contain exactly b4, b8, b12"
            )

    @classmethod
    def build_default(
        cls,
        *,
        in_channels: int,
        fusion_dim: int,
        adapter_bottleneck_dim: int,
        adapter_projection_dim: int,
        output_size: tuple[int, int] = (512, 512),
        adapter_kernel_size: int = 3,
        gamma_init: float = 0.0,
        adapter_bias: bool = True,
        share_projection_across_views: bool = True,
        validate: bool = True,
    ) -> "CachedFeatureTrainingModel":
        """Build the cached-feature trainable pipeline with the Day-04 Adapter.

        Parameters
        ----------
        in_channels:
            Frozen DINO feature width ``C``.

        fusion_dim:
            Downstream width used by ``SixFeatureProjection`` and
            ``MSILADay2Head``.  This is NOT the Day-04 Adapter variable ``d``.

        adapter_bottleneck_dim:
            Day-04 Adapter screening variable ``r``.

        adapter_projection_dim:
            Day-04 Adapter screening variable ``d``.

        output_size:
            Default dense anomaly-logit resolution.  During ``forward()``, a
            batch mask can still provide the runtime output size.

        adapter_kernel_size:
            Odd depthwise-convolution kernel size inside every Adapter.

        gamma_init:
            Initial residual gate.  ``0.0`` preserves exact identity at
            initialization.

        adapter_bias:
            Bias policy for all Adapter Conv2d layers.  Keep fixed across the
            r×d screen.

        share_projection_across_views:
            Whether Local/Context reuse one downstream C->fusion_dim projector
            per DINO block.  Keep fixed across candidates.

        validate:
            Enable the existing numerical/shape checks in downstream modules.

        Notes
        -----
        One Adapter is created per DINO block and shared between the Local and
        Context views of that block.  All three block Adapters use the same
        ``(r, d)`` candidate, which is the controlled Day-04 architecture.
        """
        from .residual_adapter import ResidualAdapter2d
        from .context_alignment import ContextToLocalAligner
        from .feature_projection import SixFeatureProjection
        from .msila import MSILADay2Head

        # Fail early with an unambiguous builder-level contract.  The Adapter
        # itself performs the same strict validation, but checking here gives a
        # clearer error before constructing any project modules.
        for name, value in (
            ("in_channels", in_channels),
            ("fusion_dim", fusion_dim),
            ("adapter_bottleneck_dim", adapter_bottleneck_dim),
            ("adapter_projection_dim", adapter_projection_dim),
            ("adapter_kernel_size", adapter_kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if adapter_kernel_size % 2 == 0:
            raise ValueError(
                "adapter_kernel_size must be odd so Adapter H×W is preserved"
            )
        if not isinstance(adapter_bias, bool):
            raise TypeError(
                f"adapter_bias must be bool, got {type(adapter_bias).__name__}"
            )

        adapters = nn.ModuleDict(
            {
                f"b{block}": ResidualAdapter2d(
                    in_dim=int(in_channels),
                    bottleneck_dim=int(adapter_bottleneck_dim),
                    projection_dim=int(adapter_projection_dim),
                    kernel_size=int(adapter_kernel_size),
                    gamma_init=float(gamma_init),
                    bias=bool(adapter_bias),
                )
                for block in (4, 8, 12)
            }
        )

        aligner = ContextToLocalAligner(
            check_finite=bool(validate),
            check_bounds=True,
        )

        # IMPORTANT:
        # fusion_dim belongs to the downstream Local/Context projection + head.
        # It must not be silently reused as Adapter projection_dim.
        projection = SixFeatureProjection(
            in_channels=int(in_channels),
            fusion_dim=int(fusion_dim),
            blocks=(4, 8, 12),
            share_across_views=bool(share_projection_across_views),
            check_finite=bool(validate),
        )

        head = MSILADay2Head(
            fusion_dim=int(fusion_dim),
            output_size=output_size,
            validate=bool(validate),
        )

        return cls(
            in_channels=in_channels,
            adapters=adapters,
            aligner=aligner,
            projection=projection,
            head=head,
            validate=validate,
        )

    @property
    def fusion(self) -> nn.Module:
        return self.head.fusion

    @property
    def decoder(self) -> nn.Module:
        return self.head.decoder

    def _adapt(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        local: dict[str, Tensor] = {}
        context: dict[str, Tensor] = {}

        for block in self.blocks:
            adapter = self.adapters[f"b{block}"]

            # One block-specific Adapter is shared between Local and Context
            # because both features come from the same frozen DINO channel
            # basis.  ResidualAdapter2d preserves [B,C,H,W], so geometry
            # alignment still operates in the original backbone feature space.
            local_input = batch[f"local_b{block}"]
            context_input = batch[f"context_b{block}"]

            local_output = adapter(local_input)
            context_output = adapter(context_input)

            if self.validate:
                if local_output.shape != local_input.shape:
                    raise CachedTrainingContractError(
                        f"b{block} Local Adapter changed shape: "
                        f"{tuple(local_input.shape)} -> {tuple(local_output.shape)}"
                    )
                if context_output.shape != context_input.shape:
                    raise CachedTrainingContractError(
                        f"b{block} Context Adapter changed shape: "
                        f"{tuple(context_input.shape)} -> {tuple(context_output.shape)}"
                    )
                if not bool(torch.isfinite(local_output).all()):
                    raise CachedTrainingContractError(
                        f"b{block} Local Adapter output contains NaN/Inf"
                    )
                if not bool(torch.isfinite(context_output).all()):
                    raise CachedTrainingContractError(
                        f"b{block} Context Adapter output contains NaN/Inf"
                    )

            local[f"L{block}"] = local_output
            context[f"C{block}"] = context_output

        return local, context

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        output_size: tuple[int, int] | None = None,
        return_trace: bool = False,
    ):
        """
        Run cached raw features -> Adapter -> Alignment -> Projection -> Head.
        """
        batch_size, feature_h, feature_w = _validate_cached_features(
            batch,
            in_channels=self.in_channels,
        )

        local, context = self._adapt(batch)

        meta = batch.get("meta")
        if meta is None:
            raise CachedTrainingContractError(
                "cached batch must contain 'meta' from cached_collate_fn"
            )

        geometry = cached_meta_to_aligner_geometry(
            meta,
            device=local["L4"].device,
            matrix_dtype=(
                torch.float64
                if local["L4"].dtype == torch.float64
                else torch.float32
            ),
        )

        if int(geometry["local_to_context"].shape[0]) != batch_size:
            raise CachedTrainingContractError(
                "metadata batch size does not match feature batch size"
            )

        aligned_context = self.aligner(
            context,
            geometry,
            target_hw=(feature_h, feature_w),
        )

        projected = self.projection(
            local,
            aligned_context,
        )

        if output_size is None:
            mask = batch.get("mask")
            if isinstance(mask, Tensor):
                if mask.ndim != 4 or mask.shape[1] != 1:
                    raise CachedTrainingContractError(
                        "batch['mask'] must be [B,1,H,W]"
                    )
                output_size = (
                    int(mask.shape[-2]),
                    int(mask.shape[-1]),
                )

        logits, head_trace = self.head(
            projected,
            output_size=output_size,
            return_trace=True,
        )

        if not return_trace:
            return logits

        trace = {
            "adapted_local": local,
            "adapted_context": context,
            "aligned_context": aligned_context,
            "projected": projected,
            **head_trace,
        }
        return logits, trace
