"""Projection bridge for MS-ILA Local/Context representation sources.

Project-relative path:
    msila/src/models/feature_projection.py

This keeps the Day-2 projection itself unchanged mathematically: every selected
DINO feature is mapped with a learned 1x1 Conv2d to the common ``fusion_dim``.
The Day-05 extension is only an interface/mapping layer so the same module can
serve R0 (1 source), R1 (3 sources), and R2 (6 sources) without forcing Local-
only candidates to provide unused Context tensors.

Raw extractor/alignment keys:
    Local   : L4, L8, L12
    Context : C4_to_L, C8_to_L, C12_to_L

Canonical Day-05 keys returned to FeatureSelector/Fusion:
    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

No interpolation/resizing occurs here. Context geometry must already have been
aligned by ``ContextToLocalAligner``. Therefore all selected inputs must already
share B/H/W before the 1x1 projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import torch
from torch import Tensor, nn

__all__ = [
    "CANONICAL_SOURCE_ORDER",
    "SOURCE_TO_RAW",
    "SixFeatureProjection",
]

CANONICAL_SOURCE_ORDER: Final[tuple[str, ...]] = (
    "local_b4",
    "local_b8",
    "local_b12",
    "context_b4",
    "context_b8",
    "context_b12",
)

SOURCE_TO_RAW: Final[dict[str, tuple[str, int, str]]] = {
    "local_b4": ("local", 4, "L4"),
    "local_b8": ("local", 8, "L8"),
    "local_b12": ("local", 12, "L12"),
    "context_b4": ("context", 4, "C4_to_L"),
    "context_b8": ("context", 8, "C8_to_L"),
    "context_b12": ("context", 12, "C12_to_L"),
}


class SixFeatureProjection(nn.Module):
    """Project selected Local/aligned-Context features to one common dimension.

    ``forward(local, aligned_context)`` preserves the Day-2 API and returns all
    six sources. Day 05 should call :meth:`project_sources` with the exact
    representation source keys. The projector weights are still one common
    module; no R0/R1/R2-specific projection module is created.
    """

    def __init__(
        self,
        in_channels: int,
        fusion_dim: int,
        *,
        blocks: Sequence[int] = (4, 8, 12),
        share_across_views: bool = True,
        bias: bool = True,
        check_finite: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.fusion_dim = int(fusion_dim)
        self.blocks = tuple(int(b) for b in blocks)
        self.share_across_views = bool(share_across_views)
        self.check_finite = bool(check_finite)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.fusion_dim <= 0:
            raise ValueError("fusion_dim must be > 0.")
        if self.blocks != (4, 8, 12):
            raise ValueError(
                "MS-ILA locked projection requires blocks=(4, 8, 12); "
                f"got {self.blocks}."
            )

        if self.share_across_views:
            self.projectors = nn.ModuleDict(
                {
                    f"b{block}": nn.Conv2d(
                        self.in_channels,
                        self.fusion_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )
                    for block in self.blocks
                }
            )
            self.local_projectors = None
            self.context_projectors = None
        else:
            self.projectors = None
            self.local_projectors = nn.ModuleDict(
                {
                    f"b{block}": nn.Conv2d(
                        self.in_channels,
                        self.fusion_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )
                    for block in self.blocks
                }
            )
            self.context_projectors = nn.ModuleDict(
                {
                    f"b{block}": nn.Conv2d(
                        self.in_channels,
                        self.fusion_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )
                    for block in self.blocks
                }
            )

    @staticmethod
    def _validate_tensor(name: str, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(x)!r}.")
        if x.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W], got {tuple(x.shape)}.")
        if any(int(v) <= 0 for v in x.shape):
            raise ValueError(f"{name} has an empty dimension: {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {x.dtype}.")

    def _get_projector(self, *, block: int, view: str) -> nn.Conv2d:
        key = f"b{block}"
        if self.share_across_views:
            assert self.projectors is not None
            return self.projectors[key]
        if view == "local":
            assert self.local_projectors is not None
            return self.local_projectors[key]
        assert self.context_projectors is not None
        return self.context_projectors[key]

    def _resolve_input(
        self,
        source_key: str,
        local_features: Mapping[str, Tensor],
        aligned_context_features: Mapping[str, Tensor] | None,
    ) -> tuple[Tensor, int, str, str]:
        if source_key not in SOURCE_TO_RAW:
            raise KeyError(
                f"Unknown Day-05 source {source_key!r}; expected one of "
                f"{CANONICAL_SOURCE_ORDER}."
            )

        view, block, raw_key = SOURCE_TO_RAW[source_key]
        if view == "local":
            source = local_features
        else:
            if aligned_context_features is None:
                raise KeyError(
                    f"{source_key} requires aligned Context features. Run "
                    "ContextToLocalAligner before projection."
                )
            source = aligned_context_features

        if raw_key not in source:
            raise KeyError(f"Missing {raw_key!r} required for {source_key!r}.")

        x = source[raw_key]
        self._validate_tensor(raw_key, x)
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"{raw_key}: expected C={self.in_channels}, got C={x.shape[1]}."
            )
        if self.check_finite and not bool(torch.isfinite(x).all()):
            raise ValueError(f"{raw_key} contains NaN/Inf before projection.")

        return x, block, view, raw_key

    def project_sources(
        self,
        local_features: Mapping[str, Tensor],
        aligned_context_features: Mapping[str, Tensor] | None = None,
        *,
        source_keys: Sequence[str],
    ) -> dict[str, Tensor]:
        """Project exactly the requested R0/R1/R2 source set.

        The method does not resize, align, select alternative layers, or change
        representation policy. It only maps channel dimension C -> fusion_dim.
        """
        if not isinstance(local_features, Mapping):
            raise TypeError("local_features must be a Mapping[str, Tensor].")
        if aligned_context_features is not None and not isinstance(
            aligned_context_features, Mapping
        ):
            raise TypeError("aligned_context_features must be a Mapping or None.")

        keys = tuple(str(k) for k in source_keys)
        if not keys:
            raise ValueError("source_keys must contain at least one source.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"source_keys contains duplicates: {keys}.")

        # Day-05 permits only the three scientifically pre-declared sets.
        allowed_sets = {
            ("local_b12",),
            ("local_b4", "local_b8", "local_b12"),
            CANONICAL_SOURCE_ORDER,
        }
        if keys not in allowed_sets:
            raise ValueError(
                "Day-05 projection only accepts R0/R1/R2 source sets; "
                f"got {keys}."
            )

        resolved: list[tuple[str, Tensor, int, str, str]] = []
        for source_key in keys:
            x, block, view, raw_key = self._resolve_input(
                source_key,
                local_features,
                aligned_context_features,
            )
            resolved.append((source_key, x, block, view, raw_key))

        ref_name, ref = resolved[0][4], resolved[0][1]
        b, _, h, w = ref.shape
        for _, x, _, _, raw_key in resolved[1:]:
            if x.shape[0] != b or x.shape[-2:] != (h, w):
                raise ValueError(
                    "All selected sources must already share B/H/W before projection; "
                    f"reference {ref_name}={tuple(ref.shape)}, "
                    f"{raw_key}={tuple(x.shape)}. Context must be aligned first."
                )
            if x.dtype != ref.dtype:
                raise ValueError(
                    f"Selected source dtype mismatch: {ref_name}={ref.dtype}, "
                    f"{raw_key}={x.dtype}."
                )
            if x.device != ref.device:
                raise ValueError(
                    f"Selected source device mismatch: {ref_name}={ref.device}, "
                    f"{raw_key}={x.device}."
                )

        out: dict[str, Tensor] = {}
        expected_shape = (b, self.fusion_dim, h, w)
        for source_key, x, block, view, _ in resolved:
            y = self._get_projector(block=block, view=view)(x)
            if tuple(y.shape) != expected_shape:
                raise RuntimeError(
                    f"{source_key}: projection contract violated; expected "
                    f"{expected_shape}, got {tuple(y.shape)}."
                )
            if self.check_finite and not bool(torch.isfinite(y).all()):
                raise RuntimeError(f"{source_key} contains NaN/Inf after projection.")
            out[source_key] = y

        return out

    def forward(
        self,
        local_features: Mapping[str, Tensor],
        aligned_context_features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Backward-compatible Day-2 API: project all six sources."""
        return self.project_sources(
            local_features,
            aligned_context_features,
            source_keys=CANONICAL_SOURCE_ORDER,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, fusion_dim={self.fusion_dim}, "
            f"blocks={self.blocks}, share_across_views={self.share_across_views}"
        )
