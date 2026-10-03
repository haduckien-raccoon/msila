"""Flexible parameter-free Mean Fusion for MS-ILA Day 05.

The same module handles R0/R1/R2. There are no candidate-specific fusion
parameters. R0 is an exact identity: with one source, ``F_fused = F12``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import torch
from torch import Tensor, nn

__all__ = ["MeanFusion"]

_ALLOWED_COUNTS: Final[tuple[int, ...]] = (1, 3, 6)
_LEGACY_ORDER: Final[tuple[str, ...]] = ("b4", "b8", "b12")


class MeanFusion(nn.Module):
    """Parameter-free fusion for 1, 3, or 6 already-compatible feature maps.

    Parameters
    ----------
    validate:
        Enforce BCHW, same shape/dtype/device, finite tensors, and source count.
    return_weights:
        If True, return ``(fused, weights)``. ``weights`` is a length-N tensor
        with fixed uniform coefficients. Defaults to False for compatibility
        with the older Day-1 MeanFusion call that returned only ``fused``.
    """

    def __init__(self, validate: bool = True, *, return_weights: bool = False) -> None:
        super().__init__()
        self.validate = bool(validate)
        self.return_weights = bool(return_weights)

    @staticmethod
    def _ordered_tensors(
        features: Sequence[Tensor] | Mapping[str, Tensor],
    ) -> list[Tensor]:
        if isinstance(features, Mapping):
            if set(features.keys()) == set(_LEGACY_ORDER):
                return [features[k] for k in _LEGACY_ORDER]
            # Python mappings preserve insertion order. Day-05 projection and
            # FeatureSelector both use deterministic scientific source order.
            return list(features.values())
        if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
            return list(features)
        raise TypeError(
            "MeanFusion expects Sequence[Tensor] or Mapping[str, Tensor], "
            f"got {type(features)!r}."
        )

    @staticmethod
    def _validate(xs: Sequence[Tensor]) -> None:
        if len(xs) not in _ALLOWED_COUNTS:
            raise ValueError(
                f"Day-05 MeanFusion requires len(features) in {_ALLOWED_COUNTS}; "
                f"got {len(xs)}."
            )
        ref: Tensor | None = None
        for i, x in enumerate(xs):
            if not isinstance(x, Tensor):
                raise TypeError(f"features[{i}] must be torch.Tensor.")
            if x.ndim != 4:
                raise ValueError(
                    f"features[{i}] must be BCHW [B,C,H,W], got {tuple(x.shape)}."
                )
            if not x.is_floating_point():
                raise TypeError(f"features[{i}] must be floating point, got {x.dtype}.")
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f"features[{i}] contains NaN/Inf.")
            if ref is None:
                ref = x
                continue
            if x.shape != ref.shape:
                raise ValueError(
                    "All MeanFusion inputs must have identical BCHW shape; "
                    f"reference={tuple(ref.shape)}, features[{i}]={tuple(x.shape)}."
                )
            if x.dtype != ref.dtype:
                raise ValueError("All MeanFusion inputs must share dtype.")
            if x.device != ref.device:
                raise ValueError("All MeanFusion inputs must share device.")

    def forward(
        self,
        features: Sequence[Tensor] | Mapping[str, Tensor],
    ) -> Tensor | tuple[Tensor, Tensor]:
        xs = self._ordered_tensors(features)
        if self.validate:
            self._validate(xs)
        elif len(xs) == 0:
            raise ValueError("MeanFusion requires at least one source.")

        # R0: no fake attention/averaging path. Keep exact tensor identity.
        if len(xs) == 1:
            fused = xs[0]
        else:
            fused = torch.stack(xs, dim=0).mean(dim=0)

        if not self.return_weights:
            return fused

        weights = torch.full(
            (len(xs),),
            1.0 / float(len(xs)),
            dtype=fused.dtype,
            device=fused.device,
        )
        return fused, weights
