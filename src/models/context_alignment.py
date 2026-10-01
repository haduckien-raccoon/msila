"""Geometry-aware Context -> Local feature alignment.

Project contract
----------------
Input Context features come from the frozen DINOv3 extractor:

    C4, C8, C12 : [B, C, Hc, Wc]

Geometry metadata comes from ``geometry/view_meta.py`` and contains the
homogeneous transform ``local_to_context`` defined in continuous pixel-edge
coordinates of the model inputs.

The aligned outputs are:

    C4_to_L, C8_to_L, C12_to_L : [B, C, Hl, Wl]

For the current project, both Local and Context are fed to DINOv3 at 512x512,
so ViT-S/16 gives Hl=Wl=Hc=Wc=32. The *field of view* is nevertheless different,
therefore an explicit geometric alignment is still required before fusion.

Core idea
---------
For each Local feature-cell center:

1. convert the cell center to Local input pixel-edge coordinates;
2. map that point with the 3x3 Local->Context homogeneous transform;
3. normalize Context coordinates to [-1, 1];
4. bilinearly sample the Context feature map with ``grid_sample`` using
   ``align_corners=False``.

The module is parameter-free and keeps autograd enabled. That matters if later
experiments insert trainable adapters before alignment.

References
----------
- Jaderberg et al., Spatial Transformer Networks, NeurIPS 2015.
  https://arxiv.org/abs/1506.02025
- PyTorch ``torch.nn.functional.grid_sample`` documentation.
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
- Richard Szeliski, Computer Vision: Algorithms and Applications, 2nd ed.,
  Sections 2.1 and 3.6 (homogeneous coordinates / geometric transforms).
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from geometry.view_meta import ViewGeometryMeta
except Exception:  # pragma: no cover - type convenience for standalone reuse
    ViewGeometryMeta = Any  # type: ignore[misc,assignment]

__all__ = [
    "ContextToLocalAligner",
    "align_context_to_local",
]


_CONTEXT_KEYS = ("C4", "C8", "C12")
_OUTPUT_KEYS = ("C4_to_L", "C8_to_L", "C12_to_L")


def _as_batched_matrix(
    value: Any,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    m = torch.as_tensor(value, device=device, dtype=dtype)
    if m.shape == (3, 3):
        m = m.unsqueeze(0).expand(batch, -1, -1)
    if m.shape != (batch, 3, 3):
        raise ValueError(
            f"{name} must have shape [3,3] or [B,3,3]; got {tuple(m.shape)} "
            f"for B={batch}."
        )
    if not bool(torch.isfinite(m).all()):
        raise ValueError(f"{name} contains NaN/Inf.")
    return m


def _as_batched_hw(
    value: Any,
    *,
    batch: int,
    device: torch.device,
    name: str,
) -> Tensor:
    hw = torch.as_tensor(value, device=device, dtype=torch.float64)
    if hw.shape == (2,):
        hw = hw.unsqueeze(0).expand(batch, -1)
    if hw.shape != (batch, 2):
        raise ValueError(
            f"{name} must have shape [2] or [B,2]; got {tuple(hw.shape)} "
            f"for B={batch}."
        )
    if bool((hw <= 0).any()) or not bool(torch.isfinite(hw).all()):
        raise ValueError(f"{name} must contain finite positive (H,W) values.")
    return hw


def _extract_geometry(
    geometry: Any,
    *,
    batch: int,
    device: torch.device,
    matrix_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return batched (local_to_context, local_input_hw, context_input_hw)."""
    if isinstance(geometry, Mapping):
        required = ("local_to_context", "local_input_hw", "context_input_hw")
        missing = [k for k in required if k not in geometry]
        if missing:
            raise KeyError(f"geometry mapping missing required keys: {missing}")
        l2c_value = geometry["local_to_context"]
        local_hw_value = geometry["local_input_hw"]
        context_hw_value = geometry["context_input_hw"]
    else:
        # Works with ViewGeometryMeta and with compatible dataclass-like objects.
        required = ("local_to_context", "local_input_hw", "context_input_hw")
        missing = [k for k in required if not hasattr(geometry, k)]
        if missing:
            raise TypeError(
                "geometry must be ViewGeometryMeta-like or a mapping with "
                f"{required}; missing {missing}."
            )
        l2c_value = geometry.local_to_context
        local_hw_value = geometry.local_input_hw
        context_hw_value = geometry.context_input_hw

    l2c = _as_batched_matrix(
        l2c_value,
        batch=batch,
        device=device,
        dtype=matrix_dtype,
        name="local_to_context",
    )
    local_hw = _as_batched_hw(
        local_hw_value,
        batch=batch,
        device=device,
        name="local_input_hw",
    )
    context_hw = _as_batched_hw(
        context_hw_value,
        batch=batch,
        device=device,
        name="context_input_hw",
    )
    return l2c, local_hw, context_hw


def _require_constant_hw(hw: Tensor, *, name: str) -> tuple[int, int]:
    """A dense batch tensor cannot have a different output H/W per sample."""
    ref = hw[0]
    if not bool(torch.allclose(hw, ref.expand_as(hw), atol=0.0, rtol=0.0)):
        raise ValueError(
            f"All samples in one batch must share the same {name}; got {hw.tolist()}."
        )
    return int(round(float(ref[0]))), int(round(float(ref[1])))


def _infer_target_hw(
    *,
    context_feature_hw: tuple[int, int],
    local_input_hw: Tensor,
    context_input_hw: Tensor,
) -> tuple[int, int]:
    """Infer Local feature size assuming the same backbone patch stride.

    If Context feature resolution is Hc x Wc for a Context model input
    Hctx x Wctx, the effective patch stride is Hctx/Hc, Wctx/Wc. The same
    frozen backbone applied to Local therefore yields:

        Hl = Hlocal / (Hctx / Hc)
        Wl = Wlocal / (Wctx / Wc)
    """
    local_h, local_w = _require_constant_hw(local_input_hw, name="local_input_hw")
    ctx_h, ctx_w = _require_constant_hw(context_input_hw, name="context_input_hw")
    feat_h, feat_w = context_feature_hw

    target_h_f = float(local_h) * float(feat_h) / float(ctx_h)
    target_w_f = float(local_w) * float(feat_w) / float(ctx_w)
    target_h = int(round(target_h_f))
    target_w = int(round(target_w_f))

    if abs(target_h_f - target_h) > 1e-6 or abs(target_w_f - target_w) > 1e-6:
        raise ValueError(
            "Cannot infer an integer Local feature grid from model-input sizes. "
            f"Computed ({target_h_f:.6f}, {target_w_f:.6f}). Pass target_hw explicitly."
        )
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Inferred invalid target_hw={(target_h, target_w)}.")
    return target_h, target_w


class ContextToLocalAligner(nn.Module):
    """Align C4/C8/C12 onto the Local feature lattice.

    Parameters
    ----------
    mode:
        Sampling mode. ``"bilinear"`` is the recommended default for dense
        continuous feature alignment.
    padding_mode:
        Passed to ``grid_sample``. ``"zeros"`` is deliberately strict: geometry
        bugs are not silently hidden by border replication.
    align_corners:
        Must remain ``False`` for the pixel-edge coordinate convention used by
        ``geometry/view_meta.py`` and for resolution-agnostic sampling.
    check_finite:
        Debug check for NaN/Inf. Disabled by default to avoid GPU synchronization
        overhead during training.
    check_bounds:
        Verify that the requested Local sampling grid lies inside Context FOV.
    bounds_tolerance:
        Numerical tolerance for the normalized [-1,1] grid bound check.
    """

    def __init__(
        self,
        *,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        check_finite: bool = False,
        check_bounds: bool = True,
        bounds_tolerance: float = 1e-5,
    ) -> None:
        super().__init__()
        if mode not in {"bilinear", "nearest"}:
            raise ValueError("mode must be 'bilinear' or 'nearest'.")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("Unsupported padding_mode.")
        if align_corners:
            raise ValueError(
                "This project uses pixel-edge geometry; align_corners must be False."
            )
        if bounds_tolerance < 0:
            raise ValueError("bounds_tolerance must be >= 0.")

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = False
        self.check_finite = bool(check_finite)
        self.check_bounds = bool(check_bounds)
        self.bounds_tolerance = float(bounds_tolerance)

    @staticmethod
    def _validate_feature(name: str, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(x)!r}.")
        if x.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W], got {tuple(x.shape)}.")
        if x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError(f"{name} has an invalid empty dimension: {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {x.dtype}.")

    def build_sampling_grid(
        self,
        context_feature: Tensor,
        geometry: Any,
        *,
        target_hw: Sequence[int] | None = None,
    ) -> Tensor:
        """Build a Local-output -> Context-input normalized sampling grid.

        Returns
        -------
        Tensor
            Shape ``[B, Hl, Wl, 2]`` in PyTorch grid-sample order ``(x, y)``.
        """
        self._validate_feature("context_feature", context_feature)
        b, _, hc, wc = context_feature.shape

        # float32 is enough for feature-grid construction; use float64 only when
        # the feature itself is float64. This avoids unnecessary fp64 GPU work.
        matrix_dtype = (
            torch.float64 if context_feature.dtype == torch.float64 else torch.float32
        )
        l2c, local_input_hw, context_input_hw = _extract_geometry(
            geometry,
            batch=b,
            device=context_feature.device,
            matrix_dtype=matrix_dtype,
        )

        if target_hw is None:
            hl, wl = _infer_target_hw(
                context_feature_hw=(hc, wc),
                local_input_hw=local_input_hw,
                context_input_hw=context_input_hw,
            )
        else:
            if len(target_hw) != 2:
                raise ValueError(f"target_hw must be (H,W), got {target_hw!r}.")
            hl, wl = int(target_hw[0]), int(target_hw[1])
            if hl <= 0 or wl <= 0:
                raise ValueError(f"target_hw must be positive, got {(hl, wl)}.")

        # A dense tensor requires common input sizes across the batch.
        _require_constant_hw(local_input_hw, name="local_input_hw")
        _require_constant_hw(context_input_hw, name="context_input_hw")

        # Feature-cell centers represented in Local model-input pixel-edge coords:
        # x_local = (j + 1/2) * W_local_input / W_local_feature
        # y_local = (i + 1/2) * H_local_input / H_local_feature
        ys = (torch.arange(hl, device=context_feature.device, dtype=matrix_dtype) + 0.5)
        xs = (torch.arange(wl, device=context_feature.device, dtype=matrix_dtype) + 0.5)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        local_h = local_input_hw[:, 0].to(matrix_dtype).view(b, 1, 1)
        local_w = local_input_hw[:, 1].to(matrix_dtype).view(b, 1, 1)
        x_local = xx.unsqueeze(0) * (local_w / float(wl))
        y_local = yy.unsqueeze(0) * (local_h / float(hl))
        ones = torch.ones_like(x_local)

        points = torch.stack((x_local, y_local, ones), dim=-1)  # [B,H,W,3]
        mapped = torch.einsum("bij,bhwj->bhwi", l2c, points)
        denom = mapped[..., 2]

        eps = torch.finfo(matrix_dtype).eps * 32
        if bool((denom.abs() <= eps).any()):
            raise ValueError("Local->Context transform maps a sample point to infinity.")

        x_context = mapped[..., 0] / denom
        y_context = mapped[..., 1] / denom

        context_h = context_input_hw[:, 0].to(matrix_dtype).view(b, 1, 1)
        context_w = context_input_hw[:, 1].to(matrix_dtype).view(b, 1, 1)

        # For grid_sample(..., align_corners=False), normalized -1/+1 correspond
        # to the *outer pixel edges*. This exactly matches our pixel-edge geometry.
        grid_x = 2.0 * x_context / context_w - 1.0
        grid_y = 2.0 * y_context / context_h - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)

        if not bool(torch.isfinite(grid).all()):
            raise ValueError("Sampling grid contains NaN/Inf.")

        if self.check_bounds:
            tol = self.bounds_tolerance
            lo = float(grid.amin())
            hi = float(grid.amax())
            if lo < -1.0 - tol or hi > 1.0 + tol:
                raise RuntimeError(
                    "Local sampling grid falls outside Context FOV: "
                    f"range=[{lo:.6f}, {hi:.6f}], expected within [-1,1] "
                    f"(tolerance={tol}). Check geometry metadata."
                )

        # grid_sample expects grid and input to use compatible floating dtypes.
        return grid.to(dtype=context_feature.dtype)

    def _validate_feature_set(self, context_features: Mapping[str, Tensor]) -> None:
        missing = [k for k in _CONTEXT_KEYS if k not in context_features]
        if missing:
            raise KeyError(f"context_features missing required keys: {missing}")

        ref: Tensor | None = None
        for key in _CONTEXT_KEYS:
            x = context_features[key]
            self._validate_feature(key, x)
            if self.check_finite and not bool(torch.isfinite(x).all()):
                raise ValueError(f"{key} contains NaN/Inf.")
            if ref is None:
                ref = x
                continue
            if x.shape[0] != ref.shape[0]:
                raise ValueError("C4/C8/C12 batch dimensions must match.")
            if x.shape[-2:] != ref.shape[-2:]:
                raise ValueError(
                    "C4/C8/C12 must share the same spatial grid for one-pass alignment; "
                    f"got {ref.shape[-2:]} and {x.shape[-2:]} for {key}."
                )
            if x.device != ref.device or x.dtype != ref.dtype:
                raise ValueError("C4/C8/C12 must share device and dtype.")

    def forward(
        self,
        context_features: Mapping[str, Tensor],
        geometry: Any,
        *,
        target_hw: Sequence[int] | None = None,
    ) -> dict[str, Tensor]:
        """Align all three Context levels with one sampling operation.

        Concatenating the three feature tensors along channels before a single
        ``grid_sample`` is mathematically equivalent to sampling them separately
        because bilinear sampling acts independently on channels. It reduces
        Python/kernel-launch overhead without changing the mapping.
        """
        self._validate_feature_set(context_features)

        tensors = [context_features[k] for k in _CONTEXT_KEYS]
        channels = [x.shape[1] for x in tensors]
        packed = torch.cat(tensors, dim=1)

        grid = self.build_sampling_grid(packed, geometry, target_hw=target_hw)
        aligned_packed = F.grid_sample(
            packed,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        aligned = torch.split(aligned_packed, channels, dim=1)

        out = {key: value for key, value in zip(_OUTPUT_KEYS, aligned)}

        if self.check_finite:
            for key, value in out.items():
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"{key} contains NaN/Inf after alignment.")

        return out


def align_context_to_local(
    context_features: Mapping[str, Tensor],
    geometry: Any,
    *,
    target_hw: Sequence[int] | None = None,
) -> dict[str, Tensor]:
    """Functional convenience wrapper using the project defaults."""
    return ContextToLocalAligner()(context_features, geometry, target_hw=target_hw)
