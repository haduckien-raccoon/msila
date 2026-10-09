"""Parameter-free bilinear sampling with deterministic CUDA input gradients.

PyTorch's CUDA grid_sample/interpolate backward raises under strict deterministic
algorithms. Fixed geometry lets us use gather/index_select, whose backward has
a deterministic implementation in that mode. Half-pixel coordinates, padding
and the bilinear kernel are unchanged. There is no learnable grid in MS-ILA.
"""
from __future__ import annotations

import torch
from torch import Tensor


def deterministic_bilinear_sample(
    x: Tensor, grid: Tensor, *, padding_mode: str = "zeros"
) -> Tensor:
    """Equivalent to bilinear grid_sample with align_corners=False, fixed grid.

    Supports the zero/border padding used by the project. Grid gradients are
    deliberately rejected rather than silently discarded. Input gradients stay
    enabled, including adapter gradients in R2.
    """
    if x.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("Expected BCHW input and BHW2 grid")
    if x.shape[0] != grid.shape[0] or x.device != grid.device:
        raise ValueError("Input/grid batch and device must match")
    if not x.is_floating_point() or not grid.is_floating_point():
        raise ValueError("Bilinear input/grid must be floating point")
    if grid.requires_grad:
        raise ValueError("Deterministic sampler requires fixed geometry")
    if padding_mode not in {"zeros", "border"}:
        raise ValueError("Deterministic sampler supports zeros/border padding")
    b, c, h, w = x.shape
    if min(b, c, h, w, *grid.shape[1:3]) < 1:
        raise ValueError("Bilinear input/grid dimensions must be positive")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    coords = grid.to(dtype=dtype)
    px = ((coords[..., 0] + 1) * w - 1) / 2
    py = ((coords[..., 1] + 1) * h - 1) / 2
    if padding_mode == "border":
        px, py = px.clamp(0, w - 1), py.clamp(0, h - 1)
    ix, iy = px.floor().long(), py.floor().long()
    dx, dy = px - ix, py - iy
    flat = x.to(dtype=dtype).flatten(2)

    def corner(cx: Tensor, cy: Tensor, weight: Tensor) -> Tensor:
        index = (cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1)).flatten(1)
        values = torch.gather(flat, 2, index[:, None].expand(-1, c, -1))
        if padding_mode == "zeros":
            weight = weight * ((cx >= 0) & (cx < w) & (cy >= 0) & (cy < h))
        return values * weight.flatten(1)[:, None]

    result = (corner(ix, iy, (1 - dx) * (1 - dy))
              + corner(ix + 1, iy, dx * (1 - dy))
              + corner(ix, iy + 1, (1 - dx) * dy)
              + corner(ix + 1, iy + 1, dx * dy))
    return result.reshape(b, c, *grid.shape[1:3]).to(dtype=x.dtype)


def deterministic_bilinear_resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    """Equivalent to interpolate(size=..., bilinear, align_corners=False).

    Separable interpolation uses the same half-pixel centers and edge clamping.
    Accumulation uses float32 for AMP inputs, returning the original dtype.
    """
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("Expected floating BCHW input")
    oh, ow = map(int, size)
    ih, iw = x.shape[-2:]
    if min(oh, ow, ih, iw) < 1:
        raise ValueError("Resize dimensions must be positive")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

    def indices(source: int, target: int):
        position = ((torch.arange(target, device=x.device, dtype=dtype) + .5)
                    * (source / target) - .5).clamp(min=0)
        lower = position.floor().long().clamp(max=source - 1)
        upper = (lower + 1).clamp(max=source - 1)
        return lower, upper, position - lower

    left, right, wx = indices(iw, ow)
    top, bottom, wy = indices(ih, oh)
    values = x.to(dtype=dtype)
    horizontal = (values.index_select(3, left) * (1 - wx)[None, None, None, :]
                  + values.index_select(3, right) * wx[None, None, None, :])
    resized = (horizontal.index_select(2, top) * (1 - wy)[None, None, :, None]
               + horizontal.index_select(2, bottom) * wy[None, None, :, None])
    return resized.to(dtype=x.dtype)
