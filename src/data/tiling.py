
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TileRecord:
    tile_id: int
    local_xyxy: Tuple[int, int, int, int]
    context_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]


def generate_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not (0 <= overlap < tile_size):
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)

    return starts


def generate_tile_records(
    H: int,
    W: int,
    local_size: int = 512,
    overlap: int = 128,
    context_size: int = 768,
) -> List[TileRecord]:
    ys = generate_starts(H, local_size, overlap)
    xs = generate_starts(W, local_size, overlap)

    records = []
    tile_id = 0

    for y0 in ys:
        for x0 in xs:
            local_x0 = x0
            local_y0 = y0
            local_x1 = x0 + local_size
            local_y1 = y0 + local_size

            cx = x0 + local_size / 2.0
            cy = y0 + local_size / 2.0

            context_x0 = int(round(cx - context_size / 2.0))
            context_y0 = int(round(cy - context_size / 2.0))
            context_x1 = context_x0 + context_size
            context_y1 = context_y0 + context_size

            records.append(
                TileRecord(
                    tile_id=tile_id,
                    local_xyxy=(local_x0, local_y0, local_x1, local_y1),
                    context_xyxy=(context_x0, context_y0, context_x1, context_y1),
                    center_xy=(cx, cy),
                )
            )
            tile_id += 1

    return records


def crop_with_padding(
    chw: torch.Tensor,
    xyxy: Sequence[int],
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
) -> torch.Tensor:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(chw.shape)}")

    x0, y0, x1, y1 = map(int, xyxy)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid box: {xyxy}")

    C, H, W = chw.shape

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)

    crop = chw[:, iy0:iy1, ix0:ix1]

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
        # Reflect requires input dimension > padding in some edge cases.
        # Fall back to replicate for very small images.
        effective_mode = pad_mode
        if crop.shape[-1] <= max(pad_left, pad_right) or crop.shape[-2] <= max(pad_top, pad_bottom):
            effective_mode = "replicate"

        if effective_mode == "constant":
            crop = F.pad(
                crop,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=pad_value,
            )
        else:
            crop = F.pad(
                crop,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode=effective_mode,
            )

    expected_h = y1 - y0
    expected_w = x1 - x0

    if tuple(crop.shape[-2:]) != (expected_h, expected_w):
        raise AssertionError(
            f"Crop shape {tuple(crop.shape[-2:])} != {(expected_h,expected_w)}"
        )

    return crop


def extract_local_context(
    image_chw: torch.Tensor,
    record: TileRecord,
    local_out_size: int = 512,
    context_out_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    local = crop_with_padding(
        image_chw,
        record.local_xyxy,
        pad_mode="reflect",
    )

    context = crop_with_padding(
        image_chw,
        record.context_xyxy,
        pad_mode="reflect",
    )

    if tuple(local.shape[-2:]) != (local_out_size, local_out_size):
        local = F.interpolate(
            local[None],
            size=(local_out_size, local_out_size),
            mode="bilinear",
            align_corners=False,
        )[0]

    context = F.interpolate(
        context[None],
        size=(context_out_size, context_out_size),
        mode="bilinear",
        align_corners=False,
    )[0]

    return local, context


def coverage_fraction_1d(
    length: int,
    tile_size: int = 512,
    overlap: int = 128,
) -> float:
    starts = generate_starts(length, tile_size, overlap)
    covered = torch.zeros(length, dtype=torch.bool)

    for s in starts:
        a = max(0, s)
        b = min(length, s + tile_size)
        covered[a:b] = True

    return float(covered.float().mean())


def coverage_fraction_2d(
    H: int,
    W: int,
    tile_size: int = 512,
    overlap: int = 128,
) -> float:
    # Grid is Cartesian product of independent x/y starts.
    return coverage_fraction_1d(H, tile_size, overlap) * coverage_fraction_1d(W, tile_size, overlap)


def hann2d(
    size: int = 512,
    eps: float = 1e-3,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    w = torch.hann_window(
        size,
        periodic=False,
        dtype=dtype,
        device=device,
    )
    w2 = torch.outer(w, w)
    return w2.clamp_min(eps)


def stitch_tiles_hann(
    tile_maps: Sequence[torch.Tensor],
    records: Sequence[TileRecord],
    out_hw: Tuple[int, int],
    local_size: int = 512,
    eps: float = 1e-8,
) -> torch.Tensor:
    if len(tile_maps) != len(records):
        raise ValueError("tile_maps and records must have same length")

    H, W = map(int, out_hw)

    if len(tile_maps) == 0:
        raise ValueError("No tile maps")

    first = tile_maps[0]

    if first.ndim == 2:
        channels = None
        accum = torch.zeros((H, W), dtype=first.dtype, device=first.device)
        weight_sum = torch.zeros_like(accum)
    elif first.ndim == 3:
        channels = first.shape[0]
        accum = torch.zeros((channels, H, W), dtype=first.dtype, device=first.device)
        weight_sum = torch.zeros((1, H, W), dtype=first.dtype, device=first.device)
    else:
        raise ValueError("Each tile map must be HW or CHW")

    window = hann2d(
        local_size,
        device=first.device,
        dtype=first.dtype,
    )

    for tile, rec in zip(tile_maps, records):
        if tile.shape[-2:] != (local_size, local_size):
            mode = "bilinear"
            if tile.ndim == 2:
                tile = F.interpolate(
                    tile[None, None],
                    size=(local_size, local_size),
                    mode=mode,
                    align_corners=False,
                )[0, 0]
            else:
                tile = F.interpolate(
                    tile[None],
                    size=(local_size, local_size),
                    mode=mode,
                    align_corners=False,
                )[0]

        x0, y0, x1, y1 = rec.local_xyxy

        ox0 = max(0, x0)
        oy0 = max(0, y0)
        ox1 = min(W, x1)
        oy1 = min(H, y1)

        if ox1 <= ox0 or oy1 <= oy0:
            continue

        tx0 = ox0 - x0
        ty0 = oy0 - y0
        tx1 = tx0 + (ox1 - ox0)
        ty1 = ty0 + (oy1 - oy0)

        w_patch = window[ty0:ty1, tx0:tx1]

        if tile.ndim == 2:
            t_patch = tile[ty0:ty1, tx0:tx1]
            accum[oy0:oy1, ox0:ox1] += t_patch * w_patch
            weight_sum[oy0:oy1, ox0:ox1] += w_patch
        else:
            t_patch = tile[:, ty0:ty1, tx0:tx1]
            accum[:, oy0:oy1, ox0:ox1] += t_patch * w_patch[None]
            weight_sum[:, oy0:oy1, ox0:ox1] += w_patch[None]

    if torch.any(weight_sum <= 0):
        raise AssertionError("Found uncovered pixels during stitching")

    return accum / weight_sum.clamp_min(eps)


def crop_mask_tiles(
    mask_hw: torch.Tensor,
    records: Sequence[TileRecord],
    local_size: int = 512,
) -> List[torch.Tensor]:
    if mask_hw.ndim != 2:
        raise ValueError("mask must be HW")

    x = mask_hw[None].float()

    tiles = []
    for rec in records:
        tile = crop_with_padding(
            x,
            rec.local_xyxy,
            pad_mode="constant",
            pad_value=0.0,
        )[0]

        if tuple(tile.shape) != (local_size, local_size):
            tile = F.interpolate(
                tile[None, None],
                size=(local_size, local_size),
                mode="nearest",
            )[0, 0]

        tiles.append((tile > 0.5).to(mask_hw.dtype))

    return tiles


def stitch_mask_tiles(
    mask_tiles: Sequence[torch.Tensor],
    records: Sequence[TileRecord],
    out_hw: Tuple[int, int],
    local_size: int = 512,
) -> torch.Tensor:
    H, W = map(int, out_hw)
    out = torch.zeros((H, W), dtype=mask_tiles[0].dtype, device=mask_tiles[0].device)

    for tile, rec in zip(mask_tiles, records):
        if tile.shape != (local_size, local_size):
            tile = F.interpolate(
                tile[None, None].float(),
                size=(local_size, local_size),
                mode="nearest",
            )[0, 0].to(mask_tiles[0].dtype)

        x0, y0, x1, y1 = rec.local_xyxy

        ox0 = max(0, x0)
        oy0 = max(0, y0)
        ox1 = min(W, x1)
        oy1 = min(H, y1)

        tx0 = ox0 - x0
        ty0 = oy0 - y0
        tx1 = tx0 + (ox1 - ox0)
        ty1 = ty0 + (oy1 - oy0)

        patch = tile[ty0:ty1, tx0:tx1]
        out[oy0:oy1, ox0:ox1] = torch.maximum(
            out[oy0:oy1, ox0:ox1],
            patch,
        )

    return out
