
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tiling import (
    generate_tile_records,
    crop_mask_tiles,
    stitch_mask_tiles,
)


def iou(a, b):
    a = a > 0
    b = b > 0
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    return inter / union if union else 1.0


def _roundtrip(mask):
    H, W = mask.shape

    records = generate_tile_records(
        H, W,
        local_size=512,
        overlap=128,
        context_size=768,
    )

    tiles = crop_mask_tiles(
        mask,
        records,
        local_size=512,
    )

    recon = stitch_mask_tiles(
        tiles,
        records,
        out_hw=(H, W),
        local_size=512,
    )

    return recon


def test_small_anomaly():
    mask = torch.zeros(900, 1300, dtype=torch.uint8)
    mask[451:454, 711:715] = 1

    recon = _roundtrip(mask)

    assert torch.equal(recon, mask)
    assert iou(recon, mask) == 1.0


def test_border_anomaly():
    mask = torch.zeros(777, 999, dtype=torch.uint8)
    mask[0:15, 0:21] = 1
    mask[-13:, -17:] = 1

    recon = _roundtrip(mask)

    assert torch.equal(recon, mask)
    assert iou(recon, mask) == 1.0


def test_anomaly_crossing_multiple_tiles():
    mask = torch.zeros(1200, 1600, dtype=torch.uint8)

    # Cross both x and y tile boundaries.
    mask[360:850, 350:1100] = 1

    recon = _roundtrip(mask)

    assert torch.equal(recon, mask)
    assert iou(recon, mask) == 1.0
