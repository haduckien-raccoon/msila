
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tiling import (
    generate_tile_records,
    coverage_fraction_2d,
    extract_local_context,
    stitch_tiles_hann,
)


def test_coverage_multiple_resolutions():
    sizes = [
        (320, 480),
        (512, 512),
        (513, 777),
        (768, 1024),
        (1200, 1600),
        (2048, 2448),
    ]

    for H, W in sizes:
        cov = coverage_fraction_2d(
            H,
            W,
            tile_size=512,
            overlap=128,
        )
        assert cov == 1.0


def test_local_context_shapes_and_coordinates():
    H, W = 900, 1300
    image = torch.rand(3, H, W)

    records = generate_tile_records(
        H,
        W,
        local_size=512,
        overlap=128,
        context_size=768,
    )

    assert len(records) > 0

    for rec in records[:5]:
        local, context = extract_local_context(
            image,
            rec,
            local_out_size=512,
            context_out_size=512,
        )
        assert local.shape == (3, 512, 512)
        assert context.shape == (3, 512, 512)


def test_hann_reconstruction_error():
    sizes = [
        (320, 480),
        (513, 777),
        (768, 1024),
        (1025, 1537),
    ]

    for H, W in sizes:
        # Smooth synthetic field + random component.
        torch.manual_seed(0)
        source = torch.rand(H, W)

        records = generate_tile_records(
            H,
            W,
            local_size=512,
            overlap=128,
            context_size=768,
        )

        tile_maps = []
        padded = source[None]

        from src.data.tiling import crop_with_padding

        for rec in records:
            tile = crop_with_padding(
                padded,
                rec.local_xyxy,
                pad_mode="constant",
                pad_value=0.0,
            )[0]
            tile_maps.append(tile)

        recon = stitch_tiles_hann(
            tile_maps,
            records,
            out_hw=(H, W),
            local_size=512,
        )

        err = torch.max(torch.abs(recon - source)).item()
        assert err < 1e-5, (H, W, err)
