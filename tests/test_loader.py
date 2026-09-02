
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_rgb_native,
    load_mask_native,
    normalize_dinov3,
    DINOV3_MEAN,
    DINOV3_STD,
)


def test_grayscale_to_three_channels(tmp_path):
    arr = np.arange(35, dtype=np.uint8).reshape(5, 7)
    p = tmp_path / "gray.png"
    Image.fromarray(arr, mode="L").save(p)

    x = load_rgb_native(p)

    assert x.shape == (3, 5, 7)
    assert x.dtype == torch.float32
    assert torch.allclose(x[0], x[1])
    assert torch.allclose(x[1], x[2])
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0


def test_rgb_native_shape(tmp_path):
    arr = np.zeros((13, 17, 3), dtype=np.uint8)
    arr[..., 0] = 10
    arr[..., 1] = 20
    arr[..., 2] = 30

    p = tmp_path / "rgb.png"
    Image.fromarray(arr, mode="RGB").save(p)

    x = load_rgb_native(p)

    assert x.shape == (3, 13, 17)
    assert torch.isclose(x[0].mean(), torch.tensor(10/255.0), atol=1e-6)


def test_mask_nearest_neighbor_and_shape(tmp_path):
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:3, 2:5] = 255

    p = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(p)

    m = load_mask_native(p, target_hw=(10, 14))

    assert m.shape == (10, 14)
    assert set(torch.unique(m).tolist()).issubset({0, 1})


def test_dinov3_normalization():
    x = torch.zeros(3, 4, 5)

    y = normalize_dinov3(
        x,
        mean=DINOV3_MEAN,
        std=DINOV3_STD,
    )

    expected = -torch.tensor(DINOV3_MEAN) / torch.tensor(DINOV3_STD)

    assert y.shape == x.shape
    assert torch.allclose(
        y[:, 0, 0],
        expected,
        atol=1e-7,
    )
