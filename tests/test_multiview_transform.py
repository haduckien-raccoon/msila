"""Unit tests for data/multiview_transform.py.

Run from project root:
    pytest -q tests/test_multiview_transform.py
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multiview_transform import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    MultiViewConfig,
    NestedMultiViewTransform,
    validate_multiview_sample,
)


def make_coordinate_image(h: int = 1000, w: int = 1200) -> torch.Tensor:
    """Synthetic RGB image whose pixel values depend on source coordinates."""
    yy = torch.arange(h, dtype=torch.int32).view(h, 1)
    xx = torch.arange(w, dtype=torch.int32).view(1, w)
    return torch.stack(
        [
            ((xx + yy) % 256).to(torch.uint8),
            ((2 * xx + yy) % 256).to(torch.uint8),
            ((xx + 2 * yy) % 256).to(torch.uint8),
        ]
    )


def test_required_shapes_and_finite() -> None:
    tf = NestedMultiViewTransform(MultiViewConfig(sampling="center"))
    out = tf(make_coordinate_image())

    assert out["x_local"].shape == (3, 512, 512)
    assert out["x_context"].shape == (3, 512, 512)
    assert torch.isfinite(out["x_local"]).all()
    assert torch.isfinite(out["x_context"]).all()


def test_local_is_exact_source_crop_and_centered_inside_context() -> None:
    image = make_coordinate_image()
    tf = NestedMultiViewTransform(
        MultiViewConfig(sampling="center", normalize=False)
    )
    out = tf(image)

    c = out["meta"]["context_box_xyxy"].tolist()
    l = out["meta"]["local_box_xyxy"].tolist()
    rel = out["meta"]["local_box_in_context_xyxy"].tolist()

    assert c[0] <= l[0] < l[2] <= c[2]
    assert c[1] <= l[1] < l[3] <= c[3]
    assert rel == [128, 128, 640, 640]

    src = tf._to_chw_float01(image)
    expected_local = src[:, l[1] : l[3], l[0] : l[2]]
    assert torch.equal(out["x_local"], expected_local)


def test_context_source_fov_and_resize_scale_metadata() -> None:
    tf = NestedMultiViewTransform(
        MultiViewConfig(sampling="center", normalize=False)
    )
    out = tf(make_coordinate_image())

    c = out["meta"]["context_box_xyxy"]
    assert int(c[2] - c[0]) == 768
    assert int(c[3] - c[1]) == 768

    scale = out["meta"]["context_to_input_scale"]
    assert torch.isclose(scale, torch.tensor(2.0 / 3.0))

    expected = torch.tensor([128, 128, 640, 640], dtype=torch.float32) * (2.0 / 3.0)
    assert torch.allclose(
        out["meta"]["local_box_in_context_input_xyxy"], expected
    )


def test_batch_contract_is_B3HW() -> None:
    tf = NestedMultiViewTransform(MultiViewConfig(sampling="center"))
    batch = default_collate(
        [tf(make_coordinate_image()), tf(make_coordinate_image())]
    )

    assert batch["x_local"].shape == (2, 3, 512, 512)
    assert batch["x_context"].shape == (2, 3, 512, 512)


def test_random_sampling_is_reproducible_with_torch_seed() -> None:
    image = make_coordinate_image()
    tf = NestedMultiViewTransform(
        MultiViewConfig(sampling="random", normalize=False)
    )

    torch.manual_seed(20260930)
    a = tf(image)["meta"]["context_box_xyxy"]
    torch.manual_seed(20260930)
    b = tf(image)["meta"]["context_box_xyxy"]

    assert torch.equal(a, b)


def test_supported_input_types() -> None:
    arr = np.zeros((800, 900, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    chw = torch.from_numpy(arr).permute(2, 0, 1)

    tf = NestedMultiViewTransform(
        MultiViewConfig(sampling="center", normalize=False)
    )

    for image in (arr, pil, chw):
        out = tf(image)
        assert out["x_local"].shape == (3, 512, 512)
        assert out["x_context"].shape == (3, 512, 512)


def test_grayscale_and_rgba_are_converted_to_rgb() -> None:
    tf = NestedMultiViewTransform(
        MultiViewConfig(sampling="center", normalize=False)
    )

    gray = np.zeros((800, 800), dtype=np.uint8)
    rgba = np.zeros((800, 800, 4), dtype=np.uint8)

    assert tf(gray)["x_local"].shape[0] == 3
    assert tf(rgba)["x_local"].shape[0] == 3


def test_normalization_formula_matches_imagenet_constants() -> None:
    image = torch.zeros((3, 800, 800), dtype=torch.float32)
    tf = NestedMultiViewTransform(MultiViewConfig(sampling="center", normalize=True))
    out = tf(image)

    expected = torch.tensor(
        [-IMAGENET_MEAN[i] / IMAGENET_STD[i] for i in range(3)],
        dtype=torch.float32,
    )
    observed = out["x_local"][:, 0, 0]

    assert torch.allclose(observed, expected, atol=1e-6, rtol=0.0)


def test_validator_accepts_valid_sample() -> None:
    cfg = MultiViewConfig(sampling="center")
    tf = NestedMultiViewTransform(cfg)
    out = tf(make_coordinate_image())
    validate_multiview_sample(out, cfg)


def test_small_image_fails_fast() -> None:
    tf = NestedMultiViewTransform(MultiViewConfig(sampling="center"))
    with pytest.raises(ValueError, match="too small"):
        tf(torch.zeros((3, 700, 700), dtype=torch.uint8))


def test_invalid_float_range_fails() -> None:
    tf = NestedMultiViewTransform(MultiViewConfig(sampling="center"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tf(torch.full((3, 800, 800), 2.0, dtype=torch.float32))


def test_invalid_nested_geometry_config_fails() -> None:
    with pytest.raises(ValueError, match="even"):
        MultiViewConfig(local_size=511, context_size=768)
