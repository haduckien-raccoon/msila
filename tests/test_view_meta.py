"""Unit tests for geometry/view_meta.py.

Run from project root:
    pytest -q tests/test_view_meta.py
"""

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.view_meta import (  # noqa: E402
    build_view_meta,
    build_view_meta_from_transform_meta,
    crop_resize_matrix,
    transform_box_xyxy,
    transform_points_xy,
    validate_view_meta,
)


SOURCE_HW = (1000, 1200)
CONTEXT_BOX = (200, 100, 968, 868)  # 768x768
LOCAL_BOX = (328, 228, 840, 740)    # centered 512x512


def make_meta():
    return build_view_meta(
        source_hw=SOURCE_HW,
        local_box_xyxy=LOCAL_BOX,
        context_box_xyxy=CONTEXT_BOX,
        local_input_hw=(512, 512),
        context_input_hw=(512, 512),
    )


def test_locked_crop_boxes_scales_and_containment() -> None:
    meta = make_meta()

    assert meta.local_box_xyxy == LOCAL_BOX
    assert meta.context_box_xyxy == CONTEXT_BOX
    assert torch.allclose(meta.local_scale_xy, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(
        meta.context_scale_xy,
        torch.tensor([2.0 / 3.0, 2.0 / 3.0], dtype=torch.float64),
    )
    assert torch.allclose(
        meta.local_box_in_context_source_xyxy,
        torch.tensor([128.0, 128.0, 640.0, 640.0], dtype=torch.float64),
    )


def test_expected_source_to_local_matrix() -> None:
    meta = make_meta()
    expected = torch.tensor(
        [
            [1.0, 0.0, -328.0],
            [0.0, 1.0, -228.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(meta.source_to_local, expected, atol=1e-12, rtol=0)


def test_expected_source_to_context_matrix() -> None:
    meta = make_meta()
    s = 2.0 / 3.0
    expected = torch.tensor(
        [
            [s, 0.0, -s * 200.0],
            [0.0, s, -s * 100.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(meta.source_to_context, expected, atol=1e-12, rtol=0)


def test_local_to_context_mapping_and_box_are_correct() -> None:
    meta = make_meta()
    s = 2.0 / 3.0
    offset = 128.0 * s
    expected_l2c = torch.tensor(
        [
            [s, 0.0, offset],
            [0.0, s, offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(meta.local_to_context, expected_l2c, atol=1e-12, rtol=0)

    expected_box = torch.tensor(
        [offset, offset, 640.0 * s, 640.0 * s],
        dtype=torch.float64,
    )
    assert torch.allclose(
        meta.local_box_in_context_input_xyxy,
        expected_box,
        atol=1e-12,
        rtol=0,
    )


def test_point_round_trip_local_context_source() -> None:
    meta = make_meta()
    pts_local = torch.tensor(
        [[0.0, 0.0], [256.0, 256.0], [512.0, 512.0]], dtype=torch.float64
    )

    pts_context = transform_points_xy(pts_local, meta.local_to_context)
    back_local = transform_points_xy(pts_context, meta.context_to_local)
    assert torch.allclose(back_local, pts_local, atol=1e-10, rtol=0)

    pts_source = transform_points_xy(pts_local, meta.local_to_source)
    expected_source = torch.tensor(
        [[328.0, 228.0], [584.0, 484.0], [840.0, 740.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(pts_source, expected_source, atol=1e-10, rtol=0)


def test_crop_resize_matrix_maps_box_edges_exactly() -> None:
    m = crop_resize_matrix(CONTEXT_BOX, (512, 512))
    mapped = transform_box_xyxy(CONTEXT_BOX, m)
    assert torch.allclose(
        mapped,
        torch.tensor([0.0, 0.0, 512.0, 512.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_adapter_accepts_multiview_transform_style_meta() -> None:
    transform_meta = {
        "source_hw": torch.tensor(SOURCE_HW, dtype=torch.int64),
        "context_box_xyxy": torch.tensor(CONTEXT_BOX, dtype=torch.int64),
        "local_box_xyxy": torch.tensor(LOCAL_BOX, dtype=torch.int64),
        # Extra keys from data/multiview_transform.py are intentionally allowed.
        "context_to_input_scale": torch.tensor(2.0 / 3.0),
    }
    meta = build_view_meta_from_transform_meta(transform_meta)
    assert meta.source_hw == SOURCE_HW
    assert meta.local_box_xyxy == LOCAL_BOX
    assert meta.context_box_xyxy == CONTEXT_BOX


def test_tensor_dict_batches_cleanly() -> None:
    a = make_meta().as_tensor_dict()
    b = make_meta().as_tensor_dict()
    batch = default_collate([a, b])

    assert batch["source_to_local"].shape == (2, 3, 3)
    assert batch["local_to_context"].shape == (2, 3, 3)
    assert batch["local_box_xyxy"].shape == (2, 4)
    assert batch["local_scale_xy"].shape == (2, 2)
    assert batch["source_to_local"].dtype == torch.float64


def test_mapping_is_deterministic() -> None:
    a = make_meta()
    b = make_meta()

    assert a.source_hw == b.source_hw
    assert a.local_box_xyxy == b.local_box_xyxy
    assert a.context_box_xyxy == b.context_box_xyxy
    assert a.local_input_hw == b.local_input_hw
    assert a.context_input_hw == b.context_input_hw
    for name in (
        "local_scale_xy",
        "context_scale_xy",
        "source_to_local",
        "local_to_source",
        "source_to_context",
        "context_to_source",
        "local_to_context",
        "context_to_local",
        "local_box_in_context_source_xyxy",
        "local_box_in_context_input_xyxy",
    ):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_validator_accepts_valid_meta() -> None:
    validate_view_meta(make_meta())


def test_local_outside_context_fails() -> None:
    with pytest.raises(ValueError, match="inside Context"):
        build_view_meta(
            source_hw=SOURCE_HW,
            local_box_xyxy=(0, 0, 512, 512),
            context_box_xyxy=CONTEXT_BOX,
        )


def test_out_of_source_box_fails() -> None:
    with pytest.raises(ValueError, match="outside source"):
        build_view_meta(
            source_hw=(700, 700),
            local_box_xyxy=(100, 100, 612, 612),
            context_box_xyxy=(0, 0, 768, 768),
        )


def test_missing_transform_meta_key_fails() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        build_view_meta_from_transform_meta({"source_hw": SOURCE_HW})
