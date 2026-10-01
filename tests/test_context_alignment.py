from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.view_meta import build_view_meta
from src.models.context_alignment import ContextToLocalAligner


def _nested_meta():
    # Context 768x768, Local 512x512 centered inside it.
    # Both model inputs are 512x512.
    return build_view_meta(
        source_hw=(1024, 1024),
        context_box_xyxy=(128, 128, 896, 896),
        local_box_xyxy=(256, 256, 768, 768),
        local_input_hw=(512, 512),
        context_input_hw=(512, 512),
    )


def _normalized_feature_centers(h: int, w: int, *, dtype=torch.float32):
    y = 2.0 * (torch.arange(h, dtype=dtype) + 0.5) / h - 1.0
    x = 2.0 * (torch.arange(w, dtype=dtype) + 0.5) / w - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def _affine_field(h: int = 32, w: int = 32) -> torch.Tensor:
    """Bilinear sampling reproduces affine coordinate fields exactly."""
    xx, yy = _normalized_feature_centers(h, w)
    c0 = 0.70 * xx + 1.30 * yy + 0.10
    c1 = -0.25 * xx + 0.40 * yy - 0.20
    return torch.stack((c0, c1), dim=0).unsqueeze(0)  # [1,2,H,W]


def test_sampling_grid_matches_expected_nested_geometry():
    meta = _nested_meta()
    aligner = ContextToLocalAligner()
    dummy = torch.zeros(1, 1, 32, 32)

    grid = aligner.build_sampling_grid(dummy, meta)
    assert grid.shape == (1, 32, 32, 2)
    assert torch.isfinite(grid).all()

    # For centered 512-in-768 geometry, Local->Context input mapping is:
    # x_C = (2/3) x_L + 256/3. Same for y.
    # First Local feature-cell center is x_L = 8; last is 504.
    first_ctx = (2.0 / 3.0) * 8.0 + 256.0 / 3.0
    last_ctx = (2.0 / 3.0) * 504.0 + 256.0 / 3.0
    expected_first = 2.0 * first_ctx / 512.0 - 1.0
    expected_last = 2.0 * last_ctx / 512.0 - 1.0

    assert float(grid[0, 0, 0, 0]) == pytest.approx(expected_first, abs=1e-6)
    assert float(grid[0, 0, -1, 0]) == pytest.approx(expected_last, abs=1e-6)
    assert float(grid[0, 0, 0, 1]) == pytest.approx(expected_first, abs=1e-6)
    assert float(grid[0, -1, 0, 1]) == pytest.approx(expected_last, abs=1e-6)


def test_synthetic_affine_grid_alignment_error_under_tolerance_and_visualize():
    meta = _nested_meta()
    aligner = ContextToLocalAligner(check_finite=True)
    base = _affine_field()

    context = {
        "C4": base,
        "C8": base + 0.5,
        "C12": base - 0.25,
    }
    aligned = aligner(context, meta)
    grid = aligner.build_sampling_grid(base, meta)

    gx = grid[..., 0]
    gy = grid[..., 1]
    expected0 = 0.70 * gx + 1.30 * gy + 0.10
    expected1 = -0.25 * gx + 0.40 * gy - 0.20
    expected = torch.stack((expected0, expected1), dim=1)

    errors = {}
    offsets = {"C4_to_L": 0.0, "C8_to_L": 0.5, "C12_to_L": -0.25}
    for key, offset in offsets.items():
        err = (aligned[key] - (expected + offset)).abs()
        errors[key] = float(err.max())
        assert errors[key] < 2e-6

    # Required visual QA artifact: expected / aligned / absolute error.
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "context_alignment_synthetic.png"

    exp_img = expected[0, 0].detach().cpu().numpy()
    got_img = aligned["C4_to_L"][0, 0].detach().cpu().numpy()
    err_img = abs(got_img - exp_img)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(exp_img)
    axes[0].set_title("Expected")
    axes[1].imshow(got_img)
    axes[1].set_title("Aligned C4→L")
    axes[2].imshow(err_img)
    axes[2].set_title(f"Abs error\nmax={err_img.max():.2e}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_identity_geometry_reconstructs_feature_map():
    meta = build_view_meta(
        source_hw=(512, 512),
        context_box_xyxy=(0, 0, 512, 512),
        local_box_xyxy=(0, 0, 512, 512),
        local_input_hw=(512, 512),
        context_input_hw=(512, 512),
    )
    aligner = ContextToLocalAligner()
    x = torch.randn(2, 5, 32, 32)
    out = aligner({"C4": x, "C8": x, "C12": x}, meta)

    assert torch.allclose(out["C4_to_L"], x, atol=2e-6, rtol=1e-6)
    assert torch.allclose(out["C8_to_L"], x, atol=2e-6, rtol=1e-6)
    assert torch.allclose(out["C12_to_L"], x, atol=2e-6, rtol=1e-6)


def test_batched_geometry_mapping_supported():
    meta = _nested_meta().as_tensor_dict(dtype=torch.float32)
    batched_meta = {}
    for key, value in meta.items():
        if key in {"local_to_context"}:
            batched_meta[key] = torch.stack((value, value), dim=0)
        elif key in {"local_input_hw", "context_input_hw"}:
            batched_meta[key] = torch.stack((value, value), dim=0)
        else:
            batched_meta[key] = value

    x = _affine_field().repeat(2, 1, 1, 1)
    aligner = ContextToLocalAligner()
    out = aligner({"C4": x, "C8": x, "C12": x}, batched_meta)
    assert out["C4_to_L"].shape == (2, 2, 32, 32)
    assert torch.isfinite(out["C4_to_L"]).all()


def test_out_of_context_geometry_fails_fast():
    # Deliberately corrupt a valid metadata dictionary so Local maps outside Context.
    meta = _nested_meta().as_tensor_dict(dtype=torch.float32)
    bad = dict(meta)
    bad_matrix = bad["local_to_context"].clone()
    bad_matrix[0, 2] += 1000.0
    bad["local_to_context"] = bad_matrix

    aligner = ContextToLocalAligner(check_bounds=True)
    x = torch.zeros(1, 2, 32, 32)
    with pytest.raises(RuntimeError, match="outside Context FOV"):
        aligner({"C4": x, "C8": x, "C12": x}, bad)
