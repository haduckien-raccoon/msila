from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_projection import SixFeatureProjection


def _six_inputs(
    *,
    b: int = 2,
    c: int = 8,
    h: int = 6,
    w: int = 7,
    dtype: torch.dtype = torch.float32,
):
    local = {
        "L4": torch.randn(b, c, h, w, dtype=dtype),
        "L8": torch.randn(b, c, h, w, dtype=dtype),
        "L12": torch.randn(b, c, h, w, dtype=dtype),
    }
    aligned = {
        "C4_to_L": torch.randn(b, c, h, w, dtype=dtype),
        "C8_to_L": torch.randn(b, c, h, w, dtype=dtype),
        "C12_to_L": torch.randn(b, c, h, w, dtype=dtype),
    }
    return local, aligned


def test_projection_returns_exact_six_common_shape_and_finite():
    torch.manual_seed(7)
    b, c, h, w, d = 2, 8, 6, 7, 5
    local, aligned = _six_inputs(b=b, c=c, h=h, w=w)

    projector = SixFeatureProjection(c, d, check_finite=True)
    out = projector(local, aligned)

    assert list(out) == [
        "local_b4",
        "local_b8",
        "local_b12",
        "context_b4",
        "context_b8",
        "context_b12",
    ]
    assert len(out) == 6
    for x in out.values():
        assert x.shape == (b, d, h, w)
        assert torch.isfinite(x).all()


def test_1x1_projection_matches_explicit_per_pixel_linear_map():
    """Conv1x1 must implement y[b,:,i,j] = W @ x[b,:,i,j] + b."""
    c, d = 3, 2
    local, aligned = _six_inputs(b=1, c=c, h=2, w=3)
    projector = SixFeatureProjection(c, d, share_across_views=True, bias=True)

    W = torch.tensor([[1.0, 2.0, -1.0], [-0.5, 0.25, 3.0]])
    bias = torch.tensor([0.2, -0.4])
    with torch.no_grad():
        projector.projectors["b4"].weight.copy_(W[:, :, None, None])
        projector.projectors["b4"].bias.copy_(bias)

    out = projector(local, aligned)
    x = local["L4"]
    expected = torch.einsum("oc,bchw->bohw", W, x) + bias.view(1, d, 1, 1)

    assert torch.allclose(out["local_b4"], expected, atol=1e-6, rtol=1e-6)


def test_shared_projection_uses_same_operator_for_same_block():
    b, c, h, w, d = 1, 4, 3, 3, 6
    same = torch.randn(b, c, h, w)
    local = {"L4": same, "L8": same, "L12": same}
    aligned = {
        "C4_to_L": same.clone(),
        "C8_to_L": same.clone(),
        "C12_to_L": same.clone(),
    }

    projector = SixFeatureProjection(c, d, share_across_views=True)
    out = projector(local, aligned)

    for block in (4, 8, 12):
        assert torch.equal(out[f"local_b{block}"], out[f"context_b{block}"])


def test_unshared_projection_has_independent_view_specific_operators():
    projector = SixFeatureProjection(4, 6, share_across_views=False)

    assert projector.projectors is None
    for block in (4, 8, 12):
        local_proj = projector.local_projectors[f"b{block}"]
        context_proj = projector.context_projectors[f"b{block}"]
        assert local_proj is not context_proj
        assert local_proj.weight.data_ptr() != context_proj.weight.data_ptr()


def test_projection_parameters_receive_finite_gradients():
    torch.manual_seed(1)
    local, aligned = _six_inputs(b=2, c=4, h=3, w=3)
    projector = SixFeatureProjection(4, 5, share_across_views=True)

    out = projector(local, aligned)
    loss = sum(x.square().mean() for x in out.values())
    loss.backward()

    for block in (4, 8, 12):
        layer = projector.projectors[f"b{block}"]
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert float(layer.weight.grad.abs().sum()) > 0.0
        if layer.bias is not None:
            assert layer.bias.grad is not None
            assert torch.isfinite(layer.bias.grad).all()


def test_projection_rejects_spatial_mismatch_before_fusion():
    local, aligned = _six_inputs(b=1, c=4, h=8, w=8)
    aligned["C4_to_L"] = torch.randn(1, 4, 7, 8)

    projector = SixFeatureProjection(4, 3)
    with pytest.raises(ValueError, match="share B/H/W"):
        projector(local, aligned)


def test_projection_rejects_wrong_channel_dimension():
    local, aligned = _six_inputs(b=1, c=4, h=5, w=5)
    local["L8"] = torch.randn(1, 5, 5, 5)

    projector = SixFeatureProjection(4, 3)
    with pytest.raises(ValueError, match="expected C=4"):
        projector(local, aligned)


def test_projection_rejects_nan_when_finite_check_enabled():
    local, aligned = _six_inputs(b=1, c=4, h=5, w=5)
    aligned["C12_to_L"][0, 0, 0, 0] = float("nan")

    projector = SixFeatureProjection(4, 3, check_finite=True)
    with pytest.raises(ValueError, match="NaN/Inf"):
        projector(local, aligned)


def test_projection_requires_all_six_sources():
    local, aligned = _six_inputs(b=1, c=4, h=5, w=5)
    del local["L12"]

    projector = SixFeatureProjection(4, 3)
    with pytest.raises(KeyError, match="L12"):
        projector(local, aligned)
