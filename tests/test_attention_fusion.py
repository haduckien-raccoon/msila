# tests/test_attention_fusion.py

import pytest
import torch

from src.models.attention_fusion import AttentionFusion
from src.models.contracts import (
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
)


DIM = 64
B = 2
H = 16
W = 16


def make_features(
    batch_size=B,
    dim=DIM,
    h=H,
    w=W,
    requires_grad=False,
):
    return {
        name: torch.randn(
            batch_size,
            dim,
            h,
            w,
            requires_grad=requires_grad,
        )
        for name in MULTIVIEW_FEATURE_KEYS
    }


# ============================================================
# 1. Output shapes
# ============================================================

def test_attention_fusion_shapes():

    model = AttentionFusion(dim=DIM)

    features = make_features()

    out = model(features)

    assert out["feature"].shape == (
        B, DIM, H, W
    )

    assert out["attention"].shape == (
        B, 6
    )

    assert out["logits"].shape == (
        B, 6
    )


# ============================================================
# 2. Attention must be finite
# ============================================================

def test_attention_is_finite():

    model = AttentionFusion(dim=DIM)

    out = model(make_features())

    assert torch.isfinite(
        out["attention"]
    ).all()

    assert torch.isfinite(
        out["logits"]
    ).all()

    assert torch.isfinite(
        out["feature"]
    ).all()


# ============================================================
# 3. Softmax weights must sum to one
# ============================================================

def test_attention_sum_is_one():

    model = AttentionFusion(dim=DIM)

    out = model(make_features())

    sums = out["attention"].sum(dim=1)

    expected = torch.ones_like(sums)

    assert torch.allclose(
        sums,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# 4. Attention must be non-negative
# ============================================================

def test_attention_non_negative():

    model = AttentionFusion(dim=DIM)

    out = model(make_features())

    assert (
        out["attention"] >= 0
    ).all()


# ============================================================
# 5. Initial state == Mean Fusion
# ============================================================

def test_uniform_initialization_matches_mean_fusion():

    torch.manual_seed(0)

    model = AttentionFusion(dim=DIM)

    features = make_features()

    out = model(features)

    stacked = torch.stack(
        [
            features[name]
            for name in MULTIVIEW_FEATURE_KEYS
        ],
        dim=1,
    )

    expected_mean = stacked.mean(dim=1)

    # Initial attention must be exactly 1/6.
    expected_weights = torch.full_like(
        out["attention"],
        1.0 / 6.0,
    )

    assert torch.allclose(
        out["attention"],
        expected_weights,
        atol=1e-7,
        rtol=1e-7,
    )

    assert torch.allclose(
        out["feature"],
        expected_mean,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# 6. Verify weighted-sum mathematically
# ============================================================

def test_fused_feature_matches_manual_weighted_sum():

    model = AttentionFusion(dim=DIM)

    features = make_features()

    # Force non-uniform source priors.
    with torch.no_grad():
        model.source_bias.copy_(
            torch.tensor([
                0.0,
                0.5,
                1.0,
                -0.5,
                0.25,
                -1.0,
            ])
        )

    out = model(features)

    stacked = torch.stack(
        [
            features[name]
            for name in MULTIVIEW_FEATURE_KEYS
        ],
        dim=1,
    )

    weights = out["attention"][
        :,
        :,
        None,
        None,
        None,
    ]

    expected = (
        stacked * weights
    ).sum(dim=1)

    assert torch.allclose(
        out["feature"],
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# 7. Attention parameters must receive gradient
# ============================================================

def test_attention_parameters_receive_gradient():

    torch.manual_seed(42)

    model = AttentionFusion(dim=DIM)

    features = make_features(
        requires_grad=True
    )

    out = model(features)

    target = torch.randn_like(
        out["feature"]
    )

    loss = torch.nn.functional.mse_loss(
        out["feature"],
        target,
    )

    loss.backward()

    assert model.score_proj.weight.grad is not None
    assert model.source_bias.grad is not None

    assert torch.isfinite(
        model.score_proj.weight.grad
    ).all()

    assert torch.isfinite(
        model.source_bias.grad
    ).all()

    # At least some attention parameters must
    # actually receive a non-zero learning signal.
    total_grad = (
        model.score_proj.weight.grad.abs().sum()
        +
        model.source_bias.grad.abs().sum()
    )

    assert total_grad > 0


# ============================================================
# 8. Missing source must fail
# ============================================================

def test_missing_source_fails():

    model = AttentionFusion(dim=DIM)

    features = make_features()

    del features["context_b12"]

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 9. Wrong shape must fail
# ============================================================

def test_wrong_shape_fails():

    model = AttentionFusion(dim=DIM)

    features = make_features()

    features["context_b8"] = torch.randn(
        B,
        DIM,
        8,
        8,
    )

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 10. NaN must fail
# ============================================================

def test_nan_input_fails():

    model = AttentionFusion(dim=DIM)

    features = make_features()

    features["local_b4"][
        0, 0, 0, 0
    ] = float("nan")

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 11. Batch size 1 must work
# ============================================================

def test_batch_size_one():

    model = AttentionFusion(dim=DIM)

    features = make_features(
        batch_size=1
    )

    out = model(features)

    assert out["feature"].shape == (
        1, DIM, H, W
    )

    assert out["attention"].shape == (
        1, 6
    )


# ============================================================
# 12. Debug summary
# ============================================================

def test_attention_summary():

    model = AttentionFusion(dim=DIM)

    out = model(make_features())

    summary = model.attention_summary(
        out["attention"]
    )

    assert tuple(summary.keys()) == tuple(
        MULTIVIEW_FEATURE_KEYS
    )

    assert abs(
        sum(summary.values()) - 1.0
    ) < 1e-6