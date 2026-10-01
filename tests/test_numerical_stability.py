from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.models.contracts import (
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
)
from src.models.msila import MSILADay2Head


# ============================================================
# Controlled Day-2 synthetic configuration
# ============================================================

B = 2
D = 32
H = 16
W = 16
OUTPUT_SIZE = (512, 512)


def make_features(
    *,
    batch_size: int = B,
    dim: int = D,
    h: int = H,
    w: int = W,
    scale: float = 1.0,
) -> dict[str, Tensor]:
    """
    Create six valid aligned/projected Day-2 feature maps:

        local_b4, local_b8, local_b12,
        context_b4, context_b8, context_b12

    Every tensor has shape [B,d,h,w].
    """
    return {
        name: torch.randn(
            batch_size,
            dim,
            h,
            w,
            dtype=torch.float32,
        ) * scale
        for name in MULTIVIEW_FEATURE_KEYS
    }


def assert_finite(
    x: Tensor,
    *,
    name: str,
) -> None:
    """
    Numerical-stability gate:
        NaN count == 0
        Inf count == 0
    """
    nan_count = int(torch.isnan(x).sum().item())
    inf_count = int(torch.isinf(x).sum().item())

    assert nan_count == 0, (
        f"{name}: found {nan_count} NaN values"
    )

    assert inf_count == 0, (
        f"{name}: found {inf_count} Inf values"
    )


# ============================================================
# 1. Main numerical-stability gate
# ============================================================

@pytest.mark.parametrize(
    "scale",
    [
        1e-3,   # very small but finite features
        1.0,    # normal synthetic scale
        1e2,    # large but still realistic numerical stress
    ],
)
def test_day02_pipeline_has_no_nan_or_inf(
    scale: float,
) -> None:
    """
    Task-6 gate:

        6 valid features
            -> AttentionFusion
            -> F_fused
            -> Decoder
            -> anomaly logits

    Every observable tensor must remain finite.
    """
    torch.manual_seed(42)

    model = MSILADay2Head(
        fusion_dim=D,
        output_size=OUTPUT_SIZE,
        validate=True,
    )
    model.eval()

    features = make_features(
        scale=scale,
    )

    with torch.no_grad():
        anomaly_logits, trace = model(
            features,
            return_trace=True,
        )

    # Input gate
    for name, feature in features.items():
        assert_finite(
            feature,
            name=f"input[{name}]",
        )

    # Fusion/debug gate
    assert_finite(
        trace["fused"],
        name="fused",
    )
    assert_finite(
        trace["attention"],
        name="attention",
    )
    assert_finite(
        trace["attention_logits"],
        name="attention_logits",
    )

    # Decoder-output gate
    assert_finite(
        anomaly_logits,
        name="anomaly_logits",
    )

    # Attention is a valid probability distribution.
    attention = trace["attention"]

    assert (attention >= 0).all()

    sums = attention.float().sum(dim=1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# 2. Explicit NaN input rejection
# ============================================================

def test_nan_input_is_rejected_before_fusion() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=OUTPUT_SIZE,
        validate=True,
    )

    features = make_features()

    features["local_b4"][0, 0, 0, 0] = float("nan")

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 3. Explicit Inf input rejection
# ============================================================

def test_inf_input_is_rejected_before_fusion() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=OUTPUT_SIZE,
        validate=True,
    )

    features = make_features()

    features["context_b12"][0, 0, 0, 0] = float("inf")

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 4. Backward gradients must also remain finite
# ============================================================

def test_backward_gradients_are_finite() -> None:
    """
    Task 5 checks whether gradients exist.
    This Task-6 test checks the complementary property:
    once gradients exist, they must contain no NaN/Inf.
    """
    torch.manual_seed(7)

    model = MSILADay2Head(
        fusion_dim=D,
        output_size=(128, 128),
        validate=True,
    )
    model.train()

    features = {
        name: tensor.requires_grad_(True)
        for name, tensor in make_features(
            batch_size=1,
            scale=1.0,
        ).items()
    }

    anomaly_logits = model(features)

    target = torch.zeros_like(
        anomaly_logits
    )

    loss = torch.nn.functional.mse_loss(
        anomaly_logits,
        target,
    )

    assert_finite(
        loss,
        name="loss",
    )

    loss.backward()

    # Gradients reaching the six feature sources must be finite.
    for name, feature in features.items():
        assert feature.grad is not None, (
            f"{name}: gradient is None"
        )

        assert_finite(
            feature.grad,
            name=f"grad[{name}]",
        )

    # Trainable Fusion + Decoder parameter gradients must be finite.
    for module_name, module in (
        ("fusion", model.fusion),
        ("decoder", model.decoder),
    ):
        trainable_count = 0

        for param_name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue

            trainable_count += 1

            assert parameter.grad is not None, (
                f"{module_name}.{param_name}: gradient is None"
            )

            assert_finite(
                parameter.grad,
                name=f"{module_name}.{param_name}.grad",
            )

        assert trainable_count > 0, (
            f"{module_name}: no trainable parameters found"
        )
