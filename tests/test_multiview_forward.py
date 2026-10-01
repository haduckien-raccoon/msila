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
# Controlled Day-2 tensor configuration
# ============================================================

D = 32
H = 16
W = 16
DEFAULT_OUTPUT = (512, 512)


def make_features(
    *,
    batch_size: int,
    dim: int = D,
    h: int = H,
    w: int = W,
) -> dict[str, Tensor]:
    """Create exactly the six tensors required by contracts.py."""
    return {
        name: torch.randn(
            batch_size,
            dim,
            h,
            w,
            dtype=torch.float32,
        )
        for name in MULTIVIEW_FEATURE_KEYS
    }


# ============================================================
# 1. Main Task-7 batch/shape gate
# ============================================================

@pytest.mark.parametrize(
    "batch_size",
    [1, 2],
)
def test_multiview_forward_shapes(
    batch_size: int,
) -> None:
    """
    Required Day-2 shape path:

        six sources: [B,d,h,w]
              ↓
        AttentionFusion
              ↓
        fused: [B,d,h,w]
              ↓
        Decoder
              ↓
        anomaly map: [B,1,512,512]
    """
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )
    model.eval()

    features = make_features(
        batch_size=batch_size,
    )

    with torch.no_grad():
        anomaly_logits, trace = model(
            features,
            return_trace=True,
        )

    # Six-source input contract
    assert set(features.keys()) == set(
        MULTIVIEW_FEATURE_KEYS
    )

    for name in MULTIVIEW_FEATURE_KEYS:
        assert features[name].shape == (
            batch_size,
            D,
            H,
            W,
        )

    # Attention Fusion output
    assert trace["fused"].shape == (
        batch_size,
        D,
        H,
        W,
    )

    # Debug-attention outputs
    assert trace["attention"].shape == (
        batch_size,
        6,
    )

    assert trace["attention_logits"].shape == (
        batch_size,
        6,
    )

    # Decoder output
    assert anomaly_logits.shape == (
        batch_size,
        1,
        512,
        512,
    )


# ============================================================
# 2. Batch dimension must be preserved exactly
# ============================================================

@pytest.mark.parametrize(
    "batch_size",
    [1, 2],
)
def test_batch_dimension_is_preserved(
    batch_size: int,
) -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )

    anomaly_logits, trace = model(
        make_features(
            batch_size=batch_size,
        ),
        return_trace=True,
    )

    assert anomaly_logits.shape[0] == batch_size
    assert trace["fused"].shape[0] == batch_size
    assert trace["attention"].shape[0] == batch_size
    assert trace["attention_logits"].shape[0] == batch_size


# ============================================================
# 3. Spatial feature size may vary, but all six must agree
# ============================================================

@pytest.mark.parametrize(
    "feature_hw",
    [
        (8, 8),
        (16, 16),
        (24, 20),
    ],
)
def test_feature_spatial_shapes_are_preserved_before_decoder(
    feature_hw: tuple[int, int],
) -> None:
    h, w = feature_hw

    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )

    anomaly_logits, trace = model(
        make_features(
            batch_size=1,
            h=h,
            w=w,
        ),
        return_trace=True,
    )

    # Fusion preserves the feature-grid size.
    assert trace["fused"].shape == (
        1,
        D,
        h,
        w,
    )

    # Decoder maps the fused feature back to the requested image size.
    assert anomaly_logits.shape == (
        1,
        1,
        512,
        512,
    )


# ============================================================
# 4. Custom decoder output size
# ============================================================

@pytest.mark.parametrize(
    "output_size",
    [
        (256, 256),
        (256, 320),
        (512, 512),
    ],
)
def test_output_size_override(
    output_size: tuple[int, int],
) -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )

    anomaly_logits = model(
        make_features(
            batch_size=1,
        ),
        output_size=output_size,
    )

    assert anomaly_logits.shape == (
        1,
        1,
        output_size[0],
        output_size[1],
    )


# ============================================================
# 5. One source with a different batch/shape must fail
# ============================================================

def test_mismatched_source_shape_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )

    features = make_features(
        batch_size=2,
    )

    # Break the current contracts.py requirement:
    # all six must have identical [B,d,h,w].
    features["context_b8"] = torch.randn(
        1,
        D,
        H,
        W,
    )

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 6. Wrong channel width d must fail
# ============================================================

def test_wrong_projected_channel_width_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )

    # All six tensors agree with one another,
    # but violate expected_channels=fusion_dim.
    features = make_features(
        batch_size=1,
        dim=D // 2,
    )

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 7. Missing one source must fail
# ============================================================

def test_missing_source_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=DEFAULT_OUTPUT,
        validate=True,
    )

    features = make_features(
        batch_size=1,
    )

    del features["context_b12"]

    with pytest.raises(ContractError):
        model(features)
