from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.contracts import (
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
)
from src.models.msila import MSILADay2Head


# ============================================================
# Day-2 contract used by the current contracts.py
# ============================================================

EXPECTED_KEYS = (
    "local_b4",
    "local_b8",
    "local_b12",
    "context_b4",
    "context_b8",
    "context_b12",
)

B = 2
D = 64
H = 16
W = 16


def make_features(
    *,
    batch_size: int = B,
    dim: int = D,
    h: int = H,
    w: int = W,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Create a valid six-source Day-2 feature dictionary."""
    return {
        name: torch.randn(
            batch_size,
            dim,
            h,
            w,
            dtype=dtype,
        )
        for name in MULTIVIEW_FEATURE_KEYS
    }


# ============================================================
# Test doubles: isolate Task 4 (Fusion -> Decoder wiring)
# ============================================================

class SpyFusion(nn.Module):
    """
    Minimal fusion with the SAME output contract as AttentionFusion v0:

        feature   : [B,d,h,w]
        attention : [B,6]
        logits    : [B,6]

    It uses a simple mean only so Task-4 tests do not depend on
    the internal implementation of AttentionFusion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_features: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        self.last_features = dict(features)

        stacked = torch.stack(
            [features[name] for name in MULTIVIEW_FEATURE_KEYS],
            dim=1,
        )
        fused = stacked.mean(dim=1)

        batch_size = fused.shape[0]

        attention = torch.full(
            (batch_size, len(MULTIVIEW_FEATURE_KEYS)),
            1.0 / len(MULTIVIEW_FEATURE_KEYS),
            dtype=fused.dtype,
            device=fused.device,
        )

        logits = torch.zeros_like(attention)

        return {
            "feature": fused,
            "attention": attention,
            "logits": logits,
        }


class SpyDecoder(nn.Module):
    """Record exactly what tensor reaches the decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:

        self.last_input = x

        # Produce one-channel dense logits, preserving differentiability.
        y = x.mean(dim=1, keepdim=True)

        return F.interpolate(
            y,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


# ============================================================
# 1. Contract synchronization
# ============================================================

def test_contract_has_exact_six_day02_sources() -> None:
    """
    Keep the test synchronized with the CURRENT contracts.py.
    """
    assert tuple(MULTIVIEW_FEATURE_KEYS) == EXPECTED_KEYS


# ============================================================
# 2. Core Task-4 gate: Fusion -> Decoder -> anomaly map
# ============================================================

def test_task4_fusion_to_decoder_wiring() -> None:
    """
    Scientific wiring test:

        6 valid features
            -> Fusion
            -> F_fused [B,d,h,w]
            -> Decoder
            -> [B,1,512,512]
    """
    fusion = SpyFusion()
    decoder = SpyDecoder()

    model = MSILADay2Head(
        fusion_dim=D,
        fusion=fusion,
        decoder=decoder,
        output_size=(512, 512),
        validate=True,
    )

    features = make_features()

    anomaly_logits, trace = model(
        features,
        return_trace=True,
    )

    assert anomaly_logits.shape == (B, 1, 512, 512)

    assert trace["fused"].shape == (B, D, H, W)
    assert trace["attention"].shape == (B, 6)
    assert trace["attention_logits"].shape == (B, 6)

    # Critical Task-4 assertion:
    # Decoder must receive F_fused, not one raw source feature.
    assert decoder.last_input is not None
    assert torch.equal(
        decoder.last_input,
        trace["fused"],
    )

    assert torch.isfinite(anomaly_logits).all()
    assert torch.isfinite(trace["fused"]).all()
    assert torch.isfinite(trace["attention"]).all()
    assert torch.isfinite(trace["attention_logits"]).all()


# ============================================================
# 3. Attention distribution contract propagated through MSILA
# ============================================================

def test_attention_debug_output_is_valid() -> None:
    fusion = SpyFusion()

    model = MSILADay2Head(
        fusion_dim=D,
        fusion=fusion,
        decoder=SpyDecoder(),
        validate=True,
    )

    _, trace = model(
        make_features(),
        return_trace=True,
    )

    attention = trace["attention"]

    assert attention.shape == (B, 6)
    assert (attention >= 0).all()
    assert torch.isfinite(attention).all()

    sums = attention.float().sum(dim=1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# 4. Current contracts.py: missing source must fail
# ============================================================

def test_missing_multiview_source_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        validate=True,
    )

    features = make_features()
    del features["context_b12"]

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 5. Current contracts.py: all six shapes must be identical
# ============================================================

def test_multiview_shape_mismatch_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        validate=True,
    )

    features = make_features()

    features["context_b8"] = torch.randn(
        B,
        D,
        H // 2,
        W // 2,
    )

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 6. Current contracts.py: projected channel width must equal d
# ============================================================

def test_wrong_fusion_dimension_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        validate=True,
    )

    # All six tensors agree with each other, but their channel
    # dimension does NOT match fusion_dim=D.
    features = make_features(dim=D // 2)

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 7. Current contracts.py: NaN/Inf must fail before fusion
# ============================================================

def test_nonfinite_feature_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        validate=True,
    )

    features = make_features()
    features["local_b4"][0, 0, 0, 0] = float("nan")

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 8. Current contracts.py: dtype must be consistent
# ============================================================

def test_dtype_mismatch_fails() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        validate=True,
    )

    features = make_features()
    features["context_b4"] = features["context_b4"].double()

    with pytest.raises(ContractError):
        model(features)


# ============================================================
# 9. Batch-size check
# ============================================================

@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_sizes(batch_size: int) -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        output_size=(512, 512),
        validate=True,
    )

    anomaly_logits, trace = model(
        make_features(batch_size=batch_size),
        return_trace=True,
    )

    assert anomaly_logits.shape == (
        batch_size,
        1,
        512,
        512,
    )

    assert trace["fused"].shape == (
        batch_size,
        D,
        H,
        W,
    )


# ============================================================
# 10. Optional output-size override
# ============================================================

def test_custom_output_size_is_supported() -> None:
    model = MSILADay2Head(
        fusion_dim=D,
        fusion=SpyFusion(),
        decoder=SpyDecoder(),
        output_size=(512, 512),
        validate=True,
    )

    anomaly_logits = model(
        make_features(batch_size=1),
        output_size=(256, 320),
    )

    assert anomaly_logits.shape == (
        1,
        1,
        256,
        320,
    )


# ============================================================
# 11. Real Task-2 -> Task-4 integration
# ============================================================

def test_real_attention_fusion_to_real_decoder() -> None:
    """
    Integration gate using the actual AttentionFusion + BasicDecoder
    created internally by MSILADay2Head.
    """
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=(512, 512),
        validate=True,
    )

    anomaly_logits, trace = model(
        make_features(),
        return_trace=True,
    )

    assert anomaly_logits.shape == (B, 1, 512, 512)
    assert trace["fused"].shape == (B, D, H, W)
    assert trace["attention"].shape == (B, 6)
    assert trace["attention_logits"].shape == (B, 6)

    sums = trace["attention"].float().sum(dim=1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
        rtol=1e-5,
    )

    assert torch.isfinite(anomaly_logits).all()
