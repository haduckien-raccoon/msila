from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.contracts import MULTIVIEW_FEATURE_KEYS
from src.models.msila import MSILADay2Head


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
) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(batch_size, dim, h, w)
        for name in MULTIVIEW_FEATURE_KEYS
    }


def test_day02_fusion_decoder_output_shape() -> None:
    """Task-4 gate: six features -> fusion -> [B,1,512,512]."""
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

    assert torch.isfinite(anomaly_logits).all()
    assert torch.isfinite(trace["fused"]).all()
    assert torch.isfinite(trace["attention"]).all()
    assert torch.isfinite(trace["attention_logits"]).all()


class SpyDecoder(nn.Module):
    """Minimal decoder used only to verify the wiring itself."""

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
        y = x.mean(dim=1, keepdim=True)
        return F.interpolate(
            y,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


def test_decoder_receives_exact_fused_feature() -> None:
    """Prove that F_fused, not a raw source, is what reaches the decoder."""
    decoder = SpyDecoder()
    model = MSILADay2Head(
        fusion_dim=D,
        decoder=decoder,
        output_size=(512, 512),
        validate=True,
    )

    anomaly_logits, trace = model(
        make_features(),
        return_trace=True,
    )

    assert decoder.last_input is not None
    assert torch.allclose(decoder.last_input, trace["fused"])
    assert anomaly_logits.shape == (B, 1, 512, 512)


def test_custom_output_size_is_supported() -> None:
    """Keep the head reusable while Day-2 default remains 512x512."""
    model = MSILADay2Head(
        fusion_dim=D,
        output_size=(512, 512),
        validate=True,
    )

    anomaly_logits = model(
        make_features(batch_size=1),
        output_size=(256, 320),
    )

    assert anomaly_logits.shape == (1, 1, 256, 320)
