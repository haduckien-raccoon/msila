from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.contracts import (
    DINO_FEATURE_KEYS,
    MULTIVIEW_FEATURE_KEYS,
)
from src.models.msila import MSILADay2Integrated
from src.models.residual_adapter import ResidualAdapter2d


B = 2
IMAGE_HW = 64
DINO_C = 16
FUSION_D = 8
FEATURE_HW = 8
OUTPUT_SIZE = (64, 64)


class FrozenDinoForIntegration(nn.Module):
    """Small frozen backbone used only to verify the TV1->TV2 boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.out_channels = DINO_C

        self.blocks = nn.ModuleDict({
            key: nn.Conv2d(
                3,
                DINO_C,
                kernel_size=3,
                padding=1,
            )
            for key in DINO_FEATURE_KEYS
        })

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:
        return {
            key: F.adaptive_avg_pool2d(
                self.blocks[key](x),
                (FEATURE_HW, FEATURE_HW),
            )
            for key in DINO_FEATURE_KEYS
        }


class TV1ContractPipeline(nn.Module):
    """Contract-faithful TV1 reference pipeline for integration QA.

    It exercises the same hand-off expected from the real TV1 code:

        Local/Context -> frozen backbone -> Adapter -> Projection
                      -> six [B,d,h,w] tensors

    Geometric alignment itself is not re-tested here; TV1 owns that test.
    The purpose here is to prove that real trainable upstream tensors can
    flow into MSILADay2Integrated without mocks/detach at the TV2 boundary.
    """

    def __init__(self) -> None:
        super().__init__()

        self.dino = FrozenDinoForIntegration()

        self.adapters = nn.ModuleDict({
            key: ResidualAdapter2d(
                in_channels=DINO_C,
                reduction=4,
                kernel_size=3,
                gamma_init=0.0,
            )
            for key in DINO_FEATURE_KEYS
        })

        self.projections = nn.ModuleDict({
            name: nn.Conv2d(
                DINO_C,
                FUSION_D,
                kernel_size=1,
            )
            for name in MULTIVIEW_FEATURE_KEYS
        })

    def _branch(
        self,
        image: Tensor,
        prefix: str,
    ) -> dict[str, Tensor]:
        raw = self.dino(image)

        adapted = {
            key: self.adapters[key](raw[key])
            for key in DINO_FEATURE_KEYS
        }

        return {
            f"{prefix}_{key}":
                self.projections[f"{prefix}_{key}"](
                    adapted[key]
                )
            for key in DINO_FEATURE_KEYS
        }

    def forward(
        self,
        local_image: Tensor,
        context_image: Tensor,
    ) -> tuple[
        dict[str, Tensor],
        dict[str, object],
    ]:
        local = self._branch(
            local_image,
            "local",
        )
        context = self._branch(
            context_image,
            "context",
        )

        features = {
            **local,
            **context,
        }

        return features, {
            "source": "tv1_contract_pipeline",
        }


def _assert_module_has_finite_nonzero_grad(
    module: nn.Module,
    name: str,
) -> None:
    params = [
        p
        for p in module.parameters()
        if p.requires_grad
    ]

    assert params, f"{name}: no trainable parameters"

    grads = [
        p.grad
        for p in params
    ]

    assert all(
        g is not None
        for g in grads
    ), f"{name}: at least one trainable parameter has grad=None"

    assert all(
        torch.isfinite(g).all()
        for g in grads
        if g is not None
    ), f"{name}: NaN/Inf gradient"

    assert any(
        float(g.detach().abs().sum()) > 0.0
        for g in grads
        if g is not None
    ), f"{name}: all gradients are zero"


def test_tv1_to_tv2_full_forward_backward() -> None:
    """Task-8 gate: integrated forward + backward must both pass."""

    torch.manual_seed(42)

    tv1 = TV1ContractPipeline()

    model = MSILADay2Integrated(
        feature_pipeline=tv1,
        fusion_dim=FUSION_D,
        output_size=OUTPUT_SIZE,
        validate=True,
    )
    model.train()

    local_image = torch.randn(
        B,
        3,
        IMAGE_HW,
        IMAGE_HW,
    )
    context_image = (
        torch.randn(
            B,
            3,
            IMAGE_HW,
            IMAGE_HW,
        ) * 0.8 + 0.1
    )

    anomaly_logits, trace = model(
        local_image,
        context_image,
        return_trace=True,
    )

    # ----------------------------
    # Full forward gate
    # ----------------------------
    assert anomaly_logits.shape == (
        B,
        1,
        OUTPUT_SIZE[0],
        OUTPUT_SIZE[1],
    )

    features = trace["multiview_features"]

    assert set(features.keys()) == set(
        MULTIVIEW_FEATURE_KEYS
    )

    for name in MULTIVIEW_FEATURE_KEYS:
        assert features[name].shape == (
            B,
            FUSION_D,
            FEATURE_HW,
            FEATURE_HW,
        )
        assert torch.isfinite(
            features[name]
        ).all()

    assert trace["fused"].shape == (
        B,
        FUSION_D,
        FEATURE_HW,
        FEATURE_HW,
    )
    assert trace["attention"].shape == (
        B,
        6,
    )
    assert trace["attention_logits"].shape == (
        B,
        6,
    )

    assert torch.isfinite(
        anomaly_logits
    ).all()

    # ----------------------------
    # Full backward gate
    # ----------------------------
    target = torch.ones_like(
        anomaly_logits
    )

    loss = F.mse_loss(
        anomaly_logits,
        target,
    )

    loss.backward()

    # Frozen DINO: no gradients by construction.
    for name, p in tv1.dino.named_parameters():
        assert not p.requires_grad, (
            f"DINO.{name}: expected frozen parameter"
        )
        assert p.grad is None, (
            f"DINO.{name}: frozen parameter received gradient"
        )

    # Upstream and downstream trainable components stay connected.
    _assert_module_has_finite_nonzero_grad(
        tv1.adapters,
        "Adapter",
    )
    _assert_module_has_finite_nonzero_grad(
        tv1.projections,
        "Projection",
    )
    _assert_module_has_finite_nonzero_grad(
        model.head.fusion,
        "Fusion",
    )
    _assert_module_has_finite_nonzero_grad(
        model.head.decoder,
        "Decoder",
    )


def test_integration_accepts_direct_six_feature_mapping() -> None:
    """The wrapper also accepts a TV1 module returning only the six features."""

    class DirectTV1(nn.Module):
        def forward(
            self,
            batch_size: int,
        ) -> dict[str, Tensor]:
            return {
                name: torch.randn(
                    batch_size,
                    FUSION_D,
                    FEATURE_HW,
                    FEATURE_HW,
                )
                for name in MULTIVIEW_FEATURE_KEYS
            }

    model = MSILADay2Integrated(
        feature_pipeline=DirectTV1(),
        fusion_dim=FUSION_D,
        output_size=OUTPUT_SIZE,
        validate=True,
    )

    out = model(B)

    assert out.shape == (
        B,
        1,
        OUTPUT_SIZE[0],
        OUTPUT_SIZE[1],
    )
