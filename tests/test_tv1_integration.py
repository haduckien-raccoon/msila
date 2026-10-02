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


# ---------------------------------------------------------------------------
# Controlled integration-test configuration.
#
# IMPORTANT:
#   Adapter d and downstream fusion_dim are intentionally different.
#   This prevents the test from silently collapsing two distinct Day-04
#   quantities:
#
#       adapter_projection_dim != fusion_dim
#
#   Adapter:
#       C -> r -> DWConv -> d -> C
#
#   TV1 projection / TV2 fusion:
#       C -> fusion_dim
# ---------------------------------------------------------------------------
B = 2
IMAGE_HW = 64
DINO_C = 16

ADAPTER_R = 4
ADAPTER_D = 6
ADAPTER_KERNEL_SIZE = 3
ADAPTER_GAMMA_INIT = 0.0
ADAPTER_BIAS = True

FUSION_DIM = 8

FEATURE_HW = 8
OUTPUT_SIZE = (64, 64)


class FrozenDinoForIntegration(nn.Module):
    """Small frozen backbone used only to verify the TV1 -> TV2 boundary.

    This is a controlled software-integration fixture, not a substitute for the
    real DINOv3 backbone and not evidence of anomaly-detection performance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out_channels = DINO_C

        self.blocks = nn.ModuleDict(
            {
                key: nn.Conv2d(
                    3,
                    DINO_C,
                    kernel_size=3,
                    padding=1,
                )
                for key in DINO_FEATURE_KEYS
            }
        )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

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
    """Contract-faithful TV1 reference pipeline for TV1 -> TV2 integration QA.

    This fixture verifies the hand-off expected by ``MSILADay2Integrated``:

        Local / Context
            -> frozen backbone
            -> Day-04 Adapter: C -> r -> DWConv -> d -> C
            -> projection: C -> fusion_dim
            -> six tensors [B, fusion_dim, h, w]

    One Adapter is shared between Local and Context for the same DINO block,
    matching the cached-training contract.

    Geometric alignment is intentionally not re-tested here.  Alignment has its
    own tests; this file verifies tensor contract and gradient connectivity at
    the TV1 -> TV2 boundary.
    """

    def __init__(
        self,
        *,
        adapter_bottleneck_dim: int = ADAPTER_R,
        adapter_projection_dim: int = ADAPTER_D,
        fusion_dim: int = FUSION_DIM,
    ) -> None:
        super().__init__()

        for name, value in (
            ("adapter_bottleneck_dim", adapter_bottleneck_dim),
            ("adapter_projection_dim", adapter_projection_dim),
            ("fusion_dim", fusion_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self.adapter_bottleneck_dim = int(adapter_bottleneck_dim)
        self.adapter_projection_dim = int(adapter_projection_dim)
        self.fusion_dim = int(fusion_dim)

        self.dino = FrozenDinoForIntegration()

        self.adapters = nn.ModuleDict(
            {
                key: ResidualAdapter2d(
                    in_dim=DINO_C,
                    bottleneck_dim=self.adapter_bottleneck_dim,
                    projection_dim=self.adapter_projection_dim,
                    kernel_size=ADAPTER_KERNEL_SIZE,
                    gamma_init=ADAPTER_GAMMA_INIT,
                    bias=ADAPTER_BIAS,
                )
                for key in DINO_FEATURE_KEYS
            }
        )

        self.projections = nn.ModuleDict(
            {
                name: nn.Conv2d(
                    DINO_C,
                    self.fusion_dim,
                    kernel_size=1,
                )
                for name in MULTIVIEW_FEATURE_KEYS
            }
        )

    def _branch(
        self,
        image: Tensor,
        prefix: str,
    ) -> dict[str, Tensor]:
        if prefix not in {"local", "context"}:
            raise ValueError(
                f"prefix must be 'local' or 'context', got {prefix!r}"
            )

        raw = self.dino(image)

        adapted: dict[str, Tensor] = {}
        for key in DINO_FEATURE_KEYS:
            source = raw[key]
            output = self.adapters[key](source)

            # ResidualAdapter2d must preserve the backbone feature lattice.
            assert output.shape == source.shape
            assert torch.isfinite(output).all()

            adapted[key] = output

        projected: dict[str, Tensor] = {}
        for key in DINO_FEATURE_KEYS:
            output_name = f"{prefix}_{key}"
            value = self.projections[output_name](adapted[key])

            assert value.shape == (
                int(image.shape[0]),
                self.fusion_dim,
                FEATURE_HW,
                FEATURE_HW,
            )
            assert torch.isfinite(value).all()

            projected[output_name] = value

        return projected

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
            "adapter_config": {
                "r": self.adapter_bottleneck_dim,
                "d": self.adapter_projection_dim,
            },
            "fusion_dim": self.fusion_dim,
        }


def _assert_day04_adapter_contract(
    tv1: TV1ContractPipeline,
    *,
    expected_r: int,
    expected_d: int,
) -> None:
    """Verify the exact Day-04 Adapter semantics used by the TV1 fixture."""

    assert set(tv1.adapters.keys()) == set(DINO_FEATURE_KEYS)

    for key in DINO_FEATURE_KEYS:
        adapter = tv1.adapters[key]

        assert adapter.in_dim == DINO_C
        assert adapter.r == expected_r
        assert adapter.d == expected_d

        # C -> r
        assert adapter.down_proj.in_channels == DINO_C
        assert adapter.down_proj.out_channels == expected_r

        # r -> r depthwise
        assert adapter.dwconv.in_channels == expected_r
        assert adapter.dwconv.out_channels == expected_r
        assert adapter.dwconv.groups == expected_r

        # r -> d
        assert adapter.mid_proj.in_channels == expected_r
        assert adapter.mid_proj.out_channels == expected_d

        # d -> C
        assert adapter.out_proj.in_channels == expected_d
        assert adapter.out_proj.out_channels == DINO_C

        # Closed-form count implemented by ResidualAdapter2d must agree with
        # the actual trainable parameter count for the test candidate.
        assert (
            adapter.num_trainable_parameters
            == adapter.expected_parameter_count()
        )


def _assert_module_has_finite_nonzero_grad(
    module: nn.Module,
    name: str,
) -> None:
    """Require a finite connected gradient path through a trainable module.

    For a zero-initialized residual gate, branch-weight gradients are allowed to
    be zero at the first backward pass because they are multiplied by gamma=0.
    Therefore the scientifically correct first-step gate is:

      1. every trainable parameter participating in the graph has a gradient;
      2. all gradients are finite;
      3. at least one gradient in the logical module is non-zero.

    For the Adapter, gamma provides the initial trainable path.
    """

    params = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    assert params, f"{name}: no trainable parameters"

    grads = [
        parameter.grad
        for parameter in params
    ]

    assert all(
        grad is not None
        for grad in grads
    ), f"{name}: at least one trainable parameter has grad=None"

    assert all(
        torch.isfinite(grad).all()
        for grad in grads
        if grad is not None
    ), f"{name}: NaN/Inf gradient"

    assert any(
        float(grad.detach().abs().sum()) > 0.0
        for grad in grads
        if grad is not None
    ), f"{name}: all gradients are zero"


def test_tv1_uses_explicit_day04_r_d_contract() -> None:
    """Regression gate for migration from legacy ``reduction`` to explicit r,d."""

    assert ADAPTER_D != FUSION_DIM, (
        "Test configuration must keep Adapter d distinct from fusion_dim"
    )

    tv1 = TV1ContractPipeline(
        adapter_bottleneck_dim=ADAPTER_R,
        adapter_projection_dim=ADAPTER_D,
        fusion_dim=FUSION_DIM,
    )

    _assert_day04_adapter_contract(
        tv1,
        expected_r=ADAPTER_R,
        expected_d=ADAPTER_D,
    )

    for name in MULTIVIEW_FEATURE_KEYS:
        projection = tv1.projections[name]
        assert projection.in_channels == DINO_C
        assert projection.out_channels == FUSION_DIM


def test_tv1_to_tv2_full_forward_backward() -> None:
    """Integrated TV1 -> TV2 forward/backward gate.

    This proves software-contract compatibility and gradient connectivity.
    It does not evaluate anomaly-localization quality.
    """

    torch.manual_seed(42)

    tv1 = TV1ContractPipeline(
        adapter_bottleneck_dim=ADAPTER_R,
        adapter_projection_dim=ADAPTER_D,
        fusion_dim=FUSION_DIM,
    )

    _assert_day04_adapter_contract(
        tv1,
        expected_r=ADAPTER_R,
        expected_d=ADAPTER_D,
    )

    model = MSILADay2Integrated(
        feature_pipeline=tv1,
        fusion_dim=FUSION_DIM,
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
        )
        * 0.8
        + 0.1
    )

    anomaly_logits, trace = model(
        local_image,
        context_image,
        return_trace=True,
    )

    # ------------------------------------------------------------------
    # Forward contract
    # ------------------------------------------------------------------
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
            FUSION_DIM,
            FEATURE_HW,
            FEATURE_HW,
        )
        assert torch.isfinite(features[name]).all()

    assert trace["fused"].shape == (
        B,
        FUSION_DIM,
        FEATURE_HW,
        FEATURE_HW,
    )
    assert trace["attention"].shape == (
        B,
        len(MULTIVIEW_FEATURE_KEYS),
    )
    assert trace["attention_logits"].shape == (
        B,
        len(MULTIVIEW_FEATURE_KEYS),
    )

    assert torch.isfinite(trace["fused"]).all()
    assert torch.isfinite(trace["attention"]).all()
    assert torch.isfinite(trace["attention_logits"]).all()
    assert torch.isfinite(anomaly_logits).all()

    # The TV1 trace must preserve the architecture provenance.
    assert trace["tv1"]["adapter_config"] == {
        "r": ADAPTER_R,
        "d": ADAPTER_D,
    }
    assert trace["tv1"]["fusion_dim"] == FUSION_DIM

    # AttentionFusion outputs a normalized distribution over six sources.
    attention_sum = trace["attention"].sum(dim=1)
    assert torch.allclose(
        attention_sum,
        torch.ones_like(attention_sum),
        atol=1e-5,
        rtol=1e-5,
    )

    # ------------------------------------------------------------------
    # Backward contract
    # ------------------------------------------------------------------
    target = torch.ones_like(
        anomaly_logits
    )

    loss = F.mse_loss(
        anomaly_logits,
        target,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    # Frozen backbone must remain frozen and receive no gradients.
    for name, parameter in tv1.dino.named_parameters():
        assert not parameter.requires_grad, (
            f"DINO.{name}: expected frozen parameter"
        )
        assert parameter.grad is None, (
            f"DINO.{name}: frozen parameter received gradient"
        )

    # Upstream and downstream trainable modules remain connected.
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
    """The wrapper also accepts a TV1 module returning only six features."""

    class DirectTV1(nn.Module):
        def forward(
            self,
            batch_size: int,
        ) -> dict[str, Tensor]:
            return {
                name: torch.randn(
                    batch_size,
                    FUSION_DIM,
                    FEATURE_HW,
                    FEATURE_HW,
                )
                for name in MULTIVIEW_FEATURE_KEYS
            }

    model = MSILADay2Integrated(
        feature_pipeline=DirectTV1(),
        fusion_dim=FUSION_DIM,
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
    assert torch.isfinite(out).all()
