from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.contracts import (
    DINO_FEATURE_KEYS,
    MULTIVIEW_FEATURE_KEYS,
)
from src.models.msila import MSILADay2Head
from src.models.residual_adapter import ResidualAdapter2d


# ============================================================
# Small synthetic configuration
#
# Task-4 already tests the 512x512 output contract.
# Gradient QA intentionally uses smaller tensors so CI is fast.
# ============================================================

B = 2
IN_H = 64
IN_W = 64

DINO_C = 16
FUSION_D = 8
FEAT_H = 8
FEAT_W = 8

OUTPUT_SIZE = (64, 64)


# ============================================================
# 1. Test-only frozen DINO surrogate
# ============================================================

class FrozenDinoStub(nn.Module):
    """
    Lightweight differentiable surrogate for the frozen DINOv3 backbone.

    It exposes trainable-looking parameters, but ALL are frozen
    (requires_grad=False). It returns exactly b4/b8/b12 in BCHW.

    This test does NOT validate DINO feature quality; it validates
    gradient routing:
        DINO parameters -> no grad
        downstream trainable modules -> grad
    """

    def __init__(self, out_channels: int = DINO_C) -> None:
        super().__init__()

        self.out_channels = int(out_channels)

        self.blocks = nn.ModuleDict(
            {
                key: nn.Conv2d(
                    3,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for key in DINO_FEATURE_KEYS
            }
        )

        # Make this an actual frozen backbone.
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        outputs: dict[str, Tensor] = {}

        for key in DINO_FEATURE_KEYS:
            x = self.blocks[key](image)

            # Match the controlled feature resolution required by the test.
            x = F.adaptive_avg_pool2d(
                x,
                output_size=(FEAT_H, FEAT_W),
            )

            outputs[key] = x

        return outputs


# ============================================================
# 2. Test-only projection
# ============================================================

class Projection1x1(nn.Module):
    """
    Scientific minimum for the Day-2 projection role:

        [B,C,h,w] -> [B,d,h,w]

    One independent 1x1 projection is used for each of the six
    Local/Context sources.

    This is intentionally test-local because TV1 owns the real
    feature_projection.py and Task 8 will integrate that real module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.proj = nn.ModuleDict(
            {
                name: nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=True,
                )
                for name in MULTIVIEW_FEATURE_KEYS
            }
        )

    def forward(
        self,
        features: dict[str, Tensor],
    ) -> dict[str, Tensor]:

        return {
            name: self.proj[name](features[name])
            for name in MULTIVIEW_FEATURE_KEYS
        }


# ============================================================
# 3. Minimal Day-2 gradient harness
# ============================================================

class GradientFlowHarness(nn.Module):
    """
    Controlled Day-2 path used ONLY for gradient QA:

        Local image ----\
                         Frozen DINO
        Context image --/      |
                              Adapter
                                |
                           Projection
                                |
                     6 projected features
                                |
                     Attention Fusion v0
                                |
                             Decoder

    Alignment is intentionally absent here:
    Task 5 validates autograd connectivity, not geometric correctness.
    Real TV1 alignment/projection is integrated in Task 8.
    """

    def __init__(self) -> None:
        super().__init__()

        self.dino = FrozenDinoStub(
            out_channels=DINO_C
        )

        # One adapter per DINO level, shared across Local/Context.
        # gamma_init=0.0 matches the Day-1 identity initialization.
        self.adapters = nn.ModuleDict(
            {
                key: ResidualAdapter2d(
                    in_dim=DINO_C,
                    bottleneck_dim=DINO_C // 4,
                    projection_dim=12,  # Software fixture; distinct from fusion width 8.
                    kernel_size=3,
                    gamma_init=0.0,
                )
                for key in DINO_FEATURE_KEYS
            }
        )

        self.projection = Projection1x1(
            in_channels=DINO_C,
            out_channels=FUSION_D,
        )

        # Uses the REAL AttentionFusion + BasicDecoder internally.
        self.head = MSILADay2Head(
            fusion_dim=FUSION_D,
            output_size=OUTPUT_SIZE,
            validate=True,
        )

    def _extract_and_adapt(
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
            f"{prefix}_{key}": adapted[key]
            for key in DINO_FEATURE_KEYS
        }

    def forward(
        self,
        local_image: Tensor,
        context_image: Tensor,
    ) -> tuple[Tensor, dict[str, object]]:

        local = self._extract_and_adapt(
            local_image,
            "local",
        )

        context = self._extract_and_adapt(
            context_image,
            "context",
        )

        six_features = {
            **local,
            **context,
        }

        projected = self.projection(
            six_features
        )

        logits, trace = self.head(
            projected,
            return_trace=True,
        )

        trace = dict(trace)
        trace["projected"] = projected

        return logits, trace


# ============================================================
# 4. Gradient utilities
# ============================================================

def trainable_parameters(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, p)
        for name, p in module.named_parameters()
        if p.requires_grad
    ]


def assert_frozen_module_has_no_grad(
    module: nn.Module,
    *,
    module_name: str,
) -> None:
    """
    Frozen means BOTH:
        requires_grad == False
        grad is None
    """
    params = list(module.named_parameters())

    assert params, (
        f"{module_name}: test invalid because module has no parameters"
    )

    for name, p in params:
        assert not p.requires_grad, (
            f"{module_name}.{name}: expected requires_grad=False"
        )

        assert p.grad is None, (
            f"{module_name}.{name}: frozen parameter unexpectedly "
            "received gradient"
        )


def assert_trainable_module_has_gradient(
    module: nn.Module,
    *,
    module_name: str,
) -> dict[str, float]:
    """
    Scientific gradient gate.

    Requirements:
      1. module has trainable parameters;
      2. every trainable parameter participates in the graph (grad != None);
      3. every available gradient is finite;
      4. at least one parameter receives a non-zero learning signal.

    Note:
    ResidualAdapter uses gamma_init=0. At initialization, some parameters
    inside its residual branch may legitimately have zero-valued gradients
    because they are multiplied by gamma=0. Therefore we do NOT require
    every gradient tensor to be non-zero. We require at least one non-zero
    gradient for the module as a whole.
    """
    params = trainable_parameters(module)

    assert params, (
        f"{module_name}: no trainable parameters found"
    )

    report: dict[str, float] = {}
    has_nonzero = False

    for name, p in params:
        assert p.grad is not None, (
            f"{module_name}.{name}: grad is None; "
            "parameter is disconnected from loss"
        )

        grad = p.grad.detach()

        assert torch.isfinite(grad).all(), (
            f"{module_name}.{name}: gradient contains NaN/Inf"
        )

        norm = float(grad.float().norm().item())
        report[name] = norm

        if norm > 0.0:
            has_nonzero = True

    assert has_nonzero, (
        f"{module_name}: all gradients are exactly zero"
    )

    return report


def clear_all_grads(
    modules: Iterable[nn.Module],
) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)


# ============================================================
# 5. Main Task-5 gate
# ============================================================

def test_day02_gradient_flow() -> None:
    """
    PASS criteria:

        Frozen DINO     : requires_grad=False and grad=None
        Adapter         : finite gradient, at least one non-zero
        Projection      : finite gradient, at least one non-zero
        AttentionFusion : finite gradient, at least one non-zero
        Decoder         : finite gradient, at least one non-zero
    """
    torch.manual_seed(42)

    model = GradientFlowHarness()
    model.train()

    local_image = torch.randn(
        B,
        3,
        IN_H,
        IN_W,
    )

    # Use a different Context input so Local/Context sources are not
    # artificially identical. This gives source attention a meaningful
    # gradient signal in the synthetic test.
    context_image = torch.randn(
        B,
        3,
        IN_H,
        IN_W,
    ) * 0.75 + 0.15

    clear_all_grads(
        [
            model.dino,
            model.adapters,
            model.projection,
            model.head,
        ]
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    anomaly_logits, trace = model(
        local_image,
        context_image,
    )

    assert anomaly_logits.shape == (
        B,
        1,
        OUTPUT_SIZE[0],
        OUTPUT_SIZE[1],
    )

    assert torch.isfinite(
        anomaly_logits
    ).all()

    # --------------------------------------------------------
    # Dummy loss
    #
    # A non-zero deterministic target avoids the weak test:
    #     loss = pred.mean()
    # which can accidentally give tiny/cancelled gradients.
    # --------------------------------------------------------
    target = torch.ones_like(
        anomaly_logits
    )

    loss = F.mse_loss(
        anomaly_logits,
        target,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------
    loss.backward()

    # --------------------------------------------------------
    # Frozen backbone gate
    # --------------------------------------------------------
    assert_frozen_module_has_no_grad(
        model.dino,
        module_name="DINO",
    )

    # --------------------------------------------------------
    # Trainable modules gate
    # --------------------------------------------------------
    adapter_report = assert_trainable_module_has_gradient(
        model.adapters,
        module_name="Adapter",
    )

    projection_report = assert_trainable_module_has_gradient(
        model.projection,
        module_name="Projection",
    )

    fusion_report = assert_trainable_module_has_gradient(
        model.head.fusion,
        module_name="Fusion",
    )

    decoder_report = assert_trainable_module_has_gradient(
        model.head.decoder,
        module_name="Decoder",
    )

    # --------------------------------------------------------
    # Also prove the fused feature stayed connected to autograd.
    # --------------------------------------------------------
    fused = trace["fused"]

    assert isinstance(fused, Tensor)
    assert fused.requires_grad
    assert fused.grad_fn is not None

    # Reports are intentionally retained as local variables:
    # pytest -s can print/debug them if needed.
    assert adapter_report
    assert projection_report
    assert fusion_report
    assert decoder_report


# ============================================================
# 6. Explicit check: DINO must remain frozen after backward
# ============================================================

def test_dino_is_frozen_by_construction() -> None:
    model = GradientFlowHarness()

    assert all(
        not p.requires_grad
        for p in model.dino.parameters()
    )


# ============================================================
# 7. Explicit check: downstream modules are trainable
# ============================================================

@pytest.mark.parametrize(
    "module_path",
    [
        "adapters",
        "projection",
        "fusion",
        "decoder",
    ],
)
def test_downstream_modules_have_trainable_parameters(
    module_path: str,
) -> None:

    model = GradientFlowHarness()

    if module_path == "fusion":
        module = model.head.fusion
    elif module_path == "decoder":
        module = model.head.decoder
    else:
        module = getattr(
            model,
            module_path,
        )

    assert any(
        p.requires_grad
        for p in module.parameters()
    ), f"{module_path}: expected trainable parameters"
