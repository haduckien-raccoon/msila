"""Configurable residual spatial adapter for Day-04 r×d screening.

Architecture (project-specific hybrid)
--------------------------------------
For a 2-D feature map F ∈ R^{B×C×H×W}:

    Z_r    = P_down(F)                    # 1×1: C -> r
    Z_s    = GELU(DWConv_k(Z_r))          # local spatial mixing at width r
    Z_d    = GELU(P_mid(Z_s))             # 1×1: r -> d
    ΔF     = P_out(Z_d)                   # 1×1: d -> C
    F_out  = F + γ ΔF

where:
    r = bottleneck_dim  (screening variable 1)
    d = projection_dim  (screening variable 2)

Scientific provenance
---------------------
The exact C -> r -> DWConv -> d -> C chain is a PROJECT DESIGN for controlled
ablation. It is not claimed to be copied verbatim from one paper.

Its components are grounded in:
- residual learning: He et al., CVPR 2016;
- bottleneck adapters / frozen-backbone transfer: Houlsby et al., ICML 2019;
- residual ViT adapters: AdaptFormer, NeurIPS 2022;
- convolutional visual inductive bias in ViT adapters: ConvPass, ECCV 2022;
- depthwise convolution: MobileNets, 2017;
- zero-initialized residual gate: ReZero, UAI 2021.

As of Sep-2026, lightweight adaptation/readout on frozen vision foundation
backbones remains a defensible design direction; see DINOv3 (2025), META
(2025), and LiDeRe (CVPR 2026). None of those papers establishes a universal
optimal (r, d), so r and d must be selected empirically on validation data.

Important terminology
---------------------
``r`` is called ``bottleneck_dim`` in the API because this module does NOT
factorize a weight matrix as LoRA does. Calling r an algebraic matrix "rank"
would be misleading. Run names still use r/d for compact experiment labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ResidualAdapterConfig:
    """Immutable configuration for :class:`ResidualAdapter2d`.

    Parameters
    ----------
    in_dim:
        Input/output channel dimension C.
    bottleneck_dim:
        Bottleneck width r; Day-04 screening variable 1.
    projection_dim:
        Hidden/projection width d; Day-04 screening variable 2.
    kernel_size:
        Odd depthwise-convolution kernel size. ``3`` is the conservative default.
    gamma_init:
        Initial residual gate. ``0.0`` gives exact identity at initialization.
    bias:
        Whether 1×1/DWConv layers use bias.
    """

    in_dim: int
    bottleneck_dim: int
    projection_dim: int
    kernel_size: int = 3
    gamma_init: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        _validate_hyperparameters(
            in_dim=self.in_dim,
            bottleneck_dim=self.bottleneck_dim,
            projection_dim=self.projection_dim,
            kernel_size=self.kernel_size,
            gamma_init=self.gamma_init,
        )

    @property
    def r(self) -> int:
        """Compact mathematical name for ``bottleneck_dim``."""

        return self.bottleneck_dim

    @property
    def d(self) -> int:
        """Compact mathematical name for ``projection_dim``."""

        return self.projection_dim

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ResidualAdapterConfig":
        """Build from a strict dict/YAML-like mapping.

        Strict keys prevent silent experiment drift (for example mixing stale
        ``reduction``/``hidden_dim`` fields with the locked Day-04 config).
        """

        allowed = {
            "in_dim",
            "bottleneck_dim",
            "projection_dim",
            "kernel_size",
            "gamma_init",
            "bias",
        }
        unknown = set(config) - allowed
        if unknown:
            raise KeyError(
                f"Unknown adapter config keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}"
            )

        required = {"in_dim", "bottleneck_dim", "projection_dim"}
        missing = required - set(config)
        if missing:
            raise KeyError(f"Missing required adapter config keys: {sorted(missing)}")

        return cls(**dict(config))


class ResidualAdapter2d(nn.Module):
    """Screenable residual adapter for spatial feature maps ``[B, C, H, W]``.

    ``bottleneck_dim`` (r) and ``projection_dim`` (d) are explicit constructor
    arguments; there are no hard-coded Day-04 candidate widths in this module.
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        projection_dim: int,
        kernel_size: int = 3,
        gamma_init: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        _validate_hyperparameters(
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim,
            projection_dim=projection_dim,
            kernel_size=kernel_size,
            gamma_init=gamma_init,
        )

        self.in_dim = int(in_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.projection_dim = int(projection_dim)
        self.kernel_size = int(kernel_size)
        self.gamma_init = float(gamma_init)
        self.bias = bool(bias)

        # C -> r: channel bottleneck.
        self.down_proj = nn.Conv2d(
            self.in_dim,
            self.bottleneck_dim,
            kernel_size=1,
            bias=self.bias,
        )

        # r -> r: local spatial mixing, one k×k kernel per channel.
        self.dwconv = nn.Conv2d(
            self.bottleneck_dim,
            self.bottleneck_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.bottleneck_dim,
            bias=self.bias,
        )

        self.activation = nn.GELU()

        # r -> d: second independently screenable capacity dimension.
        self.mid_proj = nn.Conv2d(
            self.bottleneck_dim,
            self.projection_dim,
            kernel_size=1,
            bias=self.bias,
        )

        # d -> C: restore backbone feature width.
        self.out_proj = nn.Conv2d(
            self.projection_dim,
            self.in_dim,
            kernel_size=1,
            bias=self.bias,
        )

        # ReZero-style scalar residual gate. gamma=0 => exact identity.
        self.gamma = nn.Parameter(torch.tensor(self.gamma_init, dtype=torch.float32))

        self.reset_parameters()

    @property
    def r(self) -> int:
        return self.bottleneck_dim

    @property
    def d(self) -> int:
        return self.projection_dim

    @classmethod
    def from_config(
        cls,
        config: ResidualAdapterConfig | Mapping[str, Any],
    ) -> "ResidualAdapter2d":
        """Instantiate from a validated dataclass or strict mapping."""

        cfg = (
            config
            if isinstance(config, ResidualAdapterConfig)
            else ResidualAdapterConfig.from_mapping(config)
            if isinstance(config, Mapping)
            else None
        )
        if cfg is None:
            raise TypeError(
                "config must be ResidualAdapterConfig or Mapping, "
                f"got {type(config).__name__}"
            )

        return cls(
            in_dim=cfg.in_dim,
            bottleneck_dim=cfg.bottleneck_dim,
            projection_dim=cfg.projection_dim,
            kernel_size=cfg.kernel_size,
            gamma_init=cfg.gamma_init,
            bias=cfg.bias,
        )

    def reset_parameters(self) -> None:
        """Use standard non-zero Conv2d initialization and reset gamma.

        We intentionally do NOT zero-initialize ``out_proj`` together with
        ``gamma``. If both were zero, the first-step gradient of gamma would
        also be zero and the adapter could be stuck at initialization.
        """

        for layer in (self.down_proj, self.dwconv, self.mid_proj, self.out_proj):
            layer.reset_parameters()

        with torch.no_grad():
            self.gamma.fill_(self.gamma_init)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "ResidualAdapter2d expects [B, C, H, W], "
                f"received {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected feature channels C={self.in_dim}, got C={x.shape[1]}"
            )

    def residual(self, x: Tensor) -> Tensor:
        """Compute only the learned residual branch ``ΔF``."""

        self._validate_input(x)
        z = self.down_proj(x)           # C -> r
        z = self.dwconv(z)              # r -> r, spatial mixing
        z = self.activation(z)
        z = self.mid_proj(z)            # r -> d
        z = self.activation(z)          # makes d a genuine nonlinear hidden width
        return self.out_proj(z)         # d -> C

    def forward(self, x: Tensor) -> Tensor:
        """Return ``F + gamma * ΔF`` with the same shape as ``x``."""

        delta = self.residual(x)
        return x + self.gamma * delta

    @property
    def num_trainable_parameters(self) -> int:
        """Trainable parameter count for candidate reporting."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def expected_parameter_count(self) -> int:
        """Closed-form parameter count for audit/debugging.

        With bias=True:
            C*r + r*k^2 + r*d + d*C + (2*r + d + C) + 1

        With bias=False:
            C*r + r*k^2 + r*d + d*C + 1

        The final ``+1`` is the learnable scalar gamma.
        """

        c, r, d, k = (
            self.in_dim,
            self.bottleneck_dim,
            self.projection_dim,
            self.kernel_size,
        )
        weights = c * r + r * (k * k) + r * d + d * c
        biases = (2 * r + d + c) if self.bias else 0
        return weights + biases + 1

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, bottleneck_dim={self.bottleneck_dim}, "
            f"projection_dim={self.projection_dim}, kernel_size={self.kernel_size}, "
            f"gamma_init={self.gamma_init}, bias={self.bias}"
        )


def make_adapter_run_name(bottleneck_dim: int, projection_dim: int) -> str:
    """Deterministic Day-04 run name: ``adapter_r{r}_d{d}``."""

    if isinstance(bottleneck_dim, bool) or not isinstance(bottleneck_dim, int):
        raise TypeError("bottleneck_dim must be an int")
    if isinstance(projection_dim, bool) or not isinstance(projection_dim, int):
        raise TypeError("projection_dim must be an int")
    if bottleneck_dim <= 0 or projection_dim <= 0:
        raise ValueError("bottleneck_dim and projection_dim must be > 0")
    return f"adapter_r{bottleneck_dim}_d{projection_dim}"


def _validate_hyperparameters(
    *,
    in_dim: int,
    bottleneck_dim: int,
    projection_dim: int,
    kernel_size: int,
    gamma_init: float,
) -> None:
    for name, value in (
        ("in_dim", in_dim),
        ("bottleneck_dim", bottleneck_dim),
        ("projection_dim", projection_dim),
        ("kernel_size", kernel_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd so H×W is preserved")
    if not math.isfinite(float(gamma_init)):
        raise ValueError(f"gamma_init must be finite, got {gamma_init}")


# Short project alias.
ResidualAdapter = ResidualAdapter2d
