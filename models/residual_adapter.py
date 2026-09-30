"""
Residual Spatial Adapter
========================

Architecture:

    x
      └──> down_proj 1x1
            └──> DWConv kxk
                  └──> GELU
                        └──> up_proj 1x1
                              └──> Δx

    y = x + gamma * Δx

Default:
    gamma = 0

Therefore at initialization:

    y = x

Expected input:
    x: Tensor [B, C, H, W]

Typical use:
    frozen vision backbone (e.g. DINOv3)
        -> feature map
        -> ResidualAdapter2d
        -> fusion / decoder

References
----------
1. He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
2. Houlsby et al., Parameter-Efficient Transfer Learning for NLP, 2019.
3. Chen et al., AdaptFormer, NeurIPS 2022.
4. Jie & Deng, ConvPass, ECCV 2022.
5. Howard et al., MobileNets, 2017.
6. Bachlechner et al., ReZero, 2020.
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


class ResidualAdapter2d(nn.Module):
    """
    Lightweight residual spatial adapter for 2-D feature maps.

    Pipeline
    --------
        x
        -> 1x1 down projection
        -> depthwise convolution
        -> GELU
        -> 1x1 up projection
        -> delta_x

        output = x + gamma * delta_x

    Parameters
    ----------
    in_channels:
        Number of input/output channels C.

    bottleneck_channels:
        Explicit bottleneck dimension d_hat.
        If None, it is computed as:

            d_hat = max(1, C // reduction)

    reduction:
        Bottleneck reduction ratio when bottleneck_channels is None.

    kernel_size:
        Spatial kernel for depthwise convolution.
        Recommended default: 3.

    gamma_init:
        Initial residual scaling coefficient.

        gamma_init = 0.0
        gives an exact identity mapping at initialization
        for finite inputs.

    Notes
    -----
    We intentionally DO NOT zero-initialize ``up_proj.weight``.

    If both:
        gamma = 0
        up_proj = 0

    then:
        delta_x = 0
        dL/dgamma = 0
        dL/d(theta_adapter) = 0

    and the adapter can become completely stuck at initialization.

    Instead:
        - adapter branch weights: normal non-zero initialization
        - gamma: zero initialization

    so gamma receives gradient on the first optimization step.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int | None = None,
        reduction: int = 4,
        kernel_size: int = 3,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be > 0, got {in_channels}"
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be > 0, got {reduction}"
            )

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer."
            )

        if bottleneck_channels is None:
            bottleneck_channels = max(
                1,
                in_channels // reduction,
            )

        if bottleneck_channels <= 0:
            raise ValueError(
                "bottleneck_channels must be > 0."
            )

        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.kernel_size = kernel_size
        self.gamma_init = float(gamma_init)

        # -----------------------------------------------------
        # 1. Channel bottleneck
        #
        # R^{C x H x W}
        #        ->
        # R^{d_hat x H x W}
        #
        # Equivalent to a per-spatial-location linear projection.
        # -----------------------------------------------------
        self.down_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # -----------------------------------------------------
        # 2. Spatial mixing
        #
        # groups = bottleneck_channels
        #
        # => each channel has its own k x k kernel.
        #
        # It introduces local spatial information without the
        # C_in * C_out cost of a conventional convolution.
        # -----------------------------------------------------
        padding = kernel_size // 2

        self.dwconv = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=bottleneck_channels,
            bias=True,
        )

        # -----------------------------------------------------
        # 3. Non-linearity
        #
        # GELU is a natural default for Transformer features.
        # -----------------------------------------------------
        self.activation = nn.GELU()

        # -----------------------------------------------------
        # 4. Restore original channel dimension
        #
        # R^{d_hat x H x W}
        #        ->
        # R^{C x H x W}
        # -----------------------------------------------------
        self.up_proj = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # -----------------------------------------------------
        # 5. Learnable residual gate
        #
        # y = x + gamma * delta_x
        #
        # gamma = 0 at initialization
        # -> y = x
        # -----------------------------------------------------
        self.gamma = nn.Parameter(
            torch.tensor(self.gamma_init, dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the adapter branch with non-zero weights,
        while keeping the residual gate gamma at zero.

        Important:
            Do NOT simultaneously zero-init gamma and up_proj.
        """

        # Reasonable initialization for a compact nonlinear branch.
        nn.init.kaiming_uniform_(
            self.down_proj.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.dwconv.weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            self.up_proj.weight,
            a=math.sqrt(5),
        )

        # Avoid unnecessary constant spatial/channel offsets
        # at the beginning of training.
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)

        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

        with torch.no_grad():
            self.gamma.fill_(self.gamma_init)

    def residual(self, x: Tensor) -> Tensor:
        """
        Compute Δx without adding the identity branch.

        Useful for:
            - debugging
            - visualization
            - ablation
            - measuring adapter magnitude
        """

        z = self.down_proj(x)
        z = self.dwconv(z)
        z = self.activation(z)
        delta_x = self.up_proj(z)

        return delta_x

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Tensor with shape [B, C, H, W].

        Returns
        -------
        Tensor:
            Adapted feature map with exactly the same shape.
        """

        if x.ndim != 4:
            raise ValueError(
                "ResidualAdapter2d expects [B, C, H, W], "
                f"but received shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected C={self.in_channels}, "
                f"but received C={x.shape[1]}."
            )

        delta_x = self.residual(x)

        return x + self.gamma * delta_x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"bottleneck_channels={self.bottleneck_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"gamma_init={self.gamma_init}"
        )


# Optional short alias for the project.
ResidualAdapter = ResidualAdapter2d