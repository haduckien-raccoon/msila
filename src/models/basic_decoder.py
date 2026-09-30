# models/basic_decoder.py

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BasicDecoder(nn.Module):
    """
    Minimal Day-1 decoder.

    Input:
        x: [B, C, h, w]

    Output:
        anomaly_logits: [B, 1, H, W]

    This decoder is intentionally simple.
    It is used only to verify that the complete MS-ILA pipeline
    can produce a dense anomaly map.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 1)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: Tensor,
        output_size: Tuple[int, int],
    ) -> Tensor:

        # x: [B,C,h,w]

        logits = self.head(x)

        # [B,1,h,w]

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        # [B,1,H,W]

        return logits