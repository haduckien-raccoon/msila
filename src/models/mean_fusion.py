# models/mean_fusion.py

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .contracts import (
    DINO_FEATURE_KEYS,
    validate_dino_features,
    validate_fused_feature,
)


class MeanFusion(nn.Module):
    """
    Parameter-free baseline fusion for Day 1.

    Input:
        {
            "b4":  Tensor [B, C, h, w],
            "b8":  Tensor [B, C, h, w],
            "b12": Tensor [B, C, h, w],
        }

    Output:
        Tensor [B, C, h, w]

    Requirement:
        All input features must have identical shapes.
    """

    def __init__(self, validate: bool = True):
        super().__init__()
        self.validate = validate

    def forward(
        self,
        features: Mapping[str, Tensor],
    ) -> Tensor:

        # ----------------------------------------------------
        # 1. Check tensor contract
        # ----------------------------------------------------
        if self.validate:
            validate_dino_features(
                features,
                require_same_shape=True,
            )

        # ----------------------------------------------------
        # 2. Collect features in deterministic order
        # ----------------------------------------------------
        xs = [
            features[key]
            for key in DINO_FEATURE_KEYS
        ]

        # xs:
        # b4  -> [B,C,h,w]
        # b8  -> [B,C,h,w]
        # b12 -> [B,C,h,w]

        # ----------------------------------------------------
        # 3. Mean fusion
        # ----------------------------------------------------
        stacked = torch.stack(xs, dim=0)

        # [3, B, C, h, w]

        fused = stacked.mean(dim=0)

        # [B, C, h, w]

        # ----------------------------------------------------
        # 4. Validate output
        # ----------------------------------------------------
        if self.validate:
            validate_fused_feature(
                fused,
                features,
            )

        return fused