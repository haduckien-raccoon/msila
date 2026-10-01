# models/attention_fusion.py

from __future__ import annotations

from typing import Mapping, TypedDict

import torch
from torch import Tensor, nn

from .contracts import (
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
    validate_multiview_features,
)


class AttentionFusionOutput(TypedDict):
    """
    feature:
        F_fused [B, d, h, w]

    attention:
        Normalized source weights [B, 6]

    logits:
        Raw source scores before softmax [B, 6]
    """
    feature: Tensor
    attention: Tensor
    logits: Tensor


class AttentionFusion(nn.Module):
    """
    Day-2 Source Attention Fusion v0.

    Input:
        {
            "local_b4":    [B,d,h,w],
            "local_b8":    [B,d,h,w],
            "local_b12":   [B,d,h,w],
            "context_b4":  [B,d,h,w],
            "context_b8":  [B,d,h,w],
            "context_b12": [B,d,h,w],
        }

    Output:
        feature   -> [B,d,h,w]
        attention -> [B,6]
        logits    -> [B,6]

    Formulation:
        z_i = GAP(F_i)

        s_i = w^T z_i + b_i

        alpha_i = softmax(s_i)

        F_fused = sum_i alpha_i * F_i
    """

    def __init__(
        self,
        dim: int,
        *,
        validate_input: bool = True,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be positive, got {dim}"
            )

        self.dim = dim
        self.validate_input = validate_input

        self.source_names = tuple(MULTIVIEW_FEATURE_KEYS)
        self.num_sources = len(self.source_names)

        # ----------------------------------------------------
        # Shared content-dependent scorer:
        #
        # z_i [d] -> scalar score
        #
        # bias=False because a shared scalar bias would be
        # cancelled by softmax.
        # ----------------------------------------------------
        self.score_proj = nn.Linear(
            dim,
            1,
            bias=False,
        )

        # ----------------------------------------------------
        # Source-specific learnable prior.
        #
        # Allows the network to learn that some DINO blocks /
        # views are generally more useful than others.
        # ----------------------------------------------------
        self.source_bias = nn.Parameter(
            torch.zeros(self.num_sources)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Start from uniform attention.

        logits = 0
        => softmax(logits) = 1 / 6

        Therefore the initial behavior is exactly Mean Fusion.
        Attention must LEARN to deviate from that baseline.
        """
        nn.init.zeros_(self.score_proj.weight)
        nn.init.zeros_(self.source_bias)

    def forward(
        self,
        features: Mapping[str, Tensor],
    ) -> AttentionFusionOutput:

        # ----------------------------------------------------
        # 1. Validate Day-2 contract
        # ----------------------------------------------------
        if self.validate_input:
            validate_multiview_features(
                features,
                expected_channels=self.dim,
                check_finite=True,
            )

        # ----------------------------------------------------
        # 2. Fixed source ordering
        #
        # [B,d,h,w] x 6
        #       ↓
        # [B,6,d,h,w]
        # ----------------------------------------------------
        stacked = torch.stack(
            [
                features[name]
                for name in self.source_names
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # 3. Global descriptor for each source
        #
        # GAP over spatial dimensions:
        #
        # [B,6,d,h,w]
        #       ↓
        # [B,6,d]
        # ----------------------------------------------------
        descriptors = stacked.mean(
            dim=(-2, -1)
        )

        # ----------------------------------------------------
        # 4. Compute source logits
        #
        # shared content score + source-specific prior
        #
        # [B,6,d]
        #    ↓
        # [B,6]
        # ----------------------------------------------------
        logits = (
            self.score_proj(descriptors)
            .squeeze(-1)
        )

        logits = logits + self.source_bias.unsqueeze(0)

        # ----------------------------------------------------
        # 5. Softmax over SIX SOURCES
        #
        # For each sample:
        #
        # sum_i alpha_i = 1
        # ----------------------------------------------------
        if logits.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            # Numerically safer under mixed precision.
            attention = torch.softmax(
                logits.float(),
                dim=1,
            ).to(dtype=logits.dtype)
        else:
            attention = torch.softmax(
                logits,
                dim=1,
            )

        # ----------------------------------------------------
        # 6. Weighted fusion
        #
        # attention:
        # [B,6]
        #
        # reshape:
        # [B,6,1,1,1]
        #
        # weighted sum:
        # [B,6,d,h,w]
        #       ↓
        # [B,d,h,w]
        # ----------------------------------------------------
        weights = attention[
            :,
            :,
            None,
            None,
            None,
        ]

        fused = (
            stacked * weights
        ).sum(dim=1)

        return {
            "feature": fused,
            "attention": attention,
            "logits": logits,
        }

    @torch.no_grad()
    def attention_summary(
        self,
        attention: Tensor,
    ) -> dict[str, float]:
        """
        Mean attention weight of each source over a batch.

        Useful for experiment logging only.
        """

        if attention.ndim != 2:
            raise ContractError(
                "attention must have shape [B,6], "
                f"got {tuple(attention.shape)}"
            )

        if attention.shape[1] != self.num_sources:
            raise ContractError(
                "attention source count mismatch: "
                f"expected {self.num_sources}, "
                f"got {attention.shape[1]}"
            )

        mean_weights = (
            attention
            .detach()
            .float()
            .mean(dim=0)
            .cpu()
        )

        return {
            name: float(mean_weights[i])
            for i, name in enumerate(self.source_names)
        }