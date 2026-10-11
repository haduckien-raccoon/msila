"""Spatial sigmoid gating of the six aligned/projected G2 feature sources."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..contracts import MULTIVIEW_FEATURE_KEYS, validate_multiview_features


class GatedFusion(nn.Module):
    """One independent C->1 spatial sigmoid gate per source.

    For gate logits z_i, weights are sigmoid(z_i)/sum_j(sigmoid(z_j)).
    Computing softmax(logsigmoid(z)) implements that normalization without
    dividing by an underflowed sum when every gate is nearly zero. This is
    distinct from attention's softmax(z). Zero-initialized gates start uniform.
    ``forward(features)`` returns one [B, dim, H, W] tensor.
    """

    def __init__(self, dim: int, *, validate_input: bool = True) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.validate_input = validate_input
        self.source_names = MULTIVIEW_FEATURE_KEYS
        self.num_sources = len(self.source_names)
        self.gates = nn.ModuleList(
            nn.Conv2d(dim, 1, kernel_size=1) for _ in self.source_names
        )
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def normalized_weights(self, features: Mapping[str, Tensor]) -> Tensor:
        """Return spatial weights [B, 6, H, W], normalized along dimension 1."""
        if self.validate_input:
            validate_multiview_features(
                features, expected_channels=self.dim, check_finite=True,
            )
        logits = torch.cat(
            [gate(features[key]) for gate, key in zip(self.gates, self.source_names)],
            dim=1,
        )
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return F.logsigmoid(logits).softmax(dim=1)

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        weights = self.normalized_weights(features)
        sources = torch.stack([features[key] for key in self.source_names], dim=1)
        weights = weights.to(dtype=sources.dtype)
        return (sources * weights[:, :, None, :, :]).sum(dim=1)
