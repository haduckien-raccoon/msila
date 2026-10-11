"""Concatenate six aligned/projected G2 sources and restore fusion width."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..contracts import MULTIVIEW_FEATURE_KEYS, validate_multiview_features


class ConcatFusion(nn.Module):
    """Canonical [B, 6*dim, H, W] concatenation followed by a 1x1 Conv.

    ``forward(features)`` returns one [B, dim, H, W] tensor. The projection
    uses PyTorch's standard Conv2d initialization and includes a bias.
    """

    def __init__(self, dim: int, *, validate_input: bool = True) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.validate_input = validate_input
        self.source_names = MULTIVIEW_FEATURE_KEYS
        self.num_sources = len(self.source_names)
        self.projection = nn.Conv2d(self.num_sources * dim, dim, kernel_size=1)

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        if self.validate_input:
            validate_multiview_features(
                features, expected_channels=self.dim, check_finite=True,
            )
        sources = torch.cat([features[key] for key in self.source_names], dim=1)
        # Preserve the shared tensor dtype even inside an autocast context.
        return self.projection(sources).to(dtype=sources.dtype)
