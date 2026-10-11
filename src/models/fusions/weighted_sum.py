"""Global scalar weighting of the six aligned/projected G2 feature sources."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..contracts import MULTIVIEW_FEATURE_KEYS, validate_multiview_features


class WeightedSumFusion(nn.Module):
    """Six learned scalar logits, initialized to uniform source weights.

    ``dim`` is the common fusion width, not the backbone or Adapter width.
    ``forward(features)`` returns one [B, dim, H, W] tensor. Source weights
    are global (shared across every image and spatial position).
    """

    def __init__(self, dim: int, *, validate_input: bool = True) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.validate_input = validate_input
        self.source_names = MULTIVIEW_FEATURE_KEYS
        self.num_sources = len(self.source_names)
        self.source_logits = nn.Parameter(torch.zeros(self.num_sources))

    def normalized_weights(self) -> Tensor:
        """Return nonnegative [6] weights whose sum is one.

        Normalize in FP32 for half/bfloat16 parameters; the forward path
        casts weights to the feature dtype without detaching gradients.
        """
        logits = self.source_logits
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return logits.softmax(dim=0)

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        if self.validate_input:
            validate_multiview_features(
                features, expected_channels=self.dim, check_finite=True,
            )
        sources = torch.stack([features[key] for key in self.source_names], dim=1)
        weights = self.normalized_weights().to(dtype=sources.dtype)
        return (sources * weights[None, :, None, None, None]).sum(dim=1)
