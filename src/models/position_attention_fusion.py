"""TV1 D9 spatial source attention; legacy global AttentionFusion stays intact."""
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .contracts import MULTIVIEW_FEATURE_KEYS, ContractError, validate_multiview_features


class PositionAwareAttentionFusion(nn.Module):
    """Shared C->1 pixel scorer plus six source priors, softmax over sources.

    No spatial pooling: logits/weights are [B,6,H,W]. Zero initialization
    starts at Average; a learned scorer can assign different weights per pixel.
    """

    def __init__(self, dim: int, *, validate_input: bool = True):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim, self.validate_input = dim, validate_input
        self.source_names = MULTIVIEW_FEATURE_KEYS
        self.num_sources = len(self.source_names)
        self.score_proj = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(self.num_sources))
        nn.init.zeros_(self.score_proj.weight)

    def forward(self, features: Mapping[str, Tensor], *, return_trace: bool = False):
        if self.validate_input:
            validate_multiview_features(features, expected_channels=self.dim, check_finite=True)
        sources = torch.stack([features[key] for key in self.source_names], dim=1)
        b, n, c, h, w = sources.shape
        logits = self.score_proj(sources.reshape(b * n, c, h, w)).reshape(b, n, h, w)
        logits = logits + self.source_bias[None, :, None, None]
        if not bool(torch.isfinite(logits).all()):
            raise ContractError("Spatial attention logits contain NaN/Inf")
        weights = logits.float().softmax(dim=1) if logits.dtype in (torch.float16, torch.bfloat16) else logits.softmax(dim=1)
        weights = weights.to(dtype=sources.dtype)
        fused = (sources * weights[:, :, None]).sum(dim=1)
        if return_trace:
            return dict(feature=fused, logits=logits, attention=weights)
        return fused
