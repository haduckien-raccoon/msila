"""TV1 six-source factory using the existing TV2 modules and canonical IDs."""
from collections.abc import Mapping
import logging

import torch
from torch import Tensor, nn

from .contracts import MULTIVIEW_FEATURE_KEYS, validate_g2_fused_feature, validate_multiview_features
from .fusions import ConcatFusion, GatedFusion, WeightedSumFusion
from .mean_fusion import MeanFusion
from .position_attention_fusion import PositionAwareAttentionFusion

FUSION_METHODS = ("mean", "weighted_sum", "concat", "gated", "attention")
ALIASES = {"average": "mean", "weighted": "weighted_sum"}


def canonical_fusion_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("Fusion method must be a string")
    method = ALIASES.get(method.strip().lower(), method.strip().lower())
    if method not in FUSION_METHODS:
        raise ValueError(f"Unknown Fusion {method!r}; expected {FUSION_METHODS}")
    return method


class SixSourceFusion(nn.Module):
    """One tensor for Decoder; optional diagnostics never alter the features."""

    def __init__(self, method: str, dim: int):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.method, self.dim = canonical_fusion_method(method), dim
        self.source_names, self.num_sources = MULTIVIEW_FEATURE_KEYS, 6
        self._reported_amp_cast = None
        types = dict(weighted_sum=WeightedSumFusion, concat=ConcatFusion,
                     gated=GatedFusion, attention=PositionAwareAttentionFusion)
        # Outer boundary validates once before handing off to TV2 modules.
        self.module = (MeanFusion(validate=True) if self.method == "mean"
                       else types[self.method](dim, validate_input=False))

    def forward(self, features: Mapping[str, Tensor], *, return_trace: bool = False):
        validate_multiview_features(features, expected_channels=self.dim, check_finite=True)
        ordered = {key: features[key] for key in self.source_names}
        diagnostics = {}
        if self.method == "attention" and return_trace:
            diagnostics = self.module(ordered, return_trace=True)
            fused = diagnostics.pop("feature")
        else:
            fused = self.module(ordered)
        reference = ordered[self.source_names[0]]
        if (isinstance(fused, Tensor) and fused.device == reference.device
                and fused.dtype != reference.dtype
                and torch.is_autocast_enabled(reference.device.type)):
            # CUDA AMP promotes reductions such as sum to FP32. Decoder's
            # shared tensor boundary still requires the projected source dtype.
            # Cast retains gradients; never move an output to another device or
            # hide a dtype error outside an autocast forward.
            signature = (fused.dtype, reference.dtype)
            if self._reported_amp_cast != signature:
                logging.getLogger(__name__).info('Fusion dtype under AMP (%s): %s -> %s',
                                                 self.method, *signature)
                self._reported_amp_cast = signature
            fused = fused.to(dtype=reference.dtype)
        validate_g2_fused_feature(fused, ordered, expected_channels=self.dim)
        if not return_trace:
            return fused
        if self.method == "weighted_sum":
            diagnostics["weights"] = self.module.normalized_weights()
        elif self.method == "gated":
            diagnostics["weights"] = self.module.normalized_weights(ordered)
        elif self.method == "mean":
            diagnostics["weights"] = fused.new_full((6,), 1 / 6)
        return dict(feature=fused, method=self.method, source_keys=self.source_names, **diagnostics)


def build_fusion(method: str, dim: int) -> SixSourceFusion:
    return SixSourceFusion(method, dim)
