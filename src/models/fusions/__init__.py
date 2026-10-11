"""Independent G2 six-source fusion modules (TV2 handoff to TV1).

Each module accepts the canonical feature mapping and returns one BCHW tensor.
Factory integration and experiment selection remain with the caller.
"""

from .concat import ConcatFusion
from .gated import GatedFusion
from .weighted_sum import WeightedSumFusion

__all__ = ["WeightedSumFusion", "ConcatFusion", "GatedFusion"]
