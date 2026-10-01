"""Project six aligned DINOv3 feature sources to a common fusion dimension.

Expected inputs
---------------
Local features:
    L4, L8, L12

Context features already aligned to Local coordinates:
    C4_to_L, C8_to_L, C12_to_L

Output contract
---------------
Exactly six tensors:

    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

All outputs have shape:

    [B, fusion_dim, H, W]

A 1x1 Conv2d is a learned linear projection along the channel dimension while
preserving the spatial grid. By default, Local and Context from the *same DINO
block* share one projector because they live in the same pretrained channel
basis. This keeps the two views directly comparable and halves projection
parameters. The behavior is configurable for ablation.

References
----------
- Lin et al., Feature Pyramid Networks for Object Detection, CVPR 2017.
  FPN uses 1x1 lateral convolutions to map backbone feature levels to a common
  channel dimension before multi-level fusion.
  https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html
- Lin, Chen, Yan, Network in Network, ICLR 2014 / arXiv:1312.4400.
  1x1 convolution as channel-wise learned transformation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

__all__ = ["SixFeatureProjection"]


class SixFeatureProjection(nn.Module):
    """Project Local + aligned Context features to a common dimension ``d``.

    Parameters
    ----------
    in_channels:
        DINOv3 dense feature dimension, e.g. 384 for ViT-S/16.
    fusion_dim:
        Common output channel dimension ``d`` used by later fusion/decoder.
    blocks:
        Human-readable DINO blocks. Project default: (4, 8, 12).
    share_across_views:
        If True (recommended default), Lk and Ck_to_L use the same 1x1
        projector for a given block k. Set False only for a view-specific
        projection ablation.
    bias:
        Conv2d bias flag. PyTorch default behavior is retained.
    check_finite:
        Debug NaN/Inf checks; disabled by default to avoid synchronization
        overhead in the hot training path.
    """

    def __init__(
        self,
        in_channels: int,
        fusion_dim: int,
        *,
        blocks: Sequence[int] = (4, 8, 12),
        share_across_views: bool = True,
        bias: bool = True,
        check_finite: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.fusion_dim = int(fusion_dim)
        self.blocks = tuple(int(b) for b in blocks)
        self.share_across_views = bool(share_across_views)
        self.check_finite = bool(check_finite)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.fusion_dim <= 0:
            raise ValueError("fusion_dim must be > 0.")
        if len(self.blocks) != 3 or len(set(self.blocks)) != 3:
            raise ValueError("Project contract requires exactly three unique blocks.")
        if tuple(sorted(self.blocks)) != self.blocks:
            raise ValueError("blocks must be strictly increasing.")

        if self.share_across_views:
            self.projectors = nn.ModuleDict(
                {
                    f"b{block}": nn.Conv2d(
                        self.in_channels,
                        self.fusion_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )
                    for block in self.blocks
                }
            )
            self.local_projectors = None
            self.context_projectors = None
        else:
            self.projectors = None
            self.local_projectors = nn.ModuleDict(
                {
                    f"b{block}": nn.Conv2d(
                        self.in_channels,
                        self.fusion_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )
                    for block in self.blocks
                }
            )
            self.context_projectors = nn.ModuleDict(
                {
                    f"b{block}": nn.Conv2d(
                        self.in_channels,
                        self.fusion_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )
                    for block in self.blocks
                }
            )

    @staticmethod
    def _validate_tensor(name: str, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(x)!r}.")
        if x.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W], got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {x.dtype}.")

    def _get_projector(self, *, block: int, view: str) -> nn.Conv2d:
        key = f"b{block}"
        if self.share_across_views:
            assert self.projectors is not None
            return self.projectors[key]
        if view == "local":
            assert self.local_projectors is not None
            return self.local_projectors[key]
        assert self.context_projectors is not None
        return self.context_projectors[key]

    def _validate_inputs(
        self,
        local_features: Mapping[str, Tensor],
        aligned_context_features: Mapping[str, Tensor],
    ) -> tuple[int, int, int]:
        local_keys = [f"L{b}" for b in self.blocks]
        context_keys = [f"C{b}_to_L" for b in self.blocks]

        missing_local = [k for k in local_keys if k not in local_features]
        missing_context = [k for k in context_keys if k not in aligned_context_features]
        if missing_local:
            raise KeyError(f"local_features missing keys: {missing_local}")
        if missing_context:
            raise KeyError(f"aligned_context_features missing keys: {missing_context}")

        all_items = [
            (k, local_features[k]) for k in local_keys
        ] + [
            (k, aligned_context_features[k]) for k in context_keys
        ]

        ref_name, ref = all_items[0]
        self._validate_tensor(ref_name, ref)
        if ref.shape[1] != self.in_channels:
            raise ValueError(
                f"{ref_name}: expected C={self.in_channels}, got C={ref.shape[1]}."
            )

        b, _, h, w = ref.shape
        for name, x in all_items:
            self._validate_tensor(name, x)
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f"{name}: expected C={self.in_channels}, got C={x.shape[1]}."
                )
            if x.shape[0] != b or x.shape[-2:] != (h, w):
                raise ValueError(
                    "All six sources must already share B/H/W before projection. "
                    f"Reference {ref_name}={tuple(ref.shape)}, {name}={tuple(x.shape)}."
                )
            if x.device != ref.device or x.dtype != ref.dtype:
                raise ValueError("All six sources must share device and dtype.")
            if self.check_finite and not bool(torch.isfinite(x).all()):
                raise ValueError(f"{name} contains NaN/Inf.")

        return b, h, w

    def forward(
        self,
        local_features: Mapping[str, Tensor],
        aligned_context_features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Return the six locked TV1 fusion sources with common shape."""
        b, h, w = self._validate_inputs(local_features, aligned_context_features)

        out: dict[str, Tensor] = {}
        for block in self.blocks:
            local = local_features[f"L{block}"]
            context = aligned_context_features[f"C{block}_to_L"]

            local_proj = self._get_projector(block=block, view="local")(local)
            context_proj = self._get_projector(block=block, view="context")(context)

            out[f"local_b{block}"] = local_proj
            out[f"context_b{block}"] = context_proj

        expected_shape = (b, self.fusion_dim, h, w)
        for name, x in out.items():
            if tuple(x.shape) != expected_shape:
                raise RuntimeError(
                    f"{name}: projection contract violated, expected {expected_shape}, "
                    f"got {tuple(x.shape)}."
                )
            if self.check_finite and not bool(torch.isfinite(x).all()):
                raise RuntimeError(f"{name} contains NaN/Inf after projection.")

        # Return in the exact logical order requested by the project.
        ordered: dict[str, Tensor] = {}
        for block in self.blocks:
            ordered[f"local_b{block}"] = out[f"local_b{block}"]
        for block in self.blocks:
            ordered[f"context_b{block}"] = out[f"context_b{block}"]
        return ordered

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, fusion_dim={self.fusion_dim}, "
            f"blocks={self.blocks}, share_across_views={self.share_across_views}"
        )
