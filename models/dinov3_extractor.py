"""
Frozen DINOv3 Multi-Layer Feature Extractor
============================================

Purpose
-------
Extract exactly three dense feature maps from DINOv3 ViT:

    block 4
    block 8
    block 12

Human block numbering is 1-based.

DINOv3's official get_intermediate_layers() API uses 0-based
block indices, therefore:

    block 4  -> index 3
    block 8  -> index 7
    block 12 -> index 11

Output
------
For DINOv3 ViT-S/16:

    input:
        x: [B, 3, H, W]

    output:
        f4 : [B, 384, H/16, W/16]
        f8 : [B, 384, H/16, W/16]
        f12: [B, 384, H/16, W/16]

assuming H and W are divisible by 16.

The backbone is completely frozen:

    param.requires_grad = False
    backbone.eval()

References
----------
DINOv3:
    Siméoni et al., "DINOv3", 2025.
    https://arxiv.org/abs/2508.10104

Official implementation:
    https://github.com/facebookresearch/dinov3

Official API:
    VisionTransformer.get_intermediate_layers()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------
# Human-readable block numbers.
#
# IMPORTANT:
# DINOv3 get_intermediate_layers() uses zero-based indices.
#
# block 4, 8, 12
#       ↓
# index 3, 7, 11
# ---------------------------------------------------------------------

DEFAULT_BLOCKS: tuple[int, int, int] = (4, 8, 12)


class DINOv3FeatureExtractor(nn.Module):
    """
    Frozen DINOv3 ViT feature extractor.

    Extracts normalized dense patch features from exactly three
    Transformer blocks.

    Parameters
    ----------
    repo_dir:
        Local path to the official facebookresearch/dinov3 repository.

    weights:
        Path or URL to the official DINOv3 checkpoint.

    model_name:
        Official torch.hub model name.

        Default:
            dinov3_vits16

    blocks:
        Human-readable 1-based Transformer block numbers.

        Default:
            (4, 8, 12)

    norm:
        Whether DINOv3's official final normalization should be
        applied to each extracted intermediate representation.

        Recommended:
            True
    """

    SUPPORTED_VIT_MODELS = {
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vitl16plus",
        "dinov3_vith16plus",
        "dinov3_vit7b16",
    }

    def __init__(
        self,
        repo_dir: str | Path,
        weights: str | Path,
        model_name: str = "dinov3_vits16",
        blocks: Sequence[int] = DEFAULT_BLOCKS,
        norm: bool = True,
    ) -> None:
        super().__init__()

        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.weights = str(weights)
        self.model_name = model_name
        self.blocks = tuple(int(b) for b in blocks)
        self.norm = bool(norm)

        self._validate_config()

        # -------------------------------------------------------------
        # Load backbone using the OFFICIAL DINOv3 torch.hub interface.
        #
        # We intentionally use source="local":
        # - reproducible repository version
        # - works with local checkpoint
        # - avoids silently changing source code
        # -------------------------------------------------------------
        self.backbone = torch.hub.load(
            repo_or_dir=str(self.repo_dir),
            model=self.model_name,
            source="local",
            weights=self.weights,
        )

        self._validate_backbone()

        # Convert human block numbering -> Python/DINOv3 indexing.
        #
        # block 4  -> index 3
        # block 8  -> index 7
        # block 12 -> index 11
        self.block_indices = tuple(
            block - 1
            for block in self.blocks
        )

        # -------------------------------------------------------------
        # Freeze DINOv3.
        # -------------------------------------------------------------
        self.freeze_backbone()

    # =================================================================
    # Validation
    # =================================================================

    def _validate_config(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repository not found: {self.repo_dir}"
            )

        if not (self.repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(
                f"{self.repo_dir} does not look like the official "
                "DINOv3 repository: hubconf.py was not found."
            )

        if self.model_name not in self.SUPPORTED_VIT_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. "
                "This extractor is designed for DINOv3 ViT backbones. "
                f"Supported: {sorted(self.SUPPORTED_VIT_MODELS)}"
            )

        if len(self.blocks) != 3:
            raise ValueError(
                "Exactly 3 blocks must be requested. "
                f"Received: {self.blocks}"
            )

        if len(set(self.blocks)) != 3:
            raise ValueError(
                f"Block numbers must be unique: {self.blocks}"
            )

        if any(block <= 0 for block in self.blocks):
            raise ValueError(
                "Block numbers use human 1-based indexing and "
                "must therefore be > 0."
            )

        if tuple(sorted(self.blocks)) != self.blocks:
            raise ValueError(
                "Block numbers must be in increasing order. "
                f"Received: {self.blocks}"
            )

    def _validate_backbone(self) -> None:
        if not hasattr(self.backbone, "blocks"):
            raise TypeError(
                "Loaded model does not expose Transformer blocks. "
                "A DINOv3 ViT backbone is required."
            )

        if not hasattr(
            self.backbone,
            "get_intermediate_layers",
        ):
            raise TypeError(
                "Loaded backbone does not implement "
                "get_intermediate_layers()."
            )

        depth = len(self.backbone.blocks)

        if max(self.blocks) > depth:
            raise ValueError(
                f"Requested block {max(self.blocks)}, "
                f"but backbone depth is only {depth}."
            )

    # =================================================================
    # Freeze
    # =================================================================

    def freeze_backbone(self) -> None:
        """
        Completely freeze the pretrained DINOv3 backbone.
        """

        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        """
        Allow parent model to enter training mode while DINOv3 remains
        permanently in eval mode.

        Example:

            full_model.train()

        must NOT accidentally switch the frozen DINO backbone back
        into training mode.
        """

        super().train(mode)

        self.backbone.eval()

        return self

    # =================================================================
    # Properties
    # =================================================================

    @property
    def depth(self) -> int:
        """Number of Transformer blocks."""

        return len(self.backbone.blocks)

    @property
    def patch_size(self) -> int:
        """DINOv3 patch size."""

        patch_size = self.backbone.patch_size

        if isinstance(patch_size, tuple):
            if patch_size[0] != patch_size[1]:
                raise RuntimeError(
                    "Only square patch size is expected."
                )

            return int(patch_size[0])

        return int(patch_size)

    @property
    def out_channels(self) -> int:
        """Feature dimensionality C."""

        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)

        raise AttributeError(
            "Cannot determine DINOv3 embed dimension."
        )

    # =================================================================
    # Input validation
    # =================================================================

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [B, 3, H, W], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                "DINOv3 expects RGB input with C=3, "
                f"received C={x.shape[1]}."
            )

        h, w = x.shape[-2:]

        p = self.patch_size

        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"Input H,W should be divisible by patch_size={p}. "
                f"Received H={h}, W={w}."
            )

    # =================================================================
    # Forward
    # =================================================================

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Extract dense features from blocks 4, 8 and 12.

        Parameters
        ----------
        x:
            Normalized RGB tensor:

                [B, 3, H, W]

        Returns
        -------
        tuple:
            (f4, f8, f12)

        Each feature has shape:

            [B, C, H/P, W/P]

        where:
            C = backbone embedding dimension
            P = patch size
        """

        self._validate_input(x)

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # Use torch.no_grad(), NOT torch.inference_mode().
        #
        # These frozen features will subsequently be consumed by
        # TRAINABLE adapters/fusion/decoder.
        #
        # no_grad():
        #     prevents construction of a graph inside DINOv3
        #     while leaving the resulting tensors usable by
        #     downstream trainable modules.
        # -------------------------------------------------------------
        with torch.no_grad():

            features = self.backbone.get_intermediate_layers(
                x,
                n=self.block_indices,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=self.norm,
            )

        if len(features) != 3:
            raise RuntimeError(
                "DINOv3 extractor contract violated: "
                f"expected exactly 3 features, got {len(features)}."
            )

        f4, f8, f12 = features

        self._validate_outputs(
            x=x,
            features=(f4, f8, f12),
        )

        return f4, f8, f12

    # =================================================================
    # Output validation
    # =================================================================

    def _validate_outputs(
        self,
        x: Tensor,
        features: tuple[Tensor, Tensor, Tensor],
    ) -> None:

        expected_h = x.shape[-2] // self.patch_size
        expected_w = x.shape[-1] // self.patch_size

        for block, feature in zip(
            self.blocks,
            features,
        ):
            if feature.ndim != 4:
                raise RuntimeError(
                    f"Block {block}: expected [B,C,H,W], "
                    f"got {tuple(feature.shape)}."
                )

            if feature.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"Block {block}: batch dimension mismatch."
                )

            if feature.shape[1] != self.out_channels:
                raise RuntimeError(
                    f"Block {block}: expected "
                    f"C={self.out_channels}, "
                    f"got C={feature.shape[1]}."
                )

            if feature.shape[-2:] != (
                expected_h,
                expected_w,
            ):
                raise RuntimeError(
                    f"Block {block}: expected spatial size "
                    f"{(expected_h, expected_w)}, "
                    f"got {feature.shape[-2:]}."
                )

    # =================================================================
    # Debugging helpers
    # =================================================================

    def backbone_is_frozen(self) -> bool:
        """
        True iff every DINOv3 parameter has requires_grad=False.
        """

        return all(
            not p.requires_grad
            for p in self.backbone.parameters()
        )

    def extra_repr(self) -> str:
        return (
            f"model={self.model_name}, "
            f"blocks={self.blocks}, "
            f"indices={self.block_indices}, "
            f"patch_size={self.patch_size}, "
            f"out_channels={self.out_channels}, "
            f"frozen={self.backbone_is_frozen()}"
        )