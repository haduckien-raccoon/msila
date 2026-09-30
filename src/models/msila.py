"""MS-ILA Day-1 end-to-end architecture assembly.

Day-1 scope only:
    RGB image -> frozen DINOv3 -> residual adapters -> mean fusion
              -> basic decoder -> dense anomaly logits

No training, AU-PRO, Local/Context branch, attention fusion, thresholding,
or post-processing is implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .basic_decoder import BasicDecoder
from .contracts import (
    DINO_FEATURE_KEYS,
    validate_anomaly_logits,
    validate_dino_features,
    validate_image,
)
from .dinov3_extractor import DINOv3FeatureExtractor
from .mean_fusion import MeanFusion
from .residual_adapter import ResidualAdapter2d


class MSILA(nn.Module):
    """Minimal MS-ILA v0 used for Day-1 Architecture QA.

    The extractor must expose ``out_channels`` and return exactly:
        {"b4": [B,C,h,w], "b8": [B,C,h,w], "b12": [B,C,h,w]}.

    The three DINO features are adapted independently, fused by an
    unweighted mean, and decoded to ``[B,1,H,W]``.
    """

    def __init__(
        self,
        extractor: nn.Module,
        *,
        adapter_reduction: int = 4,
        adapter_kernel_size: int = 3,
        gamma_init: float = 0.0,
        fusion: nn.Module | None = None,
        decoder: nn.Module | None = None,
        validate: bool = True,
    ) -> None:
        super().__init__()

        if not hasattr(extractor, "out_channels"):
            raise TypeError(
                "extractor must expose an 'out_channels' attribute/property."
            )

        channels = int(extractor.out_channels)
        if channels <= 0:
            raise ValueError(f"extractor.out_channels must be > 0, got {channels}")

        self.extractor = extractor
        self.validate = bool(validate)

        self.adapters = nn.ModuleDict(
            {
                key: ResidualAdapter2d(
                    in_channels=channels,
                    reduction=adapter_reduction,
                    kernel_size=adapter_kernel_size,
                    gamma_init=gamma_init,
                )
                for key in DINO_FEATURE_KEYS
            }
        )

        self.fusion = fusion if fusion is not None else MeanFusion(validate=validate)
        self.decoder = decoder if decoder is not None else BasicDecoder(channels)

    @property
    def out_channels(self) -> int:
        """Channel width of DINO/adapted/fused feature maps."""
        return int(self.extractor.out_channels)

    @classmethod
    def from_dinov3(
        cls,
        *,
        repo_dir: str | Path,
        weights: str | Path,
        model_name: str = "dinov3_vits16",
        blocks: tuple[int, int, int] = (4, 8, 12),
        norm: bool = True,
        **kwargs: Any,
    ) -> "MSILA":
        """Build Day-1 MS-ILA directly from the official DINOv3 extractor."""
        extractor = DINOv3FeatureExtractor(
            repo_dir=repo_dir,
            weights=weights,
            model_name=model_name,
            blocks=blocks,
            norm=norm,
        )
        return cls(extractor=extractor, **kwargs)

    def adapt_features(
        self,
        features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Apply one residual adapter to each DINO layer feature."""
        if self.validate:
            validate_dino_features(features, require_same_shape=True)

        adapted = {
            key: self.adapters[key](features[key])
            for key in DINO_FEATURE_KEYS
        }

        if self.validate:
            validate_dino_features(adapted, require_same_shape=True)

        return adapted

    def forward_features(
        self,
        image: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Run DINO -> Adapter -> MeanFusion and expose QA intermediates."""
        if self.validate:
            validate_image(image)

        features = self.extractor(image)

        if self.validate:
            validate_dino_features(features, require_same_shape=True)

        adapted = self.adapt_features(features)
        fused = self.fusion(adapted)

        return dict(features), adapted, fused

    def forward(
        self,
        image: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Run full Day-1 path and return full-resolution anomaly logits."""
        features, adapted, fused = self.forward_features(image)

        logits = self.decoder(
            fused,
            output_size=(int(image.shape[-2]), int(image.shape[-1])),
        )

        if self.validate:
            validate_anomaly_logits(logits, image)

        if not return_trace:
            return logits

        trace: dict[str, object] = {
            "dino": features,
            "adapted": adapted,
            "fused": fused,
        }
        return logits, trace

    def adapter_gammas(self) -> dict[str, float]:
        """Return scalar residual gates for QA/reporting."""
        return {
            key: float(self.adapters[key].gamma.detach().cpu().item())
            for key in DINO_FEATURE_KEYS
        }
