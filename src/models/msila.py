"""MS-ILA architecture assembly for Day-1 baseline and Day-2 fusion head.

Day-1 baseline:
    RGB image -> frozen DINOv3 -> residual adapters -> mean fusion
              -> basic decoder -> dense anomaly logits

Day-2 TV2 head:
    six aligned/projected Local-Context features -> Attention Fusion v0
              -> basic decoder -> dense anomaly logits

Local/Context crop generation, geometric alignment, feature projection,
training, AU-PRO evaluation, thresholding, and post-processing are outside
the scope of this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .attention_fusion import AttentionFusion
from .basic_decoder import BasicDecoder
from .contracts import (
    DINO_FEATURE_KEYS,
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
    validate_anomaly_logits,
    validate_dino_features,
    validate_image,
    validate_multiview_features,
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


class MSILADay2Head(nn.Module):
    """Day-2 TV2 head: six-source Attention Fusion -> Decoder.

    Expected input
    --------------
    A mapping with exactly six aligned/projected feature maps::

        local_b4, local_b8, local_b12,
        context_b4, context_b8, context_b12

    Every feature must have shape ``[B, d, h, w]`` in the same spatial
    coordinate system. The upstream alignment/projection modules are
    responsible for producing this contract.

    Output
    ------
    Dense anomaly logits with shape ``[B, 1, H, W]``. By default
    ``(H, W) = (512, 512)``.
    """

    def __init__(
        self,
        fusion_dim: int,
        *,
        fusion: nn.Module | None = None,
        decoder: nn.Module | None = None,
        output_size: tuple[int, int] = (512, 512),
        validate: bool = True,
    ) -> None:
        super().__init__()

        if fusion_dim <= 0:
            raise ValueError(f"fusion_dim must be > 0, got {fusion_dim}")

        if len(output_size) != 2:
            raise ValueError(f"output_size must be (H, W), got {output_size}")

        out_h, out_w = int(output_size[0]), int(output_size[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"output_size values must be > 0, got {(out_h, out_w)}"
            )

        self.fusion_dim = int(fusion_dim)
        self.output_size = (out_h, out_w)
        self.validate = bool(validate)

        # AttentionFusion v0 returns:
        #   feature   [B,d,h,w]
        #   attention [B,6]
        #   logits    [B,6]
        self.fusion = (
            fusion
            if fusion is not None
            else AttentionFusion(
                dim=self.fusion_dim,
                validate_input=validate,
            )
        )

        # Important: the Day-2 decoder consumes projected width d, which may
        # differ from the original DINO channel width C.
        self.decoder = (
            decoder
            if decoder is not None
            else BasicDecoder(self.fusion_dim)
        )

    def _unpack_and_validate_fusion_output(
        self,
        fusion_out: Mapping[str, Tensor],
        features: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Check the explicit AttentionFusion v0 interface."""

        required = {"feature", "attention", "logits"}
        received = set(fusion_out.keys())
        if received != required:
            raise ContractError(
                "AttentionFusion output must contain exactly "
                "{'feature', 'attention', 'logits'}. "
                f"received={sorted(received)}"
            )

        fused = fusion_out["feature"]
        attention = fusion_out["attention"]
        attention_logits = fusion_out["logits"]

        reference = features[MULTIVIEW_FEATURE_KEYS[0]]
        batch_size = int(reference.shape[0])

        if tuple(fused.shape) != tuple(reference.shape):
            raise ContractError(
                "Fused feature must preserve [B,d,h,w]. "
                f"expected={tuple(reference.shape)}, got={tuple(fused.shape)}"
            )

        expected_attention_shape = (
            batch_size,
            len(MULTIVIEW_FEATURE_KEYS),
        )
        if tuple(attention.shape) != expected_attention_shape:
            raise ContractError(
                "Attention weights have invalid shape. "
                f"expected={expected_attention_shape}, "
                f"got={tuple(attention.shape)}"
            )

        if tuple(attention_logits.shape) != expected_attention_shape:
            raise ContractError(
                "Attention logits have invalid shape. "
                f"expected={expected_attention_shape}, "
                f"got={tuple(attention_logits.shape)}"
            )

        for name, tensor in (
            ("fused_feature", fused),
            ("attention", attention),
            ("attention_logits", attention_logits),
        ):
            if not tensor.is_floating_point():
                raise ContractError(f"{name} must be floating point")
            if not torch.isfinite(tensor).all().item():
                raise ContractError(f"{name} contains NaN or Inf")

        # Softmax attention is a distribution over the six sources.
        if (attention < 0).any().item():
            raise ContractError("Attention weights must be non-negative")

        sums = attention.float().sum(dim=1)
        if not torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ContractError(
                "Attention weights must sum to 1 across the six sources"
            )

        return fused, attention, attention_logits

    @staticmethod
    def _validate_decoder_output(
        anomaly_logits: Tensor,
        *,
        batch_size: int,
        output_size: tuple[int, int],
    ) -> None:
        """Validate ``[B,1,H,W]`` without requiring the original image."""

        expected = (
            int(batch_size),
            1,
            int(output_size[0]),
            int(output_size[1]),
        )

        if anomaly_logits.ndim != 4:
            raise ContractError(
                "Decoder output must be BCHW. "
                f"got shape={tuple(anomaly_logits.shape)}"
            )

        if tuple(anomaly_logits.shape) != expected:
            raise ContractError(
                "Decoder output shape mismatch. "
                f"expected={expected}, got={tuple(anomaly_logits.shape)}"
            )

        if not anomaly_logits.is_floating_point():
            raise ContractError(
                "Decoder output must be floating point, "
                f"got dtype={anomaly_logits.dtype}"
            )

        if not torch.isfinite(anomaly_logits).all().item():
            raise ContractError("Decoder output contains NaN or Inf")

    def forward(
        self,
        features: Mapping[str, Tensor],
        *,
        output_size: tuple[int, int] | None = None,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Run ``6 features -> AttentionFusion -> Decoder``.

        ``return_trace=True`` keeps the fusion internals available for QA:
        fused feature, normalized attention weights, and pre-softmax logits.
        """

        if self.validate:
            validate_multiview_features(
                features,
                expected_channels=self.fusion_dim,
                check_finite=True,
            )

        target_size = self.output_size if output_size is None else (
            int(output_size[0]),
            int(output_size[1]),
        )
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                f"output_size values must be > 0, got {target_size}"
            )

        fusion_out = self.fusion(features)
        if not isinstance(fusion_out, Mapping):
            raise TypeError(
                "AttentionFusion must return a mapping with "
                "'feature', 'attention', and 'logits'."
            )

        if self.validate:
            fused, attention, attention_logits = (
                self._unpack_and_validate_fusion_output(
                    fusion_out,
                    features,
                )
            )
        else:
            fused = fusion_out["feature"]
            attention = fusion_out["attention"]
            attention_logits = fusion_out["logits"]

        anomaly_logits = self.decoder(
            fused,
            output_size=target_size,
        )

        if self.validate:
            self._validate_decoder_output(
                anomaly_logits,
                batch_size=int(fused.shape[0]),
                output_size=target_size,
            )

        if not return_trace:
            return anomaly_logits

        trace: dict[str, object] = {
            "fused": fused,
            "attention": attention,
            "attention_logits": attention_logits,
        }
        return anomaly_logits, trace

