"""MS-ILA architecture assembly for Day-1 baseline and Day-2 fusion head.

Day-1 baseline:
    RGB image -> frozen DINOv3 -> residual adapters -> mean fusion
              -> basic decoder -> dense anomaly logits

Day-2 TV2 head:
    six aligned/projected Local-Context features -> Attention Fusion v0
              -> basic decoder -> dense anomaly logits

Day-2 integration:
    TV1 feature pipeline -> six aligned/projected features -> TV2 head

The concrete Local/Context crop generation, geometric alignment, and feature
projection algorithms remain owned by TV1. This module only defines the clean
integration boundary. Training, AU-PRO evaluation, thresholding, and
post-processing remain outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .attention_fusion import AttentionFusion
from .adapter_factory import AdapterCandidate, AdapterFactoryConfig, ResidualAdapterFactory
from .backbone_registry import adapter_pairs, backbone_spec
from .basic_decoder import BasicDecoder
from .contracts import (
    DINO_FEATURE_KEYS,
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
    g2_input_image,
    validate_anomaly_logits,
    validate_dino_features,
    validate_image,
    validate_multiview_features,
)
from .dinov3_extractor import DINOv3FeatureExtractor
from .feature_projection import SixFeatureProjection
from .feature_selector import FeatureSelector
from .mean_fusion import MeanFusion
from .residual_adapter import ResidualAdapter2d


class E1(nn.Module):
    """G1: a frozen deepest DINOv3 feature feeds only BasicDecoder.

    No Adapter, feature fusion, projection, or context branch is constructed.
    Input is normalized RGB; output is raw logits at the input tile size.
    """

    def __init__(self, extractor: nn.Module, hidden_channels: int = 64,
                 deterministic_resize: bool = False):
        super().__init__()
        if not isinstance(extractor, nn.Module):
            raise TypeError("extractor must be a torch.nn.Module")
        if getattr(extractor, "blocks", None) != (extractor.depth,):
            raise ValueError("E1 requires feature_mode='deepest'")
        self.extractor = extractor
        self.extractor.requires_grad_(False)
        self.extractor.eval()
        self.decoder = BasicDecoder(extractor.out_channels, hidden_channels,
                                    deterministic_resize=deterministic_resize)

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()
        return self

    @property
    def feature_key(self) -> str:
        return f"b{self.extractor.depth}"

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_deep_feature(self, image: Tensor) -> Tensor:
        """Run exactly one frozen deepest feature, shared by E1 and E2."""
        with torch.no_grad():
            features = self.extractor(image)
        if not isinstance(features, Mapping) or set(features) != {self.feature_key}:
            raise ContractError("E1/E2 extractor must return exactly one deepest feature")
        return features[self.feature_key]

    def _decoder_feature(self, feature: Tensor) -> Tensor:
        return feature

    def forward(
        self, image: Tensor | Mapping[str, object], *, return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """G2: model(batch); legacy G1: model(image). Return raw logits."""
        image = g2_input_image(image)
        feature = self.extract_deep_feature(image)
        decoder_feature = self._decoder_feature(feature)
        logits = self.decoder(decoder_feature, output_size=image.shape[-2:])
        validate_anomaly_logits(logits, image)
        if return_trace:
            return logits, {
                "dino": {self.feature_key: feature},
                "decoder_feature": decoder_feature,
            }
        return logits


class E2(E1):
    """G2 E2 = E1 + one existing residual Adapter before the same decoder.

    Frozen deepest DINOv3 -> C->r->DWConv->d->C Adapter -> BasicDecoder.
    No Context, projection or Fusion module is constructed. Adapter r/d are
    independent internal widths; the decoder still consumes backbone width C.
    """

    def __init__(
        self,
        extractor: nn.Module,
        *,
        adapter_bottleneck_dim: int,
        adapter_projection_dim: int,
        hidden_channels: int = 64,
        adapter_kernel_size: int = 3,
        gamma_init: float = 0.0,
        adapter_bias: bool = True,
        deterministic_resize: bool = False,
    ) -> None:
        super().__init__(extractor, hidden_channels=hidden_channels,
                         deterministic_resize=deterministic_resize)
        factory = ResidualAdapterFactory(AdapterFactoryConfig(
            in_dim=extractor.out_channels,
            kernel_size=adapter_kernel_size,
            gamma_init=gamma_init,
            bias=adapter_bias,
        ))
        self.adapter = factory.build_rd(
            r=adapter_bottleneck_dim, d=adapter_projection_dim,
        ).model

    @property
    def adapter_r(self) -> int:
        return self.adapter.r

    @property
    def adapter_d(self) -> int:
        return self.adapter.d

    def _decoder_feature(self, feature: Tensor) -> Tensor:
        # Only DINO extraction is no_grad; Adapter and decoder remain in graph.
        return self.adapter(feature)


class E3(nn.Module):
    """Three frozen Local layers -> independent Adapters -> projection -> mean.

    Canonical b4/b8/b12 names are shallow/middle/deep slots for the reused
    projector/selector. ``source_blocks`` records the actual backbone blocks.
    No Context modules or tensors are used. Adapter d is independent of the
    common projection width ``fusion_dim`` consumed by BasicDecoder.
    """

    def __init__(
        self, extractor: nn.Module, *, adapter_bottleneck_dim: int,
        adapter_projection_dim: int, fusion_dim: int = 64, hidden_channels: int = 64,
        adapter_kernel_size: int = 3, gamma_init: float = 0.0,
        adapter_bias: bool = True, deterministic_resize: bool = False,
    ) -> None:
        super().__init__()
        expected = backbone_spec(extractor.model_name).blocks
        if tuple(extractor.blocks) != expected or extractor.feature_mode != "multilayer":
            raise ValueError(f"E3 requires three registry blocks {expected}")
        if type(fusion_dim) is not int or fusion_dim < 1:
            raise ValueError("fusion_dim must be a positive integer")
        self.extractor = extractor
        self.extractor.requires_grad_(False)
        self.extractor.eval()
        self.source_blocks = expected
        self.selector = FeatureSelector("multi_local")
        self.source_block_map = dict(zip(self.selector.source_keys, expected))
        factory = ResidualAdapterFactory(AdapterFactoryConfig(
            in_dim=extractor.out_channels, kernel_size=adapter_kernel_size,
            gamma_init=gamma_init, bias=adapter_bias,
        ))
        self.adapters = nn.ModuleDict({
            key: factory.build_rd(r=adapter_bottleneck_dim, d=adapter_projection_dim).model
            for key in self.selector.source_keys
        })
        # Three projectors; shared-view mode constructs no unused Context params.
        self.projection = SixFeatureProjection(extractor.out_channels, fusion_dim,
                                               share_across_views=True, check_finite=True)
        self.fusion = MeanFusion(validate=True)
        self.decoder = BasicDecoder(fusion_dim, hidden_channels,
                                    deterministic_resize=deterministic_resize)

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()
        return self

    @property
    def num_sources(self) -> int:
        return self.selector.num_sources

    @property
    def adapter_r(self) -> int:
        return self.adapters[self.selector.source_keys[0]].r

    @property
    def adapter_d(self) -> int:
        return self.adapters[self.selector.source_keys[0]].d

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, image: Tensor | Mapping[str, object], *, return_trace: bool = False):
        image = g2_input_image(image)
        with torch.no_grad():
            features = self.extractor(image)
        physical_keys = {f"b{block}" for block in self.source_blocks}
        if not isinstance(features, Mapping) or set(features) != physical_keys:
            raise ContractError("E3 extractor must return exactly the three registry layers")
        adapted = {key: self.adapters[key](features[f"b{block}"])
                   for key, block in self.source_block_map.items()}
        raw_slots = {f"L{slot}": adapted[key]
                     for slot, key in zip((4, 8, 12), self.selector.source_keys)}
        projected = self.projection.project_sources(raw_slots, source_keys=self.selector.source_keys)
        fused = self.fusion(self.selector(projected))
        logits = self.decoder(fused, output_size=image.shape[-2:])
        validate_anomaly_logits(logits, image)
        if return_trace:
            return logits, dict(dino=features, adapted=adapted, projected=projected,
                                decoder_feature=fused, source_keys=self.selector.source_keys,
                                source_blocks=self.source_block_map, num_sources=self.num_sources)
        return logits


def resolve_g2_config(
    config: Mapping[str, Any], *, root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve backbone metadata/checkpoint and Adapter widths without loading.

    Only backbone.name selects the mapped checkpoint. C/depth/patch size come
    from the existing registry, checked again by DINOv3FeatureExtractor when
    loaded. Null r/d use declared ratios of C; explicit integer widths override
    either ratio independently. The input config is never mutated.
    """
    resolved = deepcopy(dict(config))
    backbone = resolved["backbone"]
    spec = backbone_spec(backbone["name"])
    if backbone.get("frozen", True) is not True:
        raise ValueError("G2 requires a frozen DINOv3 backbone")
    backbone.update(channels=spec.channels, deepest_block=spec.depth,
                    feature_blocks=list(spec.blocks), patch_size=spec.patch_size)
    weights = backbone.get("weights")
    if weights is None:
        try:
            weights = backbone["checkpoints"][backbone["name"]]
        except KeyError as exc:
            raise KeyError(f"No checkpoint configured for {backbone['name']}") from exc
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    for key, value in (("repo_dir", backbone["repo_dir"]), ("weights", weights)):
        path = Path(value).expanduser()
        backbone[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())

    adapter = resolved["adapter"]
    if adapter.get("r") is None or adapter.get("d") is None:
        # Reuse the existing C-based width rounding; no second Adapter design.
        scaled_r, scaled_d = adapter_pairs(spec.channels, {
            "r_ratios": [adapter.get("r_ratio", 1/6)],
            "d_ratios": [adapter.get("d_ratio", 2/3)],
            "round_to": adapter.get("round_to", 8),
        })[0]
        if adapter.get("r") is None:
            adapter["r"] = scaled_r
        if adapter.get("d") is None:
            adapter["d"] = scaled_d
    AdapterCandidate(adapter["r"], adapter["d"])
    AdapterFactoryConfig(in_dim=spec.channels,
                         kernel_size=adapter.get("kernel_size", 3),
                         gamma_init=adapter.get("gamma_init", 0.0),
                         bias=adapter.get("bias", True))
    return resolved


def build_g2_model(
    config: Mapping[str, Any], experiment: str = "E2", *, root: str | Path | None = None,
) -> E1 | E3:
    """Build E1--E3 from one backbone selector; E4/E5 remain unimplemented."""
    if experiment not in {"E1", "E2", "E3"}:
        raise NotImplementedError("build_g2_model currently supports only E1, E2 and E3")
    resolved = resolve_g2_config(config, root=root)
    backbone, adapter = resolved["backbone"], resolved["adapter"]
    if not Path(backbone["weights"]).is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {backbone['weights']}")
    extractor = DINOv3FeatureExtractor(
        repo_dir=backbone["repo_dir"], weights=backbone["weights"],
        model_name=backbone["name"], norm=backbone.get("norm", True),
        feature_mode="multilayer" if experiment == "E3" else "deepest", check_finite=True,
    )
    hidden_channels = resolved["decoder"]["hidden_channels"]
    deterministic_resize = resolved["decoder"].get("deterministic_resize", False)
    if experiment == "E1":
        return E1(extractor, hidden_channels=hidden_channels,
                  deterministic_resize=deterministic_resize)
    model_class = E3 if experiment == "E3" else E2
    projection = {"fusion_dim": resolved["fusion"]["dim"]} if experiment == "E3" else {}
    return model_class(
        extractor, adapter_bottleneck_dim=adapter["r"], adapter_projection_dim=adapter["d"],
        adapter_kernel_size=adapter.get("kernel_size", 3),
        gamma_init=adapter.get("gamma_init", 0.0), adapter_bias=adapter.get("bias", True),
        hidden_channels=hidden_channels,
        deterministic_resize=deterministic_resize,
        **projection,
    )


class MSILA(nn.Module):
    """Minimal MS-ILA baseline using the current Day-04 residual Adapter.

    Backbone contract
    -----------------
    ``extractor`` must expose ``out_channels`` and return exactly:

        {"b4": [B,C,h,w], "b8": [B,C,h,w], "b12": [B,C,h,w]}

    Adapter contract
    ----------------
    Each DINO block has one independent ``ResidualAdapter2d`` implementing

        C -> r -> DWConv -> d -> C
        F_out = F + gamma * DeltaF

    where:

        r = adapter_bottleneck_dim
        d = adapter_projection_dim

    ``r`` and ``d`` are explicit because the current Day-04 screen treats them
    as two independent architecture variables.  The legacy ``adapter_reduction``
    API is intentionally not accepted here.

    Day-1 fusion remains an unweighted mean over the three adapted DINO
    features, so the fused channel width remains ``C``.  In particular,
    ``adapter_projection_dim`` is internal to the Adapter and is NOT a
    downstream fusion width.
    """

    def __init__(
        self,
        extractor: nn.Module,
        *,
        adapter_bottleneck_dim: int,
        adapter_projection_dim: int,
        adapter_kernel_size: int = 3,
        gamma_init: float = 0.0,
        adapter_bias: bool = True,
        fusion: nn.Module | None = None,
        decoder: nn.Module | None = None,
        validate: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(extractor, nn.Module):
            raise TypeError(
                f"extractor must be torch.nn.Module, got {type(extractor).__name__}"
            )
        if not hasattr(extractor, "out_channels"):
            raise TypeError(
                "extractor must expose an 'out_channels' attribute/property."
            )

        channels = int(extractor.out_channels)
        if channels <= 0:
            raise ValueError(
                f"extractor.out_channels must be > 0, got {channels}"
            )

        for name, value in (
            ("adapter_bottleneck_dim", adapter_bottleneck_dim),
            ("adapter_projection_dim", adapter_projection_dim),
            ("adapter_kernel_size", adapter_kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if adapter_kernel_size % 2 == 0:
            raise ValueError(
                "adapter_kernel_size must be odd so Adapter HxW is preserved"
            )
        if not isinstance(adapter_bias, bool):
            raise TypeError(
                f"adapter_bias must be bool, got {type(adapter_bias).__name__}"
            )

        self.extractor = extractor
        self.validate = bool(validate)

        # Explicit Day-04 Adapter architecture variables.
        self.adapter_bottleneck_dim = int(adapter_bottleneck_dim)
        self.adapter_projection_dim = int(adapter_projection_dim)
        self.adapter_kernel_size = int(adapter_kernel_size)
        self.adapter_bias = bool(adapter_bias)
        self.gamma_init = float(gamma_init)

        self.adapters = nn.ModuleDict(
            {
                key: ResidualAdapter2d(
                    in_dim=channels,
                    bottleneck_dim=self.adapter_bottleneck_dim,
                    projection_dim=self.adapter_projection_dim,
                    kernel_size=self.adapter_kernel_size,
                    gamma_init=self.gamma_init,
                    bias=self.adapter_bias,
                )
                for key in DINO_FEATURE_KEYS
            }
        )

        self.fusion = (
            fusion
            if fusion is not None
            else MeanFusion(validate=validate)
        )
        self.decoder = (
            decoder
            if decoder is not None
            else BasicDecoder(channels)
        )

    @property
    def out_channels(self) -> int:
        """Channel width C of DINO/adapted/mean-fused feature maps."""
        return int(self.extractor.out_channels)

    @property
    def adapter_r(self) -> int:
        """Day-04 compact notation r = bottleneck width."""
        return self.adapter_bottleneck_dim

    @property
    def adapter_d(self) -> int:
        """Day-04 compact notation d = Adapter internal projection width."""
        return self.adapter_projection_dim

    @classmethod
    def from_dinov3(
        cls,
        *,
        repo_dir: str | Path,
        weights: str | Path,
        adapter_bottleneck_dim: int,
        adapter_projection_dim: int,
        model_name: str = "dinov3_vits16",
        blocks: tuple[int, int, int] = (4, 8, 12),
        norm: bool = True,
        adapter_kernel_size: int = 3,
        gamma_init: float = 0.0,
        adapter_bias: bool = True,
        fusion: nn.Module | None = None,
        decoder: nn.Module | None = None,
        validate: bool = True,
    ) -> "MSILA":
        """Build MS-ILA directly from the official DINOv3 extractor.

        The Adapter pair ``(r,d)`` is required explicitly so a caller cannot
        silently fall back to the removed reduction-based architecture.
        """
        extractor = DINOv3FeatureExtractor(
            repo_dir=repo_dir,
            weights=weights,
            model_name=model_name,
            blocks=blocks,
            norm=norm,
        )
        return cls(
            extractor=extractor,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_projection_dim=adapter_projection_dim,
            adapter_kernel_size=adapter_kernel_size,
            gamma_init=gamma_init,
            adapter_bias=adapter_bias,
            fusion=fusion,
            decoder=decoder,
            validate=validate,
        )

    def adapt_features(
        self,
        features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Apply one Day-04 residual Adapter to each DINO block feature."""
        if self.validate:
            validate_dino_features(
                features,
                require_same_shape=True,
            )

        adapted: dict[str, Tensor] = {}
        for key in DINO_FEATURE_KEYS:
            source = features[key]
            output = self.adapters[key](source)

            if self.validate:
                if output.shape != source.shape:
                    raise ContractError(
                        f"Adapter {key} changed feature shape: "
                        f"{tuple(source.shape)} -> {tuple(output.shape)}"
                    )
                if not bool(torch.isfinite(output).all()):
                    raise ContractError(
                        f"Adapter {key} output contains NaN or Inf"
                    )

            adapted[key] = output

        if self.validate:
            validate_dino_features(
                adapted,
                require_same_shape=True,
            )

        return adapted

    def forward_features(
        self,
        image: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Run DINO -> Day-04 Adapter -> MeanFusion and expose intermediates."""
        if self.validate:
            validate_image(image)

        features = self.extractor(image)

        if self.validate:
            validate_dino_features(
                features,
                require_same_shape=True,
            )

        adapted = self.adapt_features(features)
        fused = self.fusion(adapted)

        if self.validate:
            reference = adapted[DINO_FEATURE_KEYS[0]]
            if tuple(fused.shape) != tuple(reference.shape):
                raise ContractError(
                    "Day-1 MeanFusion must preserve [B,C,h,w]. "
                    f"expected={tuple(reference.shape)}, "
                    f"got={tuple(fused.shape)}"
                )
            if not bool(torch.isfinite(fused).all()):
                raise ContractError(
                    "Day-1 fused feature contains NaN or Inf"
                )

        return dict(features), adapted, fused

    def forward(
        self,
        image: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Run the complete Day-1 baseline to full-resolution anomaly logits."""
        features, adapted, fused = self.forward_features(image)

        logits = self.decoder(
            fused,
            output_size=(
                int(image.shape[-2]),
                int(image.shape[-1]),
            ),
        )

        if self.validate:
            validate_anomaly_logits(logits, image)

        if not return_trace:
            return logits

        trace: dict[str, object] = {
            "dino": features,
            "adapted": adapted,
            "fused": fused,
            "adapter_config": {
                "r": self.adapter_bottleneck_dim,
                "d": self.adapter_projection_dim,
                "kernel_size": self.adapter_kernel_size,
                "bias": self.adapter_bias,
                "gamma_init": self.gamma_init,
            },
        }
        return logits, trace

    def adapter_gammas(self) -> dict[str, float]:
        """Return scalar residual gates for QA/reporting."""
        return {
            key: float(
                self.adapters[key].gamma.detach().cpu().item()
            )
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


class MSILADay2Integrated(nn.Module):
    """Integrate a TV1 feature pipeline with the Day-2 TV2 head.

    Contract
    --------
    ``feature_pipeline(*args, **kwargs)`` must return either:

    1. a mapping containing exactly the six Day-2 tensors::

           local_b4, local_b8, local_b12,
           context_b4, context_b8, context_b12

       where every tensor has shape ``[B,d,h,w]``; or

    2. ``(features, trace)`` where ``features`` is the mapping above and
       ``trace`` is any mapping of TV1 debug information.

    No tensor is detached in this wrapper. Therefore gradients produced by
    AttentionFusion/Decoder are allowed to propagate back through TV1's
    Projection/Adapter modules, while a frozen DINO backbone remains frozen
    through its own ``requires_grad=False`` configuration.
    """

    def __init__(
        self,
        feature_pipeline: nn.Module,
        fusion_dim: int,
        *,
        head: nn.Module | None = None,
        fusion: nn.Module | None = None,
        decoder: nn.Module | None = None,
        output_size: tuple[int, int] = (512, 512),
        validate: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(feature_pipeline, nn.Module):
            raise TypeError(
                "feature_pipeline must be an nn.Module that returns "
                "the six Day-2 Local/Context features"
            )

        if head is not None and (fusion is not None or decoder is not None):
            raise ValueError(
                "Provide either a prebuilt head OR fusion/decoder overrides, "
                "not both."
            )

        self.feature_pipeline = feature_pipeline
        self.fusion_dim = int(fusion_dim)
        self.validate = bool(validate)

        self.head = (
            head
            if head is not None
            else MSILADay2Head(
                fusion_dim=self.fusion_dim,
                fusion=fusion,
                decoder=decoder,
                output_size=output_size,
                validate=validate,
            )
        )

    @staticmethod
    def _split_tv1_output(
        tv1_output: object,
    ) -> tuple[Mapping[str, Tensor], Mapping[str, object]]:
        """Normalize TV1 output without imposing TV1's internal architecture."""

        if isinstance(tv1_output, Mapping):
            return tv1_output, {}

        if (
            isinstance(tv1_output, tuple)
            and len(tv1_output) == 2
            and isinstance(tv1_output[0], Mapping)
            and isinstance(tv1_output[1], Mapping)
        ):
            return tv1_output[0], tv1_output[1]

        raise TypeError(
            "TV1 feature_pipeline must return either "
            "features: Mapping[str, Tensor] or "
            "(features, trace)."
        )

    def forward(
        self,
        *feature_args: object,
        output_size: tuple[int, int] | None = None,
        return_trace: bool = False,
        **feature_kwargs: object,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Run TV1 -> six-feature contract -> AttentionFusion -> Decoder."""

        tv1_output = self.feature_pipeline(
            *feature_args,
            **feature_kwargs,
        )

        features, tv1_trace = self._split_tv1_output(
            tv1_output
        )

        if self.validate:
            validate_multiview_features(
                features,
                expected_channels=self.fusion_dim,
                check_finite=True,
            )

        anomaly_logits, head_trace = self.head(
            features,
            output_size=output_size,
            return_trace=True,
        )

        if not return_trace:
            return anomaly_logits

        trace: dict[str, object] = {
            "tv1": dict(tv1_trace),
            "multiview_features": dict(features),
            **head_trace,
        }

        return anomaly_logits, trace

