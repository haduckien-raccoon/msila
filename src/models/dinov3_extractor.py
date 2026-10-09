"""
Frozen DINOv3 multi-layer feature extractor for paired Local/Context views.

Project contract
----------------
Upstream preprocessing provides two normalized RGB tensors:

    x_local   : [B, 3, 512, 512]
    x_context : [B, 3, 512, 512]

Although both tensors have the same network input size, ``x_context`` represents a
larger source-image field of view (FOV). This module extracts DINOv3 dense patch
features from human-readable Transformer blocks 4, 8 and 12:

    L4, L8, L12  for Local
    C4, C8, C12  for Context

For DINOv3 ViT-S/16 and 512x512 inputs, every returned tensor is
[B, 384, 32, 32]. The implementation itself does not hard-code C=384 and works
with compatible DINOv3 ViT backbones that expose the official
``get_intermediate_layers`` API.

Important implementation choices
--------------------------------
1. Human block numbers (4, 8, 12) are converted to official zero-based indices
   (3, 7, 11).
2. ``reshape=True`` asks the official DINOv3 API for dense [B,C,H/P,W/P] maps.
3. The backbone is permanently frozen and kept in eval mode.
4. ``torch.no_grad()`` is used instead of ``torch.inference_mode()`` because the
   frozen features are intended to feed trainable adapters/fusion/decoder layers.
5. Paired Local/Context extraction defaults to a *single concatenated backbone
   pass* for better throughput. A sequential mode is provided for lower peak
   memory.

References
----------
DINOv3 paper:
    Siméoni et al., "DINOv3", 2025, arXiv:2508.10104.
    https://arxiv.org/abs/2508.10104

Official implementation / API:
    https://github.com/facebookresearch/dinov3
    dinov3/models/vision_transformer.py::get_intermediate_layers

PyTorch autograd modes:
    https://docs.pytorch.org/docs/stable/generated/torch.no_grad
    https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from .backbone_registry import backbone_spec, validate_blocks


DEFAULT_BLOCKS: tuple[int, int, int] = (4, 8, 12)
PairStrategy = Literal["concat", "sequential"]

# Bridge between the extractor-facing Local/Context names and the fixed
# feature-cache schema used by data/feature_cache.py.
ONLINE_TO_CACHE_KEYS: dict[str, str] = {
    "L4": "local_b4",
    "L8": "local_b8",
    "L12": "local_b12",
    "C4": "context_b4",
    "C8": "context_b8",
    "C12": "context_b12",
}


def map_online_features_to_cache(
    features: dict[str, Tensor],
    *,
    to_cpu: bool = False,
    blocks: Sequence[int] = DEFAULT_BLOCKS,
) -> dict[str, Tensor]:
    """Map online extractor outputs to the fixed feature-cache key names.

    Parameters
    ----------
    features:
        Output of :meth:`DINOv3FeatureExtractor.extract_local_context`, i.e.
        ``L4, L8, L12, C4, C8, C12``.
    to_cpu:
        If True, detach and move every tensor to contiguous CPU storage.
        This is useful immediately before serialization. For numerical online
        vs cache tests, leaving tensors on their current device is also valid.

    Returns
    -------
    dict[str, Tensor]
        ``local_b4, local_b8, local_b12, context_b4, context_b8, context_b12``.
    """
    # The v1 cache names are stable shallow/middle/deep slots. Their physical
    # block IDs live in the signed producer metadata, including for L/H+.
    mapping = {f"{view}{block}": f"{prefix}_b{slot}"
               for view, prefix in (("L", "local"), ("C", "context"))
               for slot, block in zip(DEFAULT_BLOCKS, blocks)}
    missing = [key for key in mapping if key not in features]
    if missing:
        raise KeyError(f"Online feature output missing required keys: {missing}")

    out: dict[str, Tensor] = {}
    for online_key, cache_key in mapping.items():
        tensor = features[online_key]
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{online_key} must be torch.Tensor, got {type(tensor)!r}."
            )
        if to_cpu:
            tensor = tensor.detach().to(device="cpu").contiguous()
        out[cache_key] = tensor
    return out


class DINOv3FeatureExtractor(nn.Module):
    """Frozen DINOv3 ViT feature extractor for Local/Context multi-view input.

    Parameters
    ----------
    repo_dir:
        Local checkout of the official ``facebookresearch/dinov3`` repository.
        Loading with ``source='local'`` makes the source revision explicit and
        prevents silently changing the implementation during experiments.
    weights:
        Local checkpoint path or a weight argument accepted by the official
        DINOv3 torch.hub entrypoint.
    model_name:
        Official DINOv3 ViT torch.hub model name, e.g. ``dinov3_vits16``.
    blocks:
        Exactly three human-readable, one-based Transformer block numbers.
        Project default: ``(4, 8, 12)``.
    norm:
        Forwarded to DINOv3 ``get_intermediate_layers``. ``True`` applies the
        official final normalization to the selected intermediate features.
    check_finite:
        If ``True``, check input/output tensors for NaN/Inf. Useful for smoke
        tests and debugging, but disabled by default because repeated GPU
        finite checks cause synchronization overhead in the training loop.
    """

    def __init__(
        self,
        repo_dir: str | Path,
        weights: str | Path,
        model_name: str = "dinov3_vits16",
        blocks: Sequence[int] | None = None,
        norm: bool = True,
        check_finite: bool = False,
    ) -> None:
        super().__init__()

        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.weights = str(weights)
        self.model_name = str(model_name)
        spec = backbone_spec(self.model_name)
        self.blocks = validate_blocks(spec.blocks if blocks is None else blocks, spec.depth)
        self.norm = bool(norm)
        self.check_finite = bool(check_finite)

        self._validate_config()

        # Reproducible source: use the user's pinned local DINOv3 checkout.
        local_weights = Path(self.weights).expanduser()
        if local_weights.is_file():
            # Official local-path loading uses load_state_dict_from_url(), whose
            # basename cache can alias different checkpoint files. Construct the
            # official architecture (including ViT-L SAT filename routing), then
            # read this exact local file; never trust another cached basename.
            self.weights = str(local_weights.resolve())
            self.backbone = torch.hub.load(
                repo_or_dir=str(self.repo_dir), model=self.model_name, source="local",
                weights=self.weights, pretrained=False,
            )
            state = torch.load(self.weights, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        else:
            self.backbone = torch.hub.load(
                repo_or_dir=str(self.repo_dir), model=self.model_name,
                source="local", weights=self.weights,
            )

        self._validate_backbone()
        actual = (self.out_channels, len(self.backbone.blocks), self.patch_size)
        expected = (spec.channels, spec.depth, spec.patch_size)
        if actual != expected:
            raise ValueError(f"Loaded {self.model_name} architecture mismatch: {actual} != {expected}")

        # Official get_intermediate_layers() consumes zero-based indices.
        self.block_indices = tuple(block - 1 for block in self.blocks)

        self.freeze_backbone()

    # ------------------------------------------------------------------
    # Configuration / backbone validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repository not found: {self.repo_dir}"
            )

        if not (self.repo_dir / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"{self.repo_dir} does not look like a DINOv3 torch.hub "
                "repository: hubconf.py was not found."
            )

        if not self.model_name:
            raise ValueError("model_name must be non-empty.")

        if len(self.blocks) != 3:
            raise ValueError(
                "Exactly 3 blocks are required by the project contract; "
                f"received {self.blocks}."
            )

        if len(set(self.blocks)) != 3:
            raise ValueError(f"Block numbers must be unique: {self.blocks}.")

        if any(block <= 0 for block in self.blocks):
            raise ValueError(
                "Block numbers are human-readable 1-based indices and must be > 0."
            )

        if tuple(sorted(self.blocks)) != self.blocks:
            raise ValueError(
                f"Block numbers must be strictly increasing; received {self.blocks}."
            )

    def _validate_backbone(self) -> None:
        if not hasattr(self.backbone, "blocks"):
            raise TypeError(
                "Loaded backbone does not expose Transformer blocks; "
                "a DINOv3 ViT backbone is required."
            )

        if not callable(getattr(self.backbone, "get_intermediate_layers", None)):
            raise TypeError(
                "Loaded backbone does not implement get_intermediate_layers()."
            )

        if not hasattr(self.backbone, "patch_size"):
            raise TypeError("Loaded backbone does not expose patch_size.")

        depth = len(self.backbone.blocks)
        if max(self.blocks) > depth:
            raise ValueError(
                f"Requested block {max(self.blocks)}, but backbone depth is only {depth}."
            )

    # ------------------------------------------------------------------
    # Frozen-backbone behavior
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all DINOv3 parameters and force evaluation mode."""
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep DINOv3 in eval mode even if a parent model calls ``train()``."""
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Backbone metadata
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self.backbone.blocks)

    @property
    def patch_size(self) -> int:
        patch_size = self.backbone.patch_size

        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise RuntimeError(
                    f"Expected square patch size, received {patch_size}."
                )
            return int(patch_size[0])

        return int(patch_size)

    @property
    def out_channels(self) -> int:
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)

        raise AttributeError(
            "Cannot determine DINOv3 feature dimension: backbone.embed_dim missing."
        )

    def backbone_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.backbone.parameters())

    # ------------------------------------------------------------------
    # Tensor validation
    # ------------------------------------------------------------------

    def _validate_input(self, x: Tensor, *, name: str = "x") -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(x)!r}.")

        if x.ndim != 4:
            raise ValueError(
                f"{name}: expected [B,3,H,W], received {tuple(x.shape)}."
            )

        if x.shape[0] <= 0:
            raise ValueError(f"{name}: batch dimension must be > 0.")

        if x.shape[1] != 3:
            raise ValueError(
                f"{name}: DINOv3 expects RGB input with C=3, got C={x.shape[1]}."
            )

        if not x.is_floating_point():
            raise TypeError(
                f"{name}: expected a normalized floating-point tensor; got {x.dtype}."
            )

        h, w = x.shape[-2:]
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"{name}: H and W must be divisible by patch_size={p}; "
                f"received H={h}, W={w}."
            )

        if self.check_finite and not bool(torch.isfinite(x).all()):
            raise ValueError(f"{name}: input contains NaN or Inf.")

    def _validate_pair(self, x_local: Tensor, x_context: Tensor) -> None:
        self._validate_input(x_local, name="x_local")
        self._validate_input(x_context, name="x_context")

        if x_local.shape != x_context.shape:
            raise ValueError(
                "Local and Context must have the same network-input shape for "
                "the paired project contract. "
                f"Got local={tuple(x_local.shape)}, context={tuple(x_context.shape)}."
            )

        if x_local.device != x_context.device:
            raise ValueError(
                f"Local/Context device mismatch: {x_local.device} vs {x_context.device}."
            )

        if x_local.dtype != x_context.dtype:
            raise ValueError(
                f"Local/Context dtype mismatch: {x_local.dtype} vs {x_context.dtype}."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def _extract(self, x: Tensor) -> dict[str, Tensor]:
        self._validate_input(x)

        # no_grad is intentionally used instead of inference_mode because these
        # tensors will be consumed by trainable downstream modules.
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                x,
                n=self.block_indices,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=self.norm,
            )

        features = tuple(features)
        if len(features) != 3:
            raise RuntimeError(
                "DINOv3 extractor contract violated: expected exactly "
                f"3 features, received {len(features)}."
            )

        self._validate_outputs(x=x, features=features)

        return {
            f"b{block}": feature
            for block, feature in zip(self.blocks, features)
        }

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Extract one view, preserving compatibility with the day-1 extractor."""
        return self._extract(x)

    def extract_local_context(
        self,
        x_local: Tensor,
        x_context: Tensor,
        *,
        strategy: PairStrategy = "concat",
    ) -> dict[str, Tensor]:
        """Extract Local and Context features from the *same frozen backbone*.

        Parameters
        ----------
        x_local, x_context:
            Normalized RGB tensors with identical network-input shape. The
            project uses [B,3,512,512] for both; Context encodes the larger FOV
            before resize, not a larger DINOv3 input tensor.
        strategy:
            ``"concat"`` (default): concatenate Local/Context along the batch
            dimension and run one backbone call. This improves throughput when
            memory permits.

            ``"sequential"``: run two backbone calls. Same numerical contract,
            lower peak activation memory, slightly more overhead.

        Returns
        -------
        dict[str, Tensor]
            ``L4, L8, L12, C4, C8, C12`` for the default block set.
        """
        self._validate_pair(x_local, x_context)

        if strategy not in ("concat", "sequential"):
            raise ValueError(
                f"strategy must be 'concat' or 'sequential', got {strategy!r}."
            )

        if strategy == "concat":
            batch = x_local.shape[0]
            pair = torch.cat((x_local, x_context), dim=0)
            pair_features = self._extract(pair)

            out: dict[str, Tensor] = {}
            for block in self.blocks:
                f = pair_features[f"b{block}"]
                out[f"L{block}"] = f[:batch].contiguous()
                out[f"C{block}"] = f[batch:].contiguous()
            return out

        local_features = self._extract(x_local)
        context_features = self._extract(x_context)

        out = {}
        for block in self.blocks:
            out[f"L{block}"] = local_features[f"b{block}"]
            out[f"C{block}"] = context_features[f"b{block}"]
        return out

    def extract_online_cache_features(
        self,
        x_local: Tensor,
        x_context: Tensor,
        *,
        strategy: PairStrategy = "concat",
        to_cpu: bool = False,
    ) -> dict[str, Tensor]:
        """Run the *real online DINOv3 extractor* and return cache-schema keys.

        This is the bridge needed by ``tests/test_feature_cache.py`` for the
        numerical gate:

            image/preprocessed pair -> online DINOv3 -> six tensors
                                                |
                                                +-> compare with cache reload

        No cached tensor is read here. The backbone is actually executed via
        :meth:`extract_local_context`.

        Returns exactly:
            local_b4, local_b8, local_b12,
            context_b4, context_b8, context_b12

        ``to_cpu=False`` is preferred for normal online use. Set ``to_cpu=True``
        immediately before cache serialization if desired; FeatureCacheWriter
        also normalizes tensors to CPU itself.
        """
        online = self.extract_local_context(
            x_local=x_local,
            x_context=x_context,
            strategy=strategy,
        )
        return map_online_features_to_cache(online, to_cpu=to_cpu, blocks=self.blocks)

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _validate_outputs(
        self,
        *,
        x: Tensor,
        features: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        expected_h = x.shape[-2] // self.patch_size
        expected_w = x.shape[-1] // self.patch_size

        for block, feature in zip(self.blocks, features):
            if not isinstance(feature, Tensor):
                raise RuntimeError(
                    f"Block {block}: expected Tensor, got {type(feature)!r}."
                )

            if feature.ndim != 4:
                raise RuntimeError(
                    f"Block {block}: expected [B,C,H,W], got {tuple(feature.shape)}."
                )

            expected = (
                x.shape[0],
                self.out_channels,
                expected_h,
                expected_w,
            )
            if tuple(feature.shape) != expected:
                raise RuntimeError(
                    f"Block {block}: expected shape {expected}, "
                    f"got {tuple(feature.shape)}."
                )

            if self.check_finite and not bool(torch.isfinite(feature).all()):
                raise RuntimeError(f"Block {block}: feature contains NaN or Inf.")

    def extra_repr(self) -> str:
        return (
            f"model={self.model_name}, blocks={self.blocks}, "
            f"indices={self.block_indices}, patch_size={self.patch_size}, "
            f"out_channels={self.out_channels}, norm={self.norm}, "
            f"frozen={self.backbone_is_frozen()}"
        )


def build_online_extractor(
    *,
    repo_dir: str | Path,
    weights: str | Path,
    device: str | torch.device | None = None,
    model_name: str = "dinov3_vits16",
    blocks: Sequence[int] | None = None,
    norm: bool = True,
    check_finite: bool = True,
) -> DINOv3FeatureExtractor:
    """Construct the real frozen online extractor used by integration tests.

    Paths are explicit arguments rather than hard-coded project-specific values.
    This avoids silently loading the wrong DINOv3 source revision/checkpoint.

    Examples
    --------
    >>> extractor = build_online_extractor(
    ...     repo_dir="third_party/dinov3",
    ...     weights="checkpoints/dinov3_vits16_pretrain_lvd1689m.pth",
    ... )
    >>> online = extractor.extract_online_cache_features(x_local, x_context)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    extractor = DINOv3FeatureExtractor(
        repo_dir=repo_dir,
        weights=weights,
        model_name=model_name,
        blocks=blocks,
        norm=norm,
        check_finite=check_finite,
    ).to(device)

    # Explicitly re-assert the frozen/eval invariant after .to(device).
    extractor.freeze_backbone()
    extractor.eval()

    if not extractor.backbone_is_frozen():
        raise RuntimeError("DINOv3 backbone must be frozen for online cache extraction.")

    return extractor
