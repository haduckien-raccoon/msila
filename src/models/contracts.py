#Nó sẽ damr nhận việc kiểm tra số chiều và báo lỗi
# models/contracts.py

from __future__ import annotations

from typing import Final, Mapping, NotRequired, TypedDict

import torch
from torch import Tensor


# ============================================================
# 1. Canonical feature names
# ============================================================

DINO_FEATURE_KEYS: Final[tuple[str, str, str]] = (
    "b4",
    "b8",
    "b12",
)


class DinoFeatures(TypedDict):
    """
    Standard output interface of DINOv3 extractor.

    Every feature MUST already be BCHW:
        [B, C, h, w]
    """
    b4: Tensor
    b8: Tensor
    b12: Tensor


class ContractError(ValueError):
    """Raised when a tensor violates the MS-ILA tensor contract."""
    pass


# ============================================================
# 2. Internal helpers
# ============================================================

def _validate_bchw(
    x: Tensor,
    name: str,
    *,
    check_finite: bool = True,
) -> None:

    if not isinstance(x, Tensor):
        raise ContractError(
            f"{name}: expected torch.Tensor, got {type(x)}"
        )

    if x.ndim != 4:
        raise ContractError(
            f"{name}: expected BCHW tensor [B,C,H,W], "
            f"got shape={tuple(x.shape)}"
        )

    B, C, H, W = x.shape

    if B <= 0 or C <= 0 or H <= 0 or W <= 0:
        raise ContractError(
            f"{name}: invalid shape={tuple(x.shape)}"
        )

    if not x.is_floating_point():
        raise ContractError(
            f"{name}: expected floating-point tensor, got {x.dtype}"
        )

    if check_finite and not torch.isfinite(x).all().item():
        raise ContractError(
            f"{name}: contains NaN or Inf"
        )


# ============================================================
# 3. Input image contract
# ============================================================

def validate_image(
    image: Tensor,
    *,
    check_finite: bool = True,
) -> None:
    """
    Input:
        image: [B, 3, H, W]
    """

    _validate_bchw(
        image,
        "input_image",
        check_finite=check_finite,
    )

    if image.shape[1] != 3:
        raise ContractError(
            "input_image: expected RGB input with C=3, "
            f"got C={image.shape[1]}"
        )


# ============================================================
# 4. DINO feature contract
# ============================================================

def validate_dino_features(
    features: Mapping[str, Tensor],
    *,
    require_same_shape: bool = True,
    check_finite: bool = True,
) -> None:
    """
    Expected:
        {
            "b4":  [B,C,h,w],
            "b8":  [B,C,h,w],
            "b12": [B,C,h,w],
        }

    Day-1 MeanFusion requires identical shapes.
    """

    expected = set(DINO_FEATURE_KEYS)
    received = set(features.keys())

    if received != expected:
        missing = expected - received
        extra = received - expected

        raise ContractError(
            "DINO features have invalid keys. "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    reference = features["b4"]

    for key in DINO_FEATURE_KEYS:
        x = features[key]

        _validate_bchw(
            x,
            f"DINO[{key}]",
            check_finite=check_finite,
        )

        if x.device != reference.device:
            raise ContractError(
                f"DINO[{key}]: device mismatch: "
                f"{x.device} != {reference.device}"
            )

        if x.dtype != reference.dtype:
            raise ContractError(
                f"DINO[{key}]: dtype mismatch: "
                f"{x.dtype} != {reference.dtype}"
            )

    if require_same_shape:
        ref_shape = reference.shape

        for key in DINO_FEATURE_KEYS[1:]:
            if features[key].shape != ref_shape:
                raise ContractError(
                    "Day-1 MeanFusion requires identical feature shapes: "
                    f"b4={tuple(reference.shape)}, "
                    f"{key}={tuple(features[key].shape)}"
                )


# ============================================================
# 5. Fusion output contract
# ============================================================

def validate_fused_feature(
    fused: Tensor,
    features: Mapping[str, Tensor],
    *,
    check_finite: bool = True,
) -> None:
    """
    Mean Fusion output:
        [B,C,h,w]

    Must preserve feature shape during Day 1.
    """

    _validate_bchw(
        fused,
        "fused_feature",
        check_finite=check_finite,
    )

    expected_shape = features["b4"].shape

    if fused.shape != expected_shape:
        raise ContractError(
            "fused_feature: shape mismatch. "
            f"expected={tuple(expected_shape)}, "
            f"got={tuple(fused.shape)}"
        )


# ============================================================
# 6. Decoder output contract
# ============================================================

def validate_anomaly_logits(
    logits: Tensor,
    input_image: Tensor,
    *,
    check_finite: bool = True,
) -> None:
    """
    Decoder output:
        [B, 1, H, W]
    """

    _validate_bchw(
        logits,
        "anomaly_logits",
        check_finite=check_finite,
    )

    B, _, H, W = input_image.shape

    expected_shape = (B, 1, H, W)

    if tuple(logits.shape) != expected_shape:
        raise ContractError(
            "anomaly_logits: invalid output shape. "
            f"expected={expected_shape}, "
            f"got={tuple(logits.shape)}"
        )

# ============================================================
# Day-2 Local-Context feature names
# ============================================================

MULTIVIEW_FEATURE_KEYS: Final[tuple[str, ...]] = (
    "local_b4",
    "local_b8",
    "local_b12",
    "context_b4",
    "context_b8",
    "context_b12",
)


class MultiViewFeatures(TypedDict):
    """
    Day-2 aligned + projected features.

    Every tensor MUST be:
        [B, d, h, w]

    Context features MUST already be aligned to Local coordinates
    before reaching this contract.
    """
    local_b4: Tensor
    local_b8: Tensor
    local_b12: Tensor

    context_b4: Tensor
    context_b8: Tensor
    context_b12: Tensor

# ============================================================
# Day-2 Local-Context feature contract
# ============================================================

def validate_multiview_features(
    features: Mapping[str, Tensor],
    *,
    expected_channels: int | None = None,
    check_finite: bool = True,
) -> None:
    """
    Validate the six aligned/projected Day-2 feature sources.

    Expected:
        {
            "local_b4":    [B,d,h,w],
            "local_b8":    [B,d,h,w],
            "local_b12":   [B,d,h,w],
            "context_b4":  [B,d,h,w],
            "context_b8":  [B,d,h,w],
            "context_b12": [B,d,h,w],
        }

    All six tensors MUST:
        - use BCHW layout
        - have identical shapes
        - have the same device
        - have the same dtype
        - contain only finite values
    """

    expected = set(MULTIVIEW_FEATURE_KEYS)
    received = set(features.keys())

    # --------------------------------------------------------
    # 1. Exactly six required sources
    # --------------------------------------------------------
    if received != expected:
        missing = expected - received
        extra = received - expected

        raise ContractError(
            "Day-2 multiview features have invalid keys. "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # Reference tensor
    reference = features["local_b4"]

    _validate_bchw(
        reference,
        "multiview[local_b4]",
        check_finite=check_finite,
    )

    ref_shape = reference.shape
    ref_device = reference.device
    ref_dtype = reference.dtype

    # --------------------------------------------------------
    # 2. Validate all six features
    # --------------------------------------------------------
    for key in MULTIVIEW_FEATURE_KEYS:
        x = features[key]

        _validate_bchw(
            x,
            f"multiview[{key}]",
            check_finite=check_finite,
        )

        # Same [B,d,h,w]
        if x.shape != ref_shape:
            raise ContractError(
                "Day-2 fusion requires all six features "
                "to have identical [B,d,h,w] shapes. "
                f"reference local_b4={tuple(ref_shape)}, "
                f"{key}={tuple(x.shape)}"
            )

        # Same device
        if x.device != ref_device:
            raise ContractError(
                f"multiview[{key}]: device mismatch: "
                f"{x.device} != {ref_device}"
            )

        # Same dtype
        if x.dtype != ref_dtype:
            raise ContractError(
                f"multiview[{key}]: dtype mismatch: "
                f"{x.dtype} != {ref_dtype}"
            )

    # --------------------------------------------------------
    # 3. Optional: lock fusion dimension d
    # --------------------------------------------------------
    if expected_channels is not None:
        actual_channels = reference.shape[1]

        if actual_channels != expected_channels:
            raise ContractError(
                "Day-2 fusion dimension mismatch. "
                f"expected d={expected_channels}, "
                f"got d={actual_channels}"
            )


# G2 E1--E5 model boundary. The existing G1 tile collator already emits
# image/mask/meta; optional context/view_meta are reserved for TV2 consumers.
G2_OUTPUT_SIZE: Final[tuple[int, int]] = (512, 512)
G2_FUSION_METHODS: Final[tuple[str, ...]] = (
    "mean", "concat", "weighted_sum", "gated", "attention",
)


class G2Batch(TypedDict):
    image: Tensor
    mask: NotRequired[Tensor]
    meta: NotRequired[list[Mapping[str, object]]]
    context: NotRequired[Tensor]
    view_meta: NotRequired[list[Mapping[str, object]]]


def validate_g2_batch(
    batch: Mapping[str, object], *, require_target: bool = True,
) -> None:
    """Lock 512px normalized RGB, binary targets and per-sample metadata.

    Inference may omit mask/meta. Training requires both. Geometry generation
    and normalization remain upstream responsibilities, outside the Model.
    """
    if not isinstance(batch, Mapping) or "image" not in batch:
        raise ContractError("G2 batch requires an 'image' tensor")
    image = batch["image"]
    validate_image(image)
    if tuple(image.shape[-2:]) != G2_OUTPUT_SIZE:
        raise ContractError("G2 image must be [B,3,512,512]")

    if require_target and not {"mask", "meta"} <= batch.keys():
        raise ContractError("G2 training batch requires 'mask' and 'meta'")
    if "mask" in batch:
        mask = batch["mask"]
        _validate_bchw(mask, "G2 mask")
        if tuple(mask.shape) != (image.shape[0], 1, *G2_OUTPUT_SIZE):
            raise ContractError("G2 mask must be [B,1,512,512]")
        if mask.device != image.device:
            raise ContractError("G2 mask/image device mismatch")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ContractError("G2 mask must be binary (0=normal, 1=anomaly)")

    if "context" in batch:
        context = batch["context"]
        validate_image(context)
        if context.shape != image.shape:
            raise ContractError("G2 context must match image [B,3,512,512]")
        if context.device != image.device or context.dtype != image.dtype:
            raise ContractError("G2 context/image device or dtype mismatch")

    for key in ("meta", "view_meta"):
        if key in batch:
            records = batch[key]
            if (not isinstance(records, list) or len(records) != image.shape[0]
                    or any(not isinstance(record, Mapping) for record in records)):
                raise ContractError(f"G2 {key} must be a list of B metadata mappings")


def g2_input_image(image_or_batch: Tensor | Mapping[str, object]) -> Tensor:
    """Accept the G2 batch boundary and preserve legacy E1 tensor calls."""
    if isinstance(image_or_batch, Mapping):
        validate_g2_batch(image_or_batch, require_target=False)
        return image_or_batch["image"]
    validate_image(image_or_batch)
    return image_or_batch


def validate_g2_fused_feature(
    fused: Tensor, features: Mapping[str, Tensor], *, expected_channels: int,
) -> None:
    """Every G2 fusion method hands one [B,C,H,W] tensor to the decoder.

    C is the common downstream fusion_dim, independent of Adapter d. A concat
    method must restore C internally before returning its feature tensor.
    """
    validate_multiview_features(features, expected_channels=expected_channels)
    _validate_bchw(fused, "G2 fused feature")
    reference = features[MULTIVIEW_FEATURE_KEYS[0]]
    if fused.shape != reference.shape:
        raise ContractError("G2 Fusion output must preserve [B,C,H,W]")
    if fused.device != reference.device or fused.dtype != reference.dtype:
        raise ContractError("G2 Fusion output device or dtype mismatch")

