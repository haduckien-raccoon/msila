#Nó sẽ damr nhận việc kiểm tra số chiều và báo lỗi
# models/contracts.py

from __future__ import annotations

from typing import Final, Mapping, TypedDict

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