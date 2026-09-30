#!/usr/bin/env python3
"""Run the Day-1 Architecture QA gate and write day01_report.json.

This script intentionally performs no training and computes no anomaly metric.
It verifies only structural correctness of:
    image -> DINOv3 -> Adapter -> MeanFusion -> BasicDecoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.models.contracts import DINO_FEATURE_KEYS
from src.models.msila import MSILA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MS-ILA Day-1 smoke test")
    p.add_argument("--dinov3-repo", type=Path, default=Path("/content/dinov3"))
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/content/checkpoints/dinov3_vits16_pretrain_lvd1689m.pth"
        ),
    )
    p.add_argument("--model-name", default="dinov3_vits16")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--output", type=Path, default=Path("day01_report.json"))
    return p.parse_args()


def shape(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.shape]


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def build_model(args: argparse.Namespace, device: torch.device) -> MSILA:
    return MSILA.from_dinov3(
        repo_dir=args.dinov3_repo,
        weights=args.checkpoint,
        model_name=args.model_name,
        blocks=(4, 8, 12),
        norm=True,
        adapter_reduction=4,
        adapter_kernel_size=3,
        gamma_init=0.0,
        validate=True,
    ).to(device).eval()


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Day-1 Architecture QA only; no training/AU-PRO",
        "device": str(device),
        "dinov3_repo": str(args.dinov3_repo),
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "checks": {},
        "status": "FAIL",
    }

    if not (args.dinov3_repo / "hubconf.py").exists():
        raise FileNotFoundError(
            f"Official/local DINOv3 repository not found: {args.dinov3_repo}"
        )
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {args.checkpoint}")

    torch.manual_seed(0)
    model = build_model(args, device)

    x = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )

    with torch.no_grad():
        y, trace = model(x, return_trace=True)

    dino = trace["dino"]
    adapted = trace["adapted"]
    fused = trace["fused"]

    identity_error = {
        key: float((adapted[key] - dino[key]).abs().max().item())
        for key in DINO_FEATURE_KEYS
    }

    gammas = model.adapter_gammas()
    expected_output = [
        args.batch_size,
        1,
        args.image_size,
        args.image_size,
    ]

    checks = {
        "dino_load": True,
        "correct_blocks_4_8_12": tuple(model.extractor.blocks) == (4, 8, 12),
        "dino_frozen": bool(model.extractor.backbone_is_frozen()),
        "gamma_init_zero": all(v == 0.0 for v in gammas.values()),
        "adapter_identity": all(v <= 1e-7 for v in identity_error.values()),
        "mean_fusion_decoder": finite(fused) and finite(y),
        "full_forward": shape(y) == expected_output and finite(y),
    }

    report.update(
        {
            "input_shape": shape(x),
            "dino_shapes": {key: shape(dino[key]) for key in DINO_FEATURE_KEYS},
            "adapter_shapes": {
                key: shape(adapted[key]) for key in DINO_FEATURE_KEYS
            },
            "mean_fusion_shape": shape(fused),
            "decoder_output_shape": shape(y),
            "nan": bool(torch.isnan(y).any().item()),
            "inf": bool(torch.isinf(y).any().item()),
            "dino_frozen": bool(model.extractor.backbone_is_frozen()),
            "adapter_gamma": gammas,
            "adapter_identity_max_abs_error": identity_error,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
    )

    return report


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = run(args)
    except Exception as exc:  # Always leave a machine-readable failure report.
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Day-1 Architecture QA only; no training/AU-PRO",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {args.output.resolve()}")

    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
