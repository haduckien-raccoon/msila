"""Tiny-defect diagnostics. New file: src/eval/tiny_analysis.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from src.eval.evaluator import MVTEC_AD2_CATEGORIES, MVTEC_AD2_SPLITS
from src.metrics.aupro import aupro

CANDIDATES = ("R0", "R1", "R2")
MAX_FPR = 0.05
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)  # Fixed 8-connectivity.


def load_array(base: Path, value: str) -> np.ndarray:
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def binary_mask(value: np.ndarray) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("GT must be a non-empty 2-D mask.")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
        raise ValueError("GT must contain finite binary values.")
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("GT encoding must be {0,1} or {0,255}.")
    return mask != 0


def defect_regions(gt: np.ndarray, tiny_area_px: int) -> list[dict]:
    """Classify GT components once; no prediction-dependent region selection."""
    if isinstance(tiny_area_px, bool) or not isinstance(tiny_area_px, int) or tiny_area_px < 1:
        raise ValueError("tiny_area_px must be a positive integer.")
    labels, count = label(binary_mask(gt), structure=CONNECTIVITY)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    return [
        {
            "region_id": region_id,
            "area_px": int(areas[region_id]),
            "is_tiny": bool(areas[region_id] <= tiny_area_px),
            "indices": np.flatnonzero(labels.ravel() == region_id),
        }
        for region_id in range(1, count + 1)
    ]


def region_aupro(normal_parts: list, region_parts: list) -> tuple[float | None, str]:
    """Reuse the existing AU-PRO with original background and selected regions."""
    if not region_parts:
        return None, "NO_REGIONS"
    normal_parts = [part for part in normal_parts if part.size]
    if not normal_parts:
        return None, "NO_NORMAL_PIXELS"

    # Score packing preserves the exact pixel sets and region weights.
    # Each selected GT component is one all-foreground array (one component).
    # Original background arrays contain only actual GT==0 pixels.
    maps = [part.reshape(1, -1) for part in normal_parts + region_parts]
    masks = [np.zeros((1, part.size), dtype=bool) for part in normal_parts]
    masks += [np.ones((1, part.size), dtype=bool) for part in region_parts]
    return float(aupro(maps, masks, max_fpr=MAX_FPR)["aupro"]), "OK"


def analyze(manifest_path: Path, protocol_path: Path) -> list[dict]:
    manifest_path, protocol_path = Path(manifest_path).resolve(), Path(protocol_path).resolve()
    manifest_raw, protocol_raw = manifest_path.read_bytes(), protocol_path.read_bytes()
    manifest, protocol = json.loads(manifest_raw), json.loads(protocol_raw)

    fields = {
        "tiny_area_px", "area_unit", "connectivity", "max_fpr",
        "locked_before_candidate_results", "threshold_basis",
    }
    if set(protocol) != fields:
        raise ValueError(
            "Use exactly one global tiny protocol; candidate-specific settings are not allowed."
        )

    area = protocol["tiny_area_px"]
    if isinstance(area, bool) or not isinstance(area, int) or area < 1:
        raise ValueError("Lock a positive integer tiny_area_px before evaluation.")
    if protocol.get("area_unit") != "original_image_pixels" or protocol.get("connectivity") != 8:
        raise ValueError("Protocol requires original_image_pixels and 8-connectivity.")
    if protocol.get("max_fpr") != MAX_FPR:
        raise ValueError("Protocol max_fpr must be 0.05.")
    if protocol.get("locked_before_candidate_results") is not True:
        raise ValueError("Confirm the area threshold was locked before candidate results.")
    if not isinstance(protocol.get("threshold_basis"), str) or not protocol["threshold_basis"].strip():
        raise ValueError("Record the prediction-independent threshold selection basis.")

    categories, samples, split = manifest["categories"], manifest["samples"], manifest["split"]
    if not categories or len(categories) != len(set(categories)) or not samples:
        raise ValueError("Provide unique categories and a non-empty common sample list.")
    if set(categories) - set(MVTEC_AD2_CATEGORIES):
        raise ValueError("Use canonical MVTec AD 2 category names.")
    if split not in MVTEC_AD2_SPLITS or split in {"test_private", "test_private_mixed"}:
        raise ValueError("Use one locked split with available local GT.")
    if {s["category"] for s in samples} != set(categories):
        raise ValueError("Sample categories differ from the declared category set.")

    ids = [s["image_id"] for s in samples]
    if not all(isinstance(i, str) and i and i == i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Sample image_id values must be non-empty and unique.")
    if any(set(s["maps"]) != set(CANDIDATES) for s in samples):
        raise ValueError("Each sample must have exactly R0/R1/R2 prediction paths.")

    norms = manifest["normalization_by_candidate"]
    if set(norms) != set(CANDIDATES) or not all(isinstance(v, str) and v.strip() for v in norms.values()):
        raise ValueError("Declare the existing normalization protocol for R0/R1/R2.")
    if len(set(norms.values())) != 1:
        raise ValueError("R0/R1/R2 must use the same normalization protocol.")

    rows = []
    metric_source = Path(aupro.__code__.co_filename)
    metric_sha = hashlib.sha256(metric_source.read_bytes()).hexdigest()

    for category in categories:
        plans = []
        for sample in (s for s in samples if s["category"] == category):
            gt = binary_mask(load_array(manifest_path.parent, sample["gt_mask"]))
            hw = sample["original_hw"]
            if len(hw) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in hw):
                raise ValueError(f"Invalid original_hw: {sample['image_id']}")
            if gt.shape != tuple(hw):
                raise ValueError(f"GT must match original_hw: {sample['image_id']}")
            plans.append((sample, gt, defect_regions(gt, area)))

        n_tiny = sum(r["is_tiny"] for _, _, regions in plans for r in regions)
        n_non_tiny = sum(not r["is_tiny"] for _, _, regions in plans for r in regions)
        n_normal = sum(int((~gt).sum()) for _, gt, _ in plans)

        for candidate in CANDIDATES:
            normal_parts, tiny_parts, non_tiny_parts = [], [], []

            for sample, gt, regions in plans:
                score = load_array(manifest_path.parent, sample["maps"][candidate])
                if score.shape != gt.shape or score.dtype.kind not in "uif":
                    raise ValueError(f"Invalid probability map: {candidate}/{sample['image_id']}")
                if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
                    raise ValueError(
                        f"Probability map must be finite in [0,1]: {candidate}/{sample['image_id']}"
                    )

                # Non-tiny defects NEVER become background.
                normal_parts.append(score[~gt])

                for region in regions:
                    parts = tiny_parts if region["is_tiny"] else non_tiny_parts
                    parts.append(score.ravel()[region["indices"]])

            tiny_value, tiny_status = region_aupro(normal_parts, tiny_parts)
            non_tiny_value, non_tiny_status = region_aupro(normal_parts, non_tiny_parts)

            rows.append({
                "candidate": candidate,
                "category": category,
                "split": split,
                "tiny_area_px": area,
                "tiny_rule": "area_px <= tiny_area_px",
                "connectivity": 8,
                "max_fpr": MAX_FPR,
                "n_images": len(plans),
                "n_tiny_regions": n_tiny,
                "n_non_tiny_regions": n_non_tiny,
                "n_normal_pixels": n_normal,
                "tiny_aupro_0.05": tiny_value,
                "tiny_status": tiny_status,
                "non_tiny_aupro_0.05": non_tiny_value,
                "non_tiny_status": non_tiny_status,
                "normalization_protocol": norms[candidate],
                "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "aupro_source_sha256": metric_sha,
            })

    if metric_sha != hashlib.sha256(metric_source.read_bytes()).hexdigest():
        raise ValueError("AU-PRO source changed during analysis.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiny-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Choose a new CSV path; existing output is not overwritten.")

    rows = analyze(args.manifest, args.tiny_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()