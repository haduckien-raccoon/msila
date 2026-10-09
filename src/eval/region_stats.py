"""Per-GT-region statistics. New file: src/eval/region_stats.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from src.eval.boundary_analysis import STRUCTURE, boundary_zone, validate_inputs
from src.eval.tiny_analysis import binary_mask, load_array

CANDIDATES = ("R0", "R1", "R2")


def component_geometry(mask, boundary_band_px=2):
    """Native-pixel geometry; boundary flag means image edge, not GT contour."""
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("Component audit requires a 2D mask")
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    rows = []
    h, w = mask.shape
    for rid in range(1, count + 1):
        y, x = np.where(labels == rid)
        edge = min(int(x.min()), int(y.min()), w-1-int(x.max()), h-1-int(y.max()))
        rows.append(dict(region_id=rid, area=int(len(x)), width=int(x.max()-x.min()+1),
                         height=int(y.max()-y.min()+1), area_ratio=float(len(x)/(h*w)),
                         bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                         centroid_xy=[float(x.mean()), float(y.mean())],
                         distance_to_image_edge_px=edge, is_boundary=edge < boundary_band_px))
    return rows
FIELDS = [
    "image_id", "category", "split", "region_id", "area",
    "is_tiny", "is_boundary", "boundary_flag", "boundary_overlap_px",
    "metric_name", "prediction_threshold", "tiny_area_px", "connectivity",
    "boundary_mode", "band_width_px", "normalization_protocol",
    "manifest_sha256", "tiny_protocol_sha256", "boundary_protocol_sha256",
]
for candidate in CANDIDATES:
    FIELDS.extend(f"{candidate}_{name}" for name in ("metric", "mean_score", "max_score", "tp", "fn"))


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_tiny_protocol(protocol):
    fields = {
        "tiny_area_px", "area_unit", "connectivity", "max_fpr",
        "locked_before_candidate_results", "threshold_basis",
    }
    if set(protocol) != fields:
        raise ValueError("Use the single shared tiny protocol from Task 4.")
    if type(protocol["tiny_area_px"]) is not int or protocol["tiny_area_px"] < 1:
        raise ValueError("tiny_area_px must be a locked positive integer.")
    if (
        protocol["area_unit"] != "original_image_pixels"
        or protocol["connectivity"] != 8
        or protocol["max_fpr"] != 0.05
    ):
        raise ValueError("Keep the existing tiny protocol: native pixels, connectivity=8, max_fpr=0.05.")
    if (
        protocol["locked_before_candidate_results"] is not True
        or not isinstance(protocol["threshold_basis"], str)
        or not protocol["threshold_basis"].strip()
    ):
        raise ValueError("Record the tiny threshold basis and its lock before candidate results.")


def component_score_stats(score, labels, areas, threshold):
    """Return overlap, mean/max score and TP/FN for every GT component."""
    if (
        score.shape != labels.shape
        or score.dtype.kind not in "uif"
        or not np.isfinite(score).all()
        or np.any(score < 0)
        or np.any(score > 1)
    ):
        raise ValueError("Prediction must be a finite native-resolution probability map in [0,1].")
    n = len(areas)
    score = score.astype(np.float64, copy=False)
    flat_labels = labels.ravel()
    tp = np.bincount(labels[score >= threshold], minlength=n)
    sums = np.bincount(flat_labels, weights=score.ravel(), minlength=n)
    maxima = np.full(n, -np.inf, dtype=np.float64)
    np.maximum.at(maxima, flat_labels, score.ravel())
    # Index 0 is background and is never exported as a defect region.
    return {
        region_id: {
            "metric": float(tp[region_id] / areas[region_id]),
            "mean_score": float(sums[region_id] / areas[region_id]),
            "max_score": float(maxima[region_id]),
            "tp": int(tp[region_id]),
            "fn": int(areas[region_id] - tp[region_id]),
        }
        for region_id in range(1, n)
    }


def analyze(manifest_path, tiny_protocol_path, boundary_protocol_path):
    manifest_path = Path(manifest_path).resolve()
    manifest, manifest_sha = read_json(manifest_path)
    tiny, tiny_sha = read_json(tiny_protocol_path)
    boundary, boundary_sha = read_json(boundary_protocol_path)
    validate_tiny_protocol(tiny)
    validate_inputs(manifest, boundary)
    threshold = boundary["prediction_threshold"]
    base, rows = manifest_path.parent, []

    for sample in manifest["samples"]:
        gt = binary_mask(load_array(base, sample["gt_mask"]))
        hw = sample["original_hw"]
        if (
            len(hw) != 2
            or any(type(x) is not int or x < 1 for x in hw)
            or gt.shape != tuple(hw)
        ):
            raise ValueError(f"GT must match original_hw: {sample['image_id']}")
        zone = boundary_zone(sample, gt.shape, base, boundary)
        # One GT labeling determines the identity/area/flags for all candidates.
        labels, count = label(gt, structure=STRUCTURE)
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        overlaps = np.bincount(labels[zone], minlength=count + 1)
        candidate_stats = {}
        for candidate in CANDIDATES:
            score = load_array(base, sample["maps"][candidate])
            try:
                candidate_stats[candidate] = component_score_stats(score, labels, areas, threshold)
            except ValueError as exc:
                raise ValueError(f"{candidate}/{sample['image_id']}: {exc}") from exc

        for region_id in range(1, count + 1):
            is_boundary = bool(overlaps[region_id] > 0)
            row = {
                "image_id": sample["image_id"], "category": sample["category"],
                "split": manifest["split"], "region_id": region_id,
                "area": int(areas[region_id]),
                "is_tiny": bool(areas[region_id] <= tiny["tiny_area_px"]),
                "is_boundary": is_boundary, "boundary_flag": is_boundary,
                "boundary_overlap_px": int(overlaps[region_id]),
                "metric_name": "region_overlap_at_locked_threshold",
                "prediction_threshold": threshold, "tiny_area_px": tiny["tiny_area_px"],
                "connectivity": 8, "boundary_mode": boundary["boundary_mode"],
                "band_width_px": boundary["band_width_px"],
                "normalization_protocol": manifest["normalization_by_candidate"]["R0"],
                "manifest_sha256": manifest_sha, "tiny_protocol_sha256": tiny_sha,
                "boundary_protocol_sha256": boundary_sha,
            }
            for candidate in CANDIDATES:
                for name, value in candidate_stats[candidate][region_id].items():
                    row[f"{candidate}_{name}"] = value
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiny-protocol", type=Path, required=True)
    parser.add_argument("--boundary-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new CSV path; existing output is not overwritten.")
    rows = analyze(args.manifest, args.tiny_protocol, args.boundary_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} GT defect-region rows to {args.output}")


if __name__ == "__main__":
    main()
