"""Boundary-defect diagnostics. New file: src/eval/boundary_analysis.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from src.eval.evaluator import MVTEC_AD2_CATEGORIES, MVTEC_AD2_SPLITS
from src.eval.tiny_analysis import binary_mask, load_array, region_aupro

CANDIDATES = ("R0", "R1", "R2")
STRUCTURE = np.ones((3, 3), dtype=bool)


def boundary_zone(sample, shape, base, protocol):
    if protocol["boundary_mode"] == "provided_zone_mask":
        zone = binary_mask(load_array(base, sample["boundary_zone_mask"]))
        if zone.shape != shape or not zone.any():
            raise ValueError("Boundary zone must be non-empty and match original_hw.")
        return zone

    width = protocol["band_width_px"]
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (
        (y < width)
        | (y >= shape[0] - width)
        | (x < width)
        | (x >= shape[1] - width)
    )


def select_regions(mask, zone, include_indices=True):
    """Select a component iff at least one pixel intersects the zone."""
    labels, count = label(mask, structure=STRUCTURE)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    overlaps = np.bincount(labels[zone], minlength=count + 1)
    selected_ids = np.flatnonzero(overlaps[1:] > 0) + 1
    selected = np.isin(labels, selected_ids)

    regions = [
        {
            "region_id": i,
            "area_px": int(areas[i]),
            "boundary_overlap_px": int(overlaps[i]),
            "is_boundary": bool(overlaps[i]),
            "indices": (
                np.flatnonzero(labels.ravel() == i)
                if include_indices and overlaps[i]
                else None
            ),
        }
        for i in range(1, count + 1)
    ]
    return selected, regions


def contour(mask):
    # Inner contour; image exterior is treated as background.
    return mask & ~binary_erosion(
        mask, structure=STRUCTURE, border_value=0
    )


def contour_counts(gt_selected, pred_selected, tolerance_px):
    gt_edge = contour(gt_selected)
    pred_edge = contour(pred_selected)

    matched_pred = (
        pred_edge & (distance_transform_edt(~gt_edge) <= tolerance_px)
        if gt_edge.any()
        else np.zeros_like(pred_edge)
    )
    matched_gt = (
        gt_edge & (distance_transform_edt(~pred_edge) <= tolerance_px)
        if pred_edge.any()
        else np.zeros_like(gt_edge)
    )

    return np.array(
        [
            pred_edge.sum(),
            matched_pred.sum(),
            gt_edge.sum(),
            matched_gt.sum(),
        ],
        dtype=np.int64,
    )


def contour_metrics(counts):
    n_pred, matched_pred, n_gt, matched_gt = map(int, counts)
    if n_gt == 0:
        return None, None, None, "NO_GT_BOUNDARY_REGIONS"

    precision = matched_pred / n_pred if n_pred else 0.0
    recall = matched_gt / n_gt
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1, "OK"


def save_visualization(
    path, sample, gt, zone, selected, scores, predictions, protocol
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Native-resolution mask for checking region classification.
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[zone] = (30, 80, 180)
    rgb[gt & ~selected] = (160, 160, 160)
    rgb[selected] = (255, 0, 180)

    Image.fromarray(rgb).save(
        path.with_name(path.stem + "_native_gt.png")
    )

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8), constrained_layout=True
    )
    axes[0, 0].imshow(rgb, interpolation="nearest")
    axes[0, 0].set_title(
        "GT component selection\n"
        "blue=zone, magenta=boundary, gray=interior",
        fontsize=10,
    )
    axes[1, 0].imshow(
        zone, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
    )
    axes[1, 0].set_title(
        "Boundary zone only\nwhite=zone, black=outside",
        fontsize=10,
    )

    gt_edge = contour(selected)
    for column, candidate in enumerate(CANDIDATES, start=1):
        probability_plot = axes[0, column].imshow(
            scores[candidate],
            cmap="magma",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"{candidate}: probability [0,1]")

        pred_edge = contour(predictions[candidate])
        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        overlay[zone] = (20, 35, 70)
        overlay[gt_edge] = (0, 255, 0)
        overlay[pred_edge] = (255, 0, 0)
        overlay[gt_edge & pred_edge] = (255, 255, 0)

        axes[1, column].imshow(overlay, interpolation="nearest")
        axes[1, column].set_title(
            f"{candidate}: contour comparison\n"
            "green=GT, red=pred, yellow=exact overlap",
            fontsize=10,
        )

    fig.colorbar(
        probability_plot,
        ax=list(axes[0, 1:]),
        shrink=0.7,
        label="probability (fixed 0-1)",
    )
    for axis in axes.ravel():
        axis.axis("off")

    fig.suptitle(
        f"{sample['image_id']}\n"
        f"{protocol['boundary_mode']} | "
        f"threshold={protocol['prediction_threshold']} | "
        f"tolerance={protocol['tolerance_px']}px",
        fontsize=11,
    )
    try:
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def validate_inputs(manifest, protocol):
    fields = {
        "boundary_mode",
        "band_width_px",
        "connectivity",
        "max_fpr",
        "prediction_threshold",
        "tolerance_px",
        "locked_before_candidate_results",
        "rule_basis",
    }
    if set(protocol) != fields:
        raise ValueError(
            "Use one global boundary protocol; "
            "candidate-specific settings are forbidden."
        )

    mode = protocol["boundary_mode"]
    width = protocol["band_width_px"]
    threshold = protocol["prediction_threshold"]
    tolerance = protocol["tolerance_px"]

    if mode not in {"image_border_band", "provided_zone_mask"}:
        raise ValueError(
            "Choose explicitly: image_border_band or provided_zone_mask."
        )
    if mode == "image_border_band" and (
        type(width) is not int or width < 1
    ):
        raise ValueError("Lock a positive integer band_width_px.")
    if mode == "provided_zone_mask" and width is not None:
        raise ValueError(
            "For provided_zone_mask, use band_width_px=null."
        )

    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Lock one probability threshold in [0,1].")
    if threshold != manifest.get("seg_f1_threshold"):
        raise ValueError(
            "Reuse the locked seg_f1_threshold from the manifest."
        )
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError(
            "Lock a non-negative integer contour tolerance in native pixels."
        )
    if protocol["connectivity"] != 8 or protocol["max_fpr"] != 0.05:
        raise ValueError("Keep 8-connectivity and max_fpr=0.05.")
    if (
        protocol["locked_before_candidate_results"] is not True
        or not isinstance(protocol["rule_basis"], str)
        or not protocol["rule_basis"].strip()
    ):
        raise ValueError(
            "Record the rule basis and its lock before candidate results."
        )

    categories = manifest["categories"]
    samples = manifest["samples"]
    if (
        not categories
        or len(set(categories)) != len(categories)
        or not samples
    ):
        raise ValueError(
            "Provide unique categories and a non-empty common sample list."
        )
    if (
        set(categories) - set(MVTEC_AD2_CATEGORIES)
        or {s["category"] for s in samples} != set(categories)
    ):
        raise ValueError("Use the declared canonical category set.")

    split = manifest["split"]
    if (
        split not in MVTEC_AD2_SPLITS
        or split in {"test_private", "test_private_mixed"}
    ):
        raise ValueError("Use one locked split with local GT.")

    ids = [s["image_id"] for s in samples]
    if (
        not all(
            isinstance(i, str) and i and i == i.strip()
            for i in ids
        )
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Sample IDs must be unique/non-empty.")
    if any(set(s["maps"]) != set(CANDIDATES) for s in samples):
        raise ValueError("Every sample requires R0/R1/R2 maps.")

    norms = manifest["normalization_by_candidate"]
    if (
        set(norms) != set(CANDIDATES)
        or not all(
            isinstance(v, str) and v.strip()
            for v in norms.values()
        )
        or len(set(norms.values())) != 1
    ):
        raise ValueError(
            "Declare the same normalization protocol for R0/R1/R2."
        )


def analyze(
    manifest_path,
    protocol_path,
    output_dir,
    visualizations_per_category=3,
):
    manifest_path = Path(manifest_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise FileExistsError("Choose a new output directory.")
    if (
        type(visualizations_per_category) is not int
        or visualizations_per_category < 1
    ):
        raise ValueError(
            "visualizations_per_category must be a positive integer."
        )

    manifest_raw = manifest_path.read_bytes()
    protocol_raw = protocol_path.read_bytes()
    manifest = json.loads(manifest_raw)
    protocol = json.loads(protocol_raw)
    validate_inputs(manifest, protocol)

    rows, output_ready = [], False
    base = manifest_path.parent
    threshold = protocol["prediction_threshold"]
    tolerance = protocol["tolerance_px"]

    for category in manifest["categories"]:
        plans = []
        for sample in (
            s for s in manifest["samples"]
            if s["category"] == category
        ):
            gt = binary_mask(load_array(base, sample["gt_mask"]))
            hw = sample["original_hw"]
            if (
                len(hw) != 2
                or any(type(x) is not int or x < 1 for x in hw)
                or gt.shape != tuple(hw)
            ):
                raise ValueError(
                    f"GT must match original_hw: {sample['image_id']}"
                )

            zone = boundary_zone(sample, gt.shape, base, protocol)
            selected, regions = select_regions(gt, zone)
            plans.append((sample, gt, zone, selected, regions))

        # QA selection depends only on GT, before candidate predictions.
        ordered = sorted(
            plans,
            key=lambda p: (not p[3].any(), p[0]["image_id"]),
        )
        qa_ids = {
            p[0]["image_id"]
            for p in ordered[:visualizations_per_category]
        }
        qa = {
            i: {"scores": {}, "predictions": {}}
            for i in qa_ids
        }
        n_regions = sum(
            sum(r["is_boundary"] for r in p[4])
            for p in plans
        )

        for candidate in CANDIDATES:
            normal_parts, boundary_parts = [], []
            counts = np.zeros(4, dtype=np.int64)
            n_pred_regions = 0

            for sample, gt, zone, selected, regions in plans:
                score = load_array(
                    base, sample["maps"][candidate]
                )
                if (
                    score.shape != gt.shape
                    or score.dtype.kind not in "uif"
                    or not np.isfinite(score).all()
                    or np.any(score < 0)
                    or np.any(score > 1)
                ):
                    raise ValueError(
                        f"Invalid probability map: "
                        f"{candidate}/{sample['image_id']}"
                    )

                # Non-boundary GT never becomes background.
                normal_parts.append(score[~gt])
                boundary_parts.extend(
                    score.ravel()[r["indices"]]
                    for r in regions if r["is_boundary"]
                )

                pred_selected, pred_regions = select_regions(
                    score >= threshold,
                    zone,
                    include_indices=False,
                )
                n_pred_regions += sum(
                    r["is_boundary"] for r in pred_regions
                )
                counts += contour_counts(
                    selected, pred_selected, tolerance
                )

                if sample["image_id"] in qa_ids:
                    item = qa[sample["image_id"]]
                    item["scores"][candidate] = score
                    item["predictions"][candidate] = pred_selected

            value, status = region_aupro(
                normal_parts, boundary_parts
            )
            precision, recall, f1, f1_status = contour_metrics(
                counts
            )

            rows.append({
                "candidate": candidate,
                "category": category,
                "split": manifest["split"],
                "boundary_mode": protocol["boundary_mode"],
                "band_width_px": protocol["band_width_px"],
                "selection_rule": "component intersects zone",
                "connectivity": 8,
                "prediction_threshold": threshold,
                "tolerance_px": tolerance,
                "max_fpr": 0.05,
                "n_images": len(plans),
                "n_gt_boundary_regions": n_regions,
                "n_pred_boundary_regions": int(n_pred_regions),
                "n_normal_pixels": sum(
                    p.size for p in normal_parts
                ),
                "boundary_aupro_0.05": value,
                "aupro_status": status,
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
                "boundary_f1_status": f1_status,
                "n_pred_contour_pixels": int(counts[0]),
                "n_matched_pred_contour_pixels": int(counts[1]),
                "n_gt_contour_pixels": int(counts[2]),
                "n_matched_gt_contour_pixels": int(counts[3]),
                "normalization_protocol":
                    manifest["normalization_by_candidate"][candidate],
                "protocol_sha256":
                    hashlib.sha256(protocol_raw).hexdigest(),
                "manifest_sha256":
                    hashlib.sha256(manifest_raw).hexdigest(),
            })

        # Save QA per category to limit retained native maps.
        if not output_ready:
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "mask_qa").mkdir()
            (output_dir / "boundary_protocol.json").write_bytes(
                protocol_raw
            )
            output_ready = True

        for sample, gt, zone, selected, _ in plans:
            if sample["image_id"] in qa_ids:
                token = hashlib.sha256(
                    sample["image_id"].encode()
                ).hexdigest()[:16]
                path = (
                    output_dir / "mask_qa"
                    / f"{sample['category']}_{token}.png"
                )
                item = qa[sample["image_id"]]
                save_visualization(
                    path, sample, gt, zone, selected,
                    item["scores"], item["predictions"], protocol,
                )

    # CSV is written only after every category finishes.
    with (output_dir / "boundary_region_metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--boundary-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--visualizations-per-category", type=int, default=3
    )
    args = parser.parse_args()

    rows = analyze(
        args.manifest,
        args.boundary_protocol,
        args.output_dir,
        args.visualizations_per_category,
    )
    print(f"Wrote {len(rows)} rows and mask QA to {args.output_dir}")


if __name__ == "__main__":
    main()