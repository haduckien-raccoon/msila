"""GT-selected diagnostics: Image | GT | R0 | R1 | R2."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label

from src.eval.boundary_analysis import STRUCTURE, boundary_zone, validate_inputs
from src.eval.region_stats import read_json, validate_tiny_protocol
from src.eval.tiny_analysis import binary_mask, load_array

CANDIDATES = ("R0", "R1", "R2")


def sample_plan(sample, base, tiny, boundary, padding):
    gt = binary_mask(load_array(base, sample["gt_mask"]))
    hw = sample["original_hw"]
    if len(hw) != 2 or any(type(x) is not int or x < 1 for x in hw) or gt.shape != tuple(hw):
        raise ValueError(f"GT must match original_hw: {sample['image_id']}")
    if not isinstance(sample.get("image_path"), str) or not sample["image_path"].strip():
        raise ValueError(f"Provide the original image_path: {sample['image_id']}")
    zone = boundary_zone(sample, gt.shape, base, boundary)
    labels, count = label(gt, structure=STRUCTURE)
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    overlaps = np.bincount(labels[zone], minlength=count + 1)[1:]
    is_tiny, is_boundary = areas <= tiny["tiny_area_px"], overlaps > 0
    focus_id, crop = None, None
    if count:
        focus_index = min(
            range(count),
            key=lambda i: (
                0 if is_tiny[i] and is_boundary[i] else 1 if is_tiny[i] else 2 if is_boundary[i] else 3,
                int(areas[i]), i,
            ),
        )
        ys, xs = find_objects(labels)[focus_index]
        focus_id = focus_index + 1
        crop = [max(0, ys.start - padding), min(hw[0], ys.stop + padding),
                max(0, xs.start - padding), min(hw[1], xs.stop + padding)]
    return {
        "sample": sample, "n_regions": int(count),
        "n_tiny_regions": int(is_tiny.sum()), "n_boundary_regions": int(is_boundary.sum()),
        "n_tiny_boundary_regions": int((is_tiny & is_boundary).sum()),
        "focus_region_id": focus_id, "crop_y0_y1_x0_x1": crop,
        "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(),
    }


def select_plans(plans, limit):
    """Selection is independent of candidate maps and scores."""
    selected = []
    for category in sorted({p["sample"]["category"] for p in plans}):
        groups = {"both": [], "tiny": [], "boundary": [], "other": []}
        for plan in sorted(plans, key=lambda p: p["sample"]["image_id"]):
            if plan["sample"]["category"] != category:
                continue
            t, b = plan["n_tiny_regions"] > 0, plan["n_boundary_regions"] > 0
            key = "both" if t and b else "tiny" if t else "boundary" if b else "other"
            groups[key].append(plan)
        alternating = [p for pair in zip_longest(groups["tiny"], groups["boundary"]) for p in pair if p is not None]
        selected.extend((groups["both"] + alternating + groups["other"])[:limit])
    return selected


def load_display_image(base, value):
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            arr = np.array(image.convert("RGB") if image.mode in {"P", "CMYK"} else image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("Original image must be grayscale, RGB or RGBA.")
    arr = arr[..., :3]
    if arr.dtype.kind == "u":
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    elif arr.dtype.kind != "f":
        raise ValueError("Use unsigned-integer or display-ready float original images.")
    if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Original image must have a valid display range.")
    return arr


def load_sample(plan, base):
    sample = plan["sample"]
    gt = binary_mask(load_array(base, sample["gt_mask"]))
    if gt.shape != tuple(sample["original_hw"]):
        raise ValueError("GT shape changed after sample selection.")
    if hashlib.sha256(gt.tobytes()).hexdigest() != plan["gt_sha256"]:
        raise ValueError("GT changed after sample selection.")
    image = load_display_image(base, sample["image_path"])
    if image.shape[:2] != gt.shape:
        raise ValueError(f"Original image and GT shapes differ: {sample['image_id']}")
    scores = {}
    for candidate in CANDIDATES:
        score = load_array(base, sample["maps"][candidate])
        if score.shape != gt.shape or score.dtype.kind not in "uif" or not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
            raise ValueError(f"Invalid probability map: {candidate}/{sample['image_id']}")
        scores[candidate] = score
    return image, gt, scores


def render_comparison(path, image, gt, scores, plan, crop=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if crop is not None:
        y0, y1, x0, x1 = crop
        image, gt = image[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        scores = {c: s[y0:y1, x0:x1] for c, s in scores.items()}
    fig = plt.figure(figsize=(18, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.04])
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    axes[0].imshow(image, interpolation="nearest")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    shared_norm = Normalize(vmin=0.0, vmax=1.0)
    for axis, candidate in zip(axes[2:], CANDIDATES):
        heatmap = axis.imshow(scores[candidate], cmap="magma", norm=shared_norm, interpolation="nearest")
    for axis, title in zip(axes, ["Image", "GT", *CANDIDATES]):
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(heatmap, cax=fig.add_subplot(grid[0, 5]), label="Probability [0,1]")
    view = "full image" if crop is None else f"GT zoom region {plan['focus_region_id']}, crop={crop}"
    fig.suptitle(
        f"{plan['sample']['image_id']}\n{view} | tiny={plan['n_tiny_regions']} | boundary={plan['n_boundary_regions']} | diagnostic only",
        fontsize=10,
    )
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def compare(manifest_path, tiny_protocol_path, boundary_protocol_path, output_dir,
            samples_per_category=3, zoom_padding_px=32):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing figures are not overwritten.")
    if type(samples_per_category) is not int or samples_per_category < 1:
        raise ValueError("samples_per_category must be a positive integer.")
    if type(zoom_padding_px) is not int or zoom_padding_px < 0:
        raise ValueError("zoom_padding_px must be a non-negative integer.")
    manifest, manifest_sha = read_json(manifest_path)
    tiny, tiny_sha = read_json(tiny_protocol_path)
    boundary, boundary_sha = read_json(boundary_protocol_path)
    validate_tiny_protocol(tiny)
    validate_inputs(manifest, boundary)
    base = manifest_path.parent
    # Finish GT-based selection and crop planning before loading any anomaly map.
    plans = [sample_plan(s, base, tiny, boundary, zoom_padding_px) for s in manifest["samples"]]
    selected = select_plans(plans, samples_per_category)
    output_ready, records = False, []
    for plan in selected:
        image, gt, scores = load_sample(plan, base)
        if not output_ready:
            output_dir.mkdir(parents=True, exist_ok=False)
            output_ready = True
        sample = plan["sample"]
        token = hashlib.sha256(sample["image_id"].encode()).hexdigest()[:16]
        name = f"comparison_{sample['category']}_{token}"
        full_path = output_dir / f"{name}.png"
        render_comparison(full_path, image, gt, scores, plan)
        zoom_filename = None
        crop = plan["crop_y0_y1_x0_x1"]
        if crop is not None and crop != [0, gt.shape[0], 0, gt.shape[1]]:
            zoom_filename = f"{name}_zoom.png"
            render_comparison(output_dir / zoom_filename, image, gt, scores, plan, crop)
        records.append({
            **{key: value for key, value in plan.items() if key != "sample"},
            "image_id": sample["image_id"], "category": sample["category"],
            "image_path": sample["image_path"], "gt_mask": sample["gt_mask"], "maps": sample["maps"],
            "full_figure": full_path.name, "zoom_figure": zoom_filename,
        })
    report = {
        "purpose": "diagnostic_only", "layout": ["Image", "GT", *CANDIDATES],
        "anomaly_map_range": [0.0, 1.0], "colormap": "magma",
        "normalization_protocol": manifest["normalization_by_candidate"]["R0"],
        "selection_rule": "GT only: both groups, alternating tiny/boundary, other; ID tie-break",
        "samples_per_category": samples_per_category, "zoom_padding_px": zoom_padding_px,
        "manifest_sha256": manifest_sha, "tiny_protocol_sha256": tiny_sha,
        "boundary_protocol_sha256": boundary_sha, "samples": records,
    }
    with (output_dir / "comparison_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiny-protocol", type=Path, required=True)
    parser.add_argument("--boundary-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=3)
    parser.add_argument("--zoom-padding-px", type=int, default=32)
    args = parser.parse_args()
    report = compare(args.manifest, args.tiny_protocol, args.boundary_protocol,
                     args.output_dir, args.samples_per_category, args.zoom_padding_px)
    print(f"Rendered {len(report['samples'])} GT-selected samples to {args.output_dir}")


if __name__ == "__main__":
    main()
