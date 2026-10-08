from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval import evaluator
from src.metrics import aupro as aupro_module
from src.metrics import segf1 as segf1_module

CANDIDATES = ("R0", "R1", "R2")


def evaluator_hashes():
    modules = (evaluator, aupro_module, segf1_module)
    names = ("src/eval/evaluator.py", "src/metrics/aupro.py", "src/metrics/segf1.py")
    return {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in zip(names, modules)
    }


def load_array(base, value):
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def evaluate_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    base = manifest_path.parent
    hashes = evaluator_hashes()
    if hashes != manifest["evaluator_sha256"]:
        raise ValueError("Evaluator/metric SHA256 differs from the Day-4 lock.")
    if evaluator.AUPRO_MAX_FPR != 0.05:
        raise ValueError("AU-PRO max_fpr must remain 0.05.")

    protocols = manifest["normalization_by_candidate"]
    if set(protocols) != set(CANDIDATES):
        raise ValueError("Normalization declarations must contain exactly R0/R1/R2.")
    if not all(isinstance(v, str) and v.strip() for v in protocols.values()):
        raise ValueError("Declare the existing Day-4 score protocol for every candidate.")
    if len(set(protocols.values())) != 1:
        raise ValueError("Normalization protocol differs between candidates.")

    categories = manifest["categories"]
    samples = manifest["samples"]
    if not categories or len(set(categories)) != len(categories) or not samples:
        raise ValueError("Categories must be unique/non-empty; samples must be non-empty.")
    if set(categories) - set(evaluator.MVTEC_AD2_CATEGORIES):
        raise ValueError("Use canonical MVTec AD 2 category names.")
    if set(s["category"] for s in samples) != set(categories):
        raise ValueError("Sample categories differ from the locked category set.")
    ids = [s["image_id"] for s in samples]
    if not all(isinstance(x, str) and x.strip() == x and x for x in ids):
        raise ValueError("image_id must be a non-empty string without surrounding spaces.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image_id in the common evaluation sample list.")
    for sample in samples:
        if set(sample["maps"]) != set(CANDIDATES):
            raise ValueError(f"Missing/unexpected candidate map: {sample['image_id']}")
        if not sample.get("gt_mask"):
            raise ValueError(f"Provide an explicit GT mask: {sample['image_id']}")

    reports = {candidate: {} for candidate in CANDIDATES}
    rows = []
    for category in categories:
        subset = [s for s in samples if s["category"] == category]
        # Load GT once and use the identical masks for all three candidates.
        masks = [load_array(base, s["gt_mask"]) for s in subset]
        for candidate in CANDIDATES:
            records = [
                {
                    "anomaly_map": load_array(base, s["maps"][candidate]),
                    "gt_mask": gt,
                    "meta": {
                        "image_id": s["image_id"],
                        "category": category,
                        "split": manifest["split"],
                        "original_hw": s["original_hw"],
                        "coordinate_space": "original_image",
                    },
                }
                for s, gt in zip(subset, masks)
            ]
            # No sigmoid, resize, clipping, min-max normalization or GT conversion here.
            report = evaluator.evaluate_segmentation_records(
                records,
                seg_f1_threshold=manifest["seg_f1_threshold"],
                expected_split=manifest["split"],
                expected_categories=(category,),
            )
            reports[candidate][category] = report
            rows.append({
                "candidate": candidate,
                "category": category,
                "aupro_0.05": report["metrics"]["aupro_0.05"]["per_category"][category],
                "n_samples": report["n_samples"],
            })
            del records

    scores = {
        c: {r["category"]: r["aupro_0.05"] for r in rows if r["candidate"] == c}
        for c in CANDIDATES
    }
    # Same unweighted category macro as the Day-4 metric protocol.
    macro = {c: float(np.mean(list(scores[c].values()))) for c in CANDIDATES}
    comparisons = []
    for category in [*categories, "macro"]:
        values = macro if category == "macro" else {c: scores[c][category] for c in CANDIDATES}
        comparisons.append({
            "category": category,
            "R0": values["R0"], "R1": values["R1"], "R2": values["R2"],
            "delta_R1_R0": values["R1"] - values["R0"],
            "delta_R2_R1": values["R2"] - values["R1"],
        })
    if evaluator_hashes() != hashes:
        raise ValueError("Evaluator/metric files changed during evaluation.")
    return {
        "primary_metric": "aupro_0.05",
        "max_fpr": 0.05,
        "split": manifest["split"],
        "categories": categories,
        "macro_scope": "all_8_categories" if len(categories) == 8 else "declared_subset",
        "normalization_by_candidate": protocols,
        "evaluator_sha256": hashes,
        "input_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_category": rows,
        "macro_by_candidate": macro,
        "comparisons": comparisons,
        "evaluator_reports": reports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing output is not overwritten.")
    result = evaluate_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for filename, key in (
        ("candidate_category.csv", "candidate_category"),
        ("comparisons.csv", "comparisons"),
    ):
        with (args.output_dir / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    for row in result["comparisons"]:
        print(row)


if __name__ == "__main__":
    main()
