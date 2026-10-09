"""Read-only inventory and cache-padding audit; TEST is never used for selection."""
import argparse
import csv
import json
from pathlib import Path

from src.data.dataset_audit import MVTEC_AD2_CATEGORIES, audit_cached_tile_padding
from src.data.loader import EXPECTED_SPLITS, audit_mvtec_ad2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        "Exit 0: requested audit passed (private GT can remain ineligible); "
        "1: findings; 2: invalid input/I/O; 3: padding mapping blocked. "
        "Default coverage: all eight categories and five official splits. "
        "For a partial export, declare its expected --categories and --splits."))
    p.add_argument("--data-root", type=Path, help="Root directly containing category directories")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--csv-output", type=Path, help="One row per recognized native image")
    p.add_argument("--categories", nargs="+", default=MVTEC_AD2_CATEGORIES,
                   help="Expected category coverage; does not filter inventory")
    p.add_argument("--splits", nargs="+", choices=sorted(EXPECTED_SPLITS), default=sorted(EXPECTED_SPLITS),
                   help="Expected split coverage in each expected category; does not change splits")
    p.add_argument("--source-manifest", type=Path,
                   help="CSV relative_path,file_size,sha256[,split,category,source_role]; hashes recomputed")
    p.add_argument("--padding-manifest", type=Path,
                   help="JSON list or samples: image_id,mask_path,source_hw,local_box (or geometry.local_box)")
    p.add_argument("--mask-root", type=Path, help="Base for relative cached tile mask paths; default manifest parent")
    a = p.parse_args(argv)
    if not a.data_root and not a.padding_manifest:
        p.error("supply --data-root and/or --padding-manifest")
    if (a.csv_output or a.source_manifest) and not a.data_root:
        p.error("--csv-output/--source-manifest requires --data-root")
    if a.mask_root and not a.padding_manifest:
        p.error("--mask-root requires --padding-manifest")
    try:
        report = audit_mvtec_ad2(a.data_root, expected_categories=a.categories,
                                expected_splits=a.splits, source_manifest=a.source_manifest) if a.data_root else {
                                    "schema": "msila.dataset_audit.v3", "status": "PASS"}
        if a.padding_manifest:
            report["cached_tile_padding"] = audit_cached_tile_padding(a.padding_manifest, mask_root=a.mask_root)
            status = report["cached_tile_padding"]["status"]
            if report["status"] != "FAIL" and status != "PASS":
                report["status"] = status
    except (OSError, ValueError, TypeError) as exc:
        report = dict(schema="msila.dataset_audit.v3", status="ERROR", errors=[dict(error=str(exc))])
    try:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if a.csv_output:
            a.csv_output.parent.mkdir(parents=True, exist_ok=True)
            fields = ["image", "category", "split", "defect_type", "native_hw", "gt_status",
                      "gt_availability", "mask", "pixel_evaluation_eligible"]
            with a.csv_output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report.get("images", []))
    except OSError as exc:
        print(json.dumps(dict(status="ERROR", error=str(exc))))
        return 2
    print(json.dumps(dict(status=report["status"], inventory=report.get("inventory", {}).get("status"),
                          image_files=report.get("inventory", {}).get("image_files"),
                          pixel_evaluation=report.get("pixel_evaluation"),
                          source_integrity=report.get("source_integrity", {}).get("status", "NOT_CHECKED"),
                          cached_tile_padding=report.get("cached_tile_padding", {}).get("status", "NOT_CHECKED"))))
    return {"PASS": 0, "FAIL": 1, "ERROR": 2, "BLOCKED": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
