#!/usr/bin/env python3
"""Audit and summarize the joint-search manifest; never select or create a lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_g2 import audit_joint_row
from src.eval.g2_joint import read_manifest, write_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="joint_search_360.csv; sibling spec JSON required")
    parser.add_argument("--output-root")
    parser.add_argument("--selected-r", type=int, help="Display only: externally selected r; no ranking/lock creation")
    parser.add_argument("--selected-d", type=int)
    args = parser.parse_args(argv)
    if (args.selected_r is None) != (args.selected_d is None):
        parser.error("--selected-r and --selected-d must be supplied together")
    manifest = Path(args.manifest).resolve()
    spec = json.loads((manifest.parent / "joint_search_spec.json").read_text())
    pair = (args.selected_r, args.selected_d) if args.selected_r is not None else None
    coverage = write_reports(read_manifest(manifest, spec), spec, args.output_root or manifest.parent,
                             audit=lambda row: audit_joint_row(row, spec), selected_pair=pair)
    print(json.dumps({k: v for k, v in coverage.items() if k != "records"}, indent=2))
    return 0 if coverage["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
