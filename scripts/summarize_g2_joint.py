#!/usr/bin/env python3
"""Audit G2 joint evidence; D10 merges owners and locks only a complete search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_g2 import audit_joint_row
from src.eval.g2_joint import ManifestError, read_manifest, write_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="joint_search_360.csv; sibling spec JSON required")
    parser.add_argument("--output-root")
    parser.add_argument("--selected-r", type=int, help="Display only: externally selected r; no ranking/lock creation")
    parser.add_argument("--selected-d", type=int)
    parser.add_argument("--tv1-manifest", help="TV1 runner's exact tv1_joint_search_180.csv")
    parser.add_argument("--tv2-manifest", help="TV2 runner's exact tv2_joint_search_180.csv")
    parser.add_argument("--relocate-evidence", action="store_true", help="Bind train/evaluation paths to owner manifest folders")
    parser.add_argument("--declare-rule", action="store_true", help="Before screening: bind the fixed tie-break to a preflight spec")
    parser.add_argument("--rule", help="Predeclared joint_selection_rule.json; required to publish a lock")
    parser.add_argument("--resource-config", default="configs/g2_joint_search.yaml", help="Unchanged TV1 screening protocol")
    parser.add_argument("--lock", action="store_true", help="Publish joint_selection_lock.json only after 360 audited PASS cells")
    args = parser.parse_args(argv)
    if (args.selected_r is None) != (args.selected_d is None):
        parser.error("--selected-r and --selected-d must be supplied together")
    merge = args.tv1_manifest is not None or args.tv2_manifest is not None
    if merge and (not args.tv1_manifest or not args.tv2_manifest or args.manifest):
        parser.error("Use --manifest OR both --tv1-manifest and --tv2-manifest")
    if not merge and not args.manifest:
        parser.error("Supply --manifest or both owner manifests")
    if merge and not args.output_root:
        parser.error("Owner merge requires a separate --output-root")
    if args.declare_rule and (merge or args.lock or not args.rule or args.selected_r is not None):
        parser.error("--declare-rule requires --manifest and --rule; cannot merge, select or lock")
    if args.lock and (not args.rule or not merge):
        parser.error("--lock requires a predeclared --rule and both owner manifests with training protocols")
    if args.relocate_evidence and not merge:
        parser.error("--relocate-evidence requires both owner manifests")
    if args.selected_r is not None and (merge or args.rule or args.lock):
        parser.error("Display-only --selected-r/d cannot override D10 selection")
    try:
        return run(args, merge=merge)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps(dict(status="BLOCKED", reason=str(exc), selection_performed=False), indent=2))
        return 2


def run(args, *, merge):
    from src.eval import g2_joint_selection as selection

    if args.declare_rule:
        manifest = Path(args.manifest).resolve()
        spec = selection.read_spec(manifest)
        # A hidden earlier attempt is not a pre-results declaration.
        train_root = manifest.parent / "train"
        if train_root.exists() and any(train_root.iterdir()) and not Path(args.rule).exists():
            raise ManifestError("Training evidence already exists; declare the rule before screening")
        rule = selection.declare_rule(spec, read_manifest(manifest, spec), args.rule, args.resource_config)
        print(json.dumps(dict(status="DECLARED", rule_sha256=rule["rule_sha256"],
                              experimental_status="NOT RUN", selection_performed=False), indent=2))
        return 0
    if merge:
        rows, spec, owners, protocols = selection.merge_owners(
            args.tv1_manifest, args.tv2_manifest, relocate_evidence=args.relocate_evidence)
        directory = Path(args.output_root).resolve()
        if any(directory == Path(p).resolve().parent for p in (args.tv1_manifest, args.tv2_manifest)):
            raise ManifestError("Merged output must preserve owner manifests in a separate folder")
        rule = json.loads(Path(args.rule).read_text()) if args.rule else None
        coverage, outcome = selection.write_decision(
            rows, spec, directory, rule=rule, owners=owners, lock=args.lock,
            audit=selection.evidence_auditor(spec, protocols))
        print(json.dumps({**{k: v for k, v in coverage.items() if k != "records"},
                          "selection_reasons": outcome["reasons"]}, indent=2))
        return 0 if coverage["status"] == "PASS" and (not args.lock or outcome["status"] == "PASS") else 2
    manifest = Path(args.manifest).resolve()
    spec = json.loads((manifest.parent / "joint_search_spec.json").read_text())
    pair = (args.selected_r, args.selected_d) if args.selected_r is not None else None
    if args.rule:
        raise ManifestError("D10 rule/selection requires both owner manifests and their training protocols")
    coverage = write_reports(read_manifest(manifest, spec), spec, args.output_root or manifest.parent,
                             audit=lambda row: audit_joint_row(row, spec), selected_pair=pair)
    print(json.dumps({k: v for k, v in coverage.items() if k != "records"}, indent=2))
    return 0 if coverage["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
