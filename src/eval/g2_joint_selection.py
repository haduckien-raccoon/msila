"""D10 report/selection over TV1 runner evidence; no training or metric code.

The merged resource policy records OOM as FAIL, not a terminal exclusion.
Consequently this protocol can lock only after 360 audited PASS results.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from src.eval import g2_joint as joint
from src.eval.evaluator import write_metrics_json
from src.train.screen_representation import read_yaml, sha256_file, sha256_json

OWNERS = {"tv1": joint.CATEGORIES[:4], "tv2": joint.CATEGORIES[4:]}
RESOURCE_POLICY = dict(batch_policy="fixed_for_all_candidates",
                       oom_policy="record_failure_no_automatic_batch_change")
ORDER = ["macro_aupro_005_desc", "trainable_params_asc", "r_asc", "d_asc", "fusion_order"]
LIMITATIONS = (
    "Screening budget and one seed can rank candidates differently from full training.",
    "Synthetic DEV is reused for checkpoint and joint selection; selection bias remains.",
    "Multi-seed confirmation, if performed, must be reported separately without changing this rule.",
    "No TEST scores are used. GPU latency/VRAM require named hardware and measurement conditions; "
    "they are not tie-break variables.",
)


def rule_body(spec):
    joint.validate_spec(spec)
    if spec["budget"] is None or not joint.digest(spec.get("integration_sha"), 40):
        raise joint.ManifestError("Selection requires a declared screening budget and pinned training Git SHA")
    return dict(version="g2_joint_selection_rule_v1", spec_sha256=sha256_json(spec),
                seed=spec["seed"], budget=deepcopy(spec["budget"]), metric="macro_synthetic_dev_aupro_0_05",
                required_categories=list(joint.CATEGORIES), expected_candidates=45, expected_cells=360,
                ordering=list(ORDER), fusion_order=list(joint.FUSIONS), tie_precision="exact_float_no_tolerance",
                resource_policy=dict(RESOURCE_POLICY), terminal_statuses=["PASS"],
                infeasible_exclusions_allowed=False, params="identical_trainable_count_across_8_categories")


def validate_rule(rule, spec):
    body = rule_body(spec)
    if (set(rule) != {"rule", "rule_sha256", "declaration"} or rule["rule"] != body
            or rule["rule_sha256"] != sha256_json(body)):
        raise joint.ManifestError("Predeclared selection rule/spec/hash differs; no post-hoc rule changes")
    declaration = rule["declaration"]
    if (not isinstance(declaration, dict) or set(declaration) != {"utc", "initial_status_counts", "results_observed"}
            or type(declaration["results_observed"]) is not int or declaration["results_observed"] != 0
            or any(k not in {"NOT RUN", "BLOCKED"} or type(v) is not int or v <= 0
                   for k, v in declaration["initial_status_counts"].items())
            or sum(declaration["initial_status_counts"].values()) != 360):
        raise joint.ManifestError("Rule lacks a declaration made before screening results")
    try:
        stamp = datetime.fromisoformat(declaration["utc"])
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise ValueError("Expected UTC timestamp")
    except (KeyError, TypeError, ValueError) as exc:
        raise joint.ManifestError("Invalid rule declaration timestamp") from exc
    return rule


def declare_rule(spec, rows, path, resource_config):
    """Call after TV1 preflight and before full screening; never infer from scores."""
    joint.validate_manifest(rows, spec)
    body = rule_body(spec)
    settings = read_yaml(resource_config)
    if ({k: settings.get(k) for k in RESOURCE_POLICY} != RESOURCE_POLICY
            or settings.get("screening_budget") != spec["budget"]):
        raise joint.ManifestError("Resource policy/budget differs from the merged TV1 protocol")
    path = Path(path)
    if path.exists():
        return validate_rule(json.loads(path.read_text()), spec)
    if any(row["status"] not in {"NOT RUN", "BLOCKED"} or row["checkpoint"] is not None
           or row["config_hash"] is not None or any(row[k] is not None for k in joint.METRIC_FIELDS)
           for row in rows):
        raise joint.ManifestError("Declare selection rule before any screening attempt/result")
    rule = dict(rule=body, rule_sha256=sha256_json(body),
                declaration=dict(utc=datetime.now(timezone.utc).isoformat(),
                                 initial_status_counts=dict(Counter(row["status"] for row in rows)),
                                 results_observed=0))
    validate_rule(rule, spec)
    write_metrics_json(rule, path)
    return rule


def read_spec(manifest):
    spec = json.loads((Path(manifest).parent / "joint_search_spec.json").read_text())
    joint.validate_spec(spec)
    if any(not joint.digest(spec.get(k)) for k in ("model_config_hash", "runner_config_hash")):
        raise joint.ManifestError("Invalid joint model/runner config hashes")
    return spec


def relocate(rows, owner_root):
    """Relocate only the exact runner layout; retain SHA-bound export contents."""
    rows = deepcopy(rows)
    for row in rows:
        for key, section, name in (("checkpoint", "train", "best.pt"),
                                   ("evaluation_json", "evaluation", "metrics.json")):
            if row[key] is None:
                continue
            suffix = (section, *Path(row["run_id"]).parts, name)
            if Path(row[key]).parts[-len(suffix):] != suffix:
                raise joint.ManifestError(f"Cannot relocate noncanonical {key} for {row['run_id']}")
            row[key] = str(Path(owner_root).resolve().joinpath(*suffix))
    return rows


def merge_owners(tv1_manifest, tv2_manifest, *, relocate_evidence=False):
    """Strict 180 + 180 intake. No partial merge, conflict preference or averages."""
    paths = {"tv1": Path(tv1_manifest).resolve(), "tv2": Path(tv2_manifest).resolve()}
    specs = {owner: read_spec(path) for owner, path in paths.items()}
    if specs["tv1"] != specs["tv2"]:
        raise joint.ManifestError("TV1/TV2 specs differ: require same commit, grid, seed, budget and protocol")
    spec = specs["tv1"]
    rows, provenance, protocols = [], {}, {}
    for owner, path in paths.items():
        part = joint.read_manifest(path, spec, categories=OWNERS[owner])
        if relocate_evidence:
            part = relocate(part, path.parent)
        protocol_path = path.parent / "training_protocol.json"
        if protocol_path.is_file():
            protocols[owner] = json.loads(protocol_path.read_text())
        elif any(row["status"] == "PASS" or row["checkpoint"] is not None for row in part):
            raise joint.ManifestError(f"{owner} requires original training_protocol.json for checkpoint evidence")
        rows.extend(part)
        provenance[owner] = dict(manifest=str(path), manifest_sha256=sha256_file(path),
                                 categories=list(OWNERS[owner]), cells=180,
                                 protocol_sha256=sha256_json(protocols[owner]) if owner in protocols else None)
    if len(protocols) == 2 and protocols["tv1"] != protocols["tv2"]:
        raise joint.ManifestError("TV1/TV2 training protocols/source/dataset checksums differ")
    if protocols:
        protocol = next(iter(protocols.values()))
        hashes = protocol.get("source_code_sha256", {})
        datasets = protocol.get("dataset_source_sha256", {})
        if (protocol.get("version") != "g2_d10_study_v1" or protocol.get("git_sha") != spec["integration_sha"]
                or protocol.get("screening_budget") != spec["budget"]
                or not isinstance(hashes, dict) or not hashes or any(not joint.digest(v) for v in hashes.values())
                or not isinstance(datasets, dict) or set(datasets) != set(joint.CATEGORIES)
                or any(not joint.digest(v) for v in datasets.values())):
            raise joint.ManifestError("Invalid training protocol provenance/budget/source checksums")
    joint.validate_manifest(rows, spec)
    index = {row["run_id"]: row for row in rows}
    rows = [index[row["run_id"]] for row in joint.skeleton(spec)]
    return rows, spec, provenance, protocols


def evidence_auditor(spec, protocols):
    """Use the D6/D9 evaluator audit plus D10 cross-owner source provenance."""
    from scripts.eval_g2 import audit_joint_row, source_identity
    from src.train.g2_joint_search import valid_training
    from src.utils.resume import load_checkpoint_payload

    def audit(row):
        if row["git_sha"] != spec["integration_sha"]:
            raise joint.ManifestError("Training Git SHA differs from the pinned joint spec")
        audit_joint_row(row, spec)
        owner = next(owner for owner, categories in OWNERS.items() if row["category"] in categories)
        if owner not in protocols:
            raise joint.ManifestError("Original owner training protocol is missing")
        payload, _ = load_checkpoint_payload(row["checkpoint"], require_sha256=True)
        cfg, protocol = payload["config"], protocols[owner]
        report = valid_training(Path(row["checkpoint"]).parent,
                                dict(config=cfg, config_sha256=row["config_hash"], expected_steps=cfg["expected_steps"]))
        if report is None or report["parameter_report"]["trainable_total"] != row["params"]:
            raise joint.ManifestError("TV1 completion/checkpoint/optimizer/params audit failed")
        if (cfg.get("study_sha256") != sha256_json(protocol)
                or sha256_json(source_identity(cfg["sources"])) != protocol["dataset_source_sha256"][row["category"]]):
            raise joint.ManifestError("Checkpoint study/source checksum differs from owner training protocol")
    return audit


def rank_complete(rows, spec):
    """Full-eight-category candidates only, with the published exact tie-break."""
    entries = joint.aggregate(rows, spec)
    ranking = []
    for entry in entries:
        if entry["status"] != "PASS":
            continue
        params = {row["params"] for row in rows
                  if (row["r"], row["d"], row["fusion"]) == (entry["r"], entry["d"], entry["fusion"])}
        if len(params) != 1:
            raise joint.ManifestError("Trainable params differ across categories for the same candidate")
        ranking.append({**entry, "params": params.pop()})
    ranking.sort(key=lambda x: (-x["aupro_005"], x["params"], x["r"], x["d"], joint.FUSIONS.index(x["fusion"])))
    return [{**entry, "rank": i + 1} for i, entry in enumerate(ranking)]


def decision(rows, spec, rule=None):
    """Pure reporting on already audited rows. Publishing requires another audit."""
    joint.validate_manifest(rows, spec)
    if rule is not None:
        validate_rule(rule, spec)
    final = sum(row["status"] == "PASS" for row in rows)
    ranking = rank_complete(rows, spec)
    complete = final == 360 and len(ranking) == 45
    reasons = []
    if not complete:
        reasons.append(f"Incomplete search: {final}/360 terminal audited cells; "
                       "FAIL/OOM/INFEASIBLE are not terminal under the predeclared TV1 resource policy")
    if rule is None:
        reasons.append("Selection rule declaration is missing")
    ready = complete and rule is not None
    return dict(status="READY" if ready else "BLOCKED", search_status="COMPLETE" if complete else "INCOMPLETE",
                accounted_cells=final, expected_cells=360, valid_metric_cells=final,
                eligible_candidates=len(ranking), expected_candidates=45,
                scope="all_45_candidates" if complete else "provisional_incomplete_search",
                reasons=reasons, excluded_infeasible_candidates=[],
                unresolved_infeasible_run_ids=[row["run_id"] for row in rows if row["status"] == "INFEASIBLE_ON_HARDWARE"],
                ranking=ranking, selected=ranking[0] if ready else None,
                rule_sha256=rule["rule_sha256"] if rule is not None else None,
                limitations=list(LIMITATIONS), selection_performed=False, selection_lock=None)


def publish_lock(rows, spec, rule, directory, *, audit):
    """Audit again at the publication boundary; preserve an existing immutable lock."""
    joint.validate_manifest(rows, spec)
    validate_rule(rule, spec)
    outcome = decision(rows, spec, rule)
    if outcome["status"] != "READY":
        raise joint.ManifestError("BLOCKED: joint selection requires 360/360 audited terminal cells")
    # Includes same-category DEV/protocol parity checks owned by the D9 reporter.
    for row in rows:
        audit(row)
    for category in joint.CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        if len({(row["protocol_hash"], row["dev_hash"]) for row in group}) != 1:
            raise joint.ManifestError("BLOCKED: synthetic DEV/protocol parity differs")
    if any(row["git_sha"] != spec["integration_sha"] for row in rows):
        raise joint.ManifestError("BLOCKED: training Git SHA differs across candidates")
    evidence = sorted((deepcopy(row) for row in rows), key=lambda row: row["run_id"])
    payload = dict(version="g2_joint_selection_lock_v1", spec_sha256=sha256_json(spec),
                   rule_sha256=sha256_json(rule), rule=rule, evidence_sha256=sha256_json(evidence),
                   training_git_sha=spec["integration_sha"], seed=spec["seed"], screening_budget=spec["budget"],
                   expected_cells=360, accounted_cells=360, valid_metric_cells=360,
                   eligible_candidates=45, excluded_infeasible_candidates=[], scope="all_45_candidates",
                   selected={key: outcome["selected"][key] for key in ("r", "d", "fusion", "aupro_005", "params")},
                   ranking=outcome["ranking"], limitations=list(LIMITATIONS), evidence=evidence)
    payload["sha256"] = sha256_json(payload)
    path = Path(directory) / "joint_selection_lock.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise joint.ManifestError("Existing joint selection lock/evidence differs; preserved without overwrite")
    else:
        write_metrics_json(payload, path)
    return path


def write_decision(rows, spec, directory, *, rule=None, owners=None, lock=False, audit):
    """Reuse the sole evaluator report, then add D10 matrix and selection gate."""
    directory = Path(directory)
    if rule is not None:
        validate_rule(rule, spec)
    coverage = joint.write_reports(rows, spec, directory, audit=audit)
    audited = joint.read_manifest(directory / "joint_search_360.csv", spec)
    outcome = decision(audited, spec, rule)
    existing = directory / "joint_selection_lock.json"
    if existing.exists() and outcome["status"] != "READY":
        outcome["reasons"].append("Existing lock has incomplete current evidence; preserved, do not use it")
        coverage["status"] = "BLOCKED"
    if (lock or existing.exists()) and outcome["status"] == "READY":
        try:
            path = publish_lock(audited, spec, rule, directory, audit=audit)
            outcome.update(status="PASS", selection_performed=True, selection_lock=str(path.resolve()))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            outcome.update(status="BLOCKED", selection_performed=False, selection_lock=None)
            outcome["reasons"].append(f"Publication audit: {exc}; any previous lock is preserved and must not be used")
            coverage["status"] = "BLOCKED"
    counts = {}
    for owner, categories in OWNERS.items():
        part = [row for row in audited if row["category"] in categories]
        counts[owner] = dict(expected=180, real_pass=sum(row["status"] == "PASS" for row in part),
                             status_counts=dict(Counter(row["status"] for row in part)), categories=list(categories))
    coverage.update(owner_coverage=counts, accounted_cells=outcome["accounted_cells"],
                    search_status=outcome["search_status"], selection_status=outcome["status"],
                    selection_performed=outcome["selection_performed"], selection_lock=outcome["selection_lock"],
                    excluded_infeasible_candidates=outcome["excluded_infeasible_candidates"],
                    unresolved_infeasible_run_ids=outcome["unresolved_infeasible_run_ids"],
                    owner_provenance=owners or {})
    write_metrics_json(coverage, directory / "joint_search_coverage.json")
    write_metrics_json(outcome, directory / "joint_selection_decision.json")
    # All 45 rows and all eight status columns remain present even with no scores.
    matrix = []
    for entry in joint.aggregate(audited, spec):
        group = {row["category"]: row for row in audited
                 if (row["r"], row["d"], row["fusion"]) == (entry["r"], entry["d"], entry["fusion"])}
        record = {key: entry[key] for key in ("r", "d", "fusion", "valid_categories")}
        record.update({c: group[c]["aupro_005"] for c in joint.CATEGORIES})
        record.update({f"{c}_status": group[c]["status"] for c in joint.CATEGORIES})
        record["macro_8"] = entry["aupro_005"]
        matrix.append(record)
    with (directory / "joint_search_matrix_45x8.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    text = ["# D10 joint selection decision", "", f"Status: **{outcome['status']}**; "
            f"audited terminal cells {outcome['accounted_cells']}/360; "
            f"eligible complete candidates {outcome['eligible_candidates']}/45.", "",
            *(f"- {owner}: PASS {item['real_pass']}/180." for owner, item in counts.items()), "",
            "Rule: higher eight-category macro synthetic DEV AU-PRO@0.05, then fewer trainable params, "
            "smaller r, smaller d, then mean/weighted_sum/concat/gated/attention. Exact ties; no tolerance.", "",
            "The merged OOM policy records FAIL without changing batch. It provides no terminal "
            "infeasibility exclusion, so FAIL/OOM cannot complete coverage or authorize a lock.", "",
            f"Scope: {outcome['scope']}. Selection lock: {outcome['selection_lock'] or 'NOT CREATED'}.", "",
            *(f"- {reason}" for reason in outcome["reasons"]), "",
            *(f"- {item}" for item in LIMITATIONS)]
    (directory / "joint_selection_decision.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if outcome["selection_performed"]:
        winner = outcome["selected"]
        # The D9 display helper audits evidence and preserves all five fusions.
        joint.write_reports(audited, spec, directory, audit=audit, selected_pair=(winner["r"], winner["d"]))
        write_metrics_json(coverage, directory / "joint_search_coverage.json")
    return coverage, outcome
