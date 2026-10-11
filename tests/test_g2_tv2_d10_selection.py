"""D10 CPU format/decision fixtures. No GPU experiments or scientific scores."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts import summarize_g2_joint as cli
from src.eval import g2_joint as joint
from src.eval import g2_joint_selection as selection
from src.train.screen_representation import sha256_json
from tests.test_g2_tv2_joint_search import accepted_format_row, checkpoint_config, spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def resource_config(spec, tmp_path):
    config = yaml.safe_load((ROOT / "configs/g2_joint_search.yaml").read_text())
    config["screening_budget"] = spec["budget"]
    path = tmp_path / "joint.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def rule(spec, resource_config, tmp_path):
    return selection.declare_rule(spec, joint.skeleton(spec), tmp_path / "rule.json", resource_config)


def format_pass_rows(spec):
    rows = [accepted_format_row(row, .5) for row in joint.skeleton(spec)]
    for row in rows:
        row["git_sha"] = spec["integration_sha"]
    return rows


def write_owner(path, rows, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "joint_search_spec.json").write_text(json.dumps(spec))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=joint.FIELDS)
        writer.writeheader()
        writer.writerows({**row, "budget": json.dumps(row["budget"])} for row in rows)


def owner_paths(tmp_path, spec, rows=None):
    rows = joint.skeleton(spec) if rows is None else rows
    paths = []
    for owner, cats in selection.OWNERS.items():
        path = tmp_path / owner / f"{owner}_joint_search_180.csv"
        write_owner(path, [row for row in rows if row["category"] in cats], spec)
        paths.append(path)
    return paths


def test_180_plus_180_has_all_360_unique_cells_and_nine_average_controls(spec, tmp_path):
    paths = owner_paths(tmp_path, spec)
    rows, loaded, owners, protocols = selection.merge_owners(*paths)
    assert loaded == spec and rows == joint.skeleton(spec) and protocols == {}
    assert len(rows) == len({r["run_id"] for r in rows}) == 360
    assert all(v["cells"] == 180 for v in owners.values())
    assert sum(r["fusion"] == "mean" for r in rows) == 72
    assert {r["category"] for r in rows if r["fusion"] == "mean"} == set(joint.CATEGORIES)


@pytest.mark.parametrize("violation", ("missing", "duplicate", "wrong_owner", "extra"))
def test_owner_missing_duplicate_extra_or_foreign_category_is_rejected(spec, tmp_path, violation):
    paths = owner_paths(tmp_path, spec)
    rows = joint.read_manifest(paths[1], spec, categories=selection.OWNERS["tv2"])
    if violation == "missing":
        rows.pop()
    elif violation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif violation == "wrong_owner":
        rows[-1] = deepcopy(joint.skeleton(spec)[0])
    else:
        rows.append(deepcopy(rows[0]))
    write_owner(paths[1], rows, spec)
    with pytest.raises(joint.ManifestError, match="Owner requires"):
        selection.merge_owners(*paths)


@pytest.mark.parametrize("key", ("seed", "budget", "integration_sha", "model_config_hash", "protocol_template"))
def test_owners_must_use_identical_locked_spec(spec, tmp_path, key):
    paths = owner_paths(tmp_path, spec)
    other = deepcopy(spec)
    if key == "seed":
        other[key] += 1
    elif key == "budget":
        other[key]["epochs"] += 1
    elif key == "protocol_template":
        other[key]["loss"]["dice_weight"] += 1
    else:
        other[key] = "b" * len(other[key])
    (paths[1].parent / "joint_search_spec.json").write_text(json.dumps(other))
    with pytest.raises(joint.ManifestError, match="specs differ"):
        selection.merge_owners(*paths)


def test_pass_intake_requires_original_training_protocol_not_just_csv(spec, tmp_path):
    paths = owner_paths(tmp_path, spec, format_pass_rows(spec))
    with pytest.raises(joint.ManifestError, match="training_protocol.json"):
        selection.merge_owners(*paths)


def protocol_format(spec):
    return dict(version="g2_d10_study_v1", git_sha=spec["integration_sha"], screening_budget=spec["budget"],
                source_code_sha256={"training.py": "a" * 64},
                dataset_source_sha256=dict.fromkeys(joint.CATEGORIES, "b" * 64))


def test_training_protocol_hashes_and_source_parity_are_checked(spec, tmp_path):
    paths = owner_paths(tmp_path, spec)
    proto = protocol_format(spec)
    for path in paths:
        (path.parent / "training_protocol.json").write_text(json.dumps(proto))
    _, _, owners, protocols = selection.merge_owners(*paths)
    assert len(protocols) == 2 and owners["tv1"]["protocol_sha256"] == sha256_json(proto)
    proto["dataset_source_sha256"]["vial"] = "c" * 64
    (paths[1].parent / "training_protocol.json").write_text(json.dumps(proto))
    with pytest.raises(joint.ManifestError, match="protocols/source/dataset"):
        selection.merge_owners(*paths)


def test_relocation_binds_exact_run_layout_without_editing_exports(spec, tmp_path):
    row = joint.skeleton(spec)[0]
    row["checkpoint"] = f"/content/local/train/{row['run_id']}/best.pt"
    row["evaluation_json"] = f"/content/local/evaluation/{row['run_id']}/metrics.json"
    changed = selection.relocate([row], tmp_path)[0]
    assert changed["checkpoint"] == str(tmp_path / "train" / row["run_id"] / "best.pt")
    assert changed["evaluation_json"] == str(tmp_path / "evaluation" / row["run_id"] / "metrics.json")
    assert row["checkpoint"].startswith("/content/local/")
    row["checkpoint"] = "/other/category/best.pt"
    with pytest.raises(joint.ManifestError, match="noncanonical"):
        selection.relocate([row], tmp_path)


def test_rule_is_declared_once_before_results_and_cannot_be_redefined(spec, resource_config, tmp_path):
    path = tmp_path / "joint_selection_rule.json"
    rule = selection.declare_rule(spec, joint.skeleton(spec), path, resource_config)
    original = path.read_bytes()
    assert rule["rule"]["fusion_order"] == list(joint.FUSIONS)
    assert rule["rule"]["terminal_statuses"] == ["PASS"]
    selection.declare_rule(spec, format_pass_rows(spec), path, resource_config)
    assert path.read_bytes() == original
    with pytest.raises(joint.ManifestError, match="before any screening"):
        selection.declare_rule(spec, format_pass_rows(spec), tmp_path / "too_late.json", resource_config)
    changed = deepcopy(rule)
    changed["rule"]["ordering"].reverse()
    changed["rule_sha256"] = sha256_json(changed["rule"])
    with pytest.raises(joint.ManifestError, match="post-hoc"):
        selection.validate_rule(changed, spec)


@pytest.mark.parametrize("key", ("screening_budget", "oom_policy", "batch_policy"))
def test_rule_cannot_claim_a_different_resource_protocol(spec, resource_config, tmp_path, key):
    config = yaml.safe_load(resource_config.read_text())
    if key == "screening_budget":
        config[key]["batch_size"] += 1
    else:
        config[key] = "allow_candidate_batch_change"
    resource_config.write_text(yaml.safe_dump(config))
    with pytest.raises(joint.ManifestError, match="Resource policy/budget"):
        selection.declare_rule(spec, joint.skeleton(spec), tmp_path / "rule.json", resource_config)


@pytest.mark.parametrize("status", ("FAIL", "INFEASIBLE_ON_HARDWARE", "BLOCKED", "NOT RUN", "SMOKE_PASS", "CPU_PASS"))
def test_359_pass_plus_one_unresolved_cell_never_locks(spec, rule, tmp_path, status):
    rows = format_pass_rows(spec)
    rows[-1] = {**joint.skeleton(spec)[-1], "status": status, "reason": "OOM fixture, not hardware evidence"}
    outcome = selection.decision(rows, spec, rule)
    assert outcome["status"] == "BLOCKED" and outcome["accounted_cells"] == 359
    assert outcome["eligible_candidates"] == 44 and outcome["selected"] is None
    assert outcome["excluded_infeasible_candidates"] == []
    with pytest.raises(joint.ManifestError, match="360/360"):
        selection.publish_lock(rows, spec, rule, tmp_path, audit=lambda row: None)
    assert not (tmp_path / "joint_selection_lock.json").exists()


def test_four_category_mean_is_never_eight_category_macro_or_selection(spec, rule):
    rows = joint.skeleton(spec)
    for i in range(4):
        rows[i] = accepted_format_row(rows[i], 1.)
    outcome = selection.decision(rows, spec, rule)
    assert outcome["eligible_candidates"] == 0 and outcome["selected"] is None
    assert joint.aggregate(rows, spec)[0]["aupro_005"] is None


def test_selection_prefers_hand_computed_eight_category_macro(spec, rule):
    rows = format_pass_rows(spec)
    for i in range(8):
        rows[i]["aupro_005"] = (i + 1) / 10
    # .1+.2+...+.8 /8 = .45, lower than .5 even though best individual is .8.
    ranking = selection.decision(rows, spec, rule)["ranking"]
    mean = next(e for e in ranking if (e["r"], e["d"], e["fusion"]) == (64, 256, "mean"))
    assert mean["aupro_005"] == pytest.approx(.45) and mean["rank"] == 45
    assert ranking[0]["fusion"] == "weighted_sum"


def test_exact_ties_use_params_then_r_then_d_then_fusion_order(spec, rule):
    rows = format_pass_rows(spec)
    result = selection.decision(rows, spec, rule)
    winner = result["selected"]
    assert (winner["r"], winner["d"], winner["fusion"]) == (64, 256, "mean")
    assert len(result["ranking"]) == 45 and result["scope"] == "all_45_candidates"
    for row in rows:
        if row["fusion"] == "attention":
            row["params"] = 122
    winner = selection.decision(rows, spec, rule)["selected"]
    assert (winner["r"], winner["d"], winner["fusion"]) == (64, 256, "attention")
    # A strict score increase wins even at a larger r/d and parameter count.
    for row in rows:
        if (row["r"], row["d"], row["fusion"]) == (256, 768, "gated"):
            row["aupro_005"] = .500000001
            row["params"] = 999
    assert selection.decision(rows, spec, rule)["selected"]["fusion"] == "gated"


def test_same_candidate_must_have_consistent_trainable_parameters(spec, rule):
    rows = format_pass_rows(spec)
    rows[0]["params"] += 1
    with pytest.raises(joint.ManifestError, match="params differ"):
        selection.decision(rows, spec, rule)


def test_full_format_without_real_checkpoint_audit_cannot_publish_lock(spec, rule, tmp_path):
    rows = format_pass_rows(spec)
    # Production auditor uses actual TV1 checkpoints/native exports, never fixtures.
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        selection.publish_lock(rows, spec, rule, tmp_path, audit=selection.evidence_auditor(spec, {}))
    assert not (tmp_path / "joint_selection_lock.json").exists()


def test_auditor_reuses_tv1_completion_and_binds_study_and_source_hashes(spec, monkeypatch):
    from scripts import eval_g2
    from src.train import g2_joint_search as runner
    from src.utils import resume

    row = format_pass_rows(spec)[0]
    cfg = checkpoint_config(spec, row)
    protocol = protocol_format(spec)
    protocol["dataset_source_sha256"][row["category"]] = sha256_json(eval_g2.source_identity(cfg["sources"]))
    cfg["study_sha256"] = sha256_json(protocol)
    report = {"parameter_report": {"trainable_total": row["params"]}}
    calls = []
    # Mock only I/O for this format unit; no acceptance output is written.
    monkeypatch.setattr(eval_g2, "audit_joint_row", lambda row, spec: calls.append("evaluator"))
    monkeypatch.setattr(resume, "load_checkpoint_payload", lambda *args, **kw: ({"config": cfg}, None))
    monkeypatch.setattr(runner, "valid_training", lambda *args: report)
    audit = selection.evidence_auditor(spec, {"tv1": protocol})
    audit(row)
    assert calls == ["evaluator"]
    report["parameter_report"]["trainable_total"] += 1
    with pytest.raises(joint.ManifestError, match="completion/checkpoint/optimizer/params"):
        audit(row)
    report["parameter_report"]["trainable_total"] -= 1
    cfg["study_sha256"] = "f" * 64
    with pytest.raises(joint.ManifestError, match="study/source checksum"):
        audit(row)


def test_lock_publication_format_is_audited_immutable_and_hash_bound(spec, rule, tmp_path):
    rows = format_pass_rows(spec)
    calls = []
    # Unit injection tests the publication boundary only. Output stays in pytest tmp.
    path = selection.publish_lock(rows, spec, rule, tmp_path, audit=lambda row: calls.append(row["run_id"]))
    assert len(calls) == 360 and len(set(calls)) == 360
    original = path.read_bytes()
    payload = json.loads(original)
    checksum = payload.pop("sha256")
    assert checksum == sha256_json(payload)
    assert payload["accounted_cells"] == 360 and payload["selected"]["fusion"] == "mean"
    selection.publish_lock(rows, spec, rule, tmp_path, audit=lambda row: None)
    assert path.read_bytes() == original
    rows[8]["aupro_005"] = .6
    with pytest.raises(joint.ManifestError, match="preserved without overwrite"):
        selection.publish_lock(rows, spec, rule, tmp_path, audit=lambda row: None)
    assert path.read_bytes() == original


def test_complete_csv_with_dev_parity_or_training_commit_drift_is_blocked(spec, rule, tmp_path):
    rows = format_pass_rows(spec)
    rows[8]["dev_hash"] = "f" * 64
    with pytest.raises(joint.ManifestError, match="parity"):
        selection.publish_lock(rows, spec, rule, tmp_path, audit=lambda row: None)
    rows[8]["dev_hash"] = rows[0]["dev_hash"]
    rows[8]["git_sha"] = "f" * 40
    with pytest.raises(joint.ManifestError, match="Git SHA"):
        selection.publish_lock(rows, spec, rule, tmp_path, audit=lambda row: None)
    assert not (tmp_path / "joint_selection_lock.json").exists()


def test_cli_predeclare_merge_and_requested_lock_reports_0_of_360_without_fakes(spec, resource_config, tmp_path):
    paths = owner_paths(tmp_path, spec)
    full = paths[0].parent / "joint_search_360.csv"
    joint.write_manifest(full, joint.skeleton(spec), spec)
    rule_path = tmp_path / "selection" / "joint_selection_rule.json"
    assert cli.main(["--manifest", str(full), "--declare-rule", "--rule", str(rule_path),
                     "--resource-config", str(resource_config)]) == 0
    report = tmp_path / "selection"
    assert cli.main(["--tv1-manifest", str(paths[0]), "--tv2-manifest", str(paths[1]),
                     "--output-root", str(report), "--rule", str(rule_path), "--lock"]) == 2
    coverage = json.loads((report / "joint_search_coverage.json").read_text())
    assert coverage["real_pass"] == coverage["accounted_cells"] == 0
    assert coverage["selection_status"] == "BLOCKED" and coverage["selection_lock"] is None
    assert all(v["real_pass"] == 0 and v["expected"] == 180 for v in coverage["owner_coverage"].values())
    macro = list(csv.DictReader((report / "joint_search_macro.csv").open()))
    matrix = list(csv.DictReader((report / "joint_search_matrix_45x8.csv").open()))
    assert len(macro) == len(matrix) == 45
    assert all(r["aupro_005"] == "" for r in macro)
    assert all(r[c] == "" and r[c + "_status"] == "NOT RUN" for r in matrix for c in joint.CATEGORIES)
    assert not (report / "joint_selection_lock.json").exists()


def test_cli_rejects_late_declaration_and_preserves_owner_outputs(spec, resource_config, tmp_path):
    paths = owner_paths(tmp_path, spec)
    full = paths[0].parent / "joint_search_360.csv"
    joint.write_manifest(full, joint.skeleton(spec), spec)
    (full.parent / "train" / "started-job").mkdir(parents=True)
    assert cli.main(["--manifest", str(full), "--declare-rule", "--rule", str(tmp_path / "late.json"),
                     "--resource-config", str(resource_config)]) == 2
    assert not (tmp_path / "late.json").exists()
    original = paths[0].read_bytes()
    assert cli.main(["--tv1-manifest", str(paths[0]), "--tv2-manifest", str(paths[1]),
                     "--output-root", str(paths[0].parent)]) == 2
    assert paths[0].read_bytes() == original


def test_stale_lock_is_preserved_but_reported_as_blocked_not_reused(spec, rule, tmp_path):
    stale = tmp_path / "joint_selection_lock.json"
    stale.write_text('{"fixture_stale_lock":true}')
    original = stale.read_bytes()
    coverage, outcome = selection.write_decision(joint.skeleton(spec), spec, tmp_path,
                                                 rule=rule, lock=True, audit=lambda row: None)
    assert stale.read_bytes() == original
    assert coverage["status"] == outcome["status"] == "BLOCKED"
    assert outcome["selection_lock"] is None and outcome["selection_performed"] is False
    assert "do not use" in outcome["reasons"][-1]


@pytest.mark.parametrize("args", (["--lock", "--manifest", "x.csv"],
                                   ["--tv1-manifest", "x.csv"],
                                   ["--manifest", "x.csv", "--relocate-evidence"]))
def test_cli_rejects_ambiguous_or_unbound_selection(args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
