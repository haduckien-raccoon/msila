"""Selection and automation tests using synthetic Day-5 artifact fixtures."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import create_representation_lock as lock_module
from tests.test_aggregate_week06_multiscale import SCRIPT, artifacts, csv_read, csv_write, m


def combined_command(artifacts, summary, lock):
    cmd = [sys.executable, "-B", str(SCRIPT),
           "--metrics-json", str(artifacts[0]), "--tiny-csv", str(artifacts[1]),
           "--boundary-csv", str(artifacts[2]), "--output", str(summary), "--lock-output", str(lock)]
    for path in artifacts[3]:
        cmd += ["--efficiency-csv", str(path)]
    return cmd


def test_highest_primary_selected_and_all_evidence_filled(artifacts):
    result = m.aggregate(*artifacts)
    lock = lock_module.build_lock(result)
    assert lock["selected"] == "R2"
    chosen = result["summary"][2]
    assert lock["evidence"]["primary"] == "au_pro_0.05"
    assert lock["evidence"]["primary_value"] == chosen["AU-PRO0.05"]
    assert lock["evidence"]["tiny"] == chosen["tiny_AU-PRO0.05"]
    assert lock["evidence"]["boundary"] == chosen["boundary_AU-PRO0.05"]
    assert lock["evidence"]["runtime"] == chosen["runtime_ms_per_batch"]
    assert lock["evidence"]["vram"] == chosen["VRAM_MiB"]
    assert lock["scientific_assessment"]["both_steps_improve"] is True
    assert lock["scientific_assessment"]["significance_test_performed"] is False
    assert lock["selection_rule"]["visualization_used"] is False


@pytest.mark.parametrize("scores,selected,directions", [
    ((.9, .8, .7), "R0", ("decreased", "decreased")),
    ((.7, .9, .8), "R1", ("improved", "decreased")),
    ((.7, .6, .9), "R2", ("decreased", "improved")),
    ((.8, .8, .8), "R0", ("tied", "tied")),
    ((.7, .8, .8), "R1", ("improved", "tied")),
    ((.8, .8+5e-13, .7), "R0", ("tied", "decreased")),
])
def test_selection_and_transition_answers(artifacts, scores, selected, directions):
    result = m.aggregate(*artifacts)
    for row, score in zip(result["summary"], scores):
        row["AU-PRO0.05"] = score
    lock = lock_module.build_lock(result)
    assert lock["selected"] == selected
    assert tuple(lock["comparisons"][key]["primary_direction"] for key in ("R0_to_R1", "R1_to_R2")) == directions
    assert lock["scientific_assessment"]["both_steps_improve"] is False


def test_diagnostics_do_not_override_primary(artifacts):
    result = m.aggregate(*artifacts)
    result["summary"][0]["tiny_AU-PRO0.05"] = 1.0
    result["summary"][0]["boundary_AU-PRO0.05"] = 1.0
    lock = lock_module.build_lock(result)
    assert lock["selected"] == "R2"
    assert lock["comparisons"]["R0_to_R1"]["delta_tiny"] < 0


def test_no_gt_tiny_preserves_yaml_null(artifacts, tmp_path):
    rows = csv_read(artifacts[1])
    for r in rows:
        r.update(n_tiny_regions="0", tiny_status="NO_REGIONS", **{"tiny_aupro_0.05": ""})
    csv_write(artifacts[1], rows)
    result = m.aggregate(*artifacts)
    summary = tmp_path / "week06_multiscale.csv"
    m.write_outputs(result, summary)
    lock = lock_module.build_lock(result)
    output = tmp_path / "representation_lock.yaml"
    lock_module.write_lock(lock, output, summary)
    saved = yaml.safe_load(output.read_text())
    assert saved["evidence"]["tiny"] is None
    assert saved["comparisons"]["R0_to_R1"]["delta_tiny"] is None
    assert saved["evaluation_scope"]["tiny_categories"] == []


def test_test_set_selection_refused(artifacts):
    result = m.aggregate(*artifacts)
    result["provenance"]["split"] = "test_public"
    with pytest.raises(ValueError, match="not test-set"):
        lock_module.build_lock(result)


def test_missing_candidate_refused(artifacts):
    result = m.aggregate(*artifacts)
    result["summary"].pop()
    with pytest.raises(ValueError, match="all three candidates"):
        lock_module.build_lock(result)


def test_combined_cli_creates_yaml_after_csv(artifacts, tmp_path):
    summary = tmp_path / "outputs" / "week06_multiscale.csv"
    output = tmp_path / "configs" / "representation_lock.yaml"
    completed = subprocess.run(combined_command(artifacts, summary, output), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    saved = yaml.safe_load(output.read_text())
    assert saved["selected"] == "R2" and summary.is_file()
    assert saved["provenance"]["aggregate_files_sha256"][str(summary)] == m.sha256(summary)
    assert "Selected R2" in completed.stdout


def test_existing_lock_is_protected_before_aggregation_write(artifacts, tmp_path):
    summary, output = tmp_path / "summary.csv", tmp_path / "representation_lock.yaml"
    output.write_text("existing user lock")
    completed = subprocess.run(combined_command(artifacts, summary, output), capture_output=True, text=True)
    assert completed.returncode == 2
    assert output.read_text() == "existing user lock"
    assert not summary.exists()


def test_incomplete_inputs_create_neither_summary_nor_lock(artifacts, tmp_path):
    csv_write(artifacts[1], csv_read(artifacts[1])[:-1])
    summary, output = tmp_path / "summary.csv", tmp_path / "representation_lock.yaml"
    completed = subprocess.run(combined_command(artifacts, summary, output), capture_output=True, text=True)
    assert completed.returncode == 2
    assert not summary.exists() and not output.exists()


def test_standalone_cli_revalidates_original_inputs(artifacts, tmp_path):
    summary, output = tmp_path / "summary.csv", tmp_path / "representation_lock.yaml"
    m.write_outputs(m.aggregate(*artifacts), summary)
    completed = subprocess.run([
        sys.executable, "-B", str(Path(lock_module.__file__)),
        "--summary-csv", str(summary), "--output", str(output),
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert yaml.safe_load(output.read_text())["selected"] == "R2"


def test_edited_summary_cannot_be_locked(artifacts, tmp_path):
    summary = tmp_path / "summary.csv"
    m.write_outputs(m.aggregate(*artifacts), summary)
    rows = csv_read(summary)
    rows[0]["AU-PRO0.05"] = ".99"
    csv_write(summary, rows)
    with pytest.raises(ValueError, match="CSV was changed"):
        lock_module.load_verified_aggregate(summary)


def test_changed_original_input_cannot_be_locked(artifacts, tmp_path):
    summary = tmp_path / "summary.csv"
    m.write_outputs(m.aggregate(*artifacts), summary)
    artifacts[0].write_text(artifacts[0].read_text() + "\n")
    with pytest.raises(ValueError, match="Input changed after aggregation"):
        lock_module.load_verified_aggregate(summary)
