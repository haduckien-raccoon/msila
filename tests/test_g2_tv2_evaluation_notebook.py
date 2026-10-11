"""Notebook validates real CLI calls, five cells, transport and GPU claim gates."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import eval_g2 as g2

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/"notebooks/G2_04_Evaluation.ipynb"


def cells():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]


def assignment(source,name,namespace):
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NOTEBOOK),"exec"),namespace)
    return namespace[name]


def test_notebook_schema_five_cells_and_no_experiment_claim():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4 and notebook["metadata"]["g2_evaluation"]["authoring_gpu_validation"] == "NOT RUN"
    assert notebook["metadata"]["g2_evaluation"]["setup_seed"] == 42
    code = cells()
    assert len(code) == 5
    for index,source in enumerate(code,1):
        ast.parse(source)
        assert f"Cell {index}" in source
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in notebook["cells"] if c["cell_type"] == "code")
    full = "\n".join(code)
    for forbidden in ("torch.load(","best_threshold","aupro(","train_e2(","G1NativeDataset(","build_g2_model("):
        assert forbidden not in full
    assert "'g2/member2'" in code[0] and "get_device_name(0)" in code[0]
    assert "prepare_archives(" in code[1] and "'--preflight'" in code[1]
    assert "'--report-only'" in code[4] and "summary['paired_pass'] == 8" in code[4]
    assert "selection-lock" not in full and "D6_STUDY" not in full
    assert "if PREFLIGHT_CODE == 1:" in code[1] and "if SMOKE_CODE == 1:" in code[2]
    assert "if not WEIGHT_SOURCE.is_file():" in code[1]


def test_notebook_calls_supported_real_cli_flags(tmp_path):
    code = cells()
    namespace = dict(E1_ROOT=tmp_path/"E1", E2_ROOT=tmp_path/"E2", EXPERIMENTS=["E1","E2"], DATA=tmp_path/"data",
                     DINO=tmp_path/"dino", WEIGHTS=tmp_path/"weights.pth", OUTPUT=tmp_path/"out")
    common = assignment(code[1],"COMMON_ARGS",namespace)
    namespace["run_eval"] = lambda args,stage: (args,stage)
    namespace.update(COMMON_ARGS=common,SMOKE_CATEGORY="rice",FULL_CATEGORIES="all")
    preflight,_ = assignment(code[1],"PREFLIGHT_CODE",namespace)
    smoke,_ = assignment(code[2],"SMOKE_CODE",namespace)
    full,_ = assignment(code[3],"FULL_CODE",namespace)
    report,_ = assignment(code[4],"REPORT_CODE",namespace)
    assert g2.parse_args(preflight).preflight
    assert g2.parse_args(smoke).smoke and g2.parse_args(smoke).categories == ["rice"]
    assert g2.parse_args(full).resume and not g2.parse_args(full).smoke
    assert g2.parse_args(full).categories == list(g2.CATEGORIES)
    assert g2.parse_args(report).report_only
    assert g2.parse_args(smoke).output_root != g2.parse_args(full).output_root
    assert g2.parse_args(full).device == "cuda"
    assert "--seed" not in common  # DEV seed must come from saved training config.
    namespace["EXPERIMENTS"] = ["E1"]
    assert g2.parse_args(assignment(code[1],"COMMON_ARGS",namespace)).experiments == ["E1"]


def test_logs_and_drive_sync_survive_failed_process(tmp_path):
    local,drive = tmp_path/"local",tmp_path/"drive"
    code = g2.run_logged([sys.executable,"-c","print('real transport test'); raise SystemExit(2)"],
                         cwd=ROOT,output_root=local,drive_output=drive,stage="smoke",sync_seconds=.02)
    assert code == 2
    assert "real transport test" in (drive/"smoke.log").read_text()
    (local/"pending.json.tmp").write_text("incomplete")
    g2.sync_tree(local,drive)
    assert not (drive/"pending.json.tmp").exists()

