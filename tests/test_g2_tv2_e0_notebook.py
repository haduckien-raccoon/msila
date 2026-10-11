"""Notebook transport/acceptance checks. Fixtures never establish GPU performance."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import threading
import io

import pytest
import torch
import yaml

from src.eval import e0
from tests.test_g2_tv2_e0_runner import fixture_config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/G2_01_E0_Baseline.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def helpers():
    namespace = dict(Path=Path, PurePosixPath=PurePosixPath, shutil=shutil, json=json,
                     csv=csv, hashlib=hashlib, tarfile=tarfile, threading=threading,
                     math=math, CATEGORIES=e0.MVTEC_AD2_CATEGORIES)
    definitions = [node for code in code_cells() for node in ast.parse(code).body
                   if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def assign(code, name, namespace):
    nodes = [node for node in ast.parse(code).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


def test_notebook_schema_syntax_six_cells_and_unexecuted_gpu_status():
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = code_cells()
    assert len(code) == 6
    for index, source in enumerate(code, 1):
        ast.parse(source)
        assert f"Cell {index}" in source
    assert all(cell.execution_count is None and not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
    assert notebook.metadata.g2_e0.authoring_gpu_validation == "NOT RUN"
    assert notebook.metadata.g2_e0.seed == 42
    assert "REQUIRE_T4" not in "\n".join(code) and "get_device_name(0)" in code[0]
    assert "e0_memory" not in {node.name for source in code for node in ast.walk(ast.parse(source))
                                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    pin = notebook.metadata.g2_e0.pinned_msila_commit
    for path in ("src/eval/e0.py", "src/eval/e0_memory.py", "scripts/run_g2_e0.py", "configs/g2_e0.yaml"):
        subprocess.run(["git", "cat-file", "-e", f"{pin}:{path}"], cwd=ROOT, check=True)


def test_real_cli_and_runtime_config_seed42_keep_dev_protocol(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path, monkeypatch)
    config = yaml.safe_load((ROOT / "configs/g2_e0.yaml").read_text())
    config["memory"].update(sampling_seed=42, device="cuda")
    path = tmp_path / "colab.yaml"
    path.write_text(yaml.safe_dump(config))
    namespace = dict(COLAB_CONFIG=path, DATA_ROOT=Path(cfg["data"]["root"]),
                     DINO_DIR=Path(cfg["backbone"]["repo_dir"]), LOCAL_CHECKPOINT=Path(cfg["backbone"]["weights"]),
                     OUTPUT_ROOT=tmp_path/"colab-output")
    cells = code_cells()
    common = assign(cells[1], "COMMON_ARGS", namespace)
    full_cfg = e0.resolve_config(e0.parse_args([*common, "--categories", "all"]))
    assert full_cfg["memory"]["sampling_seed"] == 42 and full_cfg["memory"]["device"] == "cuda"
    assert full_cfg["evaluation"]["dev_seed"] == config["evaluation"]["dev_seed"] == 17017
    assert full_cfg["data"] == dict(config["data"], root=cfg["data"]["root"])
    assert full_cfg["synthetic_protocol"] == cfg["synthetic_protocol"]
    assert full_cfg["evaluation"]["save_maps"] and full_cfg["mode"] == "full"
    namespace["run_logged"] = lambda arguments, stage: (arguments, stage)
    smoke_args, _ = assign(cells[2], "SMOKE_CODE", dict(namespace, SMOKE_CATEGORY="rice"))
    full_args, _ = assign(cells[3], "FULL_CODE", dict(namespace))
    assert e0.parse_args(smoke_args).smoke
    assert not e0.parse_args(full_args).smoke and e0.parse_args(full_args).categories == ["all"]
    assert e0.resolve_config(e0.parse_args(smoke_args))["output_root"].endswith("/smoke")
    assign(cells[1], "CONFIG_CODE", namespace)
    namespace["ACCEPTANCE"] = dict(rows=[])
    program = assign(cells[4], "VISUALIZATION_CODE", namespace)
    compile(program, "notebook visualization subprocess", "exec")
    assert "G1NativeDataset" in program and "np.array_equal(mask, sample['mask'][0].numpy())" in program
    assert "from src.metrics" not in program and "aupro(" not in program


def archive_set(tmp_path, *, flat=False):
    archive_dir = tmp_path / "drive"
    archive_dir.mkdir()
    for category in e0.MVTEC_AD2_CATEGORIES:
        with tarfile.open(archive_dir/f"{category}.tar.gz", "w:gz") as archive:
            for relative in ("TRAIN/good/a.bin", "VALIDATION/good/b.bin", "TRAIN/bad/trap.bin", "TEST_PUBLIC/good/trap.bin"):
                content = f"archive transport fixture {category}/{relative}".encode()
                prefix = "" if flat else f"wrapper/{category}/"
                member = tarfile.TarInfo(prefix+relative)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return archive_dir


@pytest.mark.parametrize("flat", [False, True])
def test_copy_all_eight_before_extraction_local_only_and_good_splits(tmp_path, monkeypatch, flat):
    archive_dir = archive_set(tmp_path, flat=flat)
    local_archives, data = tmp_path/"colab-archives", tmp_path/"data"
    namespace = helpers()
    copied = []
    original_copy, original_open = namespace["copy_atomic"], tarfile.open
    def record_copy(source, target):
        copied.append(source)
        return original_copy(source, target)
    def check_open(path, *args, **kwargs):
        assert len(copied) == 8, "Every Drive archive must be copied before opening a tar"
        assert Path(path).parent == local_archives
        return original_open(path, *args, **kwargs)
    namespace["copy_atomic"] = record_copy
    monkeypatch.setattr(tarfile, "open", check_open)
    manifest = namespace["prepare_archives"](archive_dir, {c: f"{c}.tar.gz" for c in e0.MVTEC_AD2_CATEGORIES},
                                               local_archives, data)
    assert len(manifest) == 8 and all(row["good_files"] == 2 for row in manifest)
    assert all(len(row["sha256"]) == 64 for row in manifest)
    for category in e0.MVTEC_AD2_CATEGORIES:
        assert (data/category/"TRAIN/good/a.bin").read_bytes() == f"archive transport fixture {category}/TRAIN/good/a.bin".encode()
        assert (data/category/"VALIDATION/good/b.bin").is_file()
        assert not (data/category/"TEST_PUBLIC").exists() and not (data/category/"TRAIN/bad").exists()


def test_missing_and_ambiguous_archives_fail_before_copy(tmp_path):
    archive_dir = archive_set(tmp_path)
    patterns = {category: f"*{category}*.tar.gz" for category in e0.MVTEC_AD2_CATEGORIES}
    (archive_dir/"rice_extra.tar.gz").write_bytes(b"ambiguous transport fixture")
    namespace = helpers()
    def trap(*args, **kwargs):
        raise AssertionError("Do not copy incomplete or ambiguous archive selection")
    namespace["copy_atomic"] = trap
    with pytest.raises(FileNotFoundError, match="rice"):
        namespace["prepare_archives"](archive_dir, patterns, tmp_path/"local", tmp_path/"data")
    (archive_dir/"rice_extra.tar.gz").unlink()
    (archive_dir/"rice.tar.gz").unlink()
    with pytest.raises(FileNotFoundError, match="rice"):
        namespace["prepare_archives"](archive_dir, patterns, tmp_path/"local", tmp_path/"data")


@pytest.mark.parametrize("name", ["../outside/TRAIN/good/a.bin", "/tmp/TRAIN/good/a.bin", "walnuts/TRAIN/good/a.bin"])
def test_archive_paths_cannot_escape_or_mix_categories(name):
    member = tarfile.TarInfo(name)
    with pytest.raises(ValueError, match="BLOCKED"):
        helpers()["selected_good_path"](member, "rice")


def test_atomic_backup_ignores_partial_files_and_updates_completed_files(tmp_path):
    namespace = helpers()
    local, drive = tmp_path/"outputs", tmp_path/"drive-outputs"
    local.mkdir()
    (local/"full.log").write_text("unit transport log")
    (local/"normal_memory.pt.tmp").write_text("partial bank transport fixture")
    namespace.update(OUTPUT_ROOT=local, DRIVE_OUTPUT=drive, SYNC_LOCK=threading.Lock(), SYNCED={})
    namespace["sync_outputs"]()
    assert (drive/"full.log").read_text() == "unit transport log"
    assert not (drive/"normal_memory.pt.tmp").exists()
    (local/"full.log").write_text("unit transport log: completed")
    namespace["sync_outputs"]()
    assert (drive/"full.log").read_text() == "unit transport log: completed"


def test_missing_results_report_not_run_without_numeric_placeholders(tmp_path):
    report = helpers()["inspect_results"](tmp_path, e0.MVTEC_AD2_CATEGORIES, "full", None)
    assert report["status"] == "NOT RUN" and report["real_categories_pass"] == 0
    assert report["macro_au_pro_0_05"] is None
    assert all(row["au_pro_0_05"] is None and not row["valid"] for row in report["rows"])


def test_fixture_smoke_all_eight_never_count_as_gpu_acceptance(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = fixture_config(tmp_path, monkeypatch, categories=e0.MVTEC_AD2_CATEGORIES)
        cfg["evaluation"]["save_maps"] = True
        actual = e0.run_all(cfg, e0.MVTEC_AD2_CATEGORIES, "cpu")
    finally:
        torch.set_num_threads(previous)
    assert actual["smoke_categories_pass"] == 8 and actual["real_categories_pass"] == 0
    output = Path(cfg["output_root"])
    namespace = helpers()
    inspect = namespace["inspect_results"]
    report = inspect(output, e0.MVTEC_AD2_CATEGORIES, "full", 0)
    assert report["status"] == "FAIL" and report["real_categories_pass"] == 0
    assert report["macro_au_pro_0_05"] is None
    smoke = inspect(output, ["rice"], "smoke", 0)
    assert smoke["status"] == "FAIL", "An API fixture also cannot masquerade as real smoke"
    metric_path = output/"rice/metrics.json"
    metric = json.loads(metric_path.read_text())
    metric["synthetic_dev_aupro_0_05"] = float("nan")  # Corruption fixture only.
    metric_path.write_text(json.dumps(metric))
    assert inspect(output, ["rice"], "full", 0)["rows"][0]["au_pro_0_05"] is None
    summary = json.loads((output/"summary.json").read_text())
    summary["status"] = "BLOCKED"
    (output/"summary.json").write_text(json.dumps(summary))
    assert inspect(output, e0.MVTEC_AD2_CATEGORIES, "full", 2)["status"] == "BLOCKED"
