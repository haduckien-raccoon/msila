"""D6 compatibility acceptance on CPU only; no performance measurements."""
import ast
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import g2_adapter_preflight as preflight
from src.models.adapter_factory import AdapterCandidate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d6_preflight")
    config = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    result = preflight.run_cpu(config, directory)
    assert result["cpu_pass"] == 9 and result["gpu_status"] == "NOT RUN"
    return directory


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda **kw: pytest.fail("CPU preflight must not load pretrained weights"))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("pair", preflight.PAIRS)
def test_each_pair_stock_e2_shape_two_backward_steps_params_and_hash(report, pair):
    record = json.loads((report / "cpu" / f"r{pair[0]}_d{pair[1]}.json").read_text())
    assert record["status"] == "PASS", record.get("error")
    assert record["shape"] == [2, 768, 4, 4]
    assert record["logits_shape"] == [2, 1, 512, 512]
    assert record["forward_backward_steps"] == 2
    assert record["finite_gradients"] and record["nonzero_branch_gradients_after_gate_update"]
    assert record["frozen_backbone"] and record["exact_identity"]
    assert record["total_trainable_params"] == record["params"] + record["decoder_params"]
    assert preflight.sha256_json(record["config"]) == record["config_sha256"]


def test_nine_distinct_capacities_fair_decoder_and_no_gpu_measurements(report):
    rows = list(csv.DictReader((report / "adapter_preflight.csv").open()))
    assert len(rows) == 9 and {(int(row["r"]), int(row["d"])) for row in rows} == set(preflight.PAIRS)
    assert {row["status"] for row in rows} == {"PASS"}
    assert {row["gpu_status"] for row in rows} == {"NOT RUN"}
    assert all(row["peak_vram_if_measured"] == "" and row["gpu_batch_size"] == "" for row in rows)
    assert len({row["params"] for row in rows}) == len({row["config_sha256"] for row in rows}) == 9
    records = [json.loads(path.read_text()) for path in (report / "cpu").glob("*.json")]
    assert len({row["decoder_initial_sha256"] for row in records}) == 1
    manifest = json.loads((report / "run_manifest.json").read_text())
    assert manifest["invalid_pairs"]["status"] == "PASS" and len(manifest["invalid_pairs"]["rejected_pairs"]) == 8
    assert manifest["selection_status"] == "pending_joint_selection_D10"
    assert not list(report.rglob("*.pt")) and not list(report.rglob("*selection_lock*"))


@pytest.mark.parametrize("pair", [(0, 256), (-1, 256), (64, 0), (64, -1), (True, 256), (64, False), (64.0, 256), (64, None)])
def test_invalid_pairs_rejected(pair):
    with pytest.raises((ValueError, TypeError)):
        AdapterCandidate(*pair)


def test_configuration_round_trip_and_hash_changes_for_meaningful_fields():
    config = preflight.resolved_config(yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text()))
    candidate = preflight.pair_config(config, (64, 256))
    assert preflight.sha256_json(candidate) == preflight.sha256_json(yaml.safe_load(yaml.safe_dump(candidate)))
    for section, key, value in (("adapter", "r", 128), ("adapter", "d", 512), ("adapter", "kernel_size", 5),
                                ("training", "seed", 1), ("decoder", "hidden_channels", 32)):
        changed = deepcopy(candidate); changed[section][key] = value
        assert preflight.sha256_json(changed) != preflight.sha256_json(candidate)


def test_disconnected_conv_fails_instead_of_reporting_pass(monkeypatch):
    cfg = preflight.pair_config(preflight.resolved_config(yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())), (64, 256))
    torch.manual_seed(2026)
    model = preflight.mock_e2(cfg)
    original = model.adapter.mid_proj.forward
    monkeypatch.setattr(model.adapter.mid_proj, "forward", lambda tensor: original(tensor).detach())
    with pytest.raises(ValueError, match="gradient"):
        preflight.check_model(model, cfg, preflight.fixture_batch(2, "cpu"), device="cpu")


def test_batch_probe_oom_and_headroom_backoff(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    visited = []
    def probe(batch):
        visited.append(batch)
        if batch == 32:
            raise torch.cuda.OutOfMemoryError("CPU unit fixture")
        return {"peak_reserved_bytes": batch * 100}
    size, trials = preflight.choose_batch([32, 16, 8, 4], probe, memory_limit=1000)
    assert size == 8 and visited == [32, 16, 8]
    assert [trial["status"] for trial in trials] == ["OOM", "HEADROOM", "FIT"]
    with pytest.raises(ValueError, match="model bug"):
        preflight.choose_batch([4, 2], lambda b: (_ for _ in ()).throw(ValueError("model bug")), memory_limit=1000)
    with pytest.raises(preflight.BatchCapacityError) as failure:
        preflight.choose_batch([1], lambda b: {"peak_reserved_bytes": 100}, memory_limit=99)
    assert failure.value.trials[0]["status"] == "HEADROOM"


@pytest.mark.parametrize("gib,cap", [(15, 16), (23, 32), (40, 64), (80, 128)])
def test_hardware_batch_caps(gib, cap):
    assert preflight.vram_batch_cap(gib * 2**30, 128) == cap
    assert preflight.vram_batch_cap(gib * 2**30, 8) == 8
    assert preflight.batch_candidates(cap)[-1] == 1


def test_gpu_absent_does_not_load_backbone_or_write_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(preflight, "FrozenE2Factory", lambda *a, **kw: pytest.fail("No GPU model loading"))
    with pytest.raises(FileNotFoundError, match="NOT RUN"):
        preflight.run_gpu({}, tmp_path, pairs=[(256, 768)])
    assert not list(tmp_path.iterdir())


def test_cached_gpu_factory_api_only_shares_frozen_backbone_on_cpu_fixture(tmp_path, monkeypatch):
    from src.models.msila import build_g2_model
    from tests.test_g2_tv1_model import BackboneFixture
    repo = tmp_path / "dino_fixture"; repo.mkdir()
    (repo / "hubconf.py").write_text("# CPU fixture only\n")
    weights = tmp_path / "fixture.pth"
    torch.save(BackboneFixture("dinov3_vitb16").state_dict(), weights)
    loads = []
    def load(**kwargs):
        loads.append(kwargs["model"])
        torch.rand(13)
        return BackboneFixture(kwargs["model"])
    monkeypatch.setattr(torch.hub, "load", load)
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["backbone"].update(repo_dir=str(repo), weights=str(weights))
    cfg = preflight.pair_config(preflight.resolved_config(cfg), (64, 256))
    torch.manual_seed(cfg["training"]["seed"])
    reference = build_g2_model(cfg)
    cache = preflight.FrozenE2Factory(cfg, device="cpu")
    first, second = cache(cfg), cache(cfg)
    assert len(loads) == 2 and first.extractor is second.extractor
    for group in ("adapter", "decoder"):
        assert getattr(first, group) is not getattr(second, group)
        expected = getattr(reference, group).state_dict()
        assert all(torch.equal(value, expected[key]) for key, value in getattr(first, group).state_dict().items())
    assert not any(p.requires_grad for p in cache.extractor.parameters())
    changed = deepcopy(cfg); changed["training"]["seed"] += 1
    with pytest.raises(ValueError, match="provenance"):
        cache(changed)


def test_gpu_subset_validation_and_no_screen_action():
    assert preflight.requested_pairs(["256:768", "64:256"]) == [(64, 256), (256, 768)]
    for request in ([], ["64:999"], ["64:256", "64:256"], ["bad"]):
        with pytest.raises(ValueError):
            preflight.requested_pairs(request)
    with pytest.raises(SystemExit):
        preflight.main(["--action", "screen"])


def test_csv_missing_measurements_and_atomic_drive_sync(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    result = preflight.write_report(local)
    assert result["cpu_pass"] == result["gpu_pass"] == 0
    rows = list(csv.DictReader((local / "adapter_preflight.csv").open()))
    assert len(rows) == 9 and {row["status"] for row in rows} == {"NOT RUN"}
    assert all(row["params"] == row["peak_vram_if_measured"] == "" for row in rows)
    preflight.sync_outputs(local, drive)
    assert (local / "adapter_preflight.csv").read_bytes() == (drive / "adapter_preflight.csv").read_bytes()
    with pytest.raises(ValueError, match="disjoint"):
        preflight.sync_outputs(local, local / "nested")


def test_corrupt_evidence_cannot_be_reported_as_pass(tmp_path):
    path = tmp_path / "cpu/r64_d256.json"
    preflight.save_record(dict(status="PASS", params=1), path)
    record = json.loads(path.read_text()); record["params"] = 999
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        preflight.read_record(path)
    assert preflight.write_report(tmp_path)["status"] == "FAIL"
    rows = list(csv.DictReader((tmp_path / "adapter_preflight.csv").open()))
    assert rows[0]["status"] == "FAIL" and "checksum" in rows[0]["error"]


def test_notebook_valid_syntax_guard_no_duplicate_screening_notebook():
    import nbformat
    path = ROOT / "notebooks/G2_02_Adapter_Preflight.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            ast.parse(cell.source)
            code.append(cell.source)
    combined = "\n".join(code)
    assert "RUN_GPU_SMOKE = False" in combined and "if RUN_GPU_SMOKE:" in combined
    assert "safetensors>=0.8" in combined
    assert "--action', 'gpu'" in combined
    assert "--action', 'screen'" not in combined and "train_e2(" not in combined
    assert not (ROOT / "notebooks/G2_02_Adapter_Screening.ipynb").exists()
