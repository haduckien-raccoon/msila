"""D10 CPU training/resume/format fixtures; never experimental screening PASS."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_g2
from src.eval import g2_joint as joint
from src.train import g2_e2
from src.train import g2_joint_search as search
from src.utils.resume import load_checkpoint_payload
from tests.test_g2_tv1_runner import setup, cpu_determinism, assert_nested_equal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings_path(setup, tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/g2_joint_search.yaml").read_text())
    cfg.update(base_config=str(setup["path"]), output_root=str(tmp_path / "D10"), num_workers=0)
    cfg["screening_budget"] = dict(epochs=2, batch_size=2, max_steps=4, max_minutes=None)
    path = tmp_path / "joint.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def args_for(path, *extra):
    return run_g2.parse_args(["--stage", "joint_search", "--config", str(path),
                              "--backbone", "dinov3_vitb16", "--device", "cpu", *extra])


def study_for(path, *, assets=True):
    args = args_for(path)
    cfg, runner, settings, namespace = search.load_settings(args)
    study = search.create_study(cfg, runner, settings, namespace, device="cpu", assets=assets)
    study["root"] = namespace / (study["sha256"] or "prepared") / "tv1"
    study["root"].mkdir(parents=True, exist_ok=True)
    return study


def context_for(study, fusion="attention"):
    return search.make_context(study, "can", (64, 256), fusion)


def test_user_budget_and_separate_main_budget_are_explicit(settings_path):
    declared = yaml.safe_load((ROOT / "configs/g2_joint_search.yaml").read_text())
    assert declared["screening_budget"] == dict(epochs=10, batch_size=32, max_steps=None, max_minutes=None)
    cfg, _, _, _ = search.load_settings(args_for(settings_path))
    assert cfg["screening_budget"] == dict(epochs=2, batch_size=2, max_steps=4, max_minutes=None)
    assert cfg["main_budget"]["max_steps"] is None
    assert cfg["training"]["batch_size"] == 2 and cfg["training"]["optimizer"]["name"] == "AdamW"


@pytest.mark.parametrize("owner,categories", [("tv1", joint.CATEGORIES[:4]), ("tv2", joint.CATEGORIES[4:])])
def test_prepare_keeps_360_schema_and_exact_owned_180_not_run(settings_path, owner, categories):
    args = args_for(settings_path, "--prepare", "--owner", owner)
    assert search.run_from_cli(args) == 0
    cfg, _, _, namespace = search.load_settings(args_for(settings_path))
    root = namespace / "prepared" / owner
    spec = json.loads((root / "joint_search_spec.json").read_text())
    rows = joint.read_manifest(root / "joint_search_360.csv", spec)
    assert len(rows) == 360 and {r["status"] for r in rows} == {"NOT RUN"}
    with (root / f"{owner}_joint_search_180.csv").open() as handle:
        owned = list(csv.DictReader(handle))
    assert len(owned) == 180 and {r["category"] for r in owned} == set(categories)
    assert len({r["run_id"] for r in owned}) == 180
    assert all(not r[k] for r in owned for k in joint.METRIC_FIELDS)
    assert json.loads((root / "training_coverage.json").read_text())["real_pass"] == 0
    assert not list(root.rglob("*.pt")) and not list(root.rglob("*selection_lock.json"))


def test_invalid_d9_stops_before_manifest_creation(settings_path, tmp_path):
    cfg = yaml.safe_load(settings_path.read_text())
    path = tmp_path / "invalid_d9.csv"
    rows = list(csv.DictReader((ROOT / cfg["d9_preflight"]).open()))
    rows[4].update(status="FAIL", error="Invalid spatial Attention")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    cfg["d9_preflight"] = str(path)
    settings_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="repair invalid D9"):
        search.load_settings(args_for(settings_path))


def test_invalid_d6_also_blocks_even_if_d9_is_pass(settings_path, tmp_path):
    cfg = yaml.safe_load(settings_path.read_text())
    path = tmp_path / "invalid_d6.csv"
    rows = [dict(r=r,d=d,status="FAIL" if (r,d)==(64,256) else "PASS") for r,d in search.PAIRS]
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=("r","d","status")); writer.writeheader(); writer.writerows(rows)
    cfg["d6_preflight"] = str(path)
    settings_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match="D6 reports an invalid"):
        search.load_settings(args_for(settings_path))


@pytest.mark.parametrize("extra", [("--categories", "walnuts"), ("--r", "32", "--d", "256"),
                                   ("--backbone", "dinov3_vits16"), ("--fusions", "global_attention")])
def test_candidate_and_owner_filter_validation(settings_path, extra):
    with pytest.raises(ValueError):
        search.load_settings(args_for(settings_path, *extra))


def test_partial_filters_never_shrink_declared_search(settings_path):
    args = args_for(settings_path, "--categories", "rice", "--r", "64", "--d", "256", "--fusions", "average", "weighted")
    search.load_settings(args)
    assert args.categories == ["rice"] and args.fusions == ["mean", "weighted_sum"]
    study = study_for(settings_path)
    assert len(joint.skeleton(study["spec"])) == 360


@pytest.mark.parametrize("fusion", joint.FUSIONS)
def test_all_e5_fusions_use_real_loop_frozen_backbone_and_full_checkpoint(settings_path, fusion):
    study = study_for(settings_path)
    context, pools = context_for(study, fusion)
    directory = study["root"] / fusion
    epochs = []
    result = g2_e2.train_e5(context, pools, directory, device="cpu", on_epoch=lambda path, epoch: epochs.append(epoch))
    assert result["status"] == "PASS" and result["global_step"] == 4 and result["fusion"] == fusion
    assert result["feature_sources"] == 6 and result["context_aligned"]
    assert result["verification_scope"] == "fixture" and result["scientific_evidence"] is False
    assert epochs == [1, 2]
    assert search.valid_training(directory, context)
    payload, _ = load_checkpoint_payload(directory / "last.pt")
    assert set(result["parameter_report"]["groups"]) == {"adapters", "projection", "decoder", "fusion"}
    assert not any("extractor" in key or "backbone" in key for key in payload["model_state"])
    row = joint.skeleton(study["spec"])[list(joint.FUSIONS).index(fusion)*8]
    row.update(config_hash=context["config_sha256"], git_sha=context["config"]["git_sha"])
    joint.validate_checkpoint_record(row, payload, study["spec"])


def test_e5_resume_mid_epoch_matches_model_optimizer_rng_and_cursor(settings_path, monkeypatch):
    study = study_for(settings_path)
    context, pools = context_for(study)
    full, interrupted = study["root"] / "full", study["root"] / "interrupted"
    g2_e2.train_e5(context, pools, full, device="cpu")
    original = g2_e2.Overfit16Trainer.train_step
    def crash(self, batch, *, step, epoch):
        if step == 2:
            raise RuntimeError("Injected interruption")
        return original(self, batch, step=step, epoch=epoch)
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", crash)
    with pytest.raises(RuntimeError, match="Injected"):
        g2_e2.train_e5(context, pools, interrupted, device="cpu")
    last, _ = load_checkpoint_payload(interrupted / "last.pt")
    assert last["training_state"]["global_step"] == 1 and last["metadata"]["next_batch"] == 1
    with pytest.raises(RuntimeError, match="use --resume"):
        g2_e2.train_e5(context, pools, interrupted, device="cpu")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", original)
    g2_e2.train_e5(context, pools, interrupted, device="cpu", resume=True)
    expected, _ = load_checkpoint_payload(full / "last.pt")
    actual, _ = load_checkpoint_payload(interrupted / "last.pt")
    for key in ("model_state", "optimizer_state", "rng_state", "training_state"):
        if key == "rng_state":
            assert torch.equal(expected[key]["torch_cpu"], actual[key]["torch_cpu"])
        else:
            assert_nested_equal(expected[key], actual[key])
    assert search.valid_training(interrupted, context)


def test_e5_resume_after_train_before_dev_repeats_no_optimizer_updates(settings_path, monkeypatch):
    study = study_for(settings_path)
    context, pools = context_for(study)
    directory = study["root"] / "dev_retry"
    evaluate = g2_e2.evaluate_dev
    monkeypatch.setattr(g2_e2, "evaluate_dev", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("DEV crash")))
    with pytest.raises(RuntimeError, match="DEV crash"):
        g2_e2.train_e5(context, pools, directory, device="cpu")
    payload, _ = load_checkpoint_payload(directory / "last.pt")
    assert payload["training_state"]["global_step"] == 2
    monkeypatch.setattr(g2_e2, "evaluate_dev", evaluate)
    steps = []
    original = g2_e2.Overfit16Trainer.train_step
    def track(self, batch, *, step, epoch):
        steps.append(step)
        return original(self, batch, step=step, epoch=epoch)
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", track)
    g2_e2.train_e5(context, pools, directory, device="cpu", resume=True)
    assert steps == [3, 4] and search.valid_training(directory, context)


def test_cache_is_frozen_and_fresh_heads_match_uncached_initialization(settings_path):
    study = study_for(settings_path)
    from src.models.msila import build_g2_model
    factory = g2_e2.FrozenE5Factory(study["config"], device="cpu")
    context, _ = context_for(study)
    cfg = context["config"]
    torch.manual_seed(cfg["training"]["seed"])
    first = build_g2_model(cfg, "E5")
    cached = factory(cfg)
    assert cached.extractor is factory.extractor and not cached.extractor.training
    assert_nested_equal(first.state_dict(), cached.state_dict())
    with torch.no_grad():
        cached.decoder.head[0].weight.add_(1)
    fresh = factory(cfg)
    assert fresh.decoder is not cached.decoder and fresh.extractor is cached.extractor
    assert_nested_equal(first.decoder.state_dict(), fresh.decoder.state_dict())
    changed = deepcopy(cfg); changed["training"]["seed"] += 1
    with pytest.raises(ValueError, match="provenance"):
        factory(changed)


def test_cpu_fixture_cannot_enter_experimental_manifest_or_be_retrained_on_skip(settings_path, monkeypatch):
    study = study_for(settings_path)
    row = joint.skeleton(study["spec"])[0]
    context, pools = search.make_context(study, "can", (64, 256), "mean")
    directory = search.directory_for(study, row)
    sync_epochs = []
    def record_sync(study, path=None):
        if path is not None:
            sync_epochs.append(json.loads((path / "epoch_progress.json").read_text())["epoch"])
    monkeypatch.setattr(search, "mirror", record_sync)
    with pytest.raises(ValueError, match="CPU/fixture"):
        search.execute(study, row, device="cpu", resume=True)
    assert sync_epochs == [1, 2]  # Four updates, only two epoch Drive callbacks.
    monkeypatch.setattr(search, "train_e5", lambda *a, **k: pytest.fail("Valid training must skip optimizer updates"))
    with pytest.raises(ValueError, match="CPU/fixture"):
        search.execute(study, row, device="cpu", resume=True)
    assert sync_epochs == [1, 2]


@pytest.mark.parametrize("field,value", [("fusion", "concat"), ("category", "rice"), ("global_step", 1),
                                        ("config_sha256", "0"*64), ("git_sha", "0"*40)])
def test_tampered_result_never_skips(settings_path, field, value):
    study = study_for(settings_path)
    context, pools = context_for(study)
    directory = study["root"] / "tampered"
    g2_e2.train_e5(context, pools, directory, device="cpu")
    path = directory / "metrics.json"
    report = json.loads(path.read_text()); report[field] = value
    path.write_text(json.dumps(report))
    assert search.valid_training(directory, context) is None


def test_oom_is_explicit_fail_keeps_180_cells_and_no_metrics(settings_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "CPU fixture: GPU NOT RUN")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(search, "FrozenE5Factory", lambda *a, **k: (_ for _ in ()).throw(torch.OutOfMemoryError("fixture OOM")))
    args = args_for(settings_path, "--device", "cuda", "--categories", "can", "--r", "64", "--d", "256", "--fusions", "mean", "--max-jobs", "1")
    assert search.run_from_cli(args) == 2
    root = next((Path(yaml.safe_load(settings_path.read_text())["output_root"])).glob("*/*/tv1"))
    coverage = json.loads((root / "training_coverage.json").read_text())
    assert coverage["real_pass"] == 0 and coverage["counts"] == {"FAIL": 1, "NOT RUN": 179}
    failed = json.loads(next(root.rglob("failure.json")).read_text())
    assert failed["failure_kind"] == "OOM"
    spec = json.loads((root / "joint_search_spec.json").read_text())
    rows = joint.read_manifest(root / "joint_search_360.csv", spec)
    assert len(rows) == 360 and all(r[k] is None for r in rows for k in joint.METRIC_FIELDS)


def test_cli_local_real_training_is_not_run(settings_path):
    assert search.run_from_cli(args_for(settings_path, "--max-jobs", "1")) == 2


def test_resume_blocks_changed_candidate_seed_budget_and_preserves_checkpoint(settings_path):
    study = study_for(settings_path)
    context, pools = context_for(study)
    directory = study["root"] / "immutable_resume"
    g2_e2.train_e5(context, pools, directory, device="cpu")
    original_sha = search.sha256_file(directory / "last.pt")
    for section, key, value in (("adapter", "r", 128), ("adapter", "d", 512),
                                 ("fusion", "method", "concat"), ("training", "seed", 43),
                                 ("training", "batch_size", 1), (None, "category", "rice")):
        changed = deepcopy(context)
        target = changed["config"][section] if section else changed["config"]
        target[key] = value
        changed["config"]["screening_budget"] = {k: changed["config"]["training"][k] for k in joint.BUDGET_FIELDS}
        changed["config_sha256"] = search.sha256_json(changed["config"])
        assert search.valid_training(directory, changed) is None
        with pytest.raises(ValueError, match="provenance mismatch"):
            g2_e2.train_e5(changed, pools, directory, device="cpu", resume=True)
    assert search.sha256_file(directory / "last.pt") == original_sha


def test_dataset_change_creates_new_protocol_namespace(settings_path):
    first = study_for(settings_path)
    # Changes source content without changing filenames/config; existing runs stay intact.
    cfg = first["config"]
    from PIL import Image
    path = Path(cfg["data"]["root"]) / "can/TRAIN/good/000.png"
    image = Image.open(path).copy()
    image.putpixel((0, 0), (255, 0, 1)); image.save(path)
    second = study_for(settings_path)
    assert first["sha256"] != second["sha256"]
    assert first["protocol"]["dataset_source_sha256"]["can"] != second["protocol"]["dataset_source_sha256"]["can"]


def test_notebook_valid_empty_outputs_and_orchestration_only():
    import ast
    import nbformat
    nb = nbformat.read(ROOT / "notebooks/G2_03_Training.ipynb", as_version=4)
    nbformat.validate(nb)
    sources = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            ast.parse(cell.source)
            sources.append(cell.source)
            assert "Overfit16Trainer" not in cell.source and "loss.backward" not in cell.source
    joined = "\n".join(sources)
    assert "RUN_FULL_SEARCH = False" in joined and "safetensors>=0.8" in joined
    assert "SCREENING_EPOCHS = 10" in joined and "SCREENING_BATCH_SIZE = 32" in joined
    assert "prepare_archives" in joined and "--stage','joint_search" in joined
    assert "run_logged" not in joined  # Its timer sync would violate the epoch-only policy.
    output_cells = [source for source in sources if "# @title OUTPUT" in source]
    assert len(output_cells) == 1 and "DRIVE_OUTPUT_BASE" in output_cells[0]
    assert "--preflight" in joined and "--smoke" in joined and "--report-only" in joined


def test_stale_smoke_success_cannot_hide_a_new_oom(settings_path, monkeypatch):
    study = study_for(settings_path)
    row = joint.skeleton(study["spec"])[0]
    path = search.directory_for(study, row, smoke=True) / "smoke_result.json"
    search.save_json(dict(status="SMOKE_PASS", fixture="stale unit-format file"), path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "CPU fixture: GPU NOT RUN")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(search, "FrozenE5Factory", lambda *a, **k: (_ for _ in ()).throw(torch.OutOfMemoryError("new fixture OOM")))
    args = args_for(settings_path, "--smoke", "--device", "cuda", "--categories", "can", "--r", "64", "--d", "256", "--fusions", "mean")
    assert search.run_from_cli(args) == 2
    assert json.loads((study["root"] / "training_coverage.json").read_text())["real_pass"] == 0


def test_report_only_has_no_per_row_drive_sync(settings_path, monkeypatch):
    syncs = []
    monkeypatch.setattr(search, "mirror", lambda study, directory=None: syncs.append(directory))
    assert search.run_from_cli(args_for(settings_path, "--report-only")) == 2
    assert syncs == [None, None]  # Setup/final metadata; no 180 row-by-row Drive writes.
