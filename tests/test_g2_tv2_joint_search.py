"""D9 CPU numerical/format fixtures only; no training or measured GPU results."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from scripts import eval_g2 as evaluator
from scripts import summarize_g2_joint as summarizer
from src.eval import g2_joint as joint
from src.eval.g2_benchmark import BenchmarkUnavailable, benchmark_native
from src.models.fusions import WeightedSumFusion
from src.train.g2_context import predict_native_e4
from src.utils.checkpoint import save_training_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_state():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


@pytest.fixture
def spec(tmp_path):
    model = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    # Explicit fixture budget exercises format/parity; it is not a D10 choice.
    model["screening_budget"] = dict(epochs=2, batch_size=4, max_steps=10, max_minutes=None)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(model))
    return joint.load_spec(path, ROOT / "configs/g2_runner.yaml", root=ROOT, integration_sha="a" * 40)


def accepted_format_row(row, score):
    """Format-only input for aggregation tests; audit is separately mandatory."""
    row = deepcopy(row)
    row.update(status="PASS", reason=None, checkpoint="/unit-format-only/best.pt",
               evaluation_json="/unit-format-only/metrics.json", params=123,
               config_hash="a" * 64, git_sha="b" * 40, checkpoint_sha256="c" * 64,
               protocol_hash="d" * 64, dev_hash="e" * 64,
               **{key: score for key in joint.METRIC_FIELDS})
    return row


def checkpoint_config(spec, row):
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg.update(stage="E5", mode="screening", category=row["category"], expected_steps=10)
    cfg["adapter"].update(r=row["r"], d=row["d"])
    cfg.setdefault("fusion", {})["method"] = row["fusion"]
    cfg["training"].update(spec["budget"])
    cfg["decoder"]["deterministic_resize"] = spec["protocol_template"]["decoder"]["deterministic_resize"]
    for key in ("max_train_sources", "max_dev_sources"):
        cfg["data"].setdefault(key, None)
    cfg["synthetic_protocol"] = deepcopy(spec["protocol_template"]["synthetic_protocol"])
    cfg["backbone"]["checkpoint_sha256"] = "f" * 64
    cfg["backbone"]["source_sha256"] = {"hubconf.py": "e" * 64}
    cfg["sources"] = {
        "train": [{"path": f"/unit/{row['category']}/TRAIN/good/train.png", "sha256": "a" * 64}],
        "dev": [{"path": f"/unit/{row['category']}/VALIDATION/good/dev.png", "sha256": "b" * 64}],
    }
    return cfg


def test_backbone_grids_reuse_registry_and_config(spec, tmp_path):
    assert len(spec["pairs"]) == 9
    assert spec["pairs"] == [[r, d] for r in (64, 128, 256) for d in (256, 512, 768)]
    model = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    runner = yaml.safe_load((ROOT / "configs/g2_runner.yaml").read_text())
    for name, r_values, d_values in (
        ("dinov3_vits16", (32, 64, 128), (128, 256, 384)),
        ("dinov3_vith16plus", (128, 256, 384), (384, 768, 1280)),
    ):
        model["backbone"]["name"], runner["backbone"] = name, name
        model_path, runner_path = tmp_path / "model.yaml", tmp_path / "runner.yaml"
        model_path.write_text(yaml.safe_dump(model))
        runner_path.write_text(yaml.safe_dump(runner))
        other = joint.load_spec(model_path, runner_path, root=ROOT)
        assert other["pairs"] == [[r, d] for r in r_values for d in d_values]


def test_unlocked_budget_produces_360_blocked_null_cells():
    spec = joint.load_spec(ROOT / "configs/g2_experiments.yaml", ROOT / "configs/g2_runner.yaml", root=ROOT)
    assert spec["budget"] is None and spec["budget_status"] == "BLOCKED"
    rows = joint.skeleton(spec)
    joint.validate_manifest(rows, spec)
    assert len(rows) == len({r["run_id"] for r in rows}) == 360
    assert all(row["status"] == "BLOCKED" and row["budget"] is None for row in rows)
    assert all(row[key] is None for row in rows for key in (*joint.METRIC_FIELDS, "checkpoint", "latency", "peak_vram"))


def test_locked_budget_not_run_csv_roundtrip_and_schema(spec, tmp_path):
    rows = joint.skeleton(spec)
    assert all(row["status"] == "NOT RUN" for row in rows)
    path = tmp_path / "joint_search_360.csv"
    joint.write_manifest(path, rows, spec)
    assert joint.read_manifest(path, spec) == rows
    schema = joint.manifest_schema()
    assert schema["required"] == list(joint.FIELDS)
    assert schema["properties"]["fusion"]["enum"] == list(joint.FUSIONS)
    assert schema["additionalProperties"] is False
    with path.open() as file:
        raw = list(csv.DictReader(file))
    assert len(raw) == 360 and raw[0]["aupro_005"] == "" and raw[0]["latency"] == ""


@pytest.mark.parametrize("violation", ("duplicate", "missing", "extra", "wrong_id"))
def test_duplicate_missing_and_wrong_identity_rejected(spec, violation):
    rows = joint.skeleton(spec)
    if violation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif violation == "missing":
        rows.pop()
    elif violation == "extra":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["run_id"] += "-changed"
    with pytest.raises(joint.ManifestError):
        joint.validate_manifest(rows, spec)


@pytest.mark.parametrize("key,value", (
    ("seed", 42), ("r", 99), ("fusion", "weighted"), ("category", "unknown"),
    ("config_hash", "bad"), ("git_sha", "bad"), ("params", True),
    ("status", "DONE"), ("benchmark_status", "CPU_MEASURED"),
    ("latency", float("nan")), ("peak_vram", 0), ("dice", float("inf")),
))
def test_schema_rejects_invalid_fields(spec, key, value):
    row = joint.skeleton(spec)[0]
    row[key] = value
    with pytest.raises(joint.ManifestError):
        joint.validate_row(row, spec)


@pytest.mark.parametrize("status", ("CPU_PASS", "SMOKE_PASS", "READY", "NOT RUN", "BLOCKED"))
def test_non_acceptance_cannot_publish_aupro_or_gpu_numbers(spec, status):
    row = joint.skeleton(spec)[0]
    row.update(status=status, aupro_005=.9)
    with pytest.raises(joint.ManifestError, match="Non-PASS"):
        joint.validate_row(row, spec)


@pytest.mark.parametrize("budget", (
    None, {}, {"epochs": 1}, dict(epochs=0, batch_size=4, max_steps=None, max_minutes=None),
    dict(epochs=1, batch_size=4, max_steps=True, max_minutes=None),
    dict(epochs=1, batch_size=4, max_steps=None, max_minutes=float("nan")),
))
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(joint.ManifestError):
        joint.validate_budget(budget)


def test_macro_uses_all_eight_categories_with_hand_computed_mean(spec, tmp_path):
    rows = joint.skeleton(spec)
    for i in range(8):
        rows[i] = accepted_format_row(rows[i], i / 10)
    macro = joint.aggregate(rows, spec)
    assert len(macro) == 45 and macro[0]["valid_categories"] == 8
    assert macro[0]["aupro_005"] == pytest.approx(.35)
    for key in joint.METRIC_FIELDS:
        assert macro[0][key] == pytest.approx(.35)
    rows[7] = joint.skeleton(spec)[7]
    incomplete = joint.aggregate(rows, spec)[0]
    assert incomplete["valid_categories"] == 7 and incomplete["status"] == "BLOCKED"
    assert all(incomplete[key] is None for key in joint.METRIC_FIELDS)


def test_report_audit_is_required_and_rejects_fabricated_scores(spec, tmp_path):
    rows = joint.skeleton(spec)
    for i in range(8):
        rows[i] = accepted_format_row(rows[i], .8)
    coverage = joint.write_reports(rows, spec, tmp_path, audit=lambda row: evaluator.audit_joint_row(row, spec))
    assert coverage["real_pass"] == 0 and coverage["complete_candidates"] == 0
    reported = joint.read_manifest(tmp_path / "joint_search_360.csv", spec)
    assert all(row["aupro_005"] is None for row in reported)
    assert all(row["status"] == "BLOCKED" for row in reported[:8])
    assert len(coverage["records"]) == 360


def test_same_category_dev_and_protocol_parity_block_conflicting_candidates(spec, tmp_path):
    rows = joint.skeleton(spec)
    rows[0] = accepted_format_row(rows[0], .4)
    rows[8] = accepted_format_row(rows[8], .5)
    rows[8]["dev_hash"] = "f" * 64
    coverage = joint.write_reports(rows, spec, tmp_path, audit=lambda row: None)
    assert coverage["real_pass"] == 0
    loaded = joint.read_manifest(tmp_path / "joint_search_360.csv", spec)
    assert loaded[0]["status"] == loaded[8]["status"] == "BLOCKED"
    assert "parity" in loaded[0]["reason"]


def test_selected_pair_table_has_five_fusions_eight_categories_no_partial_macro(spec, tmp_path):
    rows = joint.skeleton(spec)
    for i in range(7):
        rows[i] = accepted_format_row(rows[i], .2)
    coverage = joint.write_reports(rows, spec, tmp_path, audit=lambda row: None, selected_pair=(64, 256))
    with (tmp_path / "fusion_comparison_selected.csv").open() as file:
        table = list(csv.DictReader(file))
    assert len(table) == 5 and tuple(r["fusion"] for r in table) == joint.FUSIONS
    assert table[0]["macro_8"] == "" and table[0]["valid_categories"] == "7"
    assert all(c in table[0] for c in joint.CATEGORIES)
    assert table[0]["walnuts"] == "" and table[0]["walnuts_status"] == "NOT RUN"
    assert coverage["selection_performed"] is False
    assert not list(tmp_path.glob("*lock*"))


def test_checkpoint_config_candidate_hash_git_and_scientific_parity(spec):
    row = joint.skeleton(spec)[0]
    cfg = checkpoint_config(spec, row)
    row.update(config_hash=evaluator.sha256_json(cfg), git_sha="a" * 40)
    payload = dict(config=cfg, metadata=dict(config_sha256=row["config_hash"], git_sha=row["git_sha"]))
    assert joint.validate_checkpoint_record(row, payload, spec) is payload
    for section, key, value in (
        ("adapter", "r", 128), ("fusion", "method", "gated"), ("training", "seed", 42),
        ("training", "epochs", 3), ("loss", "dice_weight", 2.0), ("data", "overlap", 64),
        ("backbone", "name", "dinov3_vits16"), ("decoder", "hidden_channels", 32),
        ("adapter", "kernel_size", 5), ("adapter", "gamma_init", .1), ("adapter", "bias", False),
        ("fusion", "dim", 32), ("paired_views", "context_size", 1024),
        ("paired_views", "pair_strategy", "sequential"),
    ):
        changed = deepcopy(payload)
        changed["config"][section][key] = value
        candidate = deepcopy(row)
        candidate["config_hash"] = evaluator.sha256_json(changed["config"])
        changed["metadata"]["config_sha256"] = candidate["config_hash"]
        with pytest.raises(joint.ManifestError):
            joint.validate_checkpoint_record(candidate, changed, spec)
    broken = deepcopy(payload)
    broken["metadata"]["git_sha"] = "b" * 40
    with pytest.raises(joint.ManifestError, match="Git"):
        joint.validate_checkpoint_record(row, broken, spec)
    broken = deepcopy(payload)
    broken["metadata"]["config_sha256"] = "b" * 64
    with pytest.raises(joint.ManifestError, match="hash"):
        joint.validate_checkpoint_record(row, broken, spec)


def test_real_checkpoint_schema_loader_requires_sha_updates_and_final_budget(spec, tmp_path):
    row = joint.skeleton(spec)[0]
    cfg = checkpoint_config(spec, row)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    path = tmp_path / "best.pt"
    row.update(checkpoint=str(path), config_hash=evaluator.sha256_json(cfg), git_sha="a" * 40)
    metadata = dict(config_sha256=row["config_hash"], git_sha=row["git_sha"], history=[{"fixture": True}],
                    best_metric=.25, dev=dict(qa_status="PASS", aupro_max_fpr=.05, split="dev_synthetic"))
    # Actual serialization interface, zero updates: never an experiment checkpoint.
    save_training_checkpoint(path, model=model, optimizer=optimizer, epoch=0, global_step=0,
                             config=cfg, metadata=metadata)
    with pytest.raises(evaluator.Blocked, match="no optimizer updates"):
        evaluator.load_trained_checkpoint(path, "E5", row["category"], joint_record=row, joint_spec=spec)
    path.with_suffix(".pt.sha256").write_text("0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        evaluator.load_trained_checkpoint(path, "E5", row["category"], joint_record=row, joint_spec=spec)
    with pytest.raises(evaluator.Blocked, match="explicit joint"):
        evaluator.load_trained_checkpoint(path, "E5", row["category"])
    with pytest.raises(FileNotFoundError):
        evaluator.validate_joint_budget(path, dict(config=cfg))


def test_e5_restore_exact_heads_but_fixture_is_never_experiment(monkeypatch):
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    class PairedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = nn.Module()
            self.extractor.backbone = FakeBackbone()
            self.adapters = nn.ModuleDict({"b4": nn.Linear(2, 2)})
            self.projection = nn.Linear(2, 2)
            self.decoder = nn.Linear(2, 1)
            self.fusion = WeightedSumFusion(2)

    saved, built = PairedModel(), PairedModel()
    with torch.no_grad():
        for parameter in saved.parameters():
            parameter.fill_(.3)
    payload = dict(model_state=evaluator.checkpoint_modules(saved, "E5").state_dict())
    monkeypatch.setattr(evaluator, "build_g2_model", lambda *a, **k: built)
    with pytest.raises(evaluator.Blocked, match="fixture"):
        evaluator.restore_model(payload, {}, "E5", "cpu")
    actual = evaluator.checkpoint_modules(built, "E5").state_dict()
    assert all(torch.equal(actual[k], value) for k, value in payload["model_state"].items())


def test_unmerged_e5_runtime_reports_blocked(monkeypatch):
    def old_factory(*args, **kwargs):
        raise NotImplementedError("E1-E4 only")
    monkeypatch.setattr(evaluator, "build_g2_model", old_factory)
    with pytest.raises(evaluator.Blocked, match="g2/integration"):
        evaluator.restore_model({}, {}, "E5", "cpu")


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_benchmark_cannot_measure_cpu_or_missing_gpu(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbidden(*args):
        pytest.fail("No inference/timing may occur without a real CUDA GPU")
    with pytest.raises(BenchmarkUnavailable, match="CPU timings"):
        benchmark_native(nn.Linear(1, 1), torch.zeros(3, 2, 3), {}, device, predict_fn=forbidden)


@pytest.mark.parametrize("warmup,repeats", ((-1, 1), (1, 0), (True, 2), (0, 1.5)))
def test_benchmark_format_validates_iteration_counts(warmup, repeats):
    with pytest.raises(ValueError, match="integers"):
        benchmark_native(nn.Linear(1, 1), torch.zeros(3, 2, 3), {}, "cpu", predict_fn=None,
                         warmup=warmup, repeats=repeats)


@pytest.mark.parametrize("shape", ((31, 43), (529, 777), (513, 65)))
def test_e5_uses_existing_paired_hann_native_orientation_with_known_pixels(shape):
    class LocalProbability(nn.Module):
        def forward(self, batch):
            assert batch["image"].shape == batch["context"].shape
            assert len(batch["view_meta"]) == batch["image"].shape[0]
            probability = batch["image"][:, :1] * .229 + .485
            return torch.logit(probability.clamp(.0001, .9999))

    height, width = shape
    y, x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    known = .2 + .3 * x + .2 * y
    image = known.repeat(3, 1, 1)
    cfg = dict(data=dict(tile_size=512, overlap=128), paired_views=dict(context_size=768),
               evaluation=dict(tile_batch_size=2))
    actual = predict_native_e4(LocalProbability(), image, cfg, "cpu")
    assert tuple(actual.shape) == shape
    torch.testing.assert_close(actual, known, rtol=1e-6, atol=1e-6)
    assert actual[0, -1] > actual[-1, 0]  # X/Y axes must not be exchanged.


def test_joint_cli_skeleton_and_summarizer_keep_all_cells_and_no_fake_metrics(tmp_path):
    directory = tmp_path / "D9"
    code = evaluator.main(["--init-joint", "--device", "cpu", "--output-root", str(directory)])
    assert code == 0
    coverage = json.loads((directory / "joint_search_coverage.json").read_text())
    assert coverage["real_pass"] == 0 and coverage["expected_cells"] == coverage["unique_cells"] == 360
    assert coverage["status_counts"] == {"BLOCKED": 360}
    assert (directory / "joint_search_manifest.schema.json").is_file()
    manifest = directory / "joint_search_360.csv"
    before = manifest.read_bytes()
    assert evaluator.main(["--init-joint", "--output-root", str(directory)]) == 2
    assert manifest.read_bytes() == before  # Initialization cannot erase future D10 progress.
    code = summarizer.main(["--manifest", str(manifest), "--selected-r", "64", "--selected-d", "256"])
    assert code == 2 and manifest.read_bytes() == before
    with (directory / "joint_search_macro.csv").open() as file:
        macro = list(csv.DictReader(file))
    assert len(macro) == 45 and all(row["aupro_005"] == "" for row in macro)
    manifest_run = json.loads((directory / "run_manifest.json").read_text())
    assert manifest_run["training_performed"] is False


def test_joint_subset_missing_checkpoint_records_blocked_without_inference(spec, tmp_path, monkeypatch):
    directory = tmp_path / "D9"
    directory.mkdir()
    manifest = directory / "joint_search_360.csv"
    joint.write_manifest(manifest, joint.skeleton(spec), spec)
    (directory / "joint_search_spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(evaluator, "evaluate_category", lambda *a: pytest.fail("No missing-checkpoint inference"))
    assert evaluator.main(["--joint-manifest", str(manifest), "--categories", "rice",
                           "--r", "64", "--d", "256", "--fusions", "gated",
                           "--output-root", str(directory), "--device", "cpu", "--resume"]) == 2
    rows = joint.read_manifest(manifest, spec)
    blocked = [r for r in rows if r["status"] == "BLOCKED"]
    assert len(blocked) == 1 and blocked[0]["category"] == "rice" and blocked[0]["fusion"] == "gated"
    assert all(row[key] is None for row in rows for key in joint.METRIC_FIELDS)


def test_joint_result_cpu_smoke_cannot_become_scientific_pass(spec, tmp_path):
    row = joint.skeleton(spec)[0]
    for status in ("CPU_PASS", "SMOKE_PASS"):
        result = dict(status=status, acceptance_eligible=False, benchmark=dict(status="NOT RUN"),
                      **{key: .99 for key in joint.METRIC_KEYS})
        mapped = evaluator.joint_result_row(row, result, tmp_path)
        joint.validate_row(mapped, spec)
        assert all(mapped[key] is None for key in joint.METRIC_FIELDS)


def test_artifact_audit_rejects_cpu_fixture_even_when_csv_claims_pass(spec, tmp_path, monkeypatch):
    row = accepted_format_row(joint.skeleton(spec)[0], .2)
    cfg = checkpoint_config(spec, row)
    path = tmp_path / "best.pt"
    path.write_bytes(b"unit-format-only; checkpoint loader is mocked below")
    row.update(checkpoint=str(path), checkpoint_sha256=evaluator.file_sha256(path),
               config_hash=evaluator.sha256_json(cfg), evaluation_json=str(tmp_path / "metrics.json"))
    identity = dict(joint_run_id=row["run_id"], joint_spec_sha256=evaluator.sha256_json(spec),
                    checkpoint_config_sha256=row["config_hash"])
    (tmp_path / "evaluation_config.json").write_text(json.dumps(dict(identity=identity)))
    monkeypatch.setattr(evaluator, "load_trained_checkpoint", lambda *a, **k: dict(config=cfg))
    monkeypatch.setattr(evaluator, "completed_result", lambda *a: dict(
        status="CPU_PASS", acceptance_eligible=False, device="cpu", verification_scope="fixture"))
    with pytest.raises(evaluator.Blocked, match="real-pretrained CUDA"):
        evaluator.audit_joint_row(row, spec)


def test_joint_resume_skips_inference_but_cpu_cache_never_counts(spec, tmp_path, monkeypatch):
    rows = joint.skeleton(spec)
    row = next(r for r in rows if (r["category"], r["r"], r["d"], r["fusion"]) == ("rice", 64, 256, "mean"))
    row["checkpoint"] = str(tmp_path / "unit.pt")
    directory = tmp_path / "D9"
    directory.mkdir()
    manifest = directory / "joint_search_360.csv"
    joint.write_manifest(manifest, rows, spec)
    (directory / "joint_search_spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(evaluator, "prepare_job", lambda *a: (Path(row["checkpoint"]), {}, {}, {}, {"fixture": True}))
    monkeypatch.setattr(evaluator, "completed_result", lambda *a: dict(status="CPU_PASS", acceptance_eligible=False))
    monkeypatch.setattr(evaluator, "restore_model", lambda *a: pytest.fail("Valid resume must skip inference"))
    monkeypatch.setattr(evaluator, "audit_joint_row", lambda *a: None)
    assert evaluator.main(["--joint-manifest", str(manifest), "--categories", "rice", "--r", "64", "--d", "256",
                           "--fusions", "mean", "--resume", "--device", "cpu", "--output-root", str(directory)]) == 2
    coverage = json.loads((directory / "joint_search_coverage.json").read_text())
    assert coverage["real_pass"] == 0 and coverage["status_counts"]["CPU_PASS"] == 1


@pytest.mark.parametrize("args", (
    ["--r", "64"], ["--benchmark"], ["--init-joint", "--smoke"],
    ["--init-joint", "--resume"], ["--joint-manifest", "missing.csv", "--benchmark-repeats", "0"],
))
def test_cli_rejects_incompatible_modes(args):
    with pytest.raises(SystemExit):
        evaluator.parse_args(args)
