"""D6 fixture smoke and orchestration tests, never real PASS/72 evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.models.adapter_factory import AdapterFactoryConfig, ResidualAdapterFactory
from src.train import g2_e2
from src.utils.resume import load_checkpoint_payload
from tests.test_g2_tv1_model import BackboneFixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g2_runner_test", ROOT / "scripts/run_g2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def cpu_determinism():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "dino_fixture"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# Official API fixture only\n")
    data = tmp_path / "data"
    rng = np.random.default_rng(82)
    for category in runner.CATEGORIES:
        for split in ("TRAIN", "VALIDATION"):
            directory = data / category / split / "good"
            directory.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(directory / "000.png")
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: BackboneFixture(kwargs["model"]))
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["backbone"]["repo_dir"] = str(repo)
    for name in runner.GRIDS:
        weights = tmp_path / f"{name}.pth"
        torch.save(BackboneFixture(name).state_dict(), weights)
        cfg["backbone"]["checkpoints"][name] = str(weights)
    cfg["data"]["root"] = str(data)
    cfg["training"].update(epochs=2, batch_size=2)
    cfg["decoder"]["hidden_channels"] = 4
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump(cfg))
    run_cfg = yaml.safe_load((ROOT / "configs/g2_runner.yaml").read_text())
    run_cfg.update(model_config=str(model_config), output_root=str(tmp_path / "outputs"), backbone="dinov3_vits16")
    run_cfg["training"]["checkpoint_interval_steps"] = 1
    config_path = tmp_path / "runner.yaml"
    config_path.write_text(yaml.safe_dump(run_cfg))

    def study(name="dinov3_vits16"):
        args = runner.parse_args(["--config", str(config_path), "--stage", "adapter_screen",
                                  "--device", "cpu", "--backbone", name])
        resolved, config, root = runner.load_config(args)
        return runner.prepare_study(resolved, config, root, device="cpu")

    return dict(path=config_path, root=tmp_path / "outputs", study=study, cfg=cfg, runner=run_cfg)


@pytest.mark.parametrize("name,r_values,d_values", [
    ("dinov3_vits16", [32, 64, 128], [128, 256, 384]),
    ("dinov3_vitb16", [64, 128, 256], [256, 512, 768]),
    ("dinov3_vith16plus", [128, 256, 384], [384, 768, 1280]),
])
def test_exact_backbone_grid_and_fixed_category_protocol(setup, name, r_values, d_values):
    study = setup["study"](name)
    assert study["pairs"] == [(r, d) for r in r_values for d in d_values]
    assert len(study["pairs"]) * len(runner.CATEGORIES) == 72
    contexts = [runner.make_context(study, "rice", pair, "adapter_screen")[0] for pair in study["pairs"]]
    assert len({c["config_sha256"] for c in contexts}) == 9
    fixed = []
    for context in contexts:
        cfg = deepcopy(context["config"])
        cfg["adapter"].pop("r")
        cfg["adapter"].pop("d")
        fixed.append(cfg)
    assert all(cfg == fixed[0] for cfg in fixed)
    assert contexts[0]["expected_steps"] == 4
    assert fixed[0]["training"]["dev_synthetic_fixed"]
    assert fixed[0]["data"]["train_split"] == "TRAIN/good"
    assert fixed[0]["data"]["dev_split"] == "VALIDATION/good"


def test_wrong_h_grid_rejected_before_execution(setup):
    config = deepcopy(setup["runner"])
    config["adapter_grids"]["dinov3_vith16plus"]["r_values"] = [128, 256, 512]
    setup["path"].write_text(yaml.safe_dump(config))
    args = runner.parse_args(["--config", str(setup["path"]), "--stage", "adapter_screen",
                              "--backbone", "dinov3_vith16plus"])
    with pytest.raises(ValueError, match="declared D6 r/d grid"):
        runner.load_config(args)


def test_cli_nine_pair_smoke_then_skip_without_training(setup, monkeypatch):
    args = ["--config", str(setup["path"]), "--stage", "adapter_screen", "--categories", "rice",
            "--device", "cpu", "--smoke"]
    assert runner.main(args) == 0
    path = setup["root"] / "dinov3_vits16" / "smoke" / "adapter_screen_summary.json"
    summary = json.loads(path.read_text())
    assert summary["status"] == "SMOKE_PASS"
    assert len(summary["outcomes"]) == 9
    assert {r["status"] for r in summary["outcomes"]} == {"PASS"}
    assert summary["real_adapter_screen_pass"] == summary["real_E2_pass"] == 0
    assert not list(setup["root"].rglob("adapter_selection_lock.json"))
    monkeypatch.setattr(runner, "train_e2", lambda *a, **k: pytest.fail("Completed runs must not train again"))
    assert runner.main(args) == 0
    assert {r["status"] for r in json.loads(path.read_text())["outcomes"]} == {"SKIP"}


@pytest.mark.parametrize("name,pair", [("dinov3_vitb16", (64, 256)), ("dinov3_vith16plus", (128, 384))])
def test_b_h_e2_fixture_smoke_forward_backward(setup, name, pair):
    study = setup["study"](name)
    context, pools = runner.make_context(study, "rice", pair, "adapter_screen", smoke=True)
    path = runner.run_directory(study, "rice", pair, "adapter_screen", True)
    result = g2_e2.train_e2(context, pools, path, device="cpu")
    assert result["status"] == "PASS" and result["global_step"] == 2
    assert result["adapter_updated"] and result["decoder_updated"] and result["frozen_backbone_unchanged"]
    assert result["verification_scope"] == "fixture"
    assert runner.read_valid_result(path, context)
    assert not runner.real_evidence(result)


def training_job(setup, directory="adapter_r32_d128"):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "adapter_screen")
    path = study["root"] / "full" / "adapter_screen" / directory / "rice"
    return study, context, pools, path


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_partial_epoch_resume_matches_uninterrupted_model_and_optimizer(setup, monkeypatch):
    study, context, pools, full = training_job(setup, "uninterrupted")
    g2_e2.train_e2(context, pools, full, device="cpu")
    interrupted = runner.run_directory(study, "rice", (32, 128), "adapter_screen")
    original = g2_e2.Overfit16Trainer.train_step

    def fail_second(self, batch, *, step, epoch):
        if step == 2:
            raise RuntimeError("Injected failure")
        return original(self, batch, step=step, epoch=epoch)

    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", fail_second)
    with pytest.raises(RuntimeError, match="Injected"):
        g2_e2.train_e2(context, pools, interrupted, device="cpu")
    first, _ = load_checkpoint_payload(interrupted / "last.pt")
    assert first["training_state"]["global_step"] == 1
    assert first["metadata"]["next_epoch"] == 0 and first["metadata"]["next_batch"] == 1
    assert first["metadata"]["best_metric"] is None
    with pytest.raises(RuntimeError, match="use --resume"):
        g2_e2.train_e2(context, pools, interrupted, device="cpu")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", original)
    result = g2_e2.train_e2(context, pools, interrupted, device="cpu", resume=True)
    assert result["global_step"] == 4
    expected, _ = load_checkpoint_payload(full / "last.pt")
    actual, _ = load_checkpoint_payload(interrupted / "last.pt")
    assert_nested_equal(expected["model_state"], actual["model_state"])
    assert_nested_equal(expected["optimizer_state"], actual["optimizer_state"])
    assert runner.read_valid_result(interrupted, context)


def test_resume_after_final_batch_before_dev_does_not_repeat_updates(setup, monkeypatch):
    study, context, pools, path = training_job(setup)
    # Use a smoke job so the two updates exhaust the declared epoch budget.
    context, pools = runner.make_context(study, "rice", (32, 128), "adapter_screen", smoke=True)
    path = runner.run_directory(study, "rice", (32, 128), "adapter_screen", True)
    evaluate = g2_e2.evaluate_dev
    monkeypatch.setattr(g2_e2, "evaluate_dev", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("DEV crash")))
    with pytest.raises(RuntimeError, match="DEV crash"):
        g2_e2.train_e2(context, pools, path, device="cpu")
    payload, _ = load_checkpoint_payload(path / "last.pt")
    assert payload["training_state"]["global_step"] == 2
    assert payload["metadata"]["next_epoch"] == 0 and payload["metadata"]["next_batch"] == 2
    monkeypatch.setattr(g2_e2, "evaluate_dev", evaluate)
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", lambda *a, **k: pytest.fail("Update repeated"))
    assert g2_e2.train_e2(context, pools, path, device="cpu", resume=True)["status"] == "PASS"
    assert runner.read_valid_result(path, context)


def test_resume_final_checkpoint_can_recreate_metrics_without_updates(setup, monkeypatch):
    _, context, pools, path = training_job(setup)
    g2_e2.train_e2(context, pools, path, device="cpu")
    (path / "metrics.json").unlink()
    last_sha = runner.sha256_file(path / "last.pt")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", lambda *a, **k: pytest.fail("Update repeated"))
    assert g2_e2.train_e2(context, pools, path, device="cpu", resume=True)["status"] == "PASS"
    assert runner.sha256_file(path / "last.pt") == last_sha
    assert runner.read_valid_result(path, context)


@pytest.mark.parametrize("mutation", ["nan", "missing_metric", "budget", "hash", "checkpoint", "dev_uncommitted"])
def test_invalid_completed_runs_never_count_or_retrain(setup, monkeypatch, mutation):
    study, context, pools, path = training_job(setup)
    g2_e2.train_e2(context, pools, path, device="cpu")
    assert runner.read_valid_result(path, context)
    result = json.loads((path / "metrics.json").read_text())
    if mutation == "nan":
        result["best_synthetic_dev"]["synthetic_dev_aupro_0_05"] = float("nan")
    elif mutation == "missing_metric":
        result["best_synthetic_dev"]["synthetic_dev_aupro_0_05"] = None
    elif mutation == "budget":
        result["global_step"] -= 1
    elif mutation == "hash":
        result["config_sha256"] = "bad"
    elif mutation == "checkpoint":
        (path / "best.pt").write_bytes(b"corrupt checkpoint")
    else:
        # Re-sign a valid schema checkpoint with an uncommitted final DEV.
        payload, _ = load_checkpoint_payload(path / "last.pt")
        payload["metadata"]["dev_step"] = 0
        torch.save(payload, path / "last.pt")
        checksum = runner.sha256_file(path / "last.pt")
        (path / "last.pt.sha256").write_text(checksum + "\n")
        result["last_checkpoint_sha256"] = checksum
    (path / "metrics.json").write_text(json.dumps(result))
    assert runner.read_valid_result(path, context) is None
    monkeypatch.setattr(runner, "train_e2", lambda *a, **k: pytest.fail("Invalid PASS must not be overwritten"))
    with pytest.raises(runner.G2Blocked, match="completed run is invalid"):
        runner.execute_job(study, "rice", (32, 128), "adapter_screen", smoke=False, resume=True, device="cpu")


def virtual_ledger(study, monkeypatch):
    """Mock artifact validation for selection UNIT tests only, all in tmp_path."""
    original_reader = runner.read_valid_result
    ledger = {}
    factory = ResidualAdapterFactory(AdapterFactoryConfig(in_dim=384, **study["protocol"]["adapter"]))
    for pair in study["pairs"]:
        for category in runner.CATEGORIES:
            context, _ = runner.make_context(study, category, pair, "adapter_screen")
            # One outstanding category must lose to a balanced macro candidate.
            score = (1.0 if category == "can" else 0.0) if pair == (32, 128) else .1
            if pair in {(64, 256), (128, 384)}:
                score = .55
            ledger[str(runner.run_directory(study, category, pair, "adapter_screen"))] = dict(
                status="PASS", mode="full", category=category, adapter={"r": pair[0], "d": pair[1]},
                config_sha256=context["config_sha256"], metrics_sha256=f"unit-test-{category}-{pair}",
                best_checkpoint_sha256="unit-test-best", last_checkpoint_sha256="unit-test-last",
                decoder_initial_sha256=f"same-init-{category}",
                adapter_trainable_parameters=factory.build_rd(r=pair[0], d=pair[1]).trainable_params,
                best_synthetic_dev={"synthetic_dev_aupro_0_05": score},
                verification_scope="real_pretrained", device="cuda:0")

    def mock_read(path, context):
        if context["config"]["stage"] == "adapter_screen" and context["config"]["mode"] == "full":
            return ledger.get(str(path))
        return original_reader(path, context)

    monkeypatch.setattr(runner, "read_valid_result", mock_read)
    return ledger


def test_lock_only_at_72_macro_selection_parameter_tie_and_idempotence(setup, monkeypatch):
    study = setup["study"]()
    ledger = virtual_ledger(study, monkeypatch)
    key, row = ledger.popitem()
    assert len(runner.collect_results(study, "adapter_screen")[0]) == 71
    assert runner.publish_selection(study) is None
    path = study["root"] / "full" / "adapter_selection_lock.json"
    assert not path.exists()
    ledger[key] = row
    lock = runner.publish_selection(study)
    assert lock["payload"]["valid_runs"] == 72
    assert lock["payload"]["selected_pair"] == {"r": 64, "d": 256}
    assert lock["payload"]["ranking"][0]["macro_synthetic_dev_aupro_0_05"] == .55
    assert runner.validate_selection(study) == lock
    mtime = path.stat().st_mtime_ns
    assert runner.publish_selection(study) == lock
    assert path.stat().st_mtime_ns == mtime
    ledger.pop(key)
    with pytest.raises(runner.G2Blocked, match="71/72"):
        runner.validate_selection(study)


def test_lock_checksum_and_decoder_init_drift_block_selection(setup, monkeypatch):
    study = setup["study"]()
    ledger = virtual_ledger(study, monkeypatch)
    lock = runner.publish_selection(study)
    lock["payload"]["selected_pair"]["r"] = 32
    runner.save_json(lock, study["root"] / "full" / "adapter_selection_lock.json")
    with pytest.raises(runner.G2Blocked, match="checksum"):
        runner.validate_selection(study)
    next(iter(ledger.values()))["decoder_initial_sha256"] = "changed"
    with pytest.raises(ValueError, match="initialization drift"):
        runner.rank_candidates(study, list(ledger.values()))


def test_e2_all_eight_smoke_categories_use_locked_pair(setup, monkeypatch):
    study = setup["study"]()
    virtual_ledger(study, monkeypatch)
    runner.publish_selection(study)
    assert runner.main(["--config", str(setup["path"]), "--stage", "E2", "--categories", "all",
                        "--device", "cpu", "--smoke"]) == 0
    path = study["root"] / "smoke" / "E2_summary.json"
    result = json.loads(path.read_text())
    assert result["status"] == "SMOKE_PASS" and len(result["outcomes"]) == 8
    assert {row["status"] for row in result["outcomes"]} == {"PASS"}
    assert {(row["r"], row["d"]) for row in result["outcomes"]} == {(64, 256)}
    assert result["real_E2_pass"] == 0
    # The screen evidence is mock-only; none of this temp study is a real run.


def test_e2_missing_lock_blocked_before_gpu_or_training(setup):
    assert runner.main(["--config", str(setup["path"]), "--stage", "E2", "--device", "cuda"]) == 2
    result = json.loads((setup["root"] / "dinov3_vits16" / "full" / "E2_summary.json").read_text())
    assert result["status"] == "BLOCKED" and "lock" in result["reason"]
    assert result["real_adapter_screen_pass"] == result["real_E2_pass"] == 0


def test_full_without_cuda_not_run_zero_real_counts(setup, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runner.main(["--config", str(setup["path"]), "--stage", "adapter_screen", "--device", "cuda"]) == 2
    result = json.loads((setup["root"] / "dinov3_vits16" / "full" / "adapter_screen_summary.json").read_text())
    assert result["status"] == "NOT RUN"
    assert result["real_adapter_screen_pass"] == result["real_E2_pass"] == 0
    assert not list(setup["root"].rglob("*.pt"))


def test_resume_config_drift_rejected_before_writes(setup, monkeypatch):
    _, context, pools, path = training_job(setup)
    g2_e2.train_e2(context, pools, path, device="cpu")
    checksum = runner.sha256_file(path / "last.pt")
    changed = deepcopy(context)
    changed["config"]["training"]["seed"] += 1
    changed["config_sha256"] = runner.sha256_json(changed["config"])
    monkeypatch.setattr(g2_e2, "build_g2_model", lambda *a, **k: pytest.fail("Drift must fail before model loading"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        g2_e2.train_e2(changed, pools, path, device="cpu", resume=True)
    assert runner.sha256_file(path / "last.pt") == checksum
