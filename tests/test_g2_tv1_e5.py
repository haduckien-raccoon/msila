"""D9 E5/factory/45-cell technical acceptance; CPU fixtures, no selection."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import g2_fusion_preflight as preflight
from src.models.attention_fusion import AttentionFusion
from src.models.contracts import MULTIVIEW_FEATURE_KEYS, ContractError
from src.models.fusion_factory import FUSION_METHODS, build_fusion, canonical_fusion_method
from src.models.fusions import ConcatFusion, GatedFusion, WeightedSumFusion
from src.models.msila import E5, build_g2_model, resolve_g2_config
from src.models.position_attention_fusion import PositionAwareAttentionFusion
from tests.test_g2_tv1_e4 import pair_batch
from tests.test_g2_tv1_runner import setup, cpu_determinism, runner, assert_nested_equal

ROOT = Path(__file__).resolve().parents[1]


def features(dim=7, dtype=torch.float32):
    return {key: torch.randn(2, dim, 3, 5, dtype=dtype, requires_grad=True)
            for key in reversed(MULTIVIEW_FEATURE_KEYS)}


@pytest.mark.parametrize("label,canonical", [("average", "mean"), ("weighted", "weighted_sum"),
                                             *[(name, name) for name in FUSION_METHODS]])
def test_factory_aliases_and_reuses_tv2_modules(label, canonical):
    model = build_fusion(label, 7)
    assert canonical_fusion_method(label) == model.method == canonical
    types = dict(weighted_sum=WeightedSumFusion, concat=ConcatFusion, gated=GatedFusion,
                 attention=PositionAwareAttentionFusion)
    if canonical in types:
        assert isinstance(model.module, types[canonical])


@pytest.mark.parametrize("method", FUSION_METHODS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
def test_factory_shape_gradients_order_dtype_and_checkpoint(method, dtype):
    model = build_fusion(method, 7).to(dtype=dtype)
    source = features(dtype=dtype)
    out = model(source)
    assert out.shape == (2, 7, 3, 5) and out.dtype == dtype and torch.isfinite(out).all()
    assert torch.equal(out, model(dict(reversed(list(source.items())))))
    out.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in source.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
    checkpoint = io.BytesIO()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.05)
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = build_fusion(method, 7).to(dtype=dtype)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert torch.equal(model(source), restored(source))


@pytest.mark.parametrize("method", FUSION_METHODS)
def test_factory_cpu_autocast_and_actual_parameter_counts(method):
    dim = 7
    model = build_fusion(method, dim)
    expected = dict(mean=0, weighted_sum=6, concat=6*dim*dim+dim, gated=6*(dim+1), attention=dim+6)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected[method]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(features())
    assert output.dtype == torch.float32 and torch.isfinite(output).all()


def test_attention_is_spatial_and_content_dependent_with_source_softmax():
    model = PositionAwareAttentionFusion(1)
    source = {key: torch.zeros(1, 1, 2, 3) for key in MULTIVIEW_FEATURE_KEYS}
    source["local_b4"][0, 0, 0, 0] = 4
    source["local_b8"][0, 0, 1, 2] = 4  # Same GAP, different spatial positions.
    with torch.no_grad():
        model.score_proj.weight.fill_(1)
    result = model(source, return_trace=True)
    assert result["logits"].shape == result["attention"].shape == (1, 6, 2, 3)
    assert torch.equal(result["logits"][:, 0], source["local_b4"][:, 0])
    assert result["attention"][0, 0, 0, 0] > result["attention"][0, 1, 0, 0]
    assert result["attention"][0, 0, 1, 2] < result["attention"][0, 1, 1, 2]
    torch.testing.assert_close(result["attention"].sum(1), torch.ones(1, 2, 3))
    expected = (torch.stack([source[key] for key in MULTIVIEW_FEATURE_KEYS], dim=1)
                * result["attention"][:, :, None]).sum(1)
    assert torch.equal(result["feature"], expected)
    # Legacy G1/Day-2 API remains global; it is never the E5 attention candidate.
    assert AttentionFusion(1)(source)["logits"].shape == (1, 6)


def test_spatial_attention_starts_uniform_and_rejects_nonfinite_scorer():
    model = PositionAwareAttentionFusion(7)
    source = features()
    result = model(source, return_trace=True)
    torch.testing.assert_close(result["attention"], torch.full((2, 6, 3, 5), 1/6))
    torch.testing.assert_close(result["feature"], torch.stack([source[k] for k in MULTIVIEW_FEATURE_KEYS]).mean(0))
    with torch.no_grad():
        model.score_proj.weight.fill_(float("nan"))
    with pytest.raises(ContractError, match="NaN/Inf"):
        model(source)


@pytest.mark.parametrize("method", FUSION_METHODS)
@pytest.mark.parametrize("invalid", ("missing", "extra", "channels", "grid", "dtype", "nan", "inf", "integer", "not_tensor"))
def test_factory_rejects_invalid_six_source_contract(method, invalid):
    source = features()
    key = "context_b12"
    if invalid == "missing":
        source.pop(key)
    elif invalid == "extra":
        source["unknown"] = source[key]
    elif invalid == "channels":
        source[key] = source[key][:, :-1]
    elif invalid == "grid":
        source[key] = source[key][:, :, :-1]
    elif invalid == "dtype":
        source[key] = source[key].double()
    elif invalid in {"nan", "inf"}:
        with torch.no_grad():
            source[key][0, 0, 0, 0] = float(invalid)
    elif invalid == "integer":
        source[key] = source[key].long()
    else:
        source[key] = None
    with pytest.raises(ContractError):
        build_fusion(method, 7)(source)


@pytest.mark.parametrize("dim", (0, -1, True, 1.5, "64"))
def test_factory_and_spatial_attention_reject_invalid_width(dim):
    for constructor in (lambda: build_fusion("attention", dim), lambda: PositionAwareAttentionFusion(dim)):
        with pytest.raises(ValueError, match="positive integer"):
            constructor()


@pytest.mark.parametrize("method", (None, "", "global_attention", "unsupported"))
def test_unknown_fusion_is_rejected(method):
    with pytest.raises(ValueError):
        build_fusion(method, 7)


@pytest.mark.parametrize("method", FUSION_METHODS)
def test_e5_changes_only_fusion_and_preserves_e4_paired_pipeline(setup, method):
    cfg = deepcopy(setup["cfg"])
    cfg["backbone"]["name"] = "dinov3_vitb16"
    cfg["adapter"].update(r=64, d=256)
    cfg["fusion"].update(method=method, dim=64)
    cfg.update(adapter_pair_role="debug_pair", debug_pair=runner.debug_pair((64, 256)))
    torch.manual_seed(2026)
    baseline = build_g2_model(cfg, "E4")
    torch.manual_seed(2026)
    model = build_g2_model(cfg, "E5")
    assert isinstance(model, E5) and model.fusion_method == method
    for name in ("extractor", "adapters", "projection", "decoder", "aligner"):
        assert type(getattr(model, name)) is type(getattr(baseline, name))
        assert_nested_equal(getattr(model, name).state_dict(), getattr(baseline, name).state_dict())
    batch = pair_batch(1)
    logits, trace = model(batch, return_trace=True)
    assert logits.shape == (1, 1, 512, 512) and torch.isfinite(logits).all()
    assert trace["source_keys"] == MULTIVIEW_FEATURE_KEYS and trace["num_sources"] == 6
    assert trace["source_blocks"] == dict(zip(MULTIVIEW_FEATURE_KEYS, (4, 8, 12)*2))
    assert all(x.shape == (1, 64, 32, 32) for x in trace["projected"].values())
    if method == "mean":
        assert torch.equal(logits, baseline(batch))
    if method == "attention":
        assert trace["fusion_trace"]["logits"].shape == (1, 6, 32, 32)
    if method == "concat":
        assert "weights" not in trace["fusion_trace"]
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None and not p.requires_grad for p in model.extractor.parameters())
    assert not model.extractor.training


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    root = tmp_path_factory.mktemp("D9_45")
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    summary = preflight.run_cpu(cfg, root)
    assert summary["cpu_pass"] == 45, summary["errors"]
    return root


@pytest.mark.parametrize("pair", preflight.PAIRS)
@pytest.mark.parametrize("method", FUSION_METHODS)
def test_each_of_45_cpu_combinations(matrix, pair, method):
    record = json.loads((matrix / "cpu" / f"r{pair[0]}_d{pair[1]}_{method}.json").read_text())
    assert record["status"] == "PASS" and record["gpu_status"] == "NOT RUN"
    assert record["shape"] == [1, 64, 4, 4] and record["logits_shape"] == [1, 1, 512, 512]
    assert record["frozen_backbone"] and record["finite_gradients"] and record["source_gradients"] == 6
    assert record["forward_backward_probes"] == 2 and record["experiment_runs"] == 0
    assert preflight.sha256_json(record["config"]) == record["config_sha256"]
    assert record["peak_vram_if_measured"] is None
    if method == "attention":
        assert record["attention_logits_shape"] == [1, 6, 4, 4]


def test_matrix_coverage_capacity_shared_heads_hashes_and_gpu_plan(matrix):
    rows = list(csv.DictReader((matrix / "fusion_preflight_45.csv").open()))
    assert len(rows) == 45 and len({row["config_sha256"] for row in rows}) == 45
    assert {(int(row["r"]), int(row["d"]), row["fusion"]) for row in rows} == {
        (*pair, method) for pair in preflight.PAIRS for method in FUSION_METHODS}
    assert {row["status"] for row in rows} == {"PASS"} and {row["gpu_status"] for row in rows} == {"NOT RUN"}
    assert all(row["peak_vram_if_measured"] == "" for row in rows)
    assert len({row["adapter_params"] for row in rows}) == 9
    for pair in preflight.PAIRS:
        group = [json.loads((matrix / "cpu" / f"r{pair[0]}_d{pair[1]}_{method}.json").read_text()) for method in FUSION_METHODS]
        assert all(row["base_initial_sha256"] == group[0]["base_initial_sha256"] for row in group)
    manifest = json.loads((matrix / "run_manifest.json").read_text())
    assert manifest["selection_lock"] is None and manifest["experiment_runs"] == 0
    plan = json.loads((matrix / "gpu_preflight_plan.json").read_text())
    assert plan["status"] == "NOT RUN" and plan["attention_logits"] == ["B", 6, 32, 32]
    assert not list(matrix.rglob("*.pt")) and not list(matrix.rglob("*selection_lock.json"))


def test_invalid_combination_is_retained_with_reason_without_dropping_rows(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    original = preflight.check_model
    def fail_one(model, candidate):
        if (candidate["adapter"]["r"], candidate["adapter"]["d"], candidate["fusion"]["method"]) == (64, 256, "concat"):
            raise ValueError("Injected infeasible combination; keep explicit FAIL")
        return original(model, candidate)
    monkeypatch.setattr(preflight, "check_model", fail_one)
    summary = preflight.run_cpu(cfg, tmp_path)
    rows = list(csv.DictReader((tmp_path / "fusion_preflight_45.csv").open()))
    assert len(rows) == 45 and summary["cpu_pass"] == 44
    failed = [row for row in rows if row["status"] == "FAIL"]
    assert len(failed) == 1 and "infeasible" in failed[0]["error"]


def test_runner_d9_is_cpu_only_and_e5_training_pending(setup, monkeypatch):
    monkeypatch.setattr(runner, "prepare_study", lambda *a, **k: pytest.fail("D9 must not prepare dataset training"))
    # Test CLI orchestration without repeating the already tested 45 probes.
    monkeypatch.setattr(preflight, "run_cpu", lambda cfg, root: dict(status="PASS", cpu_pass=45))
    assert runner.main(["--config", str(setup["path"]), "--stage", "fusion_preflight", "--device", "cpu",
                        "--backbone", "dinov3_vitb16", "--output-root", str(setup["root"])]) == 0
    assert runner.main(["--config", str(setup["path"]), "--stage", "E5", "--device", "cpu",
                        "--output-root", str(setup["root"])]) == 2
    result = json.loads((setup["root"] / "dinov3_vits16" / "full" / "E5_summary.json").read_text())
    assert result["status"] == "BLOCKED" and result["real_E5_pass"] == 0


def test_config_serialization_hash_aliases_and_source_order():
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    mean = preflight.candidate_config(cfg, (64, 256), "average")
    assert mean["fusion"]["method"] == "mean"
    other = preflight.candidate_config(cfg, (64, 256), "weighted")
    assert other["fusion"]["method"] == "weighted_sum" and preflight.sha256_json(mean) != preflight.sha256_json(other)
    cfg["fusion"]["feature_order"] = list(reversed(MULTIVIEW_FEATURE_KEYS))
    with pytest.raises(ValueError, match="canonical source slots"):
        preflight.candidate_config(cfg, (64, 256), "mean")


def test_e2_config_without_fusion_keeps_legacy_checkpoint_contract():
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg.pop("fusion")
    resolved = resolve_g2_config(cfg)
    assert "fusion" not in resolved and "fusion" not in cfg
    assert resolved["backbone"]["channels"] == 768
