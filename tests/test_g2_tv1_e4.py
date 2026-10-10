"""D8 E4 acceptance: paired native views, alignment, gradients, resume and export.

Fixture training/selection is explicitly not real pretrained PASS/8 evidence.
"""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from src.data.loader import G1NativeDataset, G1TileDataset
from src.data.tiling import extract_local_context, generate_tile_records
from src.geometry.view_meta import build_view_meta
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.backbone_registry import BACKBONES, backbone_spec
from src.models.context_alignment import ContextToLocalAligner
from src.models.contracts import ContractError, MULTIVIEW_FEATURE_KEYS
from src.models.msila import E4, build_g2_model
from src.train import g2_e2
from src.train.g2_context import (PairedG2Tiles, paired_model_forward, paired_tile_collate,
                                  prepare_pair, predict_native_e4)
from src.utils.checkpoint import save_training_checkpoint
from src.utils.resume import load_checkpoint_payload
from tests.test_g2_tv1_e3 import LayerBackboneFixture
from tests.test_g2_tv1_runner import setup, cpu_determinism, runner, virtual_ledger, assert_nested_equal


def pair_batch(count=2):
    # Different geometry per batch item, including a padded boundary tile.
    image = torch.rand(3, 777, 931)
    records = generate_tile_records(777, 931)
    samples = []
    for record in records[:count]:
        local, context, geometry = prepare_pair(image, record, sample_id="same-native")
        mask = torch.zeros(1, 512, 512)
        mask[:, 81:97, 151:176] = 1
        samples.append(dict(image=local, context=context, view_meta=geometry, mask=mask, meta={}))
    return paired_tile_collate(samples)


@pytest.mark.parametrize("name", tuple(BACKBONES))
def test_six_sources_same_e3_parameters_decoder_and_gradient(setup, tmp_path, monkeypatch, name):
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: LayerBackboneFixture(kwargs["model"]))
    weights = tmp_path / f"{name}_layers.pth"
    torch.save(LayerBackboneFixture(name).state_dict(), weights)
    cfg = deepcopy(setup["cfg"])
    cfg["backbone"].update(name=name, weights=str(weights))
    cfg["adapter"].update(r=8, d=12)
    cfg["fusion"]["dim"] = 8
    cfg["decoder"]["deterministic_resize"] = True
    torch.manual_seed(2026)
    e3 = build_g2_model(cfg, "E3")
    torch.manual_seed(2026)
    model = build_g2_model(cfg, "E4")
    assert isinstance(model, E4) and isinstance(model.aligner, ContextToLocalAligner)
    assert model.num_sources == 6 and len(model.adapters) == 3
    assert model.num_trainable_parameters == e3.num_trainable_parameters
    assert_nested_equal(g2_e2.trainable_modules(model, "E4").state_dict(),
                        g2_e2.trainable_modules(e3, "E3").state_dict())
    assert not list(model.aligner.parameters()) and not list(model.fusion.parameters())
    assert model.projection.context_projectors is None
    batch = pair_batch()
    frozen = g2_e2.module_sha256(model.extractor)
    groups = g2_e2.trainable_modules(model, "E4")
    assert {id(p) for p in groups.parameters()} == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups.parameters(), lr=.01, weight_decay=0)
    model.train()
    logits, trace = model(batch, return_trace=True)
    assert logits.shape == (2, 1, 512, 512)
    assert model.extractor.backbone.requested == [tuple(b-1 for b in backbone_spec(name).blocks)]
    assert trace["source_keys"] == MULTIVIEW_FEATURE_KEYS and trace["num_sources"] == 6
    assert trace["source_blocks"] == dict(zip(MULTIVIEW_FEATURE_KEYS, backbone_spec(name).blocks * 2))
    for slot, block in zip((4, 8, 12), backbone_spec(name).blocks):
        assert torch.equal(trace["adapted"][f"local_b{slot}"], trace["dino"][f"L{block}"])
        assert torch.equal(trace["adapted"][f"context_b{slot}"], trace["dino"][f"C{block}"])
    # Independent reference for alignment; projection must consume the aligned tensor.
    ctx = {f"C{slot}": trace["adapted"][f"context_b{slot}"] for slot in (4, 8, 12)}
    aligned = model.aligner(ctx, trace["geometry"], target_hw=(32, 32))
    for slot in (4, 8, 12):
        assert torch.equal(aligned[f"C{slot}_to_L"], trace["aligned_context"][f"C{slot}_to_L"])
        expected = model.projection.projectors[f"b{slot}"](aligned[f"C{slot}_to_L"])
        assert torch.equal(expected, trace["projected"][f"context_b{slot}"])
    assert torch.equal(trace["decoder_feature"], torch.stack([
        trace["projected"][key] for key in MULTIVIEW_FEATURE_KEYS]).mean(0))
    assert trace["decoder_feature"].shape == (2, 8, 32, 32)
    for value in trace["adapted"].values():
        value.retain_grad()
    criterion = AnomalySegmentationLoss()
    criterion(logits, batch["mask"])["loss"].backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in trace["adapted"].values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in groups.parameters())
    for adapter in model.adapters.values():
        assert adapter.gamma.grad.abs() > 0
        assert all(torch.count_nonzero(p.grad) == 0 for key, p in adapter.named_parameters() if key != "gamma")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    criterion(model(batch), batch["mask"])["loss"].backward()
    for adapter in model.adapters.values():
        assert any(torch.count_nonzero(p.grad) > 0 for key, p in adapter.named_parameters() if key != "gamma")
    assert g2_e2.module_sha256(model.extractor) == frozen
    assert not model.extractor.training and all(p.grad is None for p in model.extractor.parameters())


def test_geometry_coordinate_ramp_and_alignment_gradient():
    # A non-centered crop detects direction/translation errors; pixel-edge ramp
    # can be sampled analytically at the Local feature centers.
    geom = build_view_meta(source_hw=(1000, 1200), local_box_xyxy=(230, 100, 742, 612),
                           context_box_xyxy=(100, 30, 868, 798)).as_tensor_dict(dtype=torch.float32)
    centers = (torch.arange(32) + .5) * 16
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    context_ramp = (x + 2*y)[None, None].requires_grad_()
    aligner = ContextToLocalAligner(deterministic_sampling=True, check_bounds=True)
    result = aligner(dict.fromkeys(("C4", "C8", "C12"), context_ramp), geom, target_hw=(32, 32))
    expected = ((x + 130) * (512/768) + 2*(y + 70) * (512/768))[None, None]
    for value in result.values():
        torch.testing.assert_close(value, expected, atol=1e-4, rtol=1e-6)
    sum(value.sum() for value in result.values()).backward()
    assert context_ramp.grad is not None and torch.isfinite(context_ramp.grad).all()
    assert context_ramp.grad.sum() == pytest.approx(3*32*32)


def test_concat_and_sequential_share_backbone_and_context_changes_logits(setup):
    model = build_g2_model(setup["cfg"], "E4").eval()
    batch = pair_batch(1)
    with torch.no_grad():
        concat = model(batch)
        model.pair_strategy = "sequential"
        sequential = model(batch)
        altered = {**batch, "context": batch["context"] + 1}
        changed = model(altered)
    torch.testing.assert_close(concat, sequential, atol=1e-6, rtol=1e-6)
    assert not torch.equal(concat, changed)
    assert model.extractor.backbone.requested == [(3, 7, 11)] * 5


def test_old_model_config_uses_existing_context_defaults(setup):
    path = Path(setup["runner"]["model_config"])
    old = yaml.safe_load(path.read_text())
    old.pop("paired_views")
    path.write_text(yaml.safe_dump(old))
    args = runner.parse_args(["--config", str(setup["path"]), "--stage", "E4"])
    cfg, _, _ = runner.load_config(args)
    assert cfg["paired_views"] == dict(context_size=768, pair_strategy="concat", deterministic_sampling=True)


@pytest.mark.parametrize("case", ["tensor", "missing_context", "missing_geometry", "singular", "wrong_size", "outside", "different_sample"])
def test_invalid_e4_pair_rejected(setup, case):
    model = build_g2_model(setup["cfg"], "E4")
    batch = pair_batch(1)
    if case == "tensor":
        batch = batch["image"]
    elif case == "missing_context":
        del batch["context"]
    elif case == "missing_geometry":
        del batch["view_meta"]
    elif case == "different_sample":
        batch["view_meta"][0]["context_sample_id"] = "different-native"
    else:
        geom = batch["view_meta"][0]["geometry"]
        if case == "singular":
            geom["local_to_context"].zero_()
        elif case == "wrong_size":
            geom["context_input_hw"] = torch.tensor([768, 768])
        elif case == "outside":
            geom["local_to_context"][0, 2] = 2000
    with pytest.raises((ContractError, ValueError, RuntimeError)):
        model(batch)


def test_paired_tiles_reuse_one_native_synthetic_and_exact_e3_local_mask(setup):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", selection_sha256="fixture-lock")
    cfg = context["config"]
    native = G1NativeDataset(pools["train"], cfg["synthetic_protocol"], seed=cfg["training"]["seed"],
                             role="train", variants=2, fixed=False)
    tiles = G1TileDataset(native)
    paired = PairedG2Tiles(tiles)
    assert len(paired) == len(tiles)
    # Count __getitem__ calls independently of native's synthetic cache.
    class CountNative:
        def __init__(self):
            self.calls = []
        def __getitem__(self, index):
            self.calls.append(index)
            return native[index]
    counted = CountNative()
    tiles.native = counted
    for index in range(len(tiles)):
        sample = paired[index]
        assert counted.calls[-1] == tiles.records[index][0]
        assert len(counted.calls) == index + 1
        original = native[tiles.records[index][0]]
        expected = G1TileDataset.__getitem__(tiles, index)
        counted.calls.pop()  # exclude the independent reference read
        assert torch.equal(sample["image"], expected["image"])
        assert torch.equal(sample["mask"], expected["mask"])
        record = generate_tile_records(*original["image"].shape[-2:])[0]
        _, raw_context = extract_local_context(original["image"], record)
        from src.data.loader import normalize_dinov3
        assert torch.equal(sample["context"], normalize_dinov3(raw_context))
        meta = sample["view_meta"]
        assert meta["local_sample_id"] == meta["context_sample_id"] == original["meta"]["sample_id"]
        assert min(meta["geometry"]["local_box_xyxy"]) >= 0
        assert torch.allclose(meta["geometry"]["local_to_context"], torch.tensor([
            [2/3, 0, 256/3], [0, 2/3, 256/3], [0, 0, 1]], dtype=torch.float64))


@pytest.mark.parametrize("hw", [(73, 101), (777, 931)])
def test_native_hann_stitch_exact_coordinates_and_full_resolution(hw):
    class NativeRamp(nn.Module):
        def forward(self, batch):
            logits = []
            for meta in batch["view_meta"]:
                x0, y0, _, _ = meta["local_native_xyxy"]
                y, x = torch.meshgrid(torch.arange(512)+y0, torch.arange(512)+x0, indexing="ij")
                logits.append(((x + 2*y)/(hw[1]+2*hw[0]) - .5)[None])
            return torch.stack(logits)
    cfg = dict(data=dict(tile_size=512, overlap=128), paired_views=dict(context_size=768),
               evaluation=dict(tile_batch_size=2))
    score = predict_native_e4(NativeRamp(), torch.rand(3, *hw), cfg, "cpu")
    y, x = torch.meshgrid(torch.arange(hw[0]), torch.arange(hw[1]), indexing="ij")
    expected = ((x+2*y)/(hw[1]+2*hw[0]) - .5).sigmoid()
    assert score.shape == hw
    torch.testing.assert_close(score, expected, atol=2e-7, rtol=1e-6)


def test_hann_overlap_uses_window_weights_instead_of_plain_average():
    h, w = 613, 777
    class DifferentTiles(nn.Module):
        def forward(self, batch):
            return torch.stack([torch.full((1, 512, 512), meta["local_native_xyxy"][0]/100 - 1)
                                for meta in batch["view_meta"]])
    cfg = dict(data=dict(tile_size=512, overlap=128), paired_views=dict(context_size=768),
               evaluation=dict(tile_batch_size=3))
    actual = predict_native_e4(DifferentTiles(), torch.rand(3, h, w), cfg, "cpu")
    window = torch.hann_window(512, periodic=False)
    window = torch.outer(window, window).clamp_min(.001)
    numerator, denominator = torch.zeros(h, w), torch.zeros(h, w)
    for record in generate_tile_records(h, w):
        x0, y0, x1, y1 = record.local_xyxy
        probability = torch.tensor(x0/100 - 1).sigmoid()
        numerator[y0:y1, x0:x1] += probability * window
        denominator[y0:y1, x0:x1] += window
    assert torch.all(denominator > 0)
    torch.testing.assert_close(actual, numerator / denominator, atol=1e-7, rtol=1e-6)
    # In this overlap pixel the left tile dominates through its Hann weight.
    assert abs(actual[306, 280].item() - (.2689414 + .8388911)/2) > .1


def test_e4_resume_exact_checkpoint_and_inference_without_training(setup, monkeypatch):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", selection_sha256="fixture-lock")
    e3_context, _ = runner.make_context(study, "rice", (32, 128), "E3", selection_sha256="fixture-lock")
    assert runner.comparison_identity(context["config"]) == runner.comparison_identity(e3_context["config"])
    control = study["root"] / "control" / "rice"
    g2_e2.train_e4(context, pools, control, device="cpu")
    path = runner.run_directory(study, "rice", (32, 128), "E4")
    original = g2_e2.Overfit16Trainer.train_step
    def interrupt(self, batch, *, step, epoch):
        if step == 2:
            raise RuntimeError("Injected E4 interruption")
        return original(self, batch, step=step, epoch=epoch)
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        g2_e2.train_e4(context, pools, path, device="cpu")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", original)
    result = g2_e2.train_e4(context, pools, path, device="cpu", resume=True)
    assert result["global_step"] == context["expected_steps"] == 4
    assert runner.read_valid_result(path, context) and result["feature_sources"] == 6
    assert result["context_aligned"] and result["projection_updated"] and all(result["adapters_updated"].values())
    assert not runner.real_evidence(result)
    actual, _ = load_checkpoint_payload(path / "last.pt")
    expected, _ = load_checkpoint_payload(control / "last.pt")
    for key in ("model_state", "optimizer_state"):
        assert_nested_equal(expected[key], actual[key])
    before = runner.sha256_file(path / "best.pt")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", lambda *a, **k: pytest.fail("Inference trained"))
    inference = g2_e2.infer_e4(context, pools, path, device="cpu")
    assert inference["metrics"] == result["best_synthetic_dev"]
    assert runner.sha256_file(path / "best.pt") == before and inference["optimizer_steps"] == 0
    for record in inference["predictions"]:
        assert np.load(record["score"]).shape == np.load(record["mask"]).shape == tuple(record["original_hw"])


def test_e4_resume_final_batch_retries_dev_without_optimizer_update(setup, monkeypatch):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", smoke=True, selection_sha256="fixture-lock")
    path = runner.run_directory(study, "rice", (32, 128), "E4", True)
    evaluate = g2_e2.evaluate_dev
    def crash(*args, **kwargs):
        raise RuntimeError("Injected E4 DEV crash")
    monkeypatch.setattr(g2_e2, "evaluate_dev", crash)
    with pytest.raises(RuntimeError, match="DEV crash"):
        g2_e2.train_e4(context, pools, path, device="cpu")
    checkpoint, _ = load_checkpoint_payload(path / "last.pt")
    assert checkpoint["training_state"]["global_step"] == context["expected_steps"] == 2
    assert checkpoint["metadata"]["dev_step"] == 0
    monkeypatch.setattr(g2_e2, "evaluate_dev", evaluate)
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", lambda *a, **k: pytest.fail("Final batch repeated"))
    assert g2_e2.train_e4(context, pools, path, device="cpu", resume=True)["status"] == "PASS"
    assert runner.read_valid_result(path, context)


def test_all_eight_fixture_smoke_skip_inference_and_tv2_comparison(setup, monkeypatch):
    study = setup["study"]()
    virtual_ledger(study, monkeypatch)  # Mock only the selection unit boundary in tmp_path.
    runner.publish_selection(study)
    common = ["--config", str(setup["path"]), "--categories", "all", "--device", "cpu", "--smoke"]
    assert runner.main([*common, "--stage", "E3"]) == 0
    assert runner.main([*common, "--stage", "E4"]) == 0
    path = study["root"] / "smoke"
    summary = json.loads((path / "E4_summary.json").read_text())
    assert summary["status"] == "SMOKE_PASS" and summary["real_E4_pass"] == 0
    assert len(summary["outcomes"]) == 8 and all(row["status"] == "PASS" for row in summary["outcomes"])
    comparison = json.loads((path / "E4_minus_E3_inputs.json").read_text())
    assert comparison["status"] == "SMOKE_READY" and comparison["ready_pairs"] == 8
    assert comparison["real_ready_pairs"] == 0 and comparison["experiments"] == ["E3", "E4"]
    with (path / "E4_minus_E3_inputs.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 8
    for row in comparison["records"]:
        assert row["e3"]["parameter_report"]["groups"] == row["e4"]["parameter_report"]["groups"]
        assert row["e3"]["sources_sha256"] == row["e4"]["sources_sha256"]
        for experiment in ("e3", "e4"):
            assert Path(row[experiment]["checkpoint"]).is_file()
    monkeypatch.setattr(runner, "train_e4", lambda *a, **k: pytest.fail("Completed E4 trained again"))
    assert runner.main([*common, "--stage", "E4", "--resume"]) == 0
    assert {row["status"] for row in json.loads((path / "E4_summary.json").read_text())["outcomes"]} == {"SKIP"}
    assert runner.main([*common, "--stage", "E4", "--inference"]) == 0
    inference = json.loads((path / "E4_inference_summary.json").read_text())
    assert inference["operation"] == "inference" and len(inference["outcomes"]) == 8
    baseline_metrics = path / "E3" / "walnuts" / "metrics.json"
    saved = baseline_metrics.read_bytes()
    baseline_metrics.unlink()
    lock = json.loads((study["root"] / "full" / "adapter_selection_lock.json").read_text())
    assert runner.export_comparison(study, lock, smoke=True, experiments=("E3", "E4"))["status"] == "INCOMPLETE"
    # Leave a complete, checksummed fixture handoff available for inspection.
    baseline_metrics.write_bytes(saved)
    assert runner.export_comparison(study, lock, smoke=True, experiments=("E3", "E4"))["status"] == "SMOKE_READY"


def test_e4_missing_lock_and_incomplete_inference_block(setup, monkeypatch):
    monkeypatch.setattr(runner, "train_e4", lambda *a, **k: pytest.fail("Missing lock trained"))
    assert runner.main(["--config", str(setup["path"]), "--stage", "E4", "--smoke", "--device", "cpu"]) == 2
    path = setup["root"] / "dinov3_vits16"
    result = json.loads((path / "smoke" / "E4_summary.json").read_text())
    assert result["status"] == "BLOCKED" and result["real_E4_pass"] == 0
    assert runner.main(["--config", str(setup["path"]), "--stage", "E4", "--device", "cpu"]) == 2
    result = json.loads((path / "full" / "E4_summary.json").read_text())
    assert result["status"] == "BLOCKED" and result["real_E4_pass"] == 0
    study = setup["study"]()
    with pytest.raises(runner.G2Blocked, match="completed training"):
        runner.execute_job(study, "rice", (32, 128), "E4", smoke=True, resume=False,
                           device="cpu", selection_sha256="fixture-lock", inference=True)
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "E3", "--inference"])


@pytest.mark.parametrize("smoke", [False, True])
def test_d7_checkpoint_config_survives_e4_extension_and_rejects_drift(setup, smoke):
    study = setup["study"]()
    historical = deepcopy(study["protocol"])
    for name in historical["source_code_sha256"]:
        if name in runner.D7_SOURCE_BASELINE:
            historical["source_code_sha256"][name] = runner.D7_SOURCE_BASELINE[name]
    runner.enforce_lock(study["root"] / "full", historical)
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E3", smoke=smoke, selection_sha256="fixture-lock")
    for name in context["config"]["architecture"]["implementation_sha256"]:
        if name in runner.D7_SOURCE_BASELINE:
            context["config"]["architecture"]["implementation_sha256"][name] = runner.D7_SOURCE_BASELINE[name]
    context["config_sha256"] = runner.sha256_json(context["config"])
    path = runner.run_directory(study, "rice", (32, 128), "E3", smoke)
    g2_e2.train_e3(context, pools, path, device="cpu")
    reloaded = setup["study"]()
    restored, _ = runner.make_context(reloaded, "rice", (32, 128), "E3", smoke=smoke, selection_sha256="fixture-lock")
    assert restored == context and runner.read_valid_result(path, restored)
    assert runner.execute_job(reloaded, "rice", (32, 128), "E3", smoke=smoke, resume=True,
                              device="cpu", selection_sha256="fixture-lock") == "SKIP"
    architecture = reloaded["e3_smoke"] if smoke else reloaded["e3"]
    changed = deepcopy(architecture)
    changed["feature_width"] += 1
    with pytest.raises(runner.G2Blocked, match="scientific E3 architecture"):
        runner.reuse_e3_architecture(changed, study["root"], study["sha256"], mode="smoke" if smoke else "full")
    changed = deepcopy(architecture)
    changed["implementation_sha256"]["src/models/feature_projection.py"] = "unaudited"
    with pytest.raises(runner.G2Blocked, match="unaudited implementation"):
        runner.reuse_e3_architecture(changed, study["root"], study["sha256"], mode="smoke" if smoke else "full")


def test_real_locked_e4_one_batch_checkpoint_if_available(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("NOT RUN: real E4 batch/checkpoint requires CUDA and a full Adapter selection lock")
    argv = ["--stage", "E4", "--device", "cuda"]
    for flag, variable in (("--data-root", "MVTEC_AD2_ROOT"), ("--repo-dir", "DINOV3_REPO"),
                           ("--weights", "DINOV3_WEIGHTS"), ("--backbone", "DINOV3_MODEL"),
                           ("--output-root", "G2_OUTPUT_ROOT")):
        if os.getenv(variable):
            argv.extend([flag, os.environ[variable]])
    args = runner.parse_args(argv)
    cfg, config, root = runner.load_config(args)
    if not (root / "full" / "adapter_selection_lock.json").is_file():
        pytest.skip("NOT RUN: real E4 integration has no Adapter selection lock")
    study = runner.prepare_study(cfg, config, root, device="cuda")
    lock = runner.validate_selection(study)
    pair = tuple(lock["payload"]["selected_pair"][key] for key in ("r", "d"))
    context, pools = runner.make_context(study, os.getenv("G2_CATEGORY", "rice"), pair, "E4", selection_sha256=lock["sha256"])
    cfg = context["config"]
    native = G1NativeDataset(pools["train"], cfg["synthetic_protocol"], seed=cfg["training"]["seed"],
                             role="train", variants=cfg["data"]["train_variants_per_image"], fixed=False)
    tiles = PairedG2Tiles(G1TileDataset(native, overlap=cfg["data"]["overlap"]), context_size=cfg["paired_views"]["context_size"])
    batch = paired_tile_collate([tiles[0], tiles[1]])
    g2_e2.seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
    model = build_g2_model(cfg, "E4").to("cuda")
    trainable = g2_e2.trainable_modules(model, "E4")
    optimizer, _ = g2_e2.build_optimizer(dict(trainable.items()), frozen_modules={"backbone": model.extractor},
                                        learning_rate=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    frozen = g2_e2.module_sha256(model.extractor)
    trainer = g2_e2.Overfit16Trainer(model=model, criterion=AnomalySegmentationLoss(**cfg["loss"]),
                                    optimizer=optimizer, device="cuda", frozen_modules={"backbone": model.extractor},
                                    model_forward=paired_model_forward)
    trainer.train_step(batch, step=1, epoch=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.parameters())
    assert g2_e2.module_sha256(model.extractor) == frozen
    checkpoint = tmp_path / "real_e4_one_batch.pt"
    save_training_checkpoint(checkpoint, model=trainable, optimizer=optimizer, epoch=0, global_step=1,
                             config=cfg, metadata={"purpose": "one batch integration; not full training"})
    _, verified = load_checkpoint_payload(checkpoint)
    assert verified
