"""D8-v3 E4 Mean(6) baseline: CPU integration, named debug pairs, no selection."""
from copy import deepcopy
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
from tests.test_g2_tv1_runner import setup, cpu_determinism, runner, assert_nested_equal


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
    pair = (128, 512) if name == "dinov3_vitb16" else (8, 12)
    cfg.update(adapter_pair_role="debug_pair", debug_pair=runner.debug_pair(pair),
               selection_status="pending_joint_selection_D10")
    cfg["adapter"].update(r=pair[0], d=pair[1])
    fusion_dim = 64 if name == "dinov3_vitb16" else 8
    cfg["fusion"]["dim"] = fusion_dim
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
    assert logits.shape == (2, 1, 512, 512) and torch.isfinite(logits).all()
    assert model.extractor.backbone.requested == [tuple(b-1 for b in backbone_spec(name).blocks)]
    assert trace["source_keys"] == MULTIVIEW_FEATURE_KEYS and trace["num_sources"] == 6
    assert tuple(trace["projected"]) == MULTIVIEW_FEATURE_KEYS
    assert all(value.shape == (2, fusion_dim, 32, 32) and torch.isfinite(value).all()
               for value in trace["projected"].values())
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
    assert trace["decoder_feature"].shape == (2, fusion_dim, 32, 32)
    for value in trace["adapted"].values():
        value.retain_grad()
    criterion = AnomalySegmentationLoss()
    loss = criterion(logits, batch["mask"])["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
               for value in trace["adapted"].values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in groups.parameters())
    for adapter in model.adapters.values():
        assert adapter.gamma.grad.abs() > 0
        assert all(torch.count_nonzero(p.grad) == 0 for key, p in adapter.named_parameters() if key != "gamma")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    criterion(model(batch), batch["mask"])["loss"].backward()
    for adapter in model.adapters.values():
        assert all(torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) > 0
                   for key, p in adapter.named_parameters() if key != "gamma")
    assert all(torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) > 0
               for module in (model.projection, model.decoder) for p in module.parameters())
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


@pytest.mark.parametrize("position", ["center", "top_left", "bottom_right", "padded_small_native"])
def test_center_and_padded_edge_alignment_uses_native_coordinates(position):
    hw = (73, 101) if position == "padded_small_native" else (1536, 1600)
    records = generate_tile_records(*hw)
    if position == "center":
        record = next(r for r in records if r.local_xyxy[:2] == (384, 384))
    else:
        record = records[-1] if position == "bottom_right" else records[0]
    _, _, meta = prepare_pair(torch.rand(3, *hw), record, sample_id="one-native")
    # Context feature encodes native coordinates. Aligning must recover the
    # Local feature centers even when the crop uses a virtual padded canvas.
    centers = (torch.arange(32, dtype=torch.float64) + .5) * 16
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    cx, cy, _, _ = record.context_xyxy
    lx, ly, _, _ = record.local_xyxy
    context_ramp = ((x * 1.5 + cx) + 2 * (y * 1.5 + cy))[None, None].requires_grad_()
    aligner = ContextToLocalAligner(deterministic_sampling=True, check_finite=True, check_bounds=True)
    aligned = aligner(dict.fromkeys(("C4", "C8", "C12"), context_ramp), meta["geometry"], target_hw=(32, 32))
    expected = ((x + lx) + 2 * (y + ly))[None, None]
    for value in aligned.values():
        torch.testing.assert_close(value, expected, atol=1e-9, rtol=1e-12)
    sum(value.sum() for value in aligned.values()).backward()
    assert torch.isfinite(context_ramp.grad).all() and context_ramp.grad.sum() == pytest.approx(3*32*32)
    assert aligner.align_corners is False
    if position == "center":
        assert meta["padding_ltrb"] == (0, 0, 0, 0)
    else:
        assert any(meta["padding_ltrb"])


@pytest.mark.parametrize("strategy,calls", [("concat", 1), ("sequential", 2)])
def test_builder_respects_configured_pair_strategy(setup, strategy, calls):
    cfg = deepcopy(setup["cfg"])
    cfg["adapter"].update(r=8, d=12)
    cfg["paired_views"]["pair_strategy"] = strategy
    model = build_g2_model(cfg, "E4")
    assert model.pair_strategy == strategy
    logits = model(pair_batch(1))
    assert logits.shape == (1, 1, 512, 512) and torch.isfinite(logits).all()
    assert len(model.extractor.backbone.requested) == calls


def test_builder_preserves_legacy_strategy_alias(setup):
    cfg = deepcopy(setup["cfg"])
    cfg["paired_views"].pop("pair_strategy")
    cfg["paired_views"]["strategy"] = "sequential"
    assert build_g2_model(cfg, "E4").pair_strategy == "sequential"


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


@pytest.mark.parametrize("case", ["tensor", "missing_context", "missing_geometry", "singular", "wrong_size", "outside",
                                 "different_sample", "nan_context", "context_dtype", "empty_view_meta"])
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
    elif case == "nan_context":
        batch["context"][0, 0, 0, 0] = float("nan")
    elif case == "context_dtype":
        batch["context"] = batch["context"].double()
    elif case == "empty_view_meta":
        batch["view_meta"] = []
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
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", smoke=True)
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
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", smoke=True)
    e3_context, _ = runner.make_context(study, "rice", (32, 128), "E3", smoke=True)
    for key in ("training", "data", "sources", "loss", "synthetic_protocol", "adapter"):
        assert context["config"][key] == e3_context["config"][key]
    control = study["root"] / "control" / "rice"
    g2_e2.train_e4(context, pools, control, device="cpu")
    path = runner.run_directory(study, "rice", (32, 128), "E4", smoke=True)
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
    assert result["global_step"] == context["expected_steps"] == 2
    assert result["debug_pair"]["name"] == "debug_pair_r32_d128" and not result["scientific_evidence"]
    assert runner.read_valid_result(path, context) and result["feature_sources"] == 6
    assert result["context_aligned"] and result["projection_updated"] and all(result["adapters_updated"].values())
    assert not runner.real_evidence(result)
    actual, _ = load_checkpoint_payload(path / "last.pt")
    expected, _ = load_checkpoint_payload(control / "last.pt")
    for key in ("model_state", "optimizer_state"):
        assert_nested_equal(expected[key], actual[key])
    assert {key.split(".")[0] for key in actual["model_state"]} == {"adapters", "projection", "decoder"}
    assert actual["config"] == expected["config"] == context["config"]
    assert runner.read_yaml(path / "resolved_config.yaml") == context["config"]
    assert actual["training_state"] == expected["training_state"]
    for key in ("next_epoch", "next_batch", "dev_step"):
        assert actual["metadata"][key] == expected["metadata"][key]
    assert_nested_equal({k: v for k, v in expected["rng_state"].items() if k != "numpy"},
                        {k: v for k, v in actual["rng_state"].items() if k != "numpy"})
    np.testing.assert_array_equal(expected["rng_state"]["numpy"][1], actual["rng_state"]["numpy"][1])
    assert expected["rng_state"]["numpy"][0] == actual["rng_state"]["numpy"][0]
    assert expected["rng_state"]["numpy"][2:] == actual["rng_state"]["numpy"][2:]
    # Actual E4 checkpoint loading reproduces logits before native inference.
    restored = build_g2_model(context["config"], "E4").eval()
    best, _ = load_checkpoint_payload(path / "best.pt")
    g2_e2.trainable_modules(restored, "E4").load_state_dict(best["model_state"], strict=True)
    batch = pair_batch(1)
    control_model = build_g2_model(context["config"], "E4").eval()
    control_best, _ = load_checkpoint_payload(control / "best.pt")
    g2_e2.trainable_modules(control_model, "E4").load_state_dict(control_best["model_state"], strict=True)
    with torch.no_grad():
        assert torch.equal(restored(batch), control_model(batch))
    before = runner.sha256_file(path / "best.pt")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", lambda *a, **k: pytest.fail("Inference trained"))
    inference = g2_e2.infer_e4(context, pools, path, device="cpu")
    assert inference["metrics"] == result["best_synthetic_dev"]
    assert runner.sha256_file(path / "best.pt") == before and inference["optimizer_steps"] == 0
    assert not inference["scientific_evidence"] and inference["debug_pair"] == context["config"]["debug_pair"]
    for record in inference["predictions"]:
        assert np.load(record["score"]).shape == np.load(record["mask"]).shape == tuple(record["original_hw"])
    changed = deepcopy(context)
    changed["config"]["paired_views"]["pair_strategy"] = "sequential"
    changed["config_sha256"] = runner.sha256_json(changed["config"])
    last_before = runner.sha256_file(path / "last.pt")
    monkeypatch.setattr(g2_e2, "build_g2_model", lambda *a, **k: pytest.fail("Config drift must reject before model load"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        g2_e2.train_e4(changed, pools, path, device="cpu", resume=True)
    assert runner.sha256_file(path / "last.pt") == last_before


def test_e4_resume_final_batch_retries_dev_without_optimizer_update(setup, monkeypatch):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", smoke=True)
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


def test_e4_debug_cli_resume_inference_without_selection_or_comparison(setup, monkeypatch):
    common = ["--config", str(setup["path"]), "--stage", "E4", "--categories", "rice",
              "--device", "cpu", "--smoke", "--debug-pair", "32", "128", "--output-root", str(setup["root"])]
    assert runner.main(common) == 0
    path = setup["root"] / "dinov3_vits16" / "debug_d8_v3" / "debug_pair_r32_d128" / "smoke"
    summary = json.loads((path / "E4_summary.json").read_text())
    assert summary["status"] == "SMOKE_PASS" and summary["real_E4_pass"] == 0
    assert len(summary["outcomes"]) == 1 and summary["outcomes"][0]["status"] == "PASS"
    assert summary["selection_lock"] is None and summary["E4_minus_E3"].startswith("NOT RUN")
    assert not list(setup["root"].rglob("*selection_lock.json"))
    assert not list(setup["root"].rglob("E4_minus_E3_inputs.*"))
    monkeypatch.setattr(runner, "train_e4", lambda *a, **k: pytest.fail("Completed E4 trained again"))
    assert runner.main([*common, "--resume"]) == 0
    assert {row["status"] for row in json.loads((path / "E4_summary.json").read_text())["outcomes"]} == {"SKIP"}
    assert runner.main([*common, "--inference"]) == 0
    inference = json.loads((path / "E4_inference_summary.json").read_text())
    assert inference["operation"] == "inference" and len(inference["outcomes"]) == 1
    checkpoint = path / "E4" / "rice" / "best.pt"
    before = runner.sha256_file(checkpoint)
    corrupted = checkpoint.read_bytes() + b"corrupt-checkpoint-fixture"
    checkpoint.write_bytes(corrupted)
    assert runner.main([*common, "--inference"]) == 2
    assert runner.sha256_file(checkpoint) != before and checkpoint.read_bytes() == corrupted


def test_e4_main_pending_d10_and_incomplete_inference_block(setup, monkeypatch):
    check_assets = runner.check_assets
    monkeypatch.setattr(runner, "train_e4", lambda *a, **k: pytest.fail("Main must not train before D10"))
    monkeypatch.setattr(runner, "check_assets", lambda *a, **k: pytest.fail("Main must block before assets"))
    path = setup["root"] / "dinov3_vits16"
    assert runner.main(["--config", str(setup["path"]), "--stage", "E4", "--device", "cpu",
                        "--output-root", str(setup["root"])]) == 2
    result = json.loads((path / "full" / "E4_summary.json").read_text())
    assert result["status"] == "BLOCKED" and result["real_E4_pass"] == 0
    assert "pending joint selection D10" in result["reason"]
    monkeypatch.setattr(runner, "check_assets", check_assets)
    study = setup["study"]()
    with pytest.raises(runner.G2Blocked, match="completed training"):
        runner.execute_job(study, "rice", (32, 128), "E4", smoke=True, resume=False,
                           device="cpu", inference=True)
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "E3", "--inference"])


def test_e4_main_preserves_legacy_checkpoints_and_adapter_lock(setup, monkeypatch):
    root = setup["root"] / "dinov3_vits16" / "full"
    files = [root / "E3" / "rice" / "last.pt", root / "E4" / "rice" / "last.pt",
             root / "adapter_selection_lock.json"]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"legacy fixture: preserve exactly; never evidence for new selection")
    before = {path: path.read_bytes() for path in files}
    monkeypatch.setattr(runner, "prepare_study", lambda *a, **k: pytest.fail("Main must not inspect legacy checkpoints"))
    assert runner.main(["--config", str(setup["path"]), "--stage", "E4", "--resume", "--device", "cpu",
                        "--output-root", str(setup["root"])]) == 2
    assert all(path.read_bytes() == data for path, data in before.items())


def test_e4_prepared_manifest_without_assets_or_selection(setup, monkeypatch):
    monkeypatch.setattr(runner, "prepare_study", lambda *a, **k: pytest.fail("Prepare must not load assets"))
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail("Prepare must not load backbone"))
    args = ["--config", str(setup["path"]), "--stage", "E4", "--prepare", "--backbone", "dinov3_vitb16",
            "--debug-pair", "128", "512", "--device", "cpu", "--output-root", str(setup["root"])]
    assert runner.main(args) == 0
    path = setup["root"] / "dinov3_vitb16" / "debug_pair_r128_d512" / "E4_prepared_manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "PREPARED" and manifest["experiment"] == "E4"
    assert manifest["config_sha256"] == runner.sha256_json(manifest["config"])
    assert manifest["architecture"]["num_sources"] == 6 and manifest["architecture"]["fusion"] == "mean"
    assert manifest["architecture"]["projected_shape"] == [None, 64, 32, 32]
    assert manifest["architecture"]["source_blocks"] == dict(zip(MULTIVIEW_FEATURE_KEYS, (4, 8, 12) * 2))
    assert manifest["architecture"]["context_alignment"] == "before_projection_and_fusion"
    assert manifest["architecture"]["native_synthetic"] == "one_sample_for_both_views"
    assert manifest["gpu_smoke"] == manifest["main_training"] == "NOT RUN"
    assert manifest["E4_minus_E3"].startswith("NOT RUN") and manifest["selection_lock"] is None
    assert manifest["debug_pair"] == runner.debug_pair((128, 512))
    assert runner.main(args) == 0
    assert not list(setup["root"].rglob("*.pt")) and not list(setup["root"].rglob("*selection_lock.json"))


@pytest.mark.parametrize("mutation", ["full", "missing_debug_pair", "legacy_lock", "too_many_steps", "too_many_epochs"])
def test_e4_rejects_unselected_main_unbounded_smoke_before_loading(setup, monkeypatch, mutation):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E4", smoke=True)
    cfg = context["config"]
    if mutation == "full":
        cfg["mode"] = "full"
    elif mutation == "missing_debug_pair":
        cfg.pop("debug_pair")
    elif mutation == "legacy_lock":
        cfg["selection_sha256"] = "legacy-adapter-lock"
    elif mutation == "too_many_steps":
        cfg["training"]["max_steps"] = 3
    else:
        cfg["training"]["epochs"] = 2
    monkeypatch.setattr(g2_e2, "build_g2_model", lambda *a, **k: pytest.fail("Reject before model/checkpoint load"))
    for operation in (g2_e2.train_e4, g2_e2.infer_e4):
        with pytest.raises(RuntimeError, match="main awaits joint selection D10"):
            operation(context, pools, study["root"] / "rejected", device="cpu")
    assert not (study["root"] / "rejected").exists()


def test_real_debug_e4_one_batch_checkpoint_if_explicitly_enabled(tmp_path):
    if os.getenv("MSILA_RUN_E4_GPU_SMOKE") != "1" or not torch.cuda.is_available():
        pytest.skip("NOT RUN: E4 real GPU batch is opt-in on Colab, never required locally")
    argv = ["--stage", "E4", "--device", "cuda", "--smoke", "--debug-pair", "128", "512"]
    for flag, variable in (("--data-root", "MVTEC_AD2_ROOT"), ("--repo-dir", "DINOV3_REPO"),
                           ("--weights", "DINOV3_WEIGHTS"), ("--backbone", "DINOV3_MODEL"),
                           ("--output-root", "G2_OUTPUT_ROOT")):
        if os.getenv(variable):
            argv.extend([flag, os.environ[variable]])
    args = runner.parse_args(argv)
    cfg, config, root = runner.load_config(args)
    study = runner.prepare_study(cfg, config, root, device="cuda")
    pair = (128, 512)
    context, pools = runner.make_context(study, os.getenv("G2_CATEGORY", "rice"), pair, "E4", smoke=True)
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
