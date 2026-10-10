"""D7 E3 architecture, resume and TV2 export gates; no real GPU evidence."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import os

import pytest
import torch
import yaml

from src.models.adapter_factory import AdapterFactoryConfig, ResidualAdapterFactory
from src.models.backbone_registry import BACKBONES, backbone_spec
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.models.feature_projection import SixFeatureProjection
from src.models.feature_selector import FeatureSelector
from src.models.mean_fusion import MeanFusion
from src.models.msila import E3, build_g2_model
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.train import g2_e2
from src.data.loader import G1NativeDataset, G1TileDataset, g1_tile_collate
from src.utils.checkpoint import save_training_checkpoint
from src.utils.resume import load_checkpoint_payload
from tests.test_g2_tv1_model import BackboneFixture
from tests.test_g2_tv1_runner import (setup, cpu_determinism, runner, virtual_ledger,
                                       assert_nested_equal)


class LayerBackboneFixture(BackboneFixture):
    def get_intermediate_layers(self, image, *, n, **kwargs):
        # Distinct physical layers make wrong slot mapping observable.
        features = super().get_intermediate_layers(image, n=n, **kwargs)
        return tuple(feature * ((index + 1) / len(self.blocks) + .5)
                     for index, feature in zip(n, features))


@pytest.mark.parametrize("name", tuple(BACKBONES))
def test_three_local_layers_gradient_params_logits_and_no_context(setup, tmp_path, monkeypatch, name):
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: LayerBackboneFixture(kwargs["model"]))
    weights = tmp_path / f"layers_{name}.pth"
    torch.save(LayerBackboneFixture(name).state_dict(), weights)
    cfg = deepcopy(setup["cfg"])
    cfg["backbone"].update(name=name, weights=str(weights))
    cfg["adapter"].update(r=8, d=12)
    cfg["fusion"]["dim"] = 8
    cfg["decoder"]["deterministic_resize"] = True
    torch.manual_seed(2026)
    model = build_g2_model(cfg, experiment="E3")
    assert isinstance(model, E3)
    assert isinstance(model.selector, FeatureSelector) and isinstance(model.projection, SixFeatureProjection)
    assert isinstance(model.fusion, MeanFusion) and model.fusion.return_weights is False
    assert model.source_blocks == backbone_spec(name).blocks
    assert model.num_sources == len(model.adapters) == 3
    assert len({id(adapter) for adapter in model.adapters.values()}) == 3
    assert model.projection.context_projectors is None
    fixed = AdapterFactoryConfig(in_dim=backbone_spec(name).channels)
    adapter_count = ResidualAdapterFactory(fixed).build_rd(r=8, d=12).trainable_params
    projection_count = 3 * (backbone_spec(name).channels * 8 + 8)
    decoder_count = 8 * 4 * 9 + 4 + 4 + 1
    assert model.num_trainable_parameters == 3 * adapter_count + projection_count + decoder_count
    assert not list(model.fusion.parameters()) and not list(model.selector.parameters())
    expected_groups = g2_e2.trainable_modules(model, "E3")
    assert {id(p) for p in expected_groups.parameters()} == {id(p) for p in model.parameters() if p.requires_grad}
    frozen = {key: value.clone() for key, value in model.extractor.state_dict().items()}
    batch = dict(image=torch.randn(2, 3, 512, 512), mask=torch.zeros(2, 1, 512, 512), meta=[{}, {}],
                 context=torch.randn(2, 3, 512, 512))
    batch["mask"][1, :, 75:91, 143:160] = 1
    optimizer = torch.optim.AdamW(expected_groups.parameters(), lr=.01, weight_decay=0)
    model.train()
    logits, trace = model(batch, return_trace=True)
    assert logits.shape == (2, 1, 512, 512) and torch.isfinite(logits).all()
    assert trace["decoder_feature"].shape == (2, 8, 32, 32)
    assert list(trace["dino"]) == [f"b{block}" for block in backbone_spec(name).blocks]
    assert trace["source_blocks"] == dict(zip(model.selector.source_keys, backbone_spec(name).blocks))
    assert model.extractor.backbone.requested == [tuple(b - 1 for b in backbone_spec(name).blocks)]
    for key, block in model.source_block_map.items():
        assert torch.equal(trace["adapted"][key], trace["dino"][f"b{block}"])
        assert not trace["dino"][f"b{block}"].requires_grad
    assert not torch.equal(trace["dino"][f"b{model.source_blocks[0]}"], trace["dino"][f"b{model.source_blocks[-1]}"])
    with torch.no_grad():
        without_context = model({k: v for k, v in batch.items() if k != "context"})
    assert torch.equal(logits, without_context)
    criterion = AnomalySegmentationLoss()
    criterion(logits, batch["mask"])["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in expected_groups.parameters())
    for adapter in model.adapters.values():
        assert adapter.gamma.grad.abs().item() > 0
        assert all(torch.count_nonzero(p.grad) == 0 for key, p in adapter.named_parameters() if key != "gamma")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    criterion(model(batch), batch["mask"])["loss"].backward()
    for adapter in model.adapters.values():
        assert any(torch.count_nonzero(p.grad) > 0 for key, p in adapter.named_parameters() if key != "gamma")
    assert all(p.grad is None and not p.requires_grad for p in model.extractor.parameters())
    assert not model.extractor.training and not model.extractor.backbone.training
    assert all(torch.equal(value, frozen[key]) for key, value in model.extractor.state_dict().items())


def test_e3_rejects_deep_only_extractor(setup):
    cfg = setup["cfg"]
    extractor = DINOv3FeatureExtractor(cfg["backbone"]["repo_dir"], cfg["backbone"]["checkpoints"]["dinov3_vits16"],
                                      feature_mode="deepest")
    with pytest.raises(ValueError, match="three registry blocks"):
        E3(extractor, adapter_bottleneck_dim=8, adapter_projection_dim=12)


def test_e3_context_matches_e2_scientific_protocol_and_requires_lock(setup):
    study = setup["study"]()
    pair = (64, 256)
    with pytest.raises(runner.G2Blocked, match="selection lock"):
        runner.make_context(study, "rice", pair, "E3")
    e2, _ = runner.make_context(study, "rice", pair, "E2", selection_sha256="fixture-lock")
    e3, _ = runner.make_context(study, "rice", pair, "E3", selection_sha256="fixture-lock")
    assert runner.comparison_identity(e2["config"]) == runner.comparison_identity(e3["config"])
    assert e2["expected_steps"] == e3["expected_steps"] == 4
    assert e3["config"]["architecture"]["adapter_sharing"] == "independent_per_layer"
    assert e3["config"]["architecture"]["feature_width"] == 64


def test_e3_checkpoint_resume_exact_and_all_trainable_groups_present(setup, monkeypatch):
    study = setup["study"]()
    context, pools = runner.make_context(study, "rice", (32, 128), "E3", selection_sha256="fixture-lock")
    full = study["root"] / "full" / "E3_control" / "rice"
    g2_e2.train_e3(context, pools, full, device="cpu")
    path = runner.run_directory(study, "rice", (32, 128), "E3")
    original = g2_e2.Overfit16Trainer.train_step

    def interrupt(self, batch, *, step, epoch):
        if step == 2:
            raise RuntimeError("Injected E3 interruption")
        return original(self, batch, step=step, epoch=epoch)

    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        g2_e2.train_e3(context, pools, path, device="cpu")
    monkeypatch.setattr(g2_e2.Overfit16Trainer, "train_step", original)
    result = g2_e2.train_e3(context, pools, path, device="cpu", resume=True)
    assert result["status"] == "PASS" and result["global_step"] == 4
    assert result["projection_updated"] and all(result["adapters_updated"].values())
    assert runner.read_valid_result(path, context)
    assert not runner.real_evidence(result)
    expected, _ = load_checkpoint_payload(full / "last.pt")
    actual, _ = load_checkpoint_payload(path / "last.pt")
    assert {key.split(".")[0] for key in actual["model_state"]} == {"adapters", "projection", "decoder"}
    assert_nested_equal(expected["model_state"], actual["model_state"])
    assert_nested_equal(expected["optimizer_state"], actual["optimizer_state"])
    # TV2 restores a real saved fixture checkpoint through the documented API.
    reloaded = build_g2_model(context["config"], experiment="E3")
    best, _ = load_checkpoint_payload(path / "best.pt")
    g2_e2.trainable_modules(reloaded, "E3").load_state_dict(best["model_state"], strict=True)
    assert reloaded(torch.randn(1, 3, 512, 512)).shape == (1, 1, 512, 512)


def test_e3_missing_lock_blocks_full_and_smoke_before_training(setup, monkeypatch):
    monkeypatch.setattr(runner, "train_e3", lambda *a, **k: pytest.fail("Missing lock must block training"))
    for extra in ([], ["--smoke"]):
        assert runner.main(["--config", str(setup["path"]), "--stage", "E3", "--device", "cpu", *extra]) == 2
        mode = "smoke" if extra else "full"
        result = json.loads((setup["root"] / "dinov3_vits16" / mode / "E3_summary.json").read_text())
        assert result["status"] == "BLOCKED" and result["real_E3_pass"] == 0
    assert not list(setup["root"].rglob("*.pt"))


def test_e3_all_eight_smoke_skip_and_complete_tv2_export(setup, monkeypatch):
    study = setup["study"]()
    virtual_ledger(study, monkeypatch)  # Selection-only unit fixture, not real GPU evidence.
    runner.publish_selection(study)
    common = ["--config", str(setup["path"]), "--categories", "all", "--device", "cpu", "--smoke"]
    assert runner.main([*common, "--stage", "E2"]) == 0
    assert runner.main([*common, "--stage", "E3"]) == 0
    path = study["root"] / "smoke"
    summary = json.loads((path / "E3_summary.json").read_text())
    assert summary["status"] == "SMOKE_PASS" and summary["real_E3_pass"] == 0
    assert len(summary["outcomes"]) == 8 and {r["status"] for r in summary["outcomes"]} == {"PASS"}
    data = json.loads((path / "E3_minus_E2_inputs.json").read_text())
    assert data["categories"] == list(runner.CATEGORIES)
    assert data["status"] == "SMOKE_READY" and data["ready_pairs"] == 8 and data["real_ready_pairs"] == 0
    assert data["split"] == "dev_synthetic" and data["synthetic"] is True
    assert {r["status"] for r in data["records"]} == {"READY"}
    with (path / "E3_minus_E2_inputs.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 8
    for row in data["records"]:
        for experiment in ("e2", "e3"):
            assert Path(row[experiment]["checkpoint"]).is_file()
            assert len(row[experiment]["checkpoint_sha256"]) == 64
            assert row[experiment]["sources_sha256"] == row["e2"]["sources_sha256"]
    # Completed runs are skipped without invoking model training again.
    monkeypatch.setattr(runner, "train_e3", lambda *a, **k: pytest.fail("Completed E3 run trained again"))
    assert runner.main([*common, "--stage", "E3", "--resume"]) == 0
    assert {r["status"] for r in json.loads((path / "E3_summary.json").read_text())["outcomes"]} == {"SKIP"}
    # Losing one paired category must prevent a complete comparison.
    (path / "E2" / "walnuts" / "metrics.json").unlink()
    lock = json.loads((study["root"] / "full" / "adapter_selection_lock.json").read_text())
    assert runner.export_comparison(study, lock, smoke=True)["status"] == "INCOMPLETE"


def test_immutable_d6_protocol_reused_only_for_known_source_extension(setup):
    study = setup["study"]()
    historical = deepcopy(study["protocol"])
    historical["source_code_sha256"].update(runner.D6_SOURCE_BASELINE)
    root = study["root"]
    runner.enforce_lock(root / "full", historical)
    reloaded = setup["study"]()
    assert reloaded["protocol"] == historical
    assert reloaded["sha256"] == runner.sha256_json(historical)
    assert reloaded["e3"]["implementation_sha256"]["src/models/msila.py"] != runner.D6_SOURCE_BASELINE["src/models/msila.py"]
    changed = deepcopy(study["protocol"])
    changed["training"]["seed"] += 1
    with pytest.raises(runner.G2Blocked, match="scientific protocol"):
        runner.reuse_d6_protocol(changed, root)
    changed = deepcopy(study["protocol"])
    changed["source_code_sha256"]["src/eval/evaluator.py"] = "different-evaluator"
    with pytest.raises(runner.G2Blocked, match="unaudited dependency"):
        runner.reuse_d6_protocol(changed, root)


def test_real_locked_e3_one_batch_checkpoint_if_available(tmp_path):
    """One real batch integration only; never a full-category PASS."""
    if not torch.cuda.is_available():
        pytest.skip("NOT RUN: real E3 batch/checkpoint requires CUDA and a valid full Adapter lock")
    argv = ["--stage", "E3", "--device", "cuda"]
    for flag, variable in (("--data-root", "MVTEC_AD2_ROOT"), ("--repo-dir", "DINOV3_REPO"),
                           ("--weights", "DINOV3_WEIGHTS"), ("--backbone", "DINOV3_MODEL"),
                           ("--output-root", "G2_OUTPUT_ROOT")):
        if os.getenv(variable):
            argv.extend([flag, os.environ[variable]])
    args = runner.parse_args(argv)
    cfg, config, root = runner.load_config(args)
    if not (root / "full" / "adapter_selection_lock.json").is_file():
        pytest.skip("NOT RUN: real E3 integration has no Adapter selection lock")
    # Explicit assets/lock that exist but are invalid must FAIL, not silently skip.
    study = runner.prepare_study(cfg, config, root, device="cuda")
    lock = runner.validate_selection(study)
    pair = tuple(lock["payload"]["selected_pair"][key] for key in ("r", "d"))
    category = os.getenv("G2_CATEGORY", "rice")
    context, pools = runner.make_context(study, category, pair, "E3", selection_sha256=lock["sha256"])
    cfg = context["config"]
    native = G1NativeDataset(pools["train"], cfg["synthetic_protocol"], seed=cfg["training"]["seed"],
                             role="train", variants=cfg["data"]["train_variants_per_image"], fixed=False)
    tiles = G1TileDataset(native, tile_size=512, overlap=cfg["data"]["overlap"])
    batch = g1_tile_collate([tiles[0], tiles[1]])
    g2_e2.seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
    model = build_g2_model(cfg, experiment="E3").to("cuda")
    trainable = g2_e2.trainable_modules(model, "E3")
    optimizer, _ = g2_e2.build_optimizer(dict(trainable.items()), frozen_modules={"backbone": model.extractor},
                                        learning_rate=cfg["training"]["learning_rate"],
                                        weight_decay=cfg["training"]["weight_decay"])
    frozen = g2_e2.module_sha256(model.extractor)
    trainer = g2_e2.Overfit16Trainer(model=model, criterion=AnomalySegmentationLoss(**cfg["loss"]),
                                    optimizer=optimizer, device="cuda", frozen_modules={"backbone": model.extractor})
    trainer.train_step(batch, step=1, epoch=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.parameters())
    assert g2_e2.module_sha256(model.extractor) == frozen
    checkpoint = tmp_path / "real_e3_one_batch.pt"
    save_training_checkpoint(checkpoint, model=trainable, optimizer=optimizer, epoch=0, global_step=1,
                             config=cfg, metadata={"purpose": "one batch integration; not full training"})
    payload, verified = load_checkpoint_payload(checkpoint)
    assert verified and {key.split(".")[0] for key in payload["model_state"]} == {"adapters", "projection", "decoder"}
