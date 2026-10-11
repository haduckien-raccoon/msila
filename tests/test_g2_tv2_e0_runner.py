"""All eight E0 categories exercised with explicit CPU/API fixtures only."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import src.eval.e0 as e0
from src.data.loader import G1NativeDataset
from src.data.synthetic_anomaly import NativeTinyDefectGenerator
from src.models.backbone_registry import BACKBONES, backbone_spec
from src.models.dinov3_extractor import DINOv3FeatureExtractor


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ContractBackbone(nn.Module):
    """Not a pretrained model; only reproduces the official dense feature API."""
    def __init__(self, name):
        super().__init__()
        spec = backbone_spec(name)
        self.embed_dim, self.patch_size = spec.channels, spec.patch_size
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(spec.depth))
        self.scale = nn.Parameter(torch.tensor(1.))
        self.channel_weight = nn.Parameter(torch.ones(spec.channels))
        self.requested = []

    def get_intermediate_layers(self, x, *, n, **kwargs):
        self.requested.append(tuple(n))
        assert kwargs == dict(reshape=True, return_class_token=False, return_extra_tokens=False, norm=True)
        feature = F.avg_pool2d(x.mean(1, keepdim=True), 16) * self.scale
        return tuple((feature * self.channel_weight[None, :, None, None]).contiguous() for _ in n)


def fixture_config(tmp_path, monkeypatch, name="dinov3_vits16", categories=("rice",)):
    repo = tmp_path / "dino"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# Official API fixture only; no pretrained weights")
    weights = tmp_path / "fixture.pth"
    torch.save(ContractBackbone(name).state_dict(), weights)
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: ContractBackbone(kwargs["model"]))
    data = tmp_path / "data"
    rng = np.random.default_rng(42)
    for category in categories:
        for split in ("TRAIN", "VALIDATION"):
            folder = data / category / split / "good"
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(40, 190, (96, 128, 3), dtype=np.uint8)).save(folder / "0.png")
        # TEST must never be decoded, hashed or used to produce memory/DEV.
        trap = data / category / "TEST_PUBLIC" / "good"
        trap.mkdir(parents=True)
        (trap / "trap.png").write_bytes(b"invalid image that must not be used")
        train_bad = data / category / "TRAIN" / "bad"
        train_bad.mkdir()
        (train_bad / "trap.png").write_bytes(b"do not read anomalous TRAIN")
    cfg = e0.resolve_config(e0.parse_args(["--smoke", "--device", "cpu", "--backbone", name,
          "--data-root", str(data), "--repo-dir", str(repo), "--weights", str(weights),
          "--output-root", str(tmp_path / "out")]))
    cfg["memory"].update(max_features=8, query_chunk_size=64, bank_chunk_size=4)
    return cfg


@pytest.mark.parametrize("name", tuple(BACKBONES))
def test_registry_deep_width_and_train_only_bank(tmp_path, monkeypatch, name):
    cfg = fixture_config(tmp_path, monkeypatch, name)
    pools, _ = e0.discover_sources(cfg, "rice")
    extractor = DINOv3FeatureExtractor(cfg["backbone"]["repo_dir"], cfg["backbone"]["weights"],
                                     name, feature_mode="deepest")
    before = e0.module_sha256(extractor)
    def synthetic_trap(*args, **kwargs):
        raise AssertionError("TRAIN memory must never generate anomalies")
    monkeypatch.setattr(NativeTinyDefectGenerator, "__call__", synthetic_trap)
    payload = e0.fit_normal_memory(extractor, pools["train"], cfg, "rice", "cpu")
    spec = backbone_spec(name)
    assert payload["vectors"].shape == (8, spec.channels)
    assert payload["patches_seen"] == 1024 and payload["deep_block"] == spec.depth
    assert extractor.backbone.requested == [(spec.depth-1,)]
    assert e0.module_sha256(extractor) == before
    assert all(not p.requires_grad and p.grad is None for p in extractor.parameters())
    assert not extractor.backbone.training
    repeat = e0.fit_normal_memory(extractor, pools["train"], cfg, "rice", "cpu")
    assert torch.equal(payload["vectors"], repeat["vectors"])
    assert torch.equal(payload["sampled_indices"], repeat["sampled_indices"])
    with pytest.raises(ValueError, match="TRAIN/good"):
        e0.fit_normal_memory(extractor, pools["dev"], cfg, "rice", "cpu")


def test_dev_protocol_fixed_and_never_uses_test(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path, monkeypatch)
    pools, manifest = e0.discover_sources(cfg, "rice")
    assert all("TEST" not in row["path"] and "/good/" in row["path"]
               for rows in manifest.values() for row in rows)
    dev = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"], seed=cfg["evaluation"]["dev_seed"],
                         role="dev", variants=3, fixed=True)
    first = [(dev[i]["image"].clone(), dev[i]["mask"].clone()) for i in range(len(dev))]
    dev.set_epoch(10)
    for i, (image, mask) in enumerate(first):
        assert torch.equal(image, dev[i]["image"]) and torch.equal(mask, dev[i]["mask"])
    assert not first[0][1].any() and all(mask.any() for _, mask in first[1:])
    assert e0.category_seed(2026, "rice") != e0.category_seed(2026, "can")
    train = Path(pools["train"][0].image_path)
    Path(pools["dev"][0].image_path).write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="leakage"):
        e0.discover_sources(cfg, "rice")


def test_all_eight_fixture_categories_maps_csv_resume_and_corruption(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path, monkeypatch, categories=e0.MVTEC_AD2_CATEGORIES)
    cfg["evaluation"]["save_maps"] = True
    first = e0.run_all(cfg, e0.MVTEC_AD2_CATEGORIES, "cpu")
    assert first["status"] == "SMOKE PASS" and first["smoke_categories_pass"] == 8
    assert first["real_categories_pass"] == 0 and first["macro_synthetic_dev_aupro_0_05"] is None
    output = Path(cfg["output_root"])
    before = {}
    for category in e0.MVTEC_AD2_CATEGORIES:
        folder = output / category
        metric = json.loads((folder / "metrics.json").read_text())
        assert metric["status"] == "PASS" and metric["verification_scope"] == "fixture"
        assert 0 <= metric["synthetic_dev_aupro_0_05"] <= 1
        assert metric["n_samples"] == 4 and metric["trainable_parameters"] == 0
        assert metric["frozen_backbone_unchanged"] and metric["qa_samples"] == 4
        assert metric["dev_mixed"]["regions"] == 3 and metric["dev_tiny"]["regions"] == 2
        assert metric["score_transform"] == e0.SCORE_TRANSFORM
        assert metric["split"] == "dev_synthetic" and metric["native_resolution"]
        assert (folder / "normal_memory.pt").is_file()
        before[category] = (folder / "metrics.json").read_bytes()
        for map_path in (folder / "maps").glob("*/anomaly_map.npy"):
            score = np.load(map_path, allow_pickle=False)
            mask = np.load(map_path.with_name("gt_mask.npy"), allow_pickle=False)
            assert score.shape == mask.shape == (96, 128)
            assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1
    rows = list(csv.DictReader((output / "e0_metrics_8categories.csv").open()))
    assert len(rows) == 8 and all(row["verification_scope"] == "fixture" for row in rows)
    assert len({row["bank_sha256"] for row in rows}) == 8
    def rerun_trap(*args, **kwargs):
        raise AssertionError("Completed run must be skipped")
    monkeypatch.setattr(e0, "fit_normal_memory", rerun_trap)
    monkeypatch.setattr(e0, "evaluate_dev", rerun_trap)
    resumed = e0.run_all(cfg, e0.MVTEC_AD2_CATEGORIES, "cpu", resume=True)
    assert resumed["smoke_categories_pass"] == 8
    assert all(row["skipped"] for row in resumed["categories"])
    assert all((output / category / "metrics.json").read_bytes() == before[category] for category in before)
    bank = output / "rice" / "normal_memory.pt"
    bank.write_bytes(bank.read_bytes() + b"corrupted")
    blocked = e0.run_all(cfg, ["rice"], "cpu", resume=True)
    assert blocked["status"] == "BLOCKED" and blocked["smoke_categories_pass"] == 0
    assert (output / "rice" / "metrics.json").read_bytes() == before["rice"]


def test_partial_bank_resume_config_drift_and_pass_preserved_without_assets(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path, monkeypatch)
    first = e0.run_all(cfg, ["rice"], "cpu")
    assert first["smoke_categories_pass"] == 1
    output = Path(cfg["output_root"]) / "rice"
    bank_sha = e0.file_sha256(output / "normal_memory.pt")
    (output / "metrics.json").unlink()  # Simulate interruption after bank save.
    def fit_trap(*args, **kwargs):
        raise AssertionError("Verified bank should be reused")
    monkeypatch.setattr(e0, "fit_normal_memory", fit_trap)
    resumed = e0.run_all(cfg, ["rice"], "cpu", resume=True)
    assert resumed["smoke_categories_pass"] == 1
    assert e0.file_sha256(output / "normal_memory.pt") == bank_sha
    previous = (output / "metrics.json").read_bytes()
    drift = copy.deepcopy(cfg)
    drift["memory"]["distance"] = "cosine"
    assert e0.run_all(drift, ["rice"], "cpu", resume=True)["status"] == "BLOCKED"
    assert (output / "metrics.json").read_bytes() == previous
    missing = copy.deepcopy(cfg)
    missing["backbone"]["weights"] = str(Path(cfg["output_root"]) / "missing.pth")
    assert e0.run_all(missing, ["rice"], "cpu", resume=True)["status"] == "NOT RUN"
    assert (output / "metrics.json").read_bytes() == previous


def test_missing_assets_writes_eight_not_run_records_with_blank_scores(tmp_path):
    cfg = e0.resolve_config(e0.parse_args(["--output-root", str(tmp_path), "--data-root", str(tmp_path / "missing")]))
    summary = e0.run_all(cfg, e0.MVTEC_AD2_CATEGORIES, "cuda")
    assert summary["status"] == "NOT RUN" and summary["real_categories_pass"] == 0
    for category in e0.MVTEC_AD2_CATEGORIES:
        result = json.loads((tmp_path / category / "metrics.json").read_text())
        assert result["status"] == "NOT RUN" and "synthetic_dev_aupro_0_05" not in result
        assert not (tmp_path / category / "normal_memory.pt").exists()
    rows = list(csv.DictReader((tmp_path / "e0_metrics_8categories.csv").open()))
    assert len(rows) == 8 and all(row["synthetic_dev_aupro_0_05"] == "" for row in rows)


def test_corrupt_metrics_blocked_and_preserved(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path, monkeypatch)
    output = Path(cfg["output_root"]) / "rice"
    output.mkdir(parents=True)
    path = output / "metrics.json"
    path.write_text('{"status":')
    summary = e0.run_all(cfg, ["rice"], "cpu", resume=True)
    assert summary["status"] == "BLOCKED" and summary["smoke_categories_pass"] == 0
    assert path.read_text() == '{"status":'
    assert not (output / "normal_memory.pt").exists()


@pytest.mark.parametrize("section,key,value", [
    ("data", "train_split", "TEST_PUBLIC/good"), ("data", "dev_split", "TRAIN/good"),
    ("data", "tile_size", 511), ("data", "overlap", 512), ("data", "max_train_sources", 0),
    ("data", "dev_variants_per_image", 0), ("memory", "max_features", 0),
    ("memory", "distance", "bad"), ("memory", "normalize", 1), ("memory", "sampling_seed", -1),
    ("evaluation", "test_for_tuning", True), ("evaluation", "max_fpr", .3),
    ("evaluation", "score_scale", 0), ("evaluation", "score_transform", "per_image_minmax")])
def test_invalid_protocol_settings_rejected(section, key, value):
    cfg = e0.resolve_config(e0.parse_args([]))
    cfg[section][key] = value
    with pytest.raises(ValueError):
        e0.validate_config(cfg)


def test_cli_backbone_categories_and_smoke_namespace():
    args = e0.parse_args(["--backbone", "dinov3_vith16plus", "--categories", "rice,can", "--distance", "cosine", "--smoke"])
    cfg = e0.resolve_config(args)
    assert cfg["backbone"]["name"] == "dinov3_vith16plus"
    assert "dinov3_vith16plus_pretrain" in cfg["backbone"]["weights"]
    assert cfg["memory"]["distance"] == "cosine" and cfg["output_root"].endswith("/smoke")
    assert e0.requested_categories(args.categories) == ["can", "rice"]
    for tokens in (["rice", "rice"], ["all", "can"], ["invalid"]):
        with pytest.raises(ValueError):
            e0.requested_categories(tokens)


def test_real_pretrained_one_category_smoke_if_available(tmp_path):
    cfg = e0.resolve_config(e0.parse_args(["--smoke", "--output-root", str(tmp_path)]))
    reasons = e0.asset_reasons(cfg, "cuda")
    if reasons:
        pytest.skip("NOT RUN real pretrained E0: " + "; ".join(reasons))
    summary = e0.run_all(cfg, ["rice"], "cuda")
    assert summary["smoke_categories_pass"] == 1
    assert summary["real_categories_pass"] == 0
    assert summary["categories"][3]["verification_scope"] == "pretrained_dinov3"
