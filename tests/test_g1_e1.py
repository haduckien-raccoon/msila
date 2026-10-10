"""G1 risk checks. Fixtures test contracts, never pretrained performance."""
from pathlib import Path
import copy
import os

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from src.data.loader import G1NativeDataset, G1TileDataset, scan_mvtec_ad2
from src.data.synthetic_anomaly import NativeTinyDefectGenerator
from src.data.tiling import crop_with_padding, generate_tile_records, stitch_tiles_hann
from src.models.backbone_registry import backbone_spec
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.models.msila import E1
from src.train.g1_e1 import (ROOT, G1_BACKBONES, resolve_config, parse_args, discover_sources,
                             predict_native, run)


class ContractBackbone(nn.Module):
    """Explicit API fixture; has no pretrained DINO representation."""
    def __init__(self, name):
        super().__init__()
        spec = backbone_spec(name)
        self.embed_dim, self.patch_size = spec.channels, spec.patch_size
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(spec.depth))
        self.scale = nn.Parameter(torch.tensor(1.))
        self.channel_weight = nn.Parameter(torch.ones(self.embed_dim))
        self.requested = []

    def get_intermediate_layers(self, x, *, n, **kwargs):
        self.requested.append(tuple(n))
        assert kwargs == dict(reshape=True, return_class_token=False, return_extra_tokens=False, norm=True)
        feature = torch.nn.functional.avg_pool2d(x.mean(1, keepdim=True), 16) * self.scale
        return tuple((feature * self.channel_weight[None, :, None, None]).contiguous() for _ in n)


def assets(tmp_path, monkeypatch, name="dinov3_vits16"):
    repo = tmp_path / "dino"
    repo.mkdir(exist_ok=True)
    (repo / "hubconf.py").write_text("# API contract fixture only")
    weights = tmp_path / f"{name}.pth"
    backbone = ContractBackbone(name)
    torch.save(backbone.state_dict(), weights)
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: ContractBackbone(kwargs["model"]))
    return repo, weights


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", G1_BACKBONES)
def test_deepest_shapes_freeze_and_decoder_gradient(tmp_path, monkeypatch, name):
    repo, weights = assets(tmp_path, monkeypatch, name)
    extractor = DINOv3FeatureExtractor(repo, weights, name, feature_mode="deepest")
    spec = backbone_spec(name)
    model = E1(extractor, hidden_channels=4)
    model.train()
    image = torch.rand(1, 3, 512, 512)
    feature = extractor(image)
    assert list(feature) == [f"b{spec.depth}"]
    assert feature[f"b{spec.depth}"].shape == (1, spec.channels, 32, 32)
    assert extractor.backbone.requested == [(spec.depth - 1,)]
    frozen = {k: v.clone() for k, v in extractor.state_dict().items()}
    before = [p.clone().detach() for p in model.decoder.parameters()]
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=.01)
    logits = model(image)
    assert logits.shape == (1, 1, 512, 512)
    logits.square().mean().backward()
    optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.decoder.parameters()))
    assert all(torch.equal(v, frozen[k]) for k, v in extractor.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in extractor.parameters())
    assert not extractor.backbone.training
    assert set(model._modules) == {"extractor", "decoder"}


def test_checkpoint_mismatch_and_unofficial_model_rejected(tmp_path, monkeypatch):
    repo, weights = assets(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="incompatible with dinov3_vitb16"):
        DINOv3FeatureExtractor(repo, weights, "dinov3_vitb16", feature_mode="deepest")
    torch.save({"wrong_key": torch.ones(1)}, weights)
    with pytest.raises(ValueError, match="incompatible with dinov3_vits16"):
        DINOv3FeatureExtractor(repo, weights, feature_mode="deepest")
    with pytest.raises(ValueError, match="Unsupported DINOv3 backbone"):
        DINOv3FeatureExtractor(repo, weights, "dinov3_vitsh16", feature_mode="deepest")
    with pytest.raises(ValueError, match="deepest block"):
        DINOv3FeatureExtractor(repo, weights, blocks=(4, 8, 12), feature_mode="deepest")


def test_native_synthetic_exact_mask_bins_and_determinism():
    protocol = yaml.safe_load((ROOT / "configs/full_scale_synthetic.yaml").read_text())
    generator = NativeTinyDefectGenerator(protocol)
    image = torch.rand(3, 529, 677, generator=torch.Generator().manual_seed(8))
    for size_bin in protocol["size_bins"]:
        for kind in protocol["defect_types"]:
            for placement in protocol["placements"]:
                kwargs = dict(seed=72, defect_type=kind, size_bin=size_bin, placement=placement)
                first, repeat = generator(image, **kwargs), generator(image, **kwargs)
                assert torch.equal(first.image, repeat.image) and torch.equal(first.mask, repeat.mask)
                assert set(first.mask.unique().tolist()) <= {0., 1.}
                area = int(first.mask.sum())
                lo, hi = protocol["size_bins"][size_bin]
                assert lo <= area <= hi
                # Foreground is exactly the changed pixel support, in native coordinates.
                assert torch.equal((first.image != image).any(0), first.mask[0].bool())
                assert first.metadata["mean_abs_change"] >= protocol["min_mean_abs_change"]
    normal = generator(image, seed=0, defect_type="pinhole", size_bin="sub_patch", placement="interior", normal=True)
    assert not normal.mask.any() and torch.equal(normal.image, image)


@pytest.mark.parametrize("hw", [(17, 29), (529, 677), (512, 512)])
def test_padding_and_Hann_preserve_asymmetric_coordinates(hw):
    h, w = hw
    field = torch.linspace(0, 1, h * w).reshape(1, h, w)
    field[:, 0, 0] = 1
    field[:, -1, -1] = .73
    records = generate_tile_records(h, w)
    tiles = [crop_with_padding(field, r.local_xyxy, pad_mode="constant")[0] for r in records]
    restored = stitch_tiles_hann(tiles, records, hw)
    assert torch.allclose(restored, field[0], atol=2e-7)
    assert torch.isfinite(restored).all()
    if h < 512:
        assert not tiles[0][h:, :].any() and not tiles[0][:, w:].any()


def config_and_dataset(tmp_path, monkeypatch):
    repo, weights = assets(tmp_path, monkeypatch)
    data = tmp_path / "data"
    rng = np.random.default_rng(9)
    for split, count in (("TRAIN", 8), ("VALIDATION", 1), ("TEST_PUBLIC", 1)):
        folder = data / "rice" / split / ("bad" if split == "TEST_PUBLIC" else "good")
        folder.mkdir(parents=True)
        for i in range(count):
            Image.fromarray(rng.integers(40, 180, (96, 128, 3), dtype=np.uint8)).save(folder / f"{i}.png")
    cfg = resolve_config(parse_args(["--data-root", str(data), "--repo-dir", str(repo),
                                     "--weights", str(weights), "--output-root", str(tmp_path / "out")]))
    cfg["data"].update(tile_size=128, overlap=32, train_variants_per_image=1, dev_variants_per_image=3)
    cfg["decoder"]["hidden_channels"] = 4
    cfg["training"].update(mode="overfit16", epochs=5, batch_size=4, learning_rate=.02, max_steps=8)
    cfg["evaluation"].update(tile_batch_size=2, example_limit=2)
    return cfg, data


def test_train_dev_disjoint_fixed16_and_native_mask_tile_alignment(tmp_path, monkeypatch):
    cfg, data = config_and_dataset(tmp_path, monkeypatch)
    pools, manifest = discover_sources(cfg)
    assert len(pools["train"]) == 8 and len(pools["dev"]) == 1
    assert not any("TEST" in r["path"] for role in manifest.values() for r in role)
    native = G1NativeDataset(pools["train"], cfg["synthetic_protocol"], seed=2026, role="train", variants=1)
    tiles = G1TileDataset(native, tile_size=128, overlap=32, overfit16=True)
    assert len(tiles) == 16
    originals = [tiles[i] for i in range(16)]
    native.set_epoch(4)
    assert all(torch.equal(originals[i]["image"], tiles[i]["image"]) and
               torch.equal(originals[i]["mask"], tiles[i]["mask"]) for i in range(16))
    assert sum(bool(r["mask"].any()) for r in originals) == 8
    dev = data / "rice/VALIDATION/good/0.png"
    dev.write_bytes((data / "rice/TRAIN/good/0.png").read_bytes())
    with pytest.raises(ValueError, match="source leakage"):
        discover_sources(cfg)


def test_existing_train_checkpoint_resume_and_DEV_path(tmp_path, monkeypatch):
    cfg, _ = config_and_dataset(tmp_path, monkeypatch)
    out = Path(cfg["output_root"]) / "rice"
    out.mkdir(parents=True)
    result = run(cfg, device="cpu", output_dir=out)
    assert result["frozen_backbone_unchanged"] and result["decoder_updated"]
    assert result["overfit_loss_decreased"]
    assert result["best_synthetic_dev"]["qa_status"] == "PASS"
    assert 0 <= result["best_synthetic_dev"]["synthetic_dev_aupro_0_05"] <= 1
    assert result["best_synthetic_dev"]["dev_tiny"]["regions"] == 2
    for name in ("best.pt", "last.pt", "resolved_config.yaml", "metrics.json", "loss_curve.png"):
        assert (out / name).is_file()
    maps = list((out / "examples").glob("*/predicted_map.npy"))
    assert maps and all(np.load(path).shape == (96, 128) for path in maps)
    # Check exact partial-epoch continuation against the same uninterrupted path.
    cfg["training"].update(epochs=6, max_steps=10)
    resumed = run(cfg, device="cpu", output_dir=out, resume=out / "last.pt")
    assert resumed["global_step"] == 10
    resumed_state = torch.load(out / "last.pt", weights_only=False)["model_state"]
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    run(cfg, device="cpu", output_dir=continuous)
    continuous_state = torch.load(continuous / "last.pt", weights_only=False)["model_state"]
    assert all(torch.equal(value, continuous_state[key]) for key, value in resumed_state.items())
    evaluated = run(cfg, device="cpu", output_dir=out, evaluate=out / "best.pt")
    assert evaluated["qa_status"] == "PASS" and (out / "evaluation_metrics.json").is_file()
    assert (out / "evaluation/qa_report.json").is_file()
    # Exercise CLI logging and its output contract through the same real code path.
    from src.train.g1_e1 import main
    raw = yaml.safe_load((ROOT / "configs/g1_e1.yaml").read_text())
    for key in ("data", "decoder", "training", "evaluation"):
        raw[key] = copy.deepcopy(cfg[key])
    raw["backbone"]["repo_dir"] = cfg["backbone"]["repo_dir"]
    raw["backbone"]["checkpoints"]["dinov3_vits16"] = cfg["backbone"]["weights"]
    raw["output_root"] = str(tmp_path / "cli")
    raw["training"].update(epochs=1, max_steps=1)
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    main(["--config", str(config_path), "--device", "cpu"])
    log = (tmp_path / "cli/rice/train.log").read_text()
    assert "BCE=" in log and "Dice=" in log and "checkpoint=" in log


def test_native_inference_orientation_with_existing_normalization():
    from src.data.loader import DINOV3_MEAN, DINOV3_STD
    class PixelProbability(nn.Module):
        def forward(self, image):
            value = image[:, :1] * DINOV3_STD[0] + DINOV3_MEAN[0]
            return torch.logit(value.clamp(.001, .999))
    cfg = resolve_config(parse_args([]))
    cfg["evaluation"]["tile_batch_size"] = 2
    image = torch.full((3, 529, 677), .2)
    image[:, 0:3, 0:7] = .8
    image[:, -7:, -2:] = .65
    score = predict_native(PixelProbability(), image, cfg, "cpu")
    assert torch.allclose(score, image[0], atol=3e-7)


@pytest.mark.integration
def test_real_pretrained_G1_smoke_when_assets_are_supplied(tmp_path):
    names = ("MVTEC_AD2_ROOT", "DINOV3_REPO", "DINOV3_WEIGHTS")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("NOT VERIFIED: real MVTec AD2/official DINOv3 assets unavailable")
    cfg = resolve_config(parse_args(["--smoke", "--data-root", os.environ[names[0]],
                                     "--repo-dir", os.environ[names[1]], "--weights", os.environ[names[2]]]))
    cfg["output_root"] = str(tmp_path)
    out = tmp_path / "rice"
    out.mkdir()
    run(cfg, device="cuda" if torch.cuda.is_available() else "cpu", output_dir=out)
