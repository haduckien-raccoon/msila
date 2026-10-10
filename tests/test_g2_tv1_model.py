"""D5-TV1 E2 architecture/autograd gates; fixtures are not pretrained DINO."""

import copy
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.adapter_factory import ResidualAdapterFactory
from src.models.backbone_registry import BACKBONES, backbone_spec
from src.models.contracts import ContractError, validate_g2_batch
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.models.msila import E1, E2, build_g2_model, resolve_g2_config
from src.models.residual_adapter import ResidualAdapter2d


ROOT = Path(__file__).resolve().parents[1]


class BackboneFixture(nn.Module):
    """Official intermediate-layer API fixture; no pretrained representation."""

    def __init__(self, name):
        super().__init__()
        spec = backbone_spec(name)
        self.embed_dim, self.patch_size = spec.channels, spec.patch_size
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(spec.depth))
        self.scale = nn.Parameter(torch.tensor(0.9))
        self.channel_weight = nn.Parameter(torch.linspace(0.5, 1.5, self.embed_dim))
        self.requested = []

    def get_intermediate_layers(self, image, *, n, **kwargs):
        self.requested.append(tuple(n))
        assert kwargs == dict(reshape=True, return_class_token=False,
                             return_extra_tokens=False, norm=True)
        feature = F.avg_pool2d(image.mean(1, keepdim=True), 16) * self.scale
        feature = feature * self.channel_weight[None, :, None, None]
        return tuple(feature.contiguous() for _ in n)


@pytest.fixture(autouse=True)
def deterministic_cpu_fixture():
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    yield
    torch.set_num_threads(previous_threads)
    torch.set_rng_state(previous_rng)


@pytest.fixture
def extractor_factory(tmp_path, monkeypatch):
    repo = tmp_path / "dino_api_fixture"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# Test-only official API fixture\n")
    monkeypatch.setattr(torch.hub, "load",
                        lambda **kwargs: BackboneFixture(kwargs["model"]))

    def build(name="dinov3_vits16", feature_mode="deepest"):
        weights = tmp_path / f"{name}.pth"
        torch.save(BackboneFixture(name).state_dict(), weights)
        return DINOv3FeatureExtractor(repo, weights, model_name=name,
                                      feature_mode=feature_mode, check_finite=True)
    return build


@pytest.fixture
def batch():
    target = torch.zeros(2, 1, 512, 512)
    target[1, :, 120:135, 235:248] = 1
    return dict(image=torch.randn(2, 3, 512, 512), mask=target,
                meta=[{"sample_id": "normal"}, {"sample_id": "positive"}])


def assert_e2_step(model, batch):
    """One actual update; shared by the fixture and real-asset integration."""
    validate_g2_batch(batch)
    model.train()
    assert set(model._modules) == {"extractor", "adapter", "decoder"}
    assert isinstance(model.adapter, ResidualAdapter2d)
    assert not model.extractor.training and not model.extractor.backbone.training
    frozen = {key: value.clone() for key, value in model.extractor.state_dict().items()}
    before_decoder = [p.detach().clone() for p in model.decoder.parameters()]
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected = list(model.adapter.parameters()) + list(model.decoder.parameters())
    assert {id(p) for p in trainable} == {id(p) for p in expected}
    optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=0.0)
    logits, trace = model(batch, return_trace=True)
    assert logits.shape == (batch["image"].shape[0], 1, 512, 512)
    assert torch.isfinite(logits).all()
    assert list(trace["dino"]) == [model.feature_key]
    deep = trace["dino"][model.feature_key]
    assert not deep.requires_grad
    assert deep.shape == (batch["image"].shape[0], model.extractor.out_channels, 32, 32)
    assert torch.equal(trace["decoder_feature"], deep)  # gamma=0 exact identity
    loss = AnomalySegmentationLoss()(logits, batch["mask"])["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert model.adapter.gamma.grad.abs().item() > 0
    for name, parameter in model.adapter.named_parameters():
        if name != "gamma":
            assert torch.count_nonzero(parameter.grad) == 0  # valid zero-gate behavior
    optimizer.step()
    assert model.adapter.gamma.item() != 0
    assert any(not torch.equal(old, new)
               for old, new in zip(before_decoder, model.decoder.parameters()))
    assert all(not p.requires_grad and p.grad is None for p in model.extractor.parameters())
    assert all(torch.equal(value, frozen[key])
               for key, value in model.extractor.state_dict().items())
    return optimizer


@pytest.mark.parametrize("name", tuple(BACKBONES))
def test_e2_deepest_forward_backward_and_frozen_update(extractor_factory, batch, name):
    extractor = extractor_factory(name)
    model = E2(extractor, adapter_bottleneck_dim=8, adapter_projection_dim=12,
               hidden_channels=4)
    assert_e2_step(model, batch)
    assert extractor.backbone.requested == [(backbone_spec(name).depth - 1,)]


@pytest.mark.parametrize("name", tuple(BACKBONES))
def test_config_builder_switches_backbone_channels_and_runs_backward(extractor_factory, batch, name):
    assets = extractor_factory(name)
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["backbone"].update(name=name, repo_dir=str(assets.repo_dir), weights=assets.weights)
    cfg["decoder"]["hidden_channels"] = 4
    resolved = resolve_g2_config(cfg)
    model = build_g2_model(cfg)
    assert model.extractor.out_channels == backbone_spec(name).channels
    assert model.decoder.head[0].in_channels == backbone_spec(name).channels
    assert (model.adapter_r, model.adapter_d) == (resolved["adapter"]["r"], resolved["adapter"]["d"])
    assert_e2_step(model, batch)


def test_e1_builder_uses_same_vitb_config(extractor_factory, batch):
    assets = extractor_factory("dinov3_vitb16")
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["backbone"].update(repo_dir=str(assets.repo_dir), weights=assets.weights)
    cfg["decoder"]["hidden_channels"] = 4
    model = build_g2_model(cfg, experiment="E1")
    assert set(model._modules) == {"extractor", "decoder"}
    assert model.decoder.head[0].in_channels == 768
    assert model(batch).shape == (2, 1, 512, 512)


def test_zero_gamma_matches_e1_and_never_extracts_context(extractor_factory, batch):
    extractor = extractor_factory()
    e1 = E1(copy.deepcopy(extractor), hidden_channels=4)
    e2 = E2(extractor, adapter_bottleneck_dim=8, adapter_projection_dim=12,
            hidden_channels=4)
    e2.decoder.load_state_dict(e1.decoder.state_dict())
    batch["context"] = torch.randn_like(batch["image"])
    with torch.no_grad():
        logits1, trace1 = e1(batch, return_trace=True)
        logits2, trace2 = e2(batch, return_trace=True)
        tensor_logits = e2(batch["image"])
    assert torch.equal(logits1, logits2) and torch.equal(logits2, tensor_logits)
    assert torch.equal(trace1["decoder_feature"], trace2["decoder_feature"])
    assert extractor.backbone.requested == [(11,), (11,)]  # one view per forward
    assert set(e1._modules) == {"extractor", "decoder"}
    assert set(e2._modules) == {"extractor", "adapter", "decoder"}


def test_adapter_branch_learns_after_first_gate_update(extractor_factory, batch):
    model = E2(extractor_factory(), adapter_bottleneck_dim=8,
               adapter_projection_dim=12, hidden_channels=4)
    optimizer = assert_e2_step(model, batch)
    optimizer.zero_grad(set_to_none=True)
    branch_before = {name: p.detach().clone() for name, p in model.adapter.named_parameters()
                     if name != "gamma"}
    AnomalySegmentationLoss()(model(batch), batch["mask"])["loss"].backward()
    for name, parameter in model.adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if name != "gamma":
            assert torch.count_nonzero(parameter.grad) > 0, name
    optimizer.step()
    assert all(not torch.equal(old, dict(model.adapter.named_parameters())[name])
               for name, old in branch_before.items())
    assert all(p.grad is None for p in model.extractor.parameters())


@pytest.mark.parametrize("r,d,k,bias", [(4, 7, 3, True), (12, 5, 5, False), (9, 21, 1, True)])
def test_configurable_rd_factory_and_parameter_counts(extractor_factory, monkeypatch, r, d, k, bias):
    builds = []
    build_rd = ResidualAdapterFactory.build_rd

    def audited_build(factory, **kwargs):
        builds.append(kwargs)
        return build_rd(factory, **kwargs)
    monkeypatch.setattr(ResidualAdapterFactory, "build_rd", audited_build)
    model = E2(extractor_factory(), adapter_bottleneck_dim=r, adapter_projection_dim=d,
               adapter_kernel_size=k, adapter_bias=bias, hidden_channels=4)
    assert builds == [{"r": r, "d": d}]
    assert (model.adapter_r, model.adapter_d) == (r, d)
    c = model.extractor.out_channels
    expected_adapter = c*r + r*k*k + r*d + d*c + (2*r + d + c if bias else 0) + 1
    expected_decoder = 4*c*9 + 4 + 4 + 1
    assert model.adapter.num_trainable_parameters == expected_adapter
    assert model.num_trainable_parameters == expected_adapter + expected_decoder
    assert sum(p.numel() for p in model.extractor.parameters() if p.requires_grad) == 0
    assert model.decoder.head[0].in_channels == c  # Adapter d is internal.


@pytest.mark.parametrize("kwargs,error", [
    ({"adapter_bottleneck_dim": 0}, ValueError),
    ({"adapter_projection_dim": -1}, ValueError),
    ({"adapter_bottleneck_dim": True}, TypeError),
    ({"adapter_projection_dim": 2.5}, TypeError),
    ({"adapter_kernel_size": 2}, ValueError),
    ({"gamma_init": float("nan")}, ValueError),
    ({"adapter_bias": 1}, TypeError),
])
def test_invalid_adapter_config_rejected(extractor_factory, kwargs, error):
    config = dict(adapter_bottleneck_dim=8, adapter_projection_dim=12)
    config.update(kwargs)
    with pytest.raises(error):
        E2(extractor_factory(), **config)


def test_multilayer_extractor_rejected(extractor_factory):
    with pytest.raises(ValueError, match="deepest"):
        E2(extractor_factory(feature_mode="multilayer"),
           adapter_bottleneck_dim=8, adapter_projection_dim=12)


def test_unexpected_extra_feature_rejected(extractor_factory, monkeypatch, batch):
    extractor = extractor_factory()
    original_forward = extractor.forward

    def extra_feature(image):
        features = original_forward(image)
        return {**features, "b4": features["b12"]}
    monkeypatch.setattr(extractor, "forward", extra_feature)
    model = E2(extractor, adapter_bottleneck_dim=8, adapter_projection_dim=12)
    with pytest.raises(ContractError, match="exactly one deepest"):
        model(batch)


@pytest.mark.integration
def test_real_e2_one_train_batch_when_assets_available():
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    for env, key in (("DINOV3_MODEL", "name"), ("DINOV3_REPO", "repo_dir"), ("DINOV3_WEIGHTS", "weights")):
        if env in os.environ:
            cfg["backbone"][key] = os.environ[env]
    cfg = resolve_g2_config(cfg)
    defaults = dict(MVTEC_AD2_ROOT=cfg["data"]["root"],
                    DINOV3_REPO=cfg["backbone"]["repo_dir"],
                    DINOV3_WEIGHTS=cfg["backbone"]["weights"])
    paths = {}
    missing = []
    for name, default in defaults.items():
        path = Path(os.environ.get(name, str(ROOT / default))).expanduser()
        valid = path.is_file() if name == "DINOV3_WEIGHTS" else path.is_dir()
        if not valid and name in os.environ:
            pytest.fail(f"Explicit {name} path is invalid: {path}")
        if not valid:
            missing.append(name)
        paths[name] = path
    if missing:
        pytest.skip("NOT RUN: real E2 batch requires " + ", ".join(missing))

    # Reuse the actual existing Data and loss code; never synthesize fake assets.
    from torch.utils.data import DataLoader
    from src.data.loader import G1NativeDataset, G1TileDataset, g1_tile_collate, scan_mvtec_ad2

    category = os.environ.get("G2_CATEGORY", "rice")
    assert category in cfg["categories"], "G2_CATEGORY must be one of the eight categories"
    sources = scan_mvtec_ad2(paths["MVTEC_AD2_ROOT"], categories=[category], split="train")
    good = [record for record in sources if record.is_normal]
    assert good, f"No real TRAIN/good sources for {category}"
    protocol = yaml.safe_load((ROOT / cfg["synthetic_protocol"]).read_text())
    native = G1NativeDataset(good[:1], protocol, seed=cfg["training"]["seed"],
                              role="train", variants=1, fixed=True)
    tiles = G1TileDataset(native, tile_size=512, overlap=128)
    # Include one real normal tile and one native synthetic-positive tile.
    indices = [next(i for i, (native_index, _) in enumerate(tiles.records)
                    if native.plan[native_index][1] == 0)]
    positive = next((i for i, (native_index, _) in enumerate(tiles.records)
                     if native.plan[native_index][1] == 1 and tiles[i]["mask"].any()), None)
    assert positive is not None, "Real synthetic source must produce a positive tile"
    indices.append(positive)
    batch = next(iter(DataLoader(torch.utils.data.Subset(tiles, indices), batch_size=2,
                                 collate_fn=g1_tile_collate, num_workers=0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in ("image", "mask"):
        batch[key] = batch[key].to(device)
    model = build_g2_model(cfg).to(device)
    assert_e2_step(model, batch)


def test_integration_harness_loader_wiring_with_fixture_assets(tmp_path, monkeypatch, extractor_factory):
    """Exercise the optional harness with fixtures, without claiming real data."""
    import numpy as np
    from PIL import Image

    extractor = extractor_factory()
    data = tmp_path / "fixture_data"
    folder = data / "rice" / "TRAIN" / "good"
    folder.mkdir(parents=True)
    pixels = np.random.default_rng(2026).integers(40, 180, (96, 128, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "source.png")
    monkeypatch.setenv("MVTEC_AD2_ROOT", str(data))
    monkeypatch.setenv("DINOV3_REPO", str(extractor.repo_dir))
    monkeypatch.setenv("DINOV3_WEIGHTS", str(extractor.weights))
    monkeypatch.setenv("G2_CATEGORY", "rice")
    monkeypatch.setenv("DINOV3_MODEL", "dinov3_vits16")
    test_real_e2_one_train_batch_when_assets_available()


def test_integration_harness_rejects_explicit_bad_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("MVTEC_AD2_ROOT", str(tmp_path / "missing_explicit_root"))
    with pytest.raises(pytest.fail.Exception, match="Explicit MVTEC_AD2_ROOT path is invalid"):
        test_real_e2_one_train_batch_when_assets_available()
