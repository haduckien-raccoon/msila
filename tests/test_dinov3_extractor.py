from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from torch import nn


MODULE_PATH = Path(__file__).resolve().parents[1] / "features" / "dinov3_extractor.py"
spec = importlib.util.spec_from_file_location("dinov3_extractor", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DINOv3FeatureExtractor = mod.DINOv3FeatureExtractor


class DummyBackbone(nn.Module):
    """Small mock implementing the DINOv3 intermediate-layer contract."""

    def __init__(self, *, nan_output: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.patch_size = 16
        self.embed_dim = 384
        self.nan_output = nan_output
        self.calls = 0
        self.last_n = None
        self.last_kwargs = None
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    def get_intermediate_layers(
        self,
        x,
        *,
        n,
        reshape=False,
        return_class_token=False,
        return_extra_tokens=False,
        norm=True,
    ):
        self.calls += 1
        self.last_n = tuple(n)
        self.last_kwargs = {
            "reshape": reshape,
            "return_class_token": return_class_token,
            "return_extra_tokens": return_extra_tokens,
            "norm": norm,
        }

        b, _, h, w = x.shape
        hh, ww = h // self.patch_size, w // self.patch_size
        base = x.mean(dim=(1, 2, 3), keepdim=True)

        outputs = []
        for idx in n:
            f = base.expand(b, self.embed_dim, hh, ww).clone() + float(idx)
            if self.nan_output:
                f[0, 0, 0, 0] = float("nan")
            outputs.append(f)
        return tuple(outputs)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "dinov3"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# test stub\n", encoding="utf-8")
    return repo


def _build_extractor(monkeypatch, tmp_path, *, check_finite=False, nan_output=False):
    backbone = DummyBackbone(nan_output=nan_output)

    def fake_hub_load(**kwargs):
        assert kwargs["source"] == "local"
        assert kwargs["model"] == "dinov3_vits16"
        return backbone

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    extractor = DINOv3FeatureExtractor(
        repo_dir=_make_repo(tmp_path),
        weights="dummy_checkpoint.pth",
        model_name="dinov3_vits16",
        check_finite=check_finite,
    )
    return extractor, backbone


def test_block_mapping_is_4_8_12_to_3_7_11(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(monkeypatch, tmp_path)
    assert extractor.blocks == (4, 8, 12)
    assert extractor.block_indices == (3, 7, 11)


def test_single_view_shapes_and_official_api_arguments(monkeypatch, tmp_path):
    extractor, backbone = _build_extractor(monkeypatch, tmp_path)
    x = torch.randn(2, 3, 512, 512)

    out = extractor(x)

    assert set(out) == {"b4", "b8", "b12"}
    for f in out.values():
        assert f.shape == (2, 384, 32, 32)
        assert not f.requires_grad

    assert backbone.last_n == (3, 7, 11)
    assert backbone.last_kwargs == {
        "reshape": True,
        "return_class_token": False,
        "return_extra_tokens": False,
        "norm": True,
    }


def test_local_context_concat_contract(monkeypatch, tmp_path):
    extractor, backbone = _build_extractor(monkeypatch, tmp_path)
    x_local = torch.ones(2, 3, 512, 512)
    x_context = torch.full((2, 3, 512, 512), 2.0)

    out = extractor.extract_local_context(x_local, x_context, strategy="concat")

    assert set(out) == {"L4", "L8", "L12", "C4", "C8", "C12"}
    assert backbone.calls == 1

    for key in ("L4", "L8", "L12", "C4", "C8", "C12"):
        assert out[key].shape == (2, 384, 32, 32)
        assert torch.isfinite(out[key]).all()

    # Dummy feature = input_mean + zero-based block index.
    assert torch.allclose(out["L4"].mean(), torch.tensor(1.0 + 3.0))
    assert torch.allclose(out["C4"].mean(), torch.tensor(2.0 + 3.0))
    assert torch.allclose(out["L8"].mean(), torch.tensor(1.0 + 7.0))
    assert torch.allclose(out["C12"].mean(), torch.tensor(2.0 + 11.0))


def test_sequential_has_same_output_but_two_backbone_calls(monkeypatch, tmp_path):
    extractor, backbone = _build_extractor(monkeypatch, tmp_path)
    x_local = torch.ones(1, 3, 512, 512)
    x_context = torch.full((1, 3, 512, 512), 2.0)

    out = extractor.extract_local_context(x_local, x_context, strategy="sequential")

    assert backbone.calls == 2
    assert out["L12"].shape == (1, 384, 32, 32)
    assert out["C12"].shape == (1, 384, 32, 32)
    assert torch.allclose(out["L12"].mean(), torch.tensor(12.0))
    assert torch.allclose(out["C12"].mean(), torch.tensor(13.0))


def test_backbone_remains_frozen_and_eval_after_parent_train(monkeypatch, tmp_path):
    extractor, backbone = _build_extractor(monkeypatch, tmp_path)

    assert extractor.backbone_is_frozen()
    assert not backbone.training

    extractor.train(True)

    assert extractor.training
    assert not backbone.training
    assert all(not p.requires_grad for p in backbone.parameters())


def test_no_grad_features_can_feed_trainable_adapter(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(monkeypatch, tmp_path)
    x = torch.randn(1, 3, 512, 512)
    f = extractor(x)["b12"]

    adapter = nn.Conv2d(384, 8, kernel_size=1)
    loss = adapter(f).square().mean()
    loss.backward()

    assert adapter.weight.grad is not None
    assert torch.isfinite(adapter.weight.grad).all()


def test_rejects_pair_shape_mismatch(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(monkeypatch, tmp_path)
    x_local = torch.randn(1, 3, 512, 512)
    x_context = torch.randn(1, 3, 768, 768)

    with pytest.raises(ValueError, match="same network-input shape"):
        extractor.extract_local_context(x_local, x_context)


def test_rejects_non_float_input(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(monkeypatch, tmp_path)
    x = torch.zeros(1, 3, 512, 512, dtype=torch.uint8)

    with pytest.raises(TypeError, match="floating-point"):
        extractor(x)


def test_rejects_size_not_divisible_by_patch(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(monkeypatch, tmp_path)
    x = torch.randn(1, 3, 510, 512)

    with pytest.raises(ValueError, match="divisible by patch_size"):
        extractor(x)


def test_debug_finite_check_detects_nan_output(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(
        monkeypatch,
        tmp_path,
        check_finite=True,
        nan_output=True,
    )
    x = torch.randn(1, 3, 512, 512)

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        extractor(x)


def test_invalid_strategy_is_rejected(monkeypatch, tmp_path):
    extractor, _ = _build_extractor(monkeypatch, tmp_path)
    x = torch.randn(1, 3, 512, 512)

    with pytest.raises(ValueError, match="strategy"):
        extractor.extract_local_context(x, x, strategy="bad")


def test_repo_without_hubconf_is_rejected(monkeypatch, tmp_path):
    repo = tmp_path / "not_dinov3"
    repo.mkdir()

    with pytest.raises(FileNotFoundError, match="hubconf.py"):
        DINOv3FeatureExtractor(repo, "weights.pth")


def test_optional_real_dinov3_smoke():
    """Optional integration test; skipped unless real repo/weights are provided."""
    repo = os.getenv("DINOV3_REPO")
    weights = os.getenv("DINOV3_WEIGHTS")
    if not repo or not weights:
        pytest.skip("Set DINOV3_REPO and DINOV3_WEIGHTS for real-backbone smoke test.")

    extractor = DINOv3FeatureExtractor(
        repo_dir=repo,
        weights=weights,
        model_name=os.getenv("DINOV3_MODEL", "dinov3_vits16"),
        check_finite=True,
    )
    x_local = torch.randn(1, 3, 512, 512)
    x_context = torch.randn(1, 3, 512, 512)
    out = extractor.extract_local_context(x_local, x_context)

    assert set(out) == {"L4", "L8", "L12", "C4", "C8", "C12"}
    assert all(torch.isfinite(v).all() for v in out.values())

def test_dino_is_really_frozen(extractor):
    # 1. requires_grad=False
    assert all(
        not p.requires_grad
        for p in extractor.backbone.parameters()
    )

    # 2. eval mode
    assert extractor.backbone.training is False

    # 3. gọi train() ở parent vẫn không bật DINO train
    extractor.train()
    assert extractor.backbone.training is False

def test_dino_has_no_grad_after_backward(
    extractor,
    projection,
):
    x = torch.randn(2, 3, 512, 512)

    features = extractor(x)

    y = projection(features["b12"])
    loss = y.mean()

    loss.backward()

    # DINO tuyệt đối không gradient
    for p in extractor.backbone.parameters():
        assert p.grad is None

    # Projection phải có gradient
    grads = [
        p.grad
        for p in projection.parameters()
        if p.requires_grad
    ]

    assert any(g is not None for g in grads)