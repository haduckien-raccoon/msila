from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.data.cached_dataset import (
    CachedFeatureDataset,
    make_cached_dataloader,
)
from src.data.feature_cache import (
    FEATURE_KEYS,
    FeatureCacheWriter,
)
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.cached_training import CachedFeatureTrainingModel
from src.train.optimizer import build_day3_msila_optimizer


# ---------------------------------------------------------------------------
# Controlled unit/integration fixture.
#
# IMPORTANT:
#   adapter r, adapter d, and downstream fusion_dim are intentionally different.
#   This prevents the regression test from accidentally treating:
#
#       adapter_projection_dim == fusion_dim
#
#   which is NOT the Day-04 architecture contract.
# ---------------------------------------------------------------------------
TEST_ADAPTER_R = 4
TEST_ADAPTER_D = 5
TEST_FUSION_DIM = 6


def _identity_geometry(mask_hw=(32, 32)) -> dict[str, Any]:
    h, w = mask_hw
    identity = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return {
        # feature-cache v1 required fields
        "local_box": [0.0, 0.0, float(w), float(h)],
        "context_box": [0.0, 0.0, float(w), float(h)],
        "context_to_local": identity,
        # input-size fields used by the alignment bridge
        "local_hw": [h, w],
        "context_hw": [h, w],
    }


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _make_cached_sample(
    *,
    sample_index: int,
    channels: int,
    feature_hw: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(1000 + sample_index)

    # Cache format stores the extraction batch dimension [1,C,h,w].
    features = {
        key: torch.randn(
            1,
            channels,
            feature_hw,
            feature_hw,
            generator=generator,
            dtype=torch.float32,
        )
        for key in FEATURE_KEYS
    }

    return {
        "image_id": f"sample_{sample_index:02d}",
        "category": "task14_qa",
        "geometry": _identity_geometry((32, 32)),
        **features,
    }


def _build_real_cache_backed_batch(tmp_path: Path):
    """Build a real FeatureCacheWriter -> CachedFeatureDataset test fixture.

    This fixture is synthetic at the *source tensor* level, but it exercises the
    real cache writer/reader/dataset/collate pipeline.  It is a software
    integration test, not an experimental Day-04 result.
    """
    cache_dir = tmp_path / "cache"
    signature = {
        "backbone": "task14-controlled-frozen-feature-source",
        "blocks": [4, 8, 12],
        "preprocess_version": "qa-v1",
    }

    channels = 8
    feature_hw = 8

    with FeatureCacheWriter(
        cache_dir,
        producer_signature=signature,
        target_shard_bytes=1024 * 1024,
    ) as writer:
        writer.add(
            _make_cached_sample(
                sample_index=0,
                channels=channels,
                feature_hw=feature_hw,
            )
        )
        writer.add(
            _make_cached_sample(
                sample_index=1,
                channels=channels,
                feature_hw=feature_hw,
            )
        )

    # sample_00 = normal -> loader creates exact zero mask.
    # sample_01 = anomaly -> non-empty exact binary mask from file.
    anomaly_mask = np.zeros((32, 32), dtype=np.uint8)
    anomaly_mask[8:24, 10:22] = 255
    _write_mask(tmp_path / "masks" / "sample_01.png", anomaly_mask)

    records = [
        {
            "image_id": "sample_00",
            "category": "task14_qa",
            "is_anomaly": False,
            "mask_hw": [32, 32],
        },
        {
            "image_id": "sample_01",
            "category": "task14_qa",
            "is_anomaly": True,
            "mask_hw": [32, 32],
            "mask_path": "masks/sample_01.png",
        },
    ]

    dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=records,
        expected_producer_signature=signature,
        mask_root=tmp_path,
        mask_hw_source="record",
        squeeze_cached_batch_dim=True,
        feature_dtype=torch.float32,
        mmap=False,
    )

    loader = make_cached_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    return next(iter(loader)), channels


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _changed(before: Mapping[str, torch.Tensor], module: nn.Module) -> bool:
    current = dict(module.named_parameters())
    return any(
        not torch.equal(value, current[name].detach())
        for name, value in before.items()
    )


def _assert_finite_gradient_exists(module: nn.Module, name: str) -> None:
    grads = [
        p.grad
        for p in module.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads, f"{name}: no trainable gradient"
    assert all(torch.isfinite(g).all() for g in grads), (
        f"{name}: NaN/Inf gradient"
    )
    assert any(float(g.abs().sum().item()) > 0.0 for g in grads), (
        f"{name}: all available gradients are zero"
    )


def _assert_day04_adapter_contract(
    model: CachedFeatureTrainingModel,
    *,
    in_channels: int,
    adapter_bottleneck_dim: int,
    adapter_projection_dim: int,
) -> None:
    """Verify that CachedFeatureTrainingModel contains the exact Day-04 r×d Adapter."""

    assert set(model.adapters.keys()) == {"b4", "b8", "b12"}

    for block in ("b4", "b8", "b12"):
        adapter = model.adapters[block]

        assert adapter.in_dim == in_channels
        assert adapter.r == adapter_bottleneck_dim
        assert adapter.d == adapter_projection_dim

        # C -> r
        assert adapter.down_proj.in_channels == in_channels
        assert adapter.down_proj.out_channels == adapter_bottleneck_dim

        # r -> r depthwise
        assert adapter.dwconv.in_channels == adapter_bottleneck_dim
        assert adapter.dwconv.out_channels == adapter_bottleneck_dim
        assert adapter.dwconv.groups == adapter_bottleneck_dim

        # r -> d
        assert adapter.mid_proj.in_channels == adapter_bottleneck_dim
        assert adapter.mid_proj.out_channels == adapter_projection_dim

        # d -> C
        assert adapter.out_proj.in_channels == adapter_projection_dim
        assert adapter.out_proj.out_channels == in_channels


def _run_one_train_step(
    *,
    batch: Mapping[str, Any],
    in_channels: int,
    adapter_bottleneck_dim: int = TEST_ADAPTER_R,
    adapter_projection_dim: int = TEST_ADAPTER_D,
    fusion_dim: int = TEST_FUSION_DIM,
):
    device = torch.device("cpu")

    model = CachedFeatureTrainingModel.build_default(
        in_channels=in_channels,

        # Day-04 Adapter variables:
        adapter_bottleneck_dim=adapter_bottleneck_dim,   # r
        adapter_projection_dim=adapter_projection_dim,   # d

        # Downstream projection/head width; NOT Adapter d.
        fusion_dim=fusion_dim,

        output_size=(
            int(batch["mask"].shape[-2]),
            int(batch["mask"].shape[-1]),
        ),
        adapter_kernel_size=3,
        gamma_init=0.0,
        adapter_bias=True,
        share_projection_across_views=True,
        validate=True,
    ).to(device)

    _assert_day04_adapter_contract(
        model,
        in_channels=in_channels,
        adapter_bottleneck_dim=adapter_bottleneck_dim,
        adapter_projection_dim=adapter_projection_dim,
    )

    criterion = AnomalySegmentationLoss(
        bce_weight=1.0,
        dice_weight=1.0,
    )

    optimizer, optimizer_report = build_day3_msila_optimizer(
        adapters=model.adapters,
        projection=model.projection,
        fusion=model.fusion,
        decoder=model.decoder,
        dino=None,  # cached training does not instantiate DINO
        optimizer_name="adamw",
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    tensor_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_batch[key] = value.to(device)
        else:
            tensor_batch[key] = value

    # Cached frozen DINO features are input data, not trainable graph roots.
    for key in FEATURE_KEYS:
        assert tensor_batch[key].requires_grad is False
        assert tensor_batch[key].grad is None

    before = {
        "adapter": _snapshot(model.adapters),
        "projection": _snapshot(model.projection),
        "fusion": _snapshot(model.fusion),
        "decoder": _snapshot(model.decoder),
    }

    optimizer.zero_grad(set_to_none=True)

    logits, trace = model(
        tensor_batch,
        return_trace=True,
    )

    assert logits.shape == tensor_batch["mask"].shape
    assert torch.isfinite(logits).all()

    # Adapter preserves the frozen-backbone C-dimensional feature space.
    for block in (4, 8, 12):
        assert trace["adapted_local"][f"L{block}"].shape == tensor_batch[
            f"local_b{block}"
        ].shape
        assert trace["adapted_context"][f"C{block}"].shape == tensor_batch[
            f"context_b{block}"
        ].shape

    # Downstream projection is independently controlled by fusion_dim.
    for projected in trace["projected"].values():
        assert int(projected.shape[1]) == fusion_dim
        assert torch.isfinite(projected).all()

    loss_out = criterion(
        logits,
        tensor_batch["mask"],
    )
    loss = loss_out["loss"]

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    # gamma_init=0 gives exact identity at initialization.  At the first
    # backward pass, the residual branch weights may have zero gradients
    # because they are multiplied by gamma=0, but gamma itself must receive a
    # gradient.  Therefore the correct first-step gate is: at least one finite,
    # non-zero trainable gradient exists in the Adapter module.
    _assert_finite_gradient_exists(model.adapters, "Adapter")
    _assert_finite_gradient_exists(model.projection, "Projection")
    _assert_finite_gradient_exists(model.fusion, "Fusion")
    _assert_finite_gradient_exists(model.decoder, "Decoder")

    optimizer.step()

    # At least gamma changes in the Adapter after the first optimizer step.
    assert _changed(before["adapter"], model.adapters)
    assert _changed(before["projection"], model.projection)
    assert _changed(before["fusion"], model.fusion)
    assert _changed(before["decoder"], model.decoder)

    # Frozen DINO is absent by design in cached training.
    assert not hasattr(model, "dino")

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "optimizer_report": optimizer_report,
        "logits": logits.detach(),
        "trace": trace,
        "loss": float(loss.detach().item()),
    }


def test_cached_builder_separates_adapter_d_from_fusion_dim(tmp_path):
    """Regression gate for the Day-04 API migration.

    Adapter d and downstream fusion_dim intentionally have different values.
    The test fails if a future refactor silently reuses fusion_dim as Adapter d.
    """
    batch, channels = _build_real_cache_backed_batch(tmp_path)

    adapter_r = 3
    adapter_d = 5
    fusion_dim = 7

    model = CachedFeatureTrainingModel.build_default(
        in_channels=channels,
        adapter_bottleneck_dim=adapter_r,
        adapter_projection_dim=adapter_d,
        fusion_dim=fusion_dim,
        output_size=tuple(batch["mask"].shape[-2:]),
        gamma_init=0.0,
        validate=True,
    )

    _assert_day04_adapter_contract(
        model,
        in_channels=channels,
        adapter_bottleneck_dim=adapter_r,
        adapter_projection_dim=adapter_d,
    )

    assert adapter_d != fusion_dim


def test_cached_feature_to_optimizer_step_end_to_end(tmp_path):
    """
    Mandatory cached-training integration gate:

        FeatureCacheWriter
        -> CachedFeatureDataset/DataLoader
        -> Day-04 Adapter(C -> r -> DWConv -> d -> C)
        -> Context->Local Alignment
        -> SixFeatureProjection(C -> fusion_dim)
        -> Attention Fusion
        -> Decoder
        -> BCE+Dice Loss
        -> backward
        -> AdamW optimizer.step

    This test verifies the real software path.  It does not claim experimental
    anomaly-detection performance.
    """
    torch.manual_seed(2026)

    batch, channels = _build_real_cache_backed_batch(tmp_path)
    result = _run_one_train_step(
        batch=batch,
        in_channels=channels,
        adapter_bottleneck_dim=TEST_ADAPTER_R,
        adapter_projection_dim=TEST_ADAPTER_D,
        fusion_dim=TEST_FUSION_DIM,
    )

    assert np.isfinite(result["loss"])
    assert result["loss"] > 0.0

    trace = result["trace"]
    assert set(trace["projected"]) == {
        "local_b4",
        "local_b8",
        "local_b12",
        "context_b4",
        "context_b8",
        "context_b12",
    }

    # Attention has six source weights: Local/Context × blocks 4/8/12.
    assert trace["attention"].shape == (2, 6)
    assert torch.allclose(
        trace["attention"].sum(dim=1),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_cached_geometry_inverse_fallback_is_exercised(tmp_path):
    """
    Cache-v1 stores context_to_local while the aligner consumes
    local_to_context. The integration bridge must explicitly invert the matrix.
    """
    batch, channels = _build_real_cache_backed_batch(tmp_path)

    # Controlled cache geometry intentionally contains context_to_local but not
    # local_to_context, so a successful step proves the fallback path executes.
    for meta in batch["meta"]:
        geometry = meta["geometry"]
        assert "context_to_local" in geometry
        assert "local_to_context" not in geometry

    result = _run_one_train_step(
        batch=batch,
        in_channels=channels,
        adapter_bottleneck_dim=TEST_ADAPTER_R,
        adapter_projection_dim=TEST_ADAPTER_D,
        fusion_dim=TEST_FUSION_DIM,
    )

    assert np.isfinite(result["loss"])


@pytest.mark.integration
def test_external_real_cache_one_training_step():
    """Acceptance gate on the cache produced by the real project builder.

    Required environment
    --------------------
    FEATURE_CACHE_DIR
    FEATURE_TRAIN_INDEX

    Optional environment
    --------------------
    FEATURE_MASK_ROOT
    FEATURE_FUSION_DIM       default: 128
    FEATURE_ADAPTER_R        default: 64
    FEATURE_ADAPTER_D        default: 256

    ``FEATURE_ADAPTER_R`` and ``FEATURE_ADAPTER_D`` are the Day-04 Adapter
    variables. ``FEATURE_FUSION_DIM`` is downstream and remains independent.

    The defaults are only an integration-test candidate.  They are NOT a
    selected/winning Day-04 configuration.
    """
    cache_dir = os.getenv("FEATURE_CACHE_DIR")
    train_index = os.getenv("FEATURE_TRAIN_INDEX")

    if not cache_dir or not train_index:
        pytest.skip(
            "Set FEATURE_CACHE_DIR and FEATURE_TRAIN_INDEX to run the real-cache "
            "acceptance gate."
        )

    mask_root = os.getenv("FEATURE_MASK_ROOT")
    fusion_dim = int(os.getenv("FEATURE_FUSION_DIM", "128"))
    adapter_r = int(os.getenv("FEATURE_ADAPTER_R", "64"))
    adapter_d = int(os.getenv("FEATURE_ADAPTER_D", "256"))

    if fusion_dim <= 0:
        raise ValueError("FEATURE_FUSION_DIM must be > 0")
    if adapter_r <= 0:
        raise ValueError("FEATURE_ADAPTER_R must be > 0")
    if adapter_d <= 0:
        raise ValueError("FEATURE_ADAPTER_D must be > 0")

    dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=train_index,
        mask_root=mask_root,
        mask_hw_source="record",
        squeeze_cached_batch_dim=True,
        feature_dtype=torch.float32,
        mmap=True,
    )

    loader = make_cached_dataloader(
        dataset,
        batch_size=min(2, len(dataset)),
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    in_channels = int(batch["local_b4"].shape[1])

    result = _run_one_train_step(
        batch=batch,
        in_channels=in_channels,
        adapter_bottleneck_dim=adapter_r,
        adapter_projection_dim=adapter_d,
        fusion_dim=fusion_dim,
    )

    assert np.isfinite(result["loss"])

    model = result["model"]
    _assert_day04_adapter_contract(
        model,
        in_channels=in_channels,
        adapter_bottleneck_dim=adapter_r,
        adapter_projection_dim=adapter_d,
    )
