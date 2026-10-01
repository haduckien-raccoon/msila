from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_cache import FEATURE_KEYS, FeatureCacheWriter
from src.data.cached_dataset import (
    CachedDatasetError,
    CachedFeatureDataset,
    MaskError,
    TrainingIndexError,
    cached_collate_fn,
    make_cached_dataloader,
)


@pytest.fixture
def signature():
    return {
        "backbone": "dinov3_vits16",
        "checkpoint_sha256": "test-checkpoint",
        "logical_layers_1based": [4, 8, 12],
        "internal_indices_0based": [3, 7, 11],
        "preprocess_version": "msila_local_context_v1",
    }


def cache_sample(i: int, category: str = "fabric"):
    g = torch.Generator().manual_seed(500 + i)

    def feat(h: int, w: int):
        # Deliberately keep extraction batch dimension = 1.
        return torch.randn((1, 8, h, w), generator=g, dtype=torch.float32)

    return {
        "image_id": f"{category}/sample_{i:03d}.png",
        "category": category,
        "local_b4": feat(16, 16),
        "local_b8": feat(16, 16),
        "local_b12": feat(16, 16),
        "context_b4": feat(24, 24),
        "context_b8": feat(24, 24),
        "context_b12": feat(24, 24),
        "geometry": {
            "image_hw": [64, 64],
            "local_hw": [32, 32],
            "context_hw": [48, 48],
            "local_box": [16, 16, 48, 48],
            "context_box": [8, 8, 56, 56],
            "context_to_local": [
                [1.0, 0.0, -8.0],
                [0.0, 1.0, -8.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }


def build_cache(root: Path, signature, n: int = 2):
    with FeatureCacheWriter(
        root,
        producer_signature=signature,
        target_shard_bytes=1024 * 1024,
    ) as writer:
        for i in range(n):
            writer.add(cache_sample(i))


def save_mask(path: Path, *, anomalous: bool = True):
    arr = np.zeros((32, 32), dtype=np.uint8)
    if anomalous:
        arr[10:20, 12:22] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def test_item_returns_six_features_mask_and_meta(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    masks = tmp_path / "dataset"
    build_cache(cache_dir, signature, n=1)

    mask_rel = Path("masks") / "sample_000.png"
    save_mask(masks / mask_rel)

    records = [{
        "image_id": "fabric/sample_000.png",
        "category": "fabric",
        "mask_path": str(mask_rel),
        "is_anomaly": True,
        "mask_hw": [32, 32],
        "meta": {"defect_type": "hole", "split": "train"},
    }]

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=records,
        expected_producer_signature=signature,
        mask_root=masks,
        mask_hw_source="record",
    )

    item = ds[0]

    assert set(FEATURE_KEYS).issubset(item.keys())
    assert "mask" in item
    assert "meta" in item

    # Extraction [1,C,H,W] must become per-sample [C,H,W].
    assert item["local_b4"].shape == (8, 16, 16)
    assert item["context_b12"].shape == (8, 24, 24)

    assert item["mask"].shape == (1, 32, 32)
    assert item["mask"].dtype == torch.float32
    assert set(torch.unique(item["mask"]).tolist()).issubset({0.0, 1.0})

    assert item["meta"]["image_id"] == "fabric/sample_000.png"
    assert item["meta"]["category"] == "fabric"
    assert item["meta"]["defect_type"] == "hole"
    assert "geometry" in item["meta"]

    # Training item deliberately contains no RGB image / extractor.
    assert "image" not in item
    assert "extractor" not in item


def test_normal_without_mask_gets_zero_mask(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    build_cache(cache_dir, signature, n=1)

    records = [{
        "image_id": "fabric/sample_000.png",
        "category": "fabric",
        "mask_path": None,
        "is_anomaly": False,
        "mask_hw": [32, 32],
    }]

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=records,
        expected_producer_signature=signature,
    )

    item = ds[0]
    assert item["mask"].shape == (1, 32, 32)
    assert torch.count_nonzero(item["mask"]).item() == 0


def test_anomaly_without_mask_fails(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    build_cache(cache_dir, signature, n=1)

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=[{
            "image_id": "fabric/sample_000.png",
            "category": "fabric",
            "mask_path": None,
            "is_anomaly": True,
            "mask_hw": [32, 32],
        }],
        expected_producer_signature=signature,
    )

    with pytest.raises(MaskError):
        _ = ds[0]


def test_collate_creates_real_training_batch(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    build_cache(cache_dir, signature, n=2)

    records = [
        {
            "image_id": f"fabric/sample_{i:03d}.png",
            "category": "fabric",
            "mask_path": None,
            "is_anomaly": False,
            "mask_hw": [32, 32],
        }
        for i in range(2)
    ]

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=records,
        expected_producer_signature=signature,
    )

    batch = cached_collate_fn([ds[0], ds[1]])

    assert batch["local_b4"].shape == (2, 8, 16, 16)
    assert batch["context_b4"].shape == (2, 8, 24, 24)
    assert batch["mask"].shape == (2, 1, 32, 32)
    assert len(batch["meta"]) == 2


def test_dataloader_helper_num_workers_zero(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    build_cache(cache_dir, signature, n=2)

    records = [
        {
            "image_id": f"fabric/sample_{i:03d}.png",
            "category": "fabric",
            "mask_path": None,
            "is_anomaly": False,
            "mask_hw": [32, 32],
        }
        for i in range(2)
    ]

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=records,
        expected_producer_signature=signature,
    )

    loader = make_cached_dataloader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))

    assert batch["local_b8"].shape == (2, 8, 16, 16)
    assert batch["mask"].shape == (2, 1, 32, 32)


def test_reader_is_dropped_when_dataset_is_pickled(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    build_cache(cache_dir, signature, n=1)

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=[{
            "image_id": "fabric/sample_000.png",
            "category": "fabric",
            "mask_path": None,
            "is_anomaly": False,
            "mask_hw": [32, 32],
        }],
        expected_producer_signature=signature,
    )

    _ = ds[0]
    assert ds._reader is not None

    restored = pickle.loads(pickle.dumps(ds))
    assert restored._reader is None
    assert restored._reader_pid is None

    # Reader is reconstructed lazily and sample still works.
    item = restored[0]
    assert item["local_b4"].shape == (8, 16, 16)


def test_missing_cache_reference_fails_before_training(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    build_cache(cache_dir, signature, n=1)

    with pytest.raises(TrainingIndexError):
        CachedFeatureDataset(
            cache_dir=cache_dir,
            records=[{
                "image_id": "fabric/not_in_cache.png",
                "category": "fabric",
                "mask_path": None,
                "is_anomaly": False,
                "mask_hw": [32, 32],
            }],
            expected_producer_signature=signature,
        )


def test_mask_shape_mismatch_is_not_silently_resized(tmp_path, signature):
    cache_dir = tmp_path / "cache"
    masks = tmp_path / "dataset"
    build_cache(cache_dir, signature, n=1)

    wrong_mask = masks / "wrong.png"
    wrong_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(wrong_mask)

    ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        records=[{
            "image_id": "fabric/sample_000.png",
            "category": "fabric",
            "mask_path": "wrong.png",
            "is_anomaly": True,
            "mask_hw": [32, 32],
        }],
        expected_producer_signature=signature,
        mask_root=masks,
    )

    with pytest.raises(MaskError):
        _ = ds[0]
