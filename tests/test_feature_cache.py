from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Support both normal repo layout and this exported artifact layout.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_cache import (
    FEATURE_KEYS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    CacheKeyError,
    FeatureCacheReader,
    FeatureCacheWriter,
    ProducerMismatchError,
    SchemaError,
    estimate_record_bytes,
    normalize_record,
    sample_key,
    validate_cache,
)


@pytest.fixture
def signature():
    return {
        "backbone": "dinov3_vits16",
        "checkpoint_sha256": "unit-test-checkpoint",
        "logical_layers_1based": [4, 8, 12],
        "internal_indices_0based": [3, 7, 11],
        "preprocess_version": "msila_local_context_v1",
    }


def make_sample(i: int = 0):
    g = torch.Generator().manual_seed(1000 + i)

    def feat(h, w):
        return torch.randn((1, 8, h, w), generator=g, dtype=torch.float32)

    return {
        "image_id": f"fabric/train/good/{i:03d}.png",
        "category": "fabric",
        "local_b4": feat(16, 16),
        "local_b8": feat(16, 16),
        "local_b12": feat(16, 16),
        "context_b4": feat(24, 24),
        "context_b8": feat(24, 24),
        "context_b12": feat(24, 24),
        "geometry": {
            "image_hw": [1024, 1024],
            "local_hw": [512, 512],
            "context_hw": [768, 768],
            "local_box": [100, 100, 612, 612],
            "context_box": [40, 40, 808, 808],
            "context_to_local": [
                [1.0, 0.0, -40.0],
                [0.0, 1.0, -40.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }


def test_roundtrip_preserves_schema_and_all_six_features(tmp_path, signature):
    sample = make_sample(0)

    with FeatureCacheWriter(
        tmp_path,
        producer_signature=signature,
        target_shard_bytes=1024 * 1024,
    ) as writer:
        key = writer.add(sample)

    reader = FeatureCacheReader(
        tmp_path,
        expected_producer_signature=signature,
        mmap=True,
    )
    loaded = reader.get(
        image_id=sample["image_id"],
        category=sample["category"],
    )

    assert len(reader) == 1
    assert key == sample_key(sample["image_id"], sample["category"])
    assert set(FEATURE_KEYS).issubset(loaded.keys())

    for feature_key in FEATURE_KEYS:
        assert loaded[feature_key].shape == sample[feature_key].shape
        assert loaded[feature_key].dtype == sample[feature_key].dtype
        assert torch.equal(loaded[feature_key], sample[feature_key])

    assert loaded["geometry"] == normalize_record(sample)["geometry"]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_name"] == SCHEMA_NAME
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["num_samples"] == 1
    assert manifest["num_shards"] == 1


def test_writer_splits_into_multiple_shards(tmp_path, signature):
    sample0 = normalize_record(make_sample(0))
    one_sample_bytes = estimate_record_bytes(sample0)

    # Threshold below 2 records => every subsequent record forces flush.
    threshold = one_sample_bytes + 1

    with FeatureCacheWriter(
        tmp_path,
        producer_signature=signature,
        target_shard_bytes=threshold,
        max_samples_per_shard=100,
    ) as writer:
        writer.add(make_sample(0))
        writer.add(make_sample(1))
        writer.add(make_sample(2))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_samples"] == 3
    assert manifest["num_shards"] == 3

    stats = validate_cache(tmp_path)
    assert stats == {
        "num_samples": 3,
        "num_shards": 3,
        "schema_version": SCHEMA_VERSION,
    }


def test_duplicate_sample_is_rejected(tmp_path, signature):
    s = make_sample(0)

    writer = FeatureCacheWriter(
        tmp_path,
        producer_signature=signature,
        target_shard_bytes=1024 * 1024,
    )
    writer.add(s)
    with pytest.raises(CacheKeyError):
        writer.add(s)
    writer.close()


def test_producer_signature_mismatch_is_rejected(tmp_path, signature):
    with FeatureCacheWriter(tmp_path, producer_signature=signature) as writer:
        writer.add(make_sample(0))

    wrong = dict(signature)
    wrong["preprocess_version"] = "changed-v2"

    with pytest.raises(ProducerMismatchError):
        FeatureCacheReader(
            tmp_path,
            expected_producer_signature=wrong,
        )


def test_bad_schema_missing_feature_is_rejected():
    s = make_sample(0)
    del s["context_b12"]

    with pytest.raises(SchemaError):
        normalize_record(s)


def test_nan_feature_is_rejected():
    s = make_sample(0)
    s["local_b8"][0, 0, 0, 0] = float("nan")

    with pytest.raises(SchemaError):
        normalize_record(s)
