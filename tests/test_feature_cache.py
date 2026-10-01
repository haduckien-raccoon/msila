from __future__ import annotations

import json
import sys
from pathlib import Path
import os

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
from src.models.dinov3_extractor import build_online_extractor

@pytest.fixture(scope="session")
def online_extractor():
    repo = os.getenv("DINOV3_REPO")
    weights = os.getenv("DINOV3_WEIGHTS")

    if not repo or not weights:
        pytest.skip(
            "Set DINOV3_REPO and DINOV3_WEIGHTS "
            "to run cache-vs-online integration test."
        )

    extractor = build_online_extractor(
        repo_dir=repo,
        weights=weights,
        model_name=os.getenv(
            "DINOV3_MODEL",
            "dinov3_vits16",
        ),
        blocks=(4, 8, 12),
        norm=True,
        check_finite=True,
    )

    return extractor

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

TOL = 1e-5


def assert_online_cache_error_below_tolerance(
    online: dict,
    cached: dict,
    tolerance: float = TOL,
):
    """
    Hard gate:
        max |F_online - F_cache| < tolerance

    Kiểm tra toàn bộ 6 feature:
        L4, L8, L12, C4, C8, C12
    """
    for key in FEATURE_KEYS:
        f_online = online[key].detach().cpu().float()
        f_cache = cached[key].detach().cpu().float()

        # 1. Shape phải giống tuyệt đối
        assert f_online.shape == f_cache.shape, (
            f"{key}: shape mismatch: "
            f"online={tuple(f_online.shape)}, "
            f"cache={tuple(f_cache.shape)}"
        )

        # 2. Không được có NaN / Inf
        assert torch.isfinite(f_online).all(), \
            f"{key}: online feature contains NaN/Inf"

        assert torch.isfinite(f_cache).all(), \
            f"{key}: cached feature contains NaN/Inf"

        # 3. Numerical error
        max_abs_error = (
            f_online - f_cache
        ).abs().max().item()

        assert max_abs_error < tolerance, (
            f"{key}: max_abs_error={max_abs_error:.8e} "
            f">= tolerance={tolerance:.1e}"
        )


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

def test_cache_matches_online_extraction(
    tmp_path,
    signature,
    online_extractor,
    real_sample,
):
    # ==================================================
    # 1. ONLINE DINO EXTRACTION
    # ==================================================
    online_extractor.eval()

    with torch.inference_mode():
        online = online_extractor.extract_online_cache_features(
            real_sample["x_local"],
            real_sample["x_context"],
        )

    assert set(online.keys()) == set(FEATURE_KEYS)

    # ==================================================
    # 2. WRITE TO CACHE
    # ==================================================
    sample_to_cache = {
        "image_id": real_sample["image_id"],
        "category": real_sample["category"],

        **{
            key: online[key]
            for key in FEATURE_KEYS
        },

        # geometry comes from preprocessing, NOT DINO
        "geometry": real_sample["geometry"],
    }

    with FeatureCacheWriter(
        tmp_path,
        producer_signature=signature,
    ) as writer:
        writer.add(sample_to_cache)

    # ==================================================
    # 3. READ FROM CACHE
    # ==================================================
    reader = FeatureCacheReader(
        tmp_path,
        expected_producer_signature=signature,
        mmap=True,
    )

    cached = reader.get(
        image_id=real_sample["image_id"],
        category=real_sample["category"],
    )

    # ==================================================
    # 4. NUMERICAL ACCEPTANCE GATE
    # ==================================================
    assert_online_cache_error_below_tolerance(
        online,
        cached,
        tolerance=1e-5,
    )