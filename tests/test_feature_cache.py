from __future__ import annotations

"""
tests/test_feature_cache.py
===========================

TV1 acceptance tests for the MS-ILA feature cache.

Two different things are tested:

1) Unit tests:
   - fixed schema/version
   - exact write -> read round-trip
   - sharding
   - duplicate rejection
   - producer-signature mismatch rejection
   - malformed/NaN feature rejection

2) Real integration acceptance test:
   PREPROCESSED REAL SAMPLE
        ├── fresh online frozen DINOv3 -> F_online
        └── cache built previously by the real cache-builder -> F_cache

   PASS iff, for all six sources:
       max_abs_error = max |F_online - F_cache| < 1e-5

Important
---------
The real integration test MUST NOT create its cache from the same `online`
tensor inside the test. That would only test serialization. Instead, it reads
an already-built cache from FEATURE_CACHE_DIR.

Required environment variables for the real integration test:
    DINOV3_REPO
    DINOV3_WEIGHTS
    REAL_SAMPLE_PT
    FEATURE_CACHE_DIR

Optional:
    DINOV3_MODEL=dinov3_vits16
    DINOV3_DEVICE=cuda

REAL_SAMPLE_PT must be a torch-saved dict with:
    {
        "image_id": str,
        "category": str,
        "x_local": Tensor[1,3,512,512],
        "x_context": Tensor[1,3,512,512],
        "geometry": {
            "local_box": ...,
            "context_box": ...,
            "context_to_local": ...,
            ...
        }
    }

`x_local` and `x_context` must be the exact deterministic tensors produced by
the same preprocessing contract used when the cache was built.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_cache import (  # noqa: E402
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
from src.models.dinov3_extractor import build_online_extractor  # noqa: E402


TOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures: unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def signature() -> dict[str, Any]:
    """Small deterministic producer signature used only by unit tests."""
    return {
        "backbone": "dinov3_vits16",
        "checkpoint_sha256": "unit-test-checkpoint",
        "logical_layers_1based": [4, 8, 12],
        "internal_indices_0based": [3, 7, 11],
        "preprocess_version": "msila_local_context_v1",
    }


def make_sample(i: int = 0) -> dict[str, Any]:
    """Synthetic feature record for cache-format unit tests only."""
    g = torch.Generator().manual_seed(1000 + i)

    def feat(h: int, w: int) -> torch.Tensor:
        return torch.randn(
            (1, 8, h, w),
            generator=g,
            dtype=torch.float32,
        )

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


# ---------------------------------------------------------------------------
# Fixtures: REAL online-vs-cache integration test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def online_extractor():
    """
    Real frozen DINOv3 extractor.

    The fixture is skipped unless the real repo/checkpoint are explicitly
    provided. This prevents CI from silently downloading or changing DINOv3.
    """
    repo = os.getenv("DINOV3_REPO")
    weights = os.getenv("DINOV3_WEIGHTS")

    if not repo or not weights:
        pytest.skip(
            "Real integration test requires DINOV3_REPO and DINOV3_WEIGHTS."
        )

    device = os.getenv(
        "DINOV3_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    extractor = build_online_extractor(
        repo_dir=repo,
        weights=weights,
        device=device,
        model_name=os.getenv("DINOV3_MODEL", "dinov3_vits16"),
        blocks=(4, 8, 12),
        norm=True,
        check_finite=True,
    )

    assert extractor.backbone_is_frozen()
    assert extractor.backbone.training is False

    return extractor


@pytest.fixture(scope="session")
def real_sample() -> dict[str, Any]:
    """
    Load ONE real, deterministic, already-preprocessed Local/Context sample.

    We intentionally do not invent/project-guess a preprocessing API here.
    REAL_SAMPLE_PT must be exported by the actual project preprocessing pipeline.
    """
    sample_path = os.getenv("REAL_SAMPLE_PT")

    if not sample_path:
        pytest.skip(
            "Real integration test requires REAL_SAMPLE_PT pointing to a "
            "preprocessed real sample."
        )

    path = Path(sample_path).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"REAL_SAMPLE_PT does not exist: {path}")

    # weights_only=True is sufficient because the contract contains only
    # tensors + primitive containers.
    sample = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(sample, Mapping):
        pytest.fail(
            f"REAL_SAMPLE_PT must contain a dict-like object, got {type(sample)!r}"
        )

    required = {
        "image_id",
        "category",
        "x_local",
        "x_context",
        "geometry",
    }
    missing = sorted(required.difference(sample.keys()))
    if missing:
        pytest.fail(f"REAL_SAMPLE_PT missing required fields: {missing}")

    x_local = sample["x_local"]
    x_context = sample["x_context"]

    for name, x in (
        ("x_local", x_local),
        ("x_context", x_context),
    ):
        assert isinstance(x, torch.Tensor), f"{name} must be torch.Tensor"
        assert x.ndim == 4, f"{name} must be [B,3,H,W], got {tuple(x.shape)}"
        assert x.shape[0] == 1, (
            f"{name}: acceptance sample must contain exactly one image; "
            f"got batch={x.shape[0]}"
        )
        assert x.shape[1] == 3, f"{name}: expected RGB C=3"
        assert tuple(x.shape[-2:]) == (512, 512), (
            f"{name}: expected project network input 512x512, "
            f"got {tuple(x.shape[-2:])}"
        )
        assert torch.is_floating_point(x), f"{name} must be floating point"
        assert torch.isfinite(x).all(), f"{name} contains NaN/Inf"

    assert x_local.shape == x_context.shape
    assert str(sample["image_id"])
    assert str(sample["category"])
    assert isinstance(sample["geometry"], Mapping)

    return dict(sample)


@pytest.fixture(scope="session")
def built_cache_dir() -> Path:
    """
    Cache produced BEFORE this test by the project's real feature-cache builder.

    This separation is essential:
        online extraction != cache-building branch.
    """
    value = os.getenv("FEATURE_CACHE_DIR")

    if not value:
        pytest.skip(
            "Real integration test requires FEATURE_CACHE_DIR pointing to the "
            "cache generated by the real feature-cache builder."
        )

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        pytest.fail(f"FEATURE_CACHE_DIR is not a directory: {path}")

    manifest = path / "manifest.json"
    if not manifest.is_file():
        pytest.fail(f"Feature-cache manifest not found: {manifest}")

    return path


# ---------------------------------------------------------------------------
# Numerical acceptance helper
# ---------------------------------------------------------------------------

def assert_online_cache_error_below_tolerance(
    online: Mapping[str, torch.Tensor],
    cached: Mapping[str, torch.Tensor],
    tolerance: float = TOL,
) -> dict[str, float]:
    """
    Hard numerical gate for all six features.

    For each feature F:
        e_max(F) = max_j |F_online[j] - F_cache[j]|

    PASS iff:
        e_max(F) < tolerance
    for every F in FEATURE_KEYS.

    Returns
    -------
    dict[str, float]
        Per-feature max absolute errors, useful for test logs/debugging.
    """
    assert tolerance > 0.0

    assert set(FEATURE_KEYS).issubset(online.keys())
    assert set(FEATURE_KEYS).issubset(cached.keys())

    errors: dict[str, float] = {}

    for key in FEATURE_KEYS:
        f_online = online[key].detach().cpu().float()
        f_cache = cached[key].detach().cpu().float()

        assert f_online.shape == f_cache.shape, (
            f"{key}: shape mismatch: "
            f"online={tuple(f_online.shape)}, "
            f"cache={tuple(f_cache.shape)}"
        )

        assert torch.isfinite(f_online).all(), (
            f"{key}: online feature contains NaN/Inf"
        )
        assert torch.isfinite(f_cache).all(), (
            f"{key}: cached feature contains NaN/Inf"
        )

        max_abs_error = (
            f_online - f_cache
        ).abs().max().item()

        errors[key] = max_abs_error

        assert max_abs_error < tolerance, (
            f"{key}: max_abs_error={max_abs_error:.8e} "
            f">= tolerance={tolerance:.1e}"
        )

    return errors


# ---------------------------------------------------------------------------
# Unit tests: cache format/storage
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_schema_and_all_six_features(
    tmp_path: Path,
    signature: dict[str, Any],
):
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
    assert key == sample_key(
        sample["image_id"],
        sample["category"],
    )
    assert set(FEATURE_KEYS).issubset(loaded.keys())

    for feature_key in FEATURE_KEYS:
        assert loaded[feature_key].shape == sample[feature_key].shape
        assert loaded[feature_key].dtype == sample[feature_key].dtype
        assert torch.equal(
            loaded[feature_key],
            sample[feature_key],
        )

    assert (
        loaded["geometry"]
        == normalize_record(sample)["geometry"]
    )

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_name"] == SCHEMA_NAME
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["num_samples"] == 1
    assert manifest["num_shards"] == 1


def test_writer_splits_into_multiple_shards(
    tmp_path: Path,
    signature: dict[str, Any],
):
    sample0 = normalize_record(make_sample(0))
    one_sample_bytes = estimate_record_bytes(sample0)

    # Below the size of two records -> subsequent record forces a flush.
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

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["num_samples"] == 3
    assert manifest["num_shards"] == 3

    stats = validate_cache(tmp_path)

    assert stats == {
        "num_samples": 3,
        "num_shards": 3,
        "schema_version": SCHEMA_VERSION,
    }


def test_duplicate_sample_is_rejected(
    tmp_path: Path,
    signature: dict[str, Any],
):
    sample = make_sample(0)

    writer = FeatureCacheWriter(
        tmp_path,
        producer_signature=signature,
        target_shard_bytes=1024 * 1024,
    )

    writer.add(sample)

    with pytest.raises(CacheKeyError):
        writer.add(sample)

    writer.close()


def test_producer_signature_mismatch_is_rejected(
    tmp_path: Path,
    signature: dict[str, Any],
):
    with FeatureCacheWriter(
        tmp_path,
        producer_signature=signature,
    ) as writer:
        writer.add(make_sample(0))

    wrong = dict(signature)
    wrong["preprocess_version"] = "changed-v2"

    with pytest.raises(ProducerMismatchError):
        FeatureCacheReader(
            tmp_path,
            expected_producer_signature=wrong,
        )


def test_bad_schema_missing_feature_is_rejected():
    sample = make_sample(0)
    del sample["context_b12"]

    with pytest.raises(SchemaError):
        normalize_record(sample)


def test_nan_feature_is_rejected():
    sample = make_sample(0)
    sample["local_b8"][0, 0, 0, 0] = float("nan")

    with pytest.raises(SchemaError):
        normalize_record(sample)


def test_numerical_comparator_rejects_error_at_or_above_tolerance():
    """
    Comparator itself must fail when max absolute error reaches tolerance.

    This protects the acceptance rule from accidentally becoming <= tolerance
    or from silently relying on torch.allclose's relative tolerance.
    """
    online = {
        key: torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        for key in FEATURE_KEYS
    }
    cached = {
        key: value.clone()
        for key, value in online.items()
    }

    cached["local_b8"][0, 0, 0, 0] = TOL

    with pytest.raises(AssertionError, match="local_b8"):
        assert_online_cache_error_below_tolerance(
            online,
            cached,
            tolerance=TOL,
        )


# ---------------------------------------------------------------------------
# REAL integration acceptance test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_built_cache_matches_fresh_online_extraction(
    online_extractor,
    real_sample: dict[str, Any],
    built_cache_dir: Path,
):
    """
    Final TV1 acceptance gate.

    IMPORTANT:
      - `F_online` is generated NOW by the real frozen DINOv3 extractor.
      - `F_cache` is loaded from FEATURE_CACHE_DIR, which must have been produced
        previously by the real cache-builder.
      - The test does NOT write `F_online` into that cache.

    Therefore this checks the actual independent paths:

        preprocessing -> online DINOv3
        preprocessing -> cache builder -> cache -> cache reader
    """

    image_id = str(real_sample["image_id"])
    category = str(real_sample["category"])

    # --------------------------------------------------
    # 1. Load actual builder-produced cache
    # --------------------------------------------------
    reader = FeatureCacheReader(
        built_cache_dir,
        mmap=True,
    )

    assert (image_id, category) in reader, (
        f"Real sample {category}/{image_id} is not present in FEATURE_CACHE_DIR. "
        "Build the cache for this exact sample first."
    )

    cached = reader.get(
        image_id=image_id,
        category=category,
    )

    # --------------------------------------------------
    # 2. Fresh ONLINE DINOv3 extraction
    # --------------------------------------------------
    device = next(
        online_extractor.parameters()
    ).device

    x_local = real_sample["x_local"].to(
        device=device,
        non_blocking=True,
    )
    x_context = real_sample["x_context"].to(
        device=device,
        non_blocking=True,
    )

    online_extractor.eval()

    with torch.inference_mode():
        online = online_extractor.extract_online_cache_features(
            x_local,
            x_context,
            strategy="concat",
            to_cpu=False,
        )

    assert set(online.keys()) == set(FEATURE_KEYS)

    # --------------------------------------------------
    # 3. Geometry identity check
    # --------------------------------------------------
    expected_geometry = normalize_record(
        {
            "image_id": image_id,
            "category": category,
            **{
                key: cached[key]
                for key in FEATURE_KEYS
            },
            "geometry": real_sample["geometry"],
        }
    )["geometry"]

    assert cached["geometry"] == expected_geometry, (
        "Cached geometry does not match REAL_SAMPLE_PT preprocessing metadata."
    )

    # --------------------------------------------------
    # 4. HARD numerical gate
    # --------------------------------------------------
    errors = assert_online_cache_error_below_tolerance(
        online,
        cached,
        tolerance=TOL,
    )

    # Helpful output with pytest -s / verbose logs.
    print("\n[cache-vs-online max_abs_error]")
    for key in FEATURE_KEYS:
        print(f"  {key:12s}: {errors[key]:.8e}")
