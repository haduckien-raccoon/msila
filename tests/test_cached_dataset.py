from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


FEATURE_KEYS = (
    "local_b4",
    "local_b8",
    "local_b12",
    "context_b4",
    "context_b8",
    "context_b12",
)


def _sample_key(image_id: str, category: str) -> str:
    return f"{category}/{image_id}"


FAKE_STORE: dict[str, dict] = {}
READER_CALLS: list[dict] = []


class FakeFeatureCacheReader:
    def __init__(
        self,
        cache_dir,
        *,
        expected_producer_signature=None,
        mmap=True,
        shard_cache_size=2,
    ):
        READER_CALLS.append(
            {
                "cache_dir": str(cache_dir),
                "expected_producer_signature": expected_producer_signature,
                "mmap": mmap,
                "shard_cache_size": shard_cache_size,
            }
        )

    def __contains__(self, item):
        image_id, category = item
        return _sample_key(image_id, category) in FAKE_STORE

    def get(self, *, image_id: str, category: str):
        return FAKE_STORE[_sample_key(image_id, category)]


class SchemaError(RuntimeError):
    pass


def _load_module():
    fake = types.ModuleType("feature_cache")
    fake.FEATURE_KEYS = FEATURE_KEYS
    fake.FeatureCacheReader = FakeFeatureCacheReader
    fake.SchemaError = SchemaError
    fake.sample_key = _sample_key
    sys.modules["feature_cache"] = fake

    path = Path(__file__).resolve().parents[1] / "data" / "cached_dataset.py"
    spec = importlib.util.spec_from_file_location("cached_dataset_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    FAKE_STORE.clear()
    READER_CALLS.clear()
    return _load_module()


def _cached_sample(i: int, category: str = "fabric") -> dict:
    sample = {
        "image_id": f"img_{i:02d}",
        "category": category,
        "geometry": {
            "local_hw": [4, 4],
            "context_hw": [4, 4],
            "image_hw": [4, 4],
        },
    }
    for j, key in enumerate(FEATURE_KEYS):
        sample[key] = torch.full((1, 2, 2, 2), float(i * 10 + j))
    return sample


def _records(n: int) -> list[dict]:
    out = []
    for i in range(n):
        cached = _cached_sample(i)
        FAKE_STORE[_sample_key(cached["image_id"], cached["category"])] = cached
        out.append(
            {
                "image_id": cached["image_id"],
                "category": cached["category"],
                "mask": torch.zeros(4, 4),
                "mask_hw": [4, 4],
                "is_anomaly": False,
            }
        )
    return out


def _collect_ids(loader) -> list[str]:
    ids = []
    for batch in loader:
        ids.extend(item["image_id"] for item in batch["meta"])
    return ids


def test_dataset_returns_cached_features_without_recomputing(mod):
    records = _records(1)
    ds = mod.CachedFeatureDataset(cache_dir="unused", records=records)

    item = ds[0]

    assert set(FEATURE_KEYS).issubset(item)
    for j, key in enumerate(FEATURE_KEYS):
        # Cached [1,C,H,W] -> per-sample [C,H,W], values unchanged.
        assert item[key].shape == (2, 2, 2)
        assert torch.equal(item[key], torch.full((2, 2, 2), float(j)))


def test_expected_producer_signature_is_forwarded(mod):
    records = _records(1)
    signature = {
        "backbone": "dinov3_vits16",
        "blocks": [4, 8, 12],
        "local_context": "locked-day03",
    }

    ds = mod.CachedFeatureDataset(
        cache_dir="unused",
        records=records,
        expected_producer_signature=signature,
    )
    _ = ds[0]  # opens lazy reader too

    assert len(READER_CALLS) >= 2
    assert all(
        call["expected_producer_signature"] == signature
        for call in READER_CALLS
    )


def test_same_seed_gives_same_shuffle_order(mod):
    records = _records(12)

    ds1 = mod.CachedFeatureDataset(cache_dir="unused", records=records)
    ds2 = mod.CachedFeatureDataset(cache_dir="unused", records=records)

    loader1 = mod.make_cached_dataloader(
        ds1,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        seed=42,
    )
    loader2 = mod.make_cached_dataloader(
        ds2,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        seed=42,
    )

    assert _collect_ids(loader1) == _collect_ids(loader2)


def test_seed_and_generator_are_mutually_exclusive(mod):
    records = _records(1)
    ds = mod.CachedFeatureDataset(cache_dir="unused", records=records)

    with pytest.raises(ValueError, match="either seed or generator"):
        mod.make_cached_dataloader(
            ds,
            batch_size=1,
            seed=42,
            generator=torch.Generator(),
        )


def test_mask_shape_mismatch_fails_fast(mod):
    records = _records(1)
    records[0]["mask"] = torch.zeros(3, 4)
    records[0]["mask_hw"] = [4, 4]

    ds = mod.CachedFeatureDataset(cache_dir="unused", records=records)

    with pytest.raises(mod.MaskError, match="mask shape mismatch"):
        _ = ds[0]


def test_default_does_not_cast_cached_feature_dtype(mod):
    records = _records(1)
    key = _sample_key("img_00", "fabric")
    for feature_key in FEATURE_KEYS:
        FAKE_STORE[key][feature_key] = FAKE_STORE[key][feature_key].to(torch.float16)

    ds = mod.CachedFeatureDataset(cache_dir="unused", records=records)
    item = ds[0]

    assert all(item[k].dtype == torch.float16 for k in FEATURE_KEYS)


def test_source_contains_no_dino_import_or_extractor_call():
    path = Path(__file__).resolve().parents[1] / "data" / "cached_dataset.py"
    source = path.read_text(encoding="utf-8").lower()

    forbidden = (
        "import dinov3",
        "from dinov3",
        "torch.hub.load(",
        "load_model(",
    )
    assert not any(token in source for token in forbidden)
