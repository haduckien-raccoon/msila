"""
data/feature_cache.py
=====================

Stable, versioned, sharded feature-cache format for MS-ILA.

Default storage:
    cache_root/
      manifest.json
      shards/
        shard-000000.pt
        shard-000001.pt
        ...

Each sample stores exactly six frozen-backbone feature sources:
    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

plus:
    image_id, category, geometry

Design goals
------------
1. Fixed schema/version.
2. PyTorch-native .pt shards.
3. Safe loading: torch.load(..., weights_only=True).
4. Lazy tensor storage loading when supported: mmap=True.
5. Atomic shard + manifest writes.
6. Per-sample random lookup through manifest index.
7. Cache provenance through producer_signature.
8. Strict validation of feature keys, dtype/finite values, geometry.
9. No nn.Module/custom Python object is serialized.

This module is a STORAGE FORMAT. It does not run DINOv3 and does not train
Adapter/Fusion/Decoder.

Tested contract target: PyTorch >= 2.6; optimized for current PyTorch 2.x
serialization semantics available by Sep 2026.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch


SCHEMA_NAME = "msila.feature_cache"
SCHEMA_VERSION = 1

FEATURE_KEYS: tuple[str, ...] = (
    "local_b4",
    "local_b8",
    "local_b12",
    "context_b4",
    "context_b8",
    "context_b12",
)

REQUIRED_GEOMETRY_KEYS: tuple[str, ...] = (
    "local_box",
    "context_box",
    "context_to_local",
)

MANIFEST_FILENAME = "manifest.json"
SHARDS_DIRNAME = "shards"


class FeatureCacheError(RuntimeError):
    """Base class for cache-format errors."""


class SchemaError(FeatureCacheError):
    """Raised when data does not satisfy the fixed schema."""


class ProducerMismatchError(FeatureCacheError):
    """Raised when a cache was produced by another extractor/preprocess config."""


class CacheKeyError(FeatureCacheError):
    """Raised when a sample key is duplicated or cannot be found."""


@dataclass(frozen=True)
class SampleRef:
    """Manifest pointer to one sample inside one .pt shard."""

    shard: str
    record_key: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    # NumPy-like values without a hard NumPy dependency.
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise TypeError(f"Not JSON-serializable: {type(value)!r}")


def producer_hash(signature: Mapping[str, Any]) -> str:
    """
    Stable hash for the frozen feature-producing pipeline.

    Include at least:
      - backbone/checkpoint identity
      - selected feature blocks
      - preprocessing version
      - local/context resize/crop policy
      - normalization policy
    """
    return hashlib.sha256(_canonical_json(signature).encode("utf-8")).hexdigest()


def sample_key(image_id: str, category: str) -> str:
    """Collision-resistant stable key; category is part of sample identity."""
    image_id = str(image_id)
    category = str(category)
    if not image_id:
        raise SchemaError("image_id must be non-empty")
    if not category:
        raise SchemaError("category must be non-empty")
    return hashlib.sha256(f"{category}\0{image_id}".encode("utf-8")).hexdigest()


def _validate_box(box: Sequence[Any], field: str) -> list[float]:
    if len(box) != 4:
        raise SchemaError(f"{field} must be [x0,y0,x1,y1]")
    values = [float(v) for v in box]
    if not all(math.isfinite(v) for v in values):
        raise SchemaError(f"{field} contains NaN/Inf")
    x0, y0, x1, y1 = values
    if not (x1 > x0 and y1 > y0):
        raise SchemaError(f"{field} requires x1>x0 and y1>y0")
    return values


def _validate_homography(matrix: Any) -> list[list[float]]:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().tolist()
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise SchemaError("geometry.context_to_local must be 3x3")
    out = [[float(v) for v in row] for row in matrix]
    if not all(math.isfinite(v) for row in out for v in row):
        raise SchemaError("geometry.context_to_local contains NaN/Inf")
    return out


def validate_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate the minimum geometry contract and retain JSON-compatible extras.

    Required:
      local_box:        [x0,y0,x1,y1]
      context_box:      [x0,y0,x1,y1]
      context_to_local: 3x3 homogeneous transform
    """
    missing = [k for k in REQUIRED_GEOMETRY_KEYS if k not in geometry]
    if missing:
        raise SchemaError(f"geometry missing keys: {missing}")

    out = _to_jsonable(dict(geometry))
    out["local_box"] = _validate_box(out["local_box"], "geometry.local_box")
    out["context_box"] = _validate_box(out["context_box"], "geometry.context_box")
    out["context_to_local"] = _validate_homography(out["context_to_local"])

    for key in ("local_hw", "context_hw", "image_hw"):
        if key in out:
            hw = out[key]
            if len(hw) != 2:
                raise SchemaError(f"geometry.{key} must be [H,W]")
            h, w = int(hw[0]), int(hw[1])
            if h <= 0 or w <= 0:
                raise SchemaError(f"geometry.{key} requires H,W>0")
            out[key] = [h, w]

    return out


def _validate_feature_tensor(t: Any, key: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise SchemaError(f"{key} must be torch.Tensor")
    if not torch.is_floating_point(t):
        raise SchemaError(f"{key} must be floating-point, got {t.dtype}")
    if t.numel() == 0:
        raise SchemaError(f"{key} is empty")
    if not torch.isfinite(t).all().item():
        raise SchemaError(f"{key} contains NaN/Inf")
    # Cache owns independent contiguous CPU storage.
    return t.detach().to(device="cpu").contiguous()


def normalize_record(sample: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert one sample to the exact serialization schema.

    Extra top-level keys are intentionally NOT persisted. This keeps v1 fixed.
    Add fields only through a future schema migration.
    """
    required = {"image_id", "category", "geometry", *FEATURE_KEYS}
    missing = sorted(required.difference(sample.keys()))
    if missing:
        raise SchemaError(f"sample missing fields: {missing}")

    image_id = str(sample["image_id"])
    category = str(sample["category"])
    if not image_id or not category:
        raise SchemaError("image_id/category must be non-empty")

    record = {
        "image_id": image_id,
        "category": category,
        "geometry": validate_geometry(sample["geometry"]),
    }
    for key in FEATURE_KEYS:
        record[key] = _validate_feature_tensor(sample[key], key)

    return record


def estimate_record_bytes(record: Mapping[str, Any]) -> int:
    """
    Approximate tensor payload bytes:
        B = sum_i numel(F_i) * element_size(F_i)

    This intentionally ignores the small Python/ZIP/JSON overhead.
    """
    return int(
        sum(record[k].numel() * record[k].element_size() for k in FEATURE_KEYS)
    )


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        # Ensure bytes reached OS before rename.
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_torch_load(path: Path, *, mmap: bool) -> dict[str, Any]:
    """
    Load only primitive containers + tensors.
    `weights_only=True` is explicit even though modern PyTorch defaults to it.
    """
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=mmap,
        )
    except TypeError:
        # Compatibility for a PyTorch version whose torch.load has no mmap kwarg.
        # Still require weights_only support for this format.
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(obj, dict):
        raise SchemaError(f"Shard {path} is not a dict payload")
    return obj


def _validate_shard_header(
    shard: Mapping[str, Any],
    *,
    expected_producer_sha256: str | None,
) -> None:
    if shard.get("schema_name") != SCHEMA_NAME:
        raise SchemaError(
            f"schema_name={shard.get('schema_name')!r}, expected {SCHEMA_NAME!r}"
        )
    if shard.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version={shard.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    if expected_producer_sha256 is not None:
        got = shard.get("producer_sha256")
        if got != expected_producer_sha256:
            raise ProducerMismatchError(
                f"producer signature mismatch: expected={expected_producer_sha256}, got={got}"
            )
    if not isinstance(shard.get("records"), dict):
        raise SchemaError("shard.records must be a dict")


def _new_manifest(signature: Mapping[str, Any]) -> dict[str, Any]:
    signature = _to_jsonable(dict(signature))
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "storage": {
            "backend": "torch_pt_shards",
            "extension": ".pt",
        },
        "producer_signature": signature,
        "producer_sha256": producer_hash(signature),
        "feature_keys": list(FEATURE_KEYS),
        "num_samples": 0,
        "num_shards": 0,
        "index": {},
    }
    if signature.get('schema') == 'msila.full_scale.cache.v2':
        manifest['shard_sha256'] = {}
    return manifest


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


class FeatureCacheWriter:
    """
    Append-only writer for versioned .pt shards.

    Parameters
    ----------
    root:
        Cache directory.
    producer_signature:
        Provenance of frozen extractor + deterministic preprocessing.
    target_shard_bytes:
        Flush when approximate tensor bytes reach this threshold.
        512 MiB is a practical default for local/NVMe training: many fewer files
        than per-sample cache while avoiding a monolithic cache.
    max_samples_per_shard:
        Secondary bound. Useful when samples are very small.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        producer_signature: Mapping[str, Any],
        target_shard_bytes: int = 512 * 1024 * 1024,
        max_samples_per_shard: int = 1024,
    ):
        if target_shard_bytes <= 0:
            raise ValueError("target_shard_bytes must be > 0")
        if max_samples_per_shard <= 0:
            raise ValueError("max_samples_per_shard must be > 0")

        self.root = Path(root)
        self.shards_dir = self.root / SHARDS_DIRNAME
        self.manifest_path = self.root / MANIFEST_FILENAME
        self.signature = _to_jsonable(dict(producer_signature))
        self.producer_sha256 = producer_hash(self.signature)
        self.target_shard_bytes = int(target_shard_bytes)
        self.max_samples_per_shard = int(max_samples_per_shard)

        self.root.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self._validate_manifest(manifest)
            if manifest["producer_sha256"] != self.producer_sha256:
                raise ProducerMismatchError(
                    "Existing cache belongs to a different producer signature. "
                    "Use a new cache directory or rebuild it."
                )
            self.manifest = manifest
        else:
            self.manifest = _new_manifest(self.signature)

        self._pending: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._pending_bytes = 0
        self._closed = False

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_name") != SCHEMA_NAME:
            raise SchemaError("Manifest schema_name mismatch")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(
                f"Manifest version {manifest.get('schema_version')} is unsupported"
            )
        if manifest.get("feature_keys") != list(FEATURE_KEYS):
            raise SchemaError("Manifest feature_keys do not match fixed v1 schema")
        if not isinstance(manifest.get("index"), dict):
            raise SchemaError("Manifest index must be a dict")

    def add(self, sample: Mapping[str, Any]) -> str:
        if self._closed:
            raise FeatureCacheError("Writer is closed")

        record = normalize_record(sample)
        key = sample_key(record["image_id"], record["category"])

        if key in self.manifest["index"] or key in self._pending:
            raise CacheKeyError(
                f"Duplicate cache sample: {record['category']} / {record['image_id']}"
            )

        record_bytes = estimate_record_bytes(record)

        # If current shard already has data and adding this record would exceed
        # either limit, flush first. A single oversized sample is still allowed.
        would_exceed_bytes = (
            bool(self._pending)
            and self._pending_bytes + record_bytes > self.target_shard_bytes
        )
        would_exceed_count = len(self._pending) >= self.max_samples_per_shard
        if would_exceed_bytes or would_exceed_count:
            self.flush()

        self._pending[key] = record
        self._pending_bytes += record_bytes
        return key

    def flush(self) -> Path | None:
        if not self._pending:
            return None

        shard_id = int(self.manifest["num_shards"])
        shard_name = f"shard-{shard_id:06d}.pt"
        shard_rel = f"{SHARDS_DIRNAME}/{shard_name}"
        shard_path = self.root / shard_rel

        payload = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "producer_sha256": self.producer_sha256,
            "records": dict(self._pending),
        }

        # 1) Commit shard.
        _atomic_torch_save(shard_path, payload)
        if 'shard_sha256' in self.manifest:
            self.manifest['shard_sha256'][shard_rel] = _file_sha256(shard_path)

        # 2) Update manifest only after shard exists.
        for key in self._pending:
            self.manifest["index"][key] = {
                "shard": shard_rel,
                "record_key": key,
            }

        self.manifest["num_samples"] = int(self.manifest["num_samples"]) + len(self._pending)
        self.manifest["num_shards"] = shard_id + 1
        _atomic_write_json(self.manifest_path, self.manifest)

        self._pending.clear()
        self._pending_bytes = 0
        return shard_path

    def close(self) -> None:
        if not self._closed:
            self.flush()
            # Ensure even an empty cache has a manifest.
            if not self.manifest_path.exists():
                _atomic_write_json(self.manifest_path, self.manifest)
            self._closed = True

    def __enter__(self) -> "FeatureCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Do not commit pending records if user's block raised.
        if exc_type is None:
            self.close()
        else:
            self._pending.clear()
            self._pending_bytes = 0
            self._closed = True


class FeatureCacheReader:
    """
    Random-access reader backed by manifest.json.

    A tiny LRU caches loaded shard dictionaries. With mmap=True, PyTorch can
    lazily map tensor storages instead of eagerly copying every storage.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_producer_signature: Mapping[str, Any] | None = None,
        mmap: bool = True,
        shard_cache_size: int = 2,
    ):
        if shard_cache_size < 0:
            raise ValueError("shard_cache_size must be >= 0")

        self.root = Path(root)
        self.manifest_path = self.root / MANIFEST_FILENAME
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        if self.manifest.get("schema_name") != SCHEMA_NAME:
            raise SchemaError("Manifest schema_name mismatch")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(
                f"Unsupported schema_version={self.manifest.get('schema_version')}"
            )
        if self.manifest.get("feature_keys") != list(FEATURE_KEYS):
            raise SchemaError("Manifest feature_keys mismatch")

        self.producer_sha256 = self.manifest["producer_sha256"]
        self._verified_shards = {}
        if expected_producer_signature is not None:
            expected_hash = producer_hash(expected_producer_signature)
            if expected_hash != self.producer_sha256:
                raise ProducerMismatchError(
                    f"producer signature mismatch: expected={expected_hash}, "
                    f"got={self.producer_sha256}"
                )

        self.mmap = bool(mmap)
        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def __len__(self) -> int:
        return int(self.manifest["num_samples"])

    def __contains__(self, item: tuple[str, str]) -> bool:
        image_id, category = item
        return sample_key(image_id, category) in self.manifest["index"]

    def keys(self) -> Iterator[str]:
        yield from self.manifest["index"].keys()

    def _get_shard(self, shard_rel: str) -> dict[str, Any]:
        if shard_rel in self._shard_cache:
            shard = self._shard_cache.pop(shard_rel)
            self._shard_cache[shard_rel] = shard
            return shard

        shard_path = self.root / shard_rel
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        if self.manifest['producer_signature'].get('schema') == 'msila.full_scale.cache.v2':
            stat = shard_path.stat()
            stamp = (stat.st_size,stat.st_mtime_ns)
            if self._verified_shards.get(shard_rel) != stamp:
                expected = self.manifest.get('shard_sha256',{}).get(shard_rel)
                if expected is None or _file_sha256(shard_path) != expected:
                    raise ProducerMismatchError(f'Full-scale cache shard checksum missing or invalid: {shard_rel}')
                self._verified_shards[shard_rel] = stamp

        shard = _safe_torch_load(shard_path, mmap=self.mmap)
        _validate_shard_header(
            shard,
            expected_producer_sha256=self.producer_sha256,
        )

        if self.shard_cache_size > 0:
            self._shard_cache[shard_rel] = shard
            while len(self._shard_cache) > self.shard_cache_size:
                self._shard_cache.popitem(last=False)

        return shard

    def get(self, *, image_id: str, category: str) -> dict[str, Any]:
        key = sample_key(image_id, category)
        ref = self.manifest["index"].get(key)
        if ref is None:
            raise CacheKeyError(f"Cache miss: {category} / {image_id}")

        shard = self._get_shard(ref["shard"])
        record = shard["records"].get(ref["record_key"])
        if record is None:
            raise SchemaError(
                f"Manifest points to missing record {ref['record_key']} "
                f"in {ref['shard']}"
            )

        # Validate at read boundary too; corruption/wrong schema must fail loudly.
        normalized = normalize_record(record)
        if normalized["image_id"] != str(image_id) or normalized["category"] != str(category):
            raise SchemaError("Manifest identity does not match loaded record")
        return normalized

    def clear_shard_cache(self) -> None:
        self._shard_cache.clear()


def validate_cache(root: str | Path) -> dict[str, int]:
    """
    Full structural validation.

    Intended for CI / pre-training gate, not every training epoch.
    """
    reader = FeatureCacheReader(root, mmap=True, shard_cache_size=1)
    seen = 0
    for key, ref in reader.manifest["index"].items():
        shard = reader._get_shard(ref["shard"])
        record = shard["records"].get(ref["record_key"])
        if record is None:
            raise SchemaError(f"Missing record {key} in {ref['shard']}")
        normalized = normalize_record(record)
        actual_key = sample_key(normalized["image_id"], normalized["category"])
        if actual_key != key:
            raise SchemaError(f"Index key mismatch for {key}")
        seen += 1

    if seen != int(reader.manifest["num_samples"]):
        raise SchemaError(
            f"manifest num_samples={reader.manifest['num_samples']}, scanned={seen}"
        )

    return {
        "num_samples": seen,
        "num_shards": int(reader.manifest["num_shards"]),
        "schema_version": SCHEMA_VERSION,
    }
