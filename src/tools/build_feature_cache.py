#!/usr/bin/env python3
"""
Feature cache for a frozen DINOv3-style multi-view extractor.

Project contract
----------------
Each image caches exactly these six feature tensors:

    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

and alignment metadata:

    image_id, category,
    geometry.local_box,
    geometry.context_box,
    geometry.context_to_local

The builder is deliberately independent of the project's dataset and DINOv3
wrapper. Your existing extractor remains the single source of truth: it must
return the six tensors + geometry. This file only makes those outputs
persistent, validates them, and reloads them for downstream Adapter/Fusion/
Decoder training.

Recommended default: cache float32 if the acceptance gate is
torch.allclose(..., atol=1e-5). Do not silently quantize to fp16/bf16.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch
from safetensors.torch import load_file, save_file


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


class FeatureCacheError(RuntimeError):
    """Base error for feature-cache failures."""


class CacheValidationError(FeatureCacheError):
    """Raised when a sample/cache does not satisfy the project contract."""


class SignatureMismatchError(CacheValidationError):
    """Raised when a cache belongs to a different extractor/preprocess setup."""


@dataclass(frozen=True)
class CacheJob:
    """
    Bundle returned by a project-specific CLI factory.

    `samples`:
        Iterable of mappings. Each sample must contain at least:
        {"image_id": str, "category": str, ...}
        Remaining fields are whatever your extractor requires.

    `extractor`:
        Callable(sample) -> mapping with FEATURE_KEYS + "geometry".

    `signature`:
        JSON-serializable mapping that uniquely identifies the frozen
        feature-generating pipeline. Include model/checkpoint/preprocess/layers.
    """

    samples: Iterable[Mapping[str, Any]]
    extractor: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    signature: Mapping[str, Any]


@dataclass(frozen=True)
class CachePaths:
    tensor_path: Path
    metadata_path: Path


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _to_jsonable(value: Any) -> Any:
    """Convert common tensor/numpy-like containers to strict JSON values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    # NumPy scalars/arrays without importing numpy as a hard dependency.
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
    raise TypeError(f"Value of type {type(value)!r} is not JSON serializable")


def signature_hash(signature: Mapping[str, Any]) -> str:
    """Stable SHA-256 fingerprint of the extractor/preprocess signature."""
    normalized = _to_jsonable(dict(signature))
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of a file, useful for checkpoint/cache provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(image_id: str, category: str) -> str:
    raw = f"{category}\0{image_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:24]
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(image_id)).stem).strip("._-")
    readable = readable[:40] or "sample"
    return f"{readable}__{digest}"


def _validate_box(box: Sequence[float], name: str) -> list[float]:
    if len(box) != 4:
        raise CacheValidationError(f"{name} must be [x0, y0, x1, y1], got {box!r}")
    vals = [float(v) for v in box]
    if not all(math.isfinite(v) for v in vals):
        raise CacheValidationError(f"{name} contains non-finite values: {vals}")
    x0, y0, x1, y1 = vals
    if not (x1 > x0 and y1 > y0):
        raise CacheValidationError(f"{name} must satisfy x1>x0 and y1>y0, got {vals}")
    return vals


def _validate_matrix_3x3(matrix: Any, name: str = "context_to_local") -> list[list[float]]:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().tolist()
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise CacheValidationError(f"{name} must be a 3x3 homogeneous matrix")
    result = [[float(v) for v in row] for row in matrix]
    if not all(math.isfinite(v) for row in result for v in row):
        raise CacheValidationError(f"{name} contains non-finite values")
    return result


def validate_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate minimum alignment metadata while preserving extra JSON metadata."""
    missing = [k for k in REQUIRED_GEOMETRY_KEYS if k not in geometry]
    if missing:
        raise CacheValidationError(f"geometry is missing keys: {missing}")

    result = _to_jsonable(dict(geometry))
    result["local_box"] = _validate_box(result["local_box"], "geometry.local_box")
    result["context_box"] = _validate_box(result["context_box"], "geometry.context_box")
    result["context_to_local"] = _validate_matrix_3x3(result["context_to_local"])

    for hw_key in ("local_hw", "context_hw"):
        if hw_key in result:
            hw = result[hw_key]
            if len(hw) != 2 or int(hw[0]) <= 0 or int(hw[1]) <= 0:
                raise CacheValidationError(f"geometry.{hw_key} must be [H, W] with H,W>0")
            result[hw_key] = [int(hw[0]), int(hw[1])]

    return result


def context_to_local_from_boxes(
    local_box: Sequence[float],
    context_box: Sequence[float],
    *,
    local_hw: Sequence[int],
    context_hw: Sequence[int],
) -> list[list[float]]:
    """
    Build a 3x3 affine map from resized context-crop coordinates to
    resized local-crop coordinates.

    Coordinate convention:
      - boxes are [x0, y0, x1, y1] in ORIGINAL-image continuous edge coords;
      - crop coordinates use [0,W] x [0,H] edge coordinates;
      - local_hw/context_hw are [H, W] AFTER resize.

    For p_C = [u_C, v_C, 1]^T:
        p_L = M_(C->L) p_C

    This is geometric metadata only. If your crop implementation uses a
    different pixel-center convention, store the matrix produced by that
    implementation instead of recomputing it here.
    """
    lx0, ly0, lx1, ly1 = _validate_box(local_box, "local_box")
    cx0, cy0, cx1, cy1 = _validate_box(context_box, "context_box")

    hl, wl = int(local_hw[0]), int(local_hw[1])
    hc, wc = int(context_hw[0]), int(context_hw[1])
    if min(hl, wl, hc, wc) <= 0:
        raise ValueError("local_hw/context_hw must contain positive integers")

    local_w_orig = lx1 - lx0
    local_h_orig = ly1 - ly0
    context_w_orig = cx1 - cx0
    context_h_orig = cy1 - cy0

    sx = (wl * context_w_orig) / (wc * local_w_orig)
    sy = (hl * context_h_orig) / (hc * local_h_orig)
    tx = wl * (cx0 - lx0) / local_w_orig
    ty = hl * (cy0 - ly0) / local_h_orig

    return [
        [float(sx), 0.0, float(tx)],
        [0.0, float(sy), float(ty)],
        [0.0, 0.0, 1.0],
    ]


def apply_homogeneous_2d(matrix_3x3: Sequence[Sequence[float]], xy: Sequence[float]) -> tuple[float, float]:
    """Apply a homogeneous 3x3 transform to one 2D point."""
    m = _validate_matrix_3x3(matrix_3x3)
    x, y = float(xy[0]), float(xy[1])
    xp = m[0][0] * x + m[0][1] * y + m[0][2]
    yp = m[1][0] * x + m[1][1] * y + m[1][2]
    wp = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(wp) < 1e-12:
        raise ZeroDivisionError("Homogeneous transform produced w≈0")
    return xp / wp, yp / wp


def _ensure_feature_dict(features: Mapping[str, Any]) -> None:
    missing = [k for k in FEATURE_KEYS if k not in features]
    if missing:
        raise CacheValidationError(f"Extractor output is missing feature keys: {missing}")

    for key in FEATURE_KEYS:
        t = features[key]
        if not isinstance(t, torch.Tensor):
            raise CacheValidationError(f"{key} must be torch.Tensor, got {type(t)!r}")
        if not torch.is_floating_point(t):
            raise CacheValidationError(f"{key} must be floating point, got dtype={t.dtype}")
        if t.numel() == 0:
            raise CacheValidationError(f"{key} is empty")
        if not torch.isfinite(t).all().item():
            raise CacheValidationError(f"{key} contains NaN/Inf")


def _cache_tensor(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=dtype).contiguous()


def _dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache dtype {name!r}; use float32/float16/bfloat16") from exc


def assert_cache_matches(
    online: Mapping[str, Any],
    cached: Mapping[str, Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """
    Hard gate: cached feature tensors must reproduce online extractor outputs.
    Raises AssertionError with shape/max-error diagnostics on first failure.
    """
    _ensure_feature_dict(online)
    _ensure_feature_dict(cached)

    for key in FEATURE_KEYS:
        a = online[key].detach().to(device="cpu", dtype=torch.float32)
        b = cached[key].detach().to(device="cpu", dtype=torch.float32)

        if a.shape != b.shape:
            raise AssertionError(
                f"{key}: shape mismatch online={tuple(a.shape)} cache={tuple(b.shape)}"
            )
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            max_abs = (a - b).abs().max().item()
            raise AssertionError(
                f"{key}: cache mismatch; max_abs={max_abs:.8g}, "
                f"atol={atol}, rtol={rtol}"
            )


class FeatureCacheStore:
    """
    Per-sample cache using:
      - .safetensors for the six dense feature tensors
      - .json for IDs, geometry, provenance, shapes/dtypes, checksum

    The pair is written atomically file-by-file. A SHA-256 recorded in JSON
    detects a stale/corrupted tensor file.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def paths_for(self, image_id: str, category: str) -> CachePaths:
        key = _cache_key(image_id=image_id, category=category)
        shard = key[-2:]
        d = self.samples_dir / shard
        return CachePaths(
            tensor_path=d / f"{key}.safetensors",
            metadata_path=d / f"{key}.json",
        )

    def exists(self, image_id: str, category: str) -> bool:
        p = self.paths_for(image_id, category)
        return p.tensor_path.is_file() and p.metadata_path.is_file()

    def save(
        self,
        *,
        image_id: str,
        category: str,
        features: Mapping[str, Any],
        geometry: Mapping[str, Any],
        signature: Mapping[str, Any],
        cache_dtype: torch.dtype = torch.float32,
        extra_metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> CachePaths:
        image_id = str(image_id)
        category = str(category)
        if not image_id:
            raise CacheValidationError("image_id must be non-empty")
        if not category:
            raise CacheValidationError("category must be non-empty")

        _ensure_feature_dict(features)
        geometry_checked = validate_geometry(geometry)

        paths = self.paths_for(image_id, category)
        paths.tensor_path.parent.mkdir(parents=True, exist_ok=True)

        if self.exists(image_id, category) and not overwrite:
            raise FileExistsError(
                f"Cache already exists for category={category!r}, image_id={image_id!r}"
            )

        tensors = {k: _cache_tensor(features[k], cache_dtype) for k in FEATURE_KEYS}
        sig_jsonable = _to_jsonable(dict(signature))
        sig_hash = signature_hash(sig_jsonable)

        tensor_tmp = paths.tensor_path.with_name(paths.tensor_path.name + f".tmp.{os.getpid()}")
        meta_tmp = paths.metadata_path.with_name(paths.metadata_path.name + f".tmp.{os.getpid()}")

        try:
            save_file(
                tensors,
                str(tensor_tmp),
                metadata={
                    "schema_version": str(SCHEMA_VERSION),
                    "signature_sha256": sig_hash,
                    "image_id": image_id,
                    "category": category,
                },
            )
            tensor_sha = sha256_file(tensor_tmp)

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "image_id": image_id,
                "category": category,
                "signature": sig_jsonable,
                "signature_sha256": sig_hash,
                "tensor_sha256": tensor_sha,
                "geometry": geometry_checked,
                "features": {
                    k: {
                        "shape": list(tensors[k].shape),
                        "dtype": str(tensors[k].dtype).removeprefix("torch."),
                    }
                    for k in FEATURE_KEYS
                },
            }
            if extra_metadata:
                metadata["extra_metadata"] = _to_jsonable(dict(extra_metadata))

            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

            # Tensor must land first; without metadata the entry is considered absent.
            os.replace(tensor_tmp, paths.tensor_path)
            os.replace(meta_tmp, paths.metadata_path)
        finally:
            for tmp in (tensor_tmp, meta_tmp):
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass

        return paths

    def load(
        self,
        *,
        image_id: str,
        category: str,
        expected_signature: Mapping[str, Any] | None = None,
        verify_checksum: bool = True,
    ) -> dict[str, Any]:
        paths = self.paths_for(image_id, category)
        if not paths.tensor_path.is_file() or not paths.metadata_path.is_file():
            raise FileNotFoundError(
                f"Cache not found for category={category!r}, image_id={image_id!r}"
            )

        with open(paths.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise CacheValidationError(
                f"Unsupported cache schema {metadata.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}"
            )
        if metadata.get("image_id") != str(image_id) or metadata.get("category") != str(category):
            raise CacheValidationError("Cache metadata ID/category does not match request")

        if expected_signature is not None:
            expected_hash = signature_hash(expected_signature)
            actual_hash = metadata.get("signature_sha256")
            if actual_hash != expected_hash:
                raise SignatureMismatchError(
                    "Feature cache signature mismatch. Rebuild cache because model/checkpoint/"
                    "preprocess/layer selection changed.\n"
                    f"expected={expected_hash}\nactual={actual_hash}"
                )

        if verify_checksum:
            actual_tensor_sha = sha256_file(paths.tensor_path)
            if actual_tensor_sha != metadata.get("tensor_sha256"):
                raise CacheValidationError(
                    f"Tensor checksum mismatch for {paths.tensor_path}; cache may be corrupted"
                )

        tensors = load_file(str(paths.tensor_path), device="cpu")
        _ensure_feature_dict(tensors)
        geometry = validate_geometry(metadata["geometry"])

        for key in FEATURE_KEYS:
            declared = metadata["features"][key]
            if list(tensors[key].shape) != declared["shape"]:
                raise CacheValidationError(
                    f"{key}: metadata shape={declared['shape']} "
                    f"but file shape={list(tensors[key].shape)}"
                )

        return {
            "image_id": metadata["image_id"],
            "category": metadata["category"],
            **{k: tensors[k] for k in FEATURE_KEYS},
            "geometry": geometry,
            "cache_metadata": metadata,
        }


class FeatureCacheBuilder:
    """
    Build/verify a cache from the project's existing frozen extractor.

    The extractor must be deterministic for a fixed sample:
        output = extractor(sample)
        output[FEATURE_KEYS] -> tensors
        output["geometry"]   -> alignment metadata

    If `extractor` is torch.nn.Module, the builder forces eval() and
    requires_grad_(False). Extraction itself runs in torch.inference_mode().
    """

    def __init__(
        self,
        *,
        extractor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        cache_dir: str | Path,
        signature: Mapping[str, Any],
        cache_dtype: torch.dtype = torch.float32,
        verify_after_write: bool = True,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.extractor = extractor
        self.store = FeatureCacheStore(cache_dir)
        self.signature = _to_jsonable(dict(signature))
        self.cache_dtype = cache_dtype
        self.verify_after_write = bool(verify_after_write)
        self.atol = float(atol)
        self.rtol = float(rtol)

        if isinstance(extractor, torch.nn.Module):
            extractor.eval()
            extractor.requires_grad_(False)

    def extract_online(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        with torch.inference_mode():
            output = self.extractor(sample)
        if not isinstance(output, Mapping):
            raise CacheValidationError(
                f"extractor(sample) must return a mapping, got {type(output)!r}"
            )
        _ensure_feature_dict(output)
        if "geometry" not in output:
            raise CacheValidationError("Extractor output must contain 'geometry'")
        validate_geometry(output["geometry"])
        return output

    def build_one(
        self,
        sample: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if "image_id" not in sample or "category" not in sample:
            raise CacheValidationError("Each sample must contain image_id and category")

        image_id = str(sample["image_id"])
        category = str(sample["category"])

        if self.store.exists(image_id, category) and not overwrite:
            # Existing cache is accepted only if provenance still matches.
            return self.store.load(
                image_id=image_id,
                category=category,
                expected_signature=self.signature,
                verify_checksum=True,
            )

        online = self.extract_online(sample)

        extra_metadata = {}
        if "extra_metadata" in online:
            extra_metadata = online["extra_metadata"]

        self.store.save(
            image_id=image_id,
            category=category,
            features=online,
            geometry=online["geometry"],
            signature=self.signature,
            cache_dtype=self.cache_dtype,
            extra_metadata=extra_metadata,
            overwrite=overwrite,
        )

        cached = self.store.load(
            image_id=image_id,
            category=category,
            expected_signature=self.signature,
            verify_checksum=True,
        )

        if self.verify_after_write:
            assert_cache_matches(
                online,
                cached,
                atol=self.atol,
                rtol=self.rtol,
            )

        return cached

    def build_many(
        self,
        samples: Iterable[Mapping[str, Any]],
        *,
        overwrite: bool = False,
        limit: int | None = None,
    ) -> dict[str, int]:
        stats = {"built_or_loaded": 0, "failed": 0}

        for i, sample in enumerate(samples):
            if limit is not None and i >= limit:
                break
            try:
                self.build_one(sample, overwrite=overwrite)
                stats["built_or_loaded"] += 1
            except Exception:
                stats["failed"] += 1
                raise

        return stats


def load_cached_sample(
    cache_dir: str | Path,
    *,
    image_id: str,
    category: str,
    expected_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function for Adapter/Fusion/Decoder datasets."""
    return FeatureCacheStore(cache_dir).load(
        image_id=image_id,
        category=category,
        expected_signature=expected_signature,
        verify_checksum=True,
    )


def _load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("--factory must be MODULE:FUNCTION")
    module_name, symbol_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and verify frozen DINOv3 feature cache."
    )
    parser.add_argument(
        "--factory",
        required=True,
        help=(
            "Project factory MODULE:FUNCTION. It must return CacheJob "
            "(samples, extractor, signature)."
        ),
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--cache-dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="Use float32 for the strict atol=1e-5 equivalence gate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    factory = _load_symbol(args.factory)
    job = factory()
    if not isinstance(job, CacheJob):
        raise TypeError(
            f"{args.factory} must return build_feature_cache.CacheJob, got {type(job)!r}"
        )

    builder = FeatureCacheBuilder(
        extractor=job.extractor,
        cache_dir=args.cache_dir,
        signature=job.signature,
        cache_dtype=_dtype_from_name(args.cache_dtype),
        verify_after_write=not args.no_verify,
        atol=1e-5,
        rtol=1e-5,
    )
    stats = builder.build_many(
        job.samples,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _main()
