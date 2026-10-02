"""
data/cached_dataset.py
======================

Training Dataset that reads PRECOMPUTED frozen-backbone features.

Output of __getitem__:
    {
        "local_b4": Tensor,
        "local_b8": Tensor,
        "local_b12": Tensor,
        "context_b4": Tensor,
        "context_b8": Tensor,
        "context_b12": Tensor,
        "mask": Tensor[1,H,W],
        "meta": dict,
    }

The module deliberately contains NO DINO/DINOv3/model-extraction code.
Training therefore becomes:

    feature cache -> CachedFeatureDataset -> DataLoader
                  -> Adapter -> Fusion -> Decoder -> Loss

It is designed to compose with data/feature_cache.py (schema v1).

Recommended environment as of Sep 2026:
    PyTorch 2.x, Pillow, NumPy.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .feature_cache import (
        FEATURE_KEYS,
        FeatureCacheReader,
        SchemaError,
        sample_key,
    )
except ImportError:  # allows direct execution/import during isolated testing
    from feature_cache import (  # type: ignore
        FEATURE_KEYS,
        FeatureCacheReader,
        SchemaError,
        sample_key,
    )


class CachedDatasetError(RuntimeError):
    """Base error for the training cache dataset."""


class TrainingIndexError(CachedDatasetError):
    """Raised when the training index is malformed."""


class MaskError(CachedDatasetError):
    """Raised when a mask is missing or incompatible with the declared sample."""


_RESERVED_RECORD_KEYS = {
    "image_id",
    "category",
    "mask",
    "mask_path",
    "mask_hw",
    "is_anomaly",
    "meta",
}


def load_training_records(
    source: Sequence[Mapping[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    """
    Load the lightweight training index.

    Accepted formats
    ----------------
    1) Sequence[Mapping]
    2) .json containing:
         [...]
       or:
         {"samples": [...]}
    3) .jsonl: one JSON object per line

    Minimum record:
        {"image_id": "...", "category": "..."}

    Typical anomaly-localization record:
        {
          "image_id": "...",
          "category": "fabric",
          "mask_path": "masks/...png",   # null for known normal samples
          "is_anomaly": true,
          "mask_hw": [512, 512],
          "meta": {...}
        }
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "samples" not in data:
                    raise TrainingIndexError(
                        f"{path}: JSON object must contain key 'samples'"
                    )
                data = data["samples"]
            if not isinstance(data, list):
                raise TrainingIndexError(f"{path}: expected a list of records")
            records = data

        elif suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TrainingIndexError(
                            f"{path}:{line_no}: invalid JSON"
                        ) from exc
                    records.append(obj)
        else:
            raise TrainingIndexError(
                f"Unsupported training-index format {suffix!r}; use .json or .jsonl"
            )
    else:
        records = list(source)

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TrainingIndexError(f"record[{i}] must be a mapping")

        if "image_id" not in record or "category" not in record:
            raise TrainingIndexError(
                f"record[{i}] must contain image_id and category"
            )

        image_id = str(record["image_id"])
        category = str(record["category"])
        if not image_id or not category:
            raise TrainingIndexError(
                f"record[{i}] image_id/category must be non-empty"
            )

        key = sample_key(image_id, category)
        if key in seen:
            raise TrainingIndexError(
                f"duplicate training record: category={category!r}, image_id={image_id!r}"
            )
        seen.add(key)

        out = dict(record)
        out["image_id"] = image_id
        out["category"] = category

        meta = out.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, Mapping):
            raise TrainingIndexError(f"record[{i}].meta must be a mapping")
        out["meta"] = dict(meta)

        if "mask_hw" in out and out["mask_hw"] is not None:
            hw = out["mask_hw"]
            if len(hw) != 2 or int(hw[0]) <= 0 or int(hw[1]) <= 0:
                raise TrainingIndexError(
                    f"record[{i}].mask_hw must be [H,W] with H,W>0"
                )
            out["mask_hw"] = [int(hw[0]), int(hw[1])]

        normalized.append(out)

    return normalized


def _resolve_path(path_value: str | Path, root: Path | None) -> Path:
    p = Path(path_value)
    if p.is_absolute() or root is None:
        return p
    return root / p


def _to_binary_mask(mask: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Normalize mask to float32 [1,H,W] with values {0,1}.

    This is appropriate for binary anomaly localization.
    """
    if not isinstance(mask, torch.Tensor):
        raise MaskError(f"mask must be torch.Tensor, got {type(mask)!r}")

    m = mask.detach().cpu()

    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 3 and m.shape[0] == 1:
        pass
    else:
        raise MaskError(
            f"mask must have shape [H,W] or [1,H,W], got {tuple(m.shape)}"
        )

    if not torch.isfinite(m.to(torch.float32)).all().item():
        raise MaskError("mask contains NaN/Inf")

    return (m.to(torch.float32) > float(threshold)).to(torch.float32).contiguous()


def load_binary_mask(path: str | Path, *, threshold: int = 0) -> torch.Tensor:
    """
    Read a grayscale mask and return float32 [1,H,W] in {0,1}.

    `threshold=0` is correct for common industrial anomaly masks encoded
    as background=0 and foreground>0 (often 255).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8, copy=True)

    mask = torch.from_numpy(arr)
    return (mask > int(threshold)).to(torch.float32).unsqueeze(0).contiguous()


def _expected_mask_hw(
    record: Mapping[str, Any],
    geometry: Mapping[str, Any],
    source: str,
) -> tuple[int, int] | None:
    """
    Resolve expected mask [H,W].

    source:
      - "record": record["mask_hw"] if present; otherwise no constraint.
      - "local": geometry["local_hw"] (required).
      - "context": geometry["context_hw"] (required).
      - "image": geometry["image_hw"] (required).
      - "none": no constraint.
    """
    if source == "none":
        return None

    if source == "record":
        hw = record.get("mask_hw")
        if hw is None:
            return None
        return int(hw[0]), int(hw[1])

    geometry_key = {
        "local": "local_hw",
        "context": "context_hw",
        "image": "image_hw",
    }.get(source)

    if geometry_key is None:
        raise ValueError(
            "mask_hw_source must be one of: record, local, context, image, none"
        )

    if geometry_key not in geometry:
        raise MaskError(
            f"mask_hw_source={source!r} requires geometry.{geometry_key}"
        )

    hw = geometry[geometry_key]
    return int(hw[0]), int(hw[1])


def _normalize_cached_feature(
    t: torch.Tensor,
    *,
    key: str,
    squeeze_cached_batch_dim: bool,
    feature_dtype: torch.dtype | None,
) -> torch.Tensor:
    """
    Return a CPU, detached, contiguous per-sample feature tensor.

    For caches created one image at a time, extractors often save:
      [1,C,H,W] or [1,N,C].
    When `squeeze_cached_batch_dim=True`, that singleton extraction-batch
    dimension is removed, so DataLoader later creates the true training batch:
      [B,C,H,W] or [B,N,C].
    """
    if not isinstance(t, torch.Tensor):
        raise SchemaError(f"{key} is not torch.Tensor")

    out = t.detach().cpu()

    if squeeze_cached_batch_dim and out.ndim in (3, 4) and out.shape[0] == 1:
        out = out.squeeze(0)

    if feature_dtype is not None and out.dtype != feature_dtype:
        out = out.to(feature_dtype)

    if not torch.is_floating_point(out):
        raise SchemaError(f"{key} must be floating point")
    if out.numel() == 0:
        raise SchemaError(f"{key} is empty")
    if not torch.isfinite(out).all().item():
        raise SchemaError(f"{key} contains NaN/Inf")

    return out.contiguous()


class CachedFeatureDataset(Dataset):
    """
    Map-style Dataset for training from cached frozen features.

    Important multiprocessing design:
      FeatureCacheReader is created lazily per process (`os.getpid()`).
      Therefore DataLoader workers do not share a stale mmap/shard-reader object.

    No image path or backbone is required at __getitem__ time.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        records: Sequence[Mapping[str, Any]] | str | Path,
        expected_producer_signature: Mapping[str, Any] | None = None,
        mask_root: str | Path | None = None,
        mask_threshold: int = 0,
        mask_hw_source: str = "record",
        allow_zero_mask_for_normal: bool = True,
        squeeze_cached_batch_dim: bool = True,
        feature_dtype: torch.dtype | None = None,
        mmap: bool = True,
        shard_cache_size: int = 2,
    ):
        self.cache_dir = Path(cache_dir)
        self.records = load_training_records(records)
        self.expected_producer_signature = (
            dict(expected_producer_signature)
            if expected_producer_signature is not None
            else None
        )
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.mask_threshold = int(mask_threshold)
        self.mask_hw_source = str(mask_hw_source)
        self.allow_zero_mask_for_normal = bool(allow_zero_mask_for_normal)
        self.squeeze_cached_batch_dim = bool(squeeze_cached_batch_dim)
        self.feature_dtype = feature_dtype
        self.mmap = bool(mmap)
        self.shard_cache_size = int(shard_cache_size)

        if self.shard_cache_size < 0:
            raise ValueError("shard_cache_size must be >= 0")

        # Fail fast on bad configuration/signature in the main process.
        probe = FeatureCacheReader(
            self.cache_dir,
            expected_producer_signature=self.expected_producer_signature,
            mmap=self.mmap,
            shard_cache_size=0,
        )

        # Fail before training if the index references a non-existent cache key.
        missing = [
            (r["category"], r["image_id"])
            for r in self.records
            if (r["image_id"], r["category"]) not in probe
        ]
        if missing:
            preview = ", ".join(f"{c}/{i}" for c, i in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise TrainingIndexError(
                f"{len(missing)} training record(s) are missing from feature cache: "
                f"{preview}{suffix}"
            )

        self._reader: FeatureCacheReader | None = None
        self._reader_pid: int | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        """
        Drop process-local reader before spawn/pickle.

        DataLoader worker gets a clean dataset object and lazily opens its own
        mmap/shard reader.
        """
        state = dict(self.__dict__)
        state["_reader"] = None
        state["_reader_pid"] = None
        return state

    def _get_reader(self) -> FeatureCacheReader:
        pid = os.getpid()
        if self._reader is None or self._reader_pid != pid:
            self._reader = FeatureCacheReader(
                self.cache_dir,
                expected_producer_signature=self.expected_producer_signature,
                mmap=self.mmap,
                shard_cache_size=self.shard_cache_size,
            )
            self._reader_pid = pid
        return self._reader

    def _load_mask(
        self,
        record: Mapping[str, Any],
        geometry: Mapping[str, Any],
    ) -> tuple[torch.Tensor, str | None]:
        expected_hw = _expected_mask_hw(
            record,
            geometry,
            self.mask_hw_source,
        )

        if "mask" in record and record["mask"] is not None:
            mask = _to_binary_mask(
                record["mask"],
                threshold=float(self.mask_threshold),
            )
            mask_path_str = None

        else:
            mask_path_value = record.get("mask_path")
            if mask_path_value not in (None, ""):
                mask_path = _resolve_path(mask_path_value, self.mask_root)
                mask = load_binary_mask(
                    mask_path,
                    threshold=self.mask_threshold,
                )
                mask_path_str = str(mask_path)
            else:
                is_anomaly = record.get(
                    "is_anomaly",
                    record.get("meta", {}).get("is_anomaly"),
                )

                if self.allow_zero_mask_for_normal and is_anomaly is False:
                    if expected_hw is None:
                        raise MaskError(
                            "Normal sample has no mask file, so a zero mask is allowed, "
                            "but mask size is unknown. Provide record.mask_hw or choose "
                            "mask_hw_source=local/context/image with matching geometry."
                        )
                    h, w = expected_hw
                    mask = torch.zeros((1, h, w), dtype=torch.float32)
                    mask_path_str = None
                else:
                    raise MaskError(
                        f"Missing mask for {record['category']}/{record['image_id']}. "
                        "A missing mask is accepted only for an explicitly normal sample "
                        "(is_anomaly=False) when allow_zero_mask_for_normal=True."
                    )

        if expected_hw is not None:
            actual_hw = tuple(int(x) for x in mask.shape[-2:])
            if actual_hw != expected_hw:
                raise MaskError(
                    f"mask shape mismatch for {record['category']}/{record['image_id']}: "
                    f"expected H,W={expected_hw}, got {actual_hw}. "
                    "Do not silently resize masks in the loader; fix preprocessing/alignment."
                )

        return mask, mask_path_str

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        reader = self._get_reader()

        cached = reader.get(
            image_id=record["image_id"],
            category=record["category"],
        )

        features = {
            key: _normalize_cached_feature(
                cached[key],
                key=key,
                squeeze_cached_batch_dim=self.squeeze_cached_batch_dim,
                feature_dtype=self.feature_dtype,
            )
            for key in FEATURE_KEYS
        }

        geometry = cached["geometry"]
        mask, resolved_mask_path = self._load_mask(record, geometry)

        user_meta = dict(record.get("meta", {}))

        # Keep non-reserved training-index fields instead of silently dropping
        # useful annotations such as split, defect_type, tile_id, etc.
        for key, value in record.items():
            if key not in _RESERVED_RECORD_KEYS and key not in user_meta:
                user_meta[key] = value

        meta = {
            **user_meta,
            "image_id": cached["image_id"],
            "category": cached["category"],
            "cache_key": sample_key(cached["image_id"], cached["category"]),
            "geometry": geometry,
            "is_anomaly": record.get(
                "is_anomaly",
                user_meta.get("is_anomaly"),
            ),
            "mask_path": resolved_mask_path,
        }

        return {
            **features,
            "mask": mask,
            "meta": meta,
        }


def cached_collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Strict collate for fixed-shape training.

    Features and masks are stacked. Metadata remains a list.

    We deliberately FAIL on shape mismatch instead of silently padding/resizing,
    because local/context alignment and segmentation masks are scientific
    invariants of the preprocessing pipeline.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    out: dict[str, Any] = {}

    for key in (*FEATURE_KEYS, "mask"):
        tensors = [item[key] for item in batch]
        shapes = [tuple(t.shape) for t in tensors]
        if len(set(shapes)) != 1:
            raise CachedDatasetError(
                f"Cannot stack {key}; sample shapes differ: {shapes}. "
                "Use a fixed preprocessing contract or a task-specific collate function."
            )
        out[key] = torch.stack(tensors, dim=0)

    out["meta"] = [dict(item["meta"]) for item in batch]
    return out



def seed_worker(worker_id: int) -> None:
    """
    Seed Python and NumPy RNGs from PyTorch's worker seed.

    PyTorch already assigns each DataLoader worker its own torch seed.
    Propagating that seed to Python/NumPy avoids duplicated stochastic
    behavior if worker-side preprocessing is introduced later.

    The current CachedFeatureDataset itself is deterministic and performs
    no random augmentation; this helper is therefore a reproducibility guard,
    not a change to cached features.
    """
    del worker_id  # worker id is already encoded in torch.initial_seed()
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_cached_dataloader(
    dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> DataLoader:
    """
    Conservative DataLoader defaults for cached-feature training.

    - CPU tensors stay on CPU inside Dataset workers.
    - pin_memory defaults to CUDA availability.
    - persistent_workers defaults to True iff num_workers > 0.
    - prefetch_factor is passed only when multiprocessing is enabled.
    - default PyTorch in-order behavior is preserved for reproducibility.
    - `seed` creates a dedicated CPU Generator for deterministic shuffling.
    - Python/NumPy worker RNGs are seeded from PyTorch via `seed_worker`.
    - pass either `seed` or an explicit `generator`, never both.

    For Day-04 r×d screening, keep `seed`, records/split, batch size,
    shuffle policy, cache signature, and all other training settings fixed.

    Benchmark num_workers/prefetch_factor on the actual storage device:
    NVMe, HDD, NFS and Google Drive can have very different optima.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")

    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both")
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers=True requires num_workers>0")

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "drop_last": bool(drop_last),
        "collate_fn": cached_collate_fn,
        "generator": generator,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["worker_init_fn"] = seed_worker

    return DataLoader(**kwargs)
