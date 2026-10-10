
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test_public",
    "test_private",
    "test_private_mixed",
}

MASK_DIR_TOKENS = {
    "ground_truth", "groundtruth", "gt",
    "mask", "masks", "label", "labels",
    "segmentation", "segmentations",
}

DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)


def _norm_token(x: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(x).strip().lower())


def _canonical_split(part: str) -> Optional[str]:
    p = _norm_token(part)
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test_public": "test_public",
        "testpublic": "test_public",
        "public_test": "test_public",
        "test_private": "test_private",
        "testprivate": "test_private",
        "private_test": "test_private",
        "test_private_mixed": "test_private_mixed",
        "testprivatemixed": "test_private_mixed",
        "private_mixed": "test_private_mixed",
    }
    return aliases.get(p)


def _is_mask_path(path: Path) -> bool:
    parts = [_norm_token(p) for p in path.parts]
    stem = _norm_token(path.stem)
    return (
        any(p in MASK_DIR_TOKENS for p in parts)
        or stem.endswith("_mask")
        or stem.endswith("_gt")
        or stem.endswith("_label")
    )


def _canonical_stem(stem: str) -> str:
    s = _norm_token(stem)
    for suffix in ("_mask", "_gt", "_label", "_seg", "_segmentation"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    category: str
    split: str
    mask_path: Optional[str]
    defect_type: str = "good"
    gt_status: str = "normal_zero"
    mask_candidates: Tuple[str, ...] = ()

    @property
    def is_normal(self) -> bool:
        return self.defect_type in {"good", "normal", "ok"}


def _layout_identity(path: Path, root: Path):
    parts = path.relative_to(root).parts
    splits = [(i, _canonical_split(x)) for i, x in enumerate(parts[:-1])
              if _canonical_split(x) is not None]
    if len(splits) != 1 or splits[0][0] == 0:
        return None
    idx, split = splits[0]
    # Supports category/split/{good,bad}/..., category/split/ground_truth/bad/...
    # and category/ground_truth/split/bad/...; unscoped GT is never guessed.
    tail = [x for x in parts[idx+1:-1] if _norm_token(x) not in MASK_DIR_TOKENS]
    defect = _norm_token(tail[0]) if tail else ("good" if split == "train" else "unknown")
    identity = tuple(tail[1:]) + (_canonical_stem(path.stem),)
    return parts[0], split, defect, identity


def scan_mvtec_ad2(data_root: str | Path, *, require_pixel_gt: bool = False,
                   split: Optional[str] = None, categories: Optional[Sequence[str]] = None) -> List[SampleRecord]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    images, masks = [], {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        key = _layout_identity(p, root)
        if key is None:
            continue
        if split is not None and key[1] != split:
            continue
        if categories is not None and key[0] not in categories:
            continue
        if _is_mask_path(p.relative_to(root)):
            masks.setdefault(key, []).append(str(p))
        else:
            images.append((p, key))
    records = []
    image_counts = {}
    for _, key in images:
        image_counts[key] = image_counts.get(key, 0) + 1
    for p, key in images:
        category, role, defect, _ = key
        candidates = tuple(masks.get(key, ()))
        normal = defect in {"good", "normal", "ok"}
        status = "normal_zero" if normal else ("matched" if len(candidates) == 1 else
                                               "ambiguous" if candidates else "missing")
        if not normal and image_counts[key] > 1:
            status = "ambiguous"
        if require_pixel_gt and not normal and status != "matched":
            raise ValueError(f"{status} pixel GT (abnormal GT): category={category}, split={role}, "
                             f"defect={defect}, image={p}, candidates={candidates}")
        records.append(SampleRecord(str(p), category, role,
                                   candidates[0] if status == "matched" else None,
                                   defect, status, candidates))
    return records


def audit_mvtec_ad2(data_root: str | Path) -> dict:
    records = scan_mvtec_ad2(data_root)
    rows, counts, errors = [], {}, []
    for r in records:
        key = f"{r.category}/{r.split}"
        count = counts.setdefault(key, dict(normal=0, abnormal=0, normal_zero=0, matched=0, missing=0, ambiguous=0))
        count["normal" if r.is_normal else "abnormal"] += 1
        if r.gt_status in count:
            count[r.gt_status] += 1
        with Image.open(r.image_path) as im:
            hw = [im.height, im.width]
        row = dict(image=r.image_path, category=r.category, split=r.split,
                   defect_type=r.defect_type, native_hw=hw, gt_status=r.gt_status,
                   mask=r.mask_path, candidates=list(r.mask_candidates))
        if r.gt_status in {"missing", "ambiguous"}:
            errors.append(dict(image=r.image_path, error=r.gt_status))
        if r.mask_path:
            with Image.open(r.mask_path) as im:
                row["mask_hw"] = [im.height, im.width]
                row['mask_requires_nearest_resize'] = row['mask_hw'] != hw
                if not np.asarray(im).any():
                    errors.append(dict(image=r.image_path, error="abnormal GT is empty"))
        rows.append(row)
    return dict(schema="msila.dataset_audit.v2", root=str(data_root), counts=counts,
                images=rows, errors=errors, status="PASS" if records and not errors else "FAIL")


def load_rgb_native(path: str | Path) -> torch.Tensor:
    # Output: float32 [3,H,W] in [0,1].
    with Image.open(path) as im:
        arr = np.array(im)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    # Robust conversion for uint8/uint16/float.
    if np.issubdtype(arr.dtype, np.integer):
        maxv = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / float(maxv)
    else:
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def normalize_dinov3(
    image_chw: torch.Tensor,
    mean: Sequence[float] = DINOV3_MEAN,
    std: Sequence[float] = DINOV3_STD,
) -> torch.Tensor:
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(image_chw.shape)}")

    mean_t = torch.tensor(mean, dtype=image_chw.dtype, device=image_chw.device)[:, None, None]
    std_t = torch.tensor(std, dtype=image_chw.dtype, device=image_chw.device)[:, None, None]
    return (image_chw - mean_t) / std_t


def load_mask_native(
    path: str | Path,
    target_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("L")

        if target_hw is not None and im.size != (target_hw[1], target_hw[0]):
            im = im.resize(
                (target_hw[1], target_hw[0]),
                resample=Image.Resampling.NEAREST,
            )

        arr = np.array(im)

    mask = torch.from_numpy((arr > 0).astype(np.uint8))
    return mask


class MVTecAD2HighResDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        mean: Sequence[float] = DINOV3_MEAN,
        std: Sequence[float] = DINOV3_STD,
        require_pixel_gt: bool = True,
    ):
        self.data_root = Path(data_root)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        records = scan_mvtec_ad2(self.data_root, require_pixel_gt=require_pixel_gt,
                                  split=split, categories=categories)

        if split is not None:
            records = [r for r in records if r.split == split]

        if categories is not None:
            keep = set(categories)
            records = [r for r in records if r.category in keep]

        self.records = records

        if len(self.records) == 0:
            raise RuntimeError(
                f"No samples found for split={split}, categories={categories}, root={self.data_root}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]

        image = load_rgb_native(rec.image_path)
        _, h, w = image.shape

        mask = torch.zeros((h, w), dtype=torch.uint8) if rec.is_normal else None
        if rec.mask_path is not None:
            mask = load_mask_native(rec.mask_path, target_hw=(h, w))
            if not rec.is_normal and not bool(mask.any()):
                raise ValueError(f"Abnormal GT is empty: {rec.image_path}, mask={rec.mask_path}")
            if tuple(mask.shape) != (h, w):
                raise AssertionError(
                    f"Mask shape {tuple(mask.shape)} != image shape {(h,w)}"
                )

        return {
            "image": image,                         # raw [0,1], native resolution
            "image_norm": normalize_dinov3(image, self.mean, self.std),
            "mask": mask,
            "meta": {
                "category": rec.category,
                "split": rec.split,
                "H": h,
                "W": w,
                "path": rec.image_path,
                "mask_path": rec.mask_path,
                "defect_type": rec.defect_type,
                "is_anomaly": not rec.is_normal,
            },
        }


def native_collate_fn(batch):
    # Native-resolution images may have different H,W.
    # Preserve as a Python list instead of torch.stack().
    return batch


class G1NativeDataset(Dataset):
    """Good-only native pairs; synthesize before any tile crop.

    The protocol YAML is passed through unchanged. TRAIN seed varies by epoch;
    DEV and Overfit-16 stay fixed. A small cache avoids regenerating adjacent
    tiles without retaining the entire native dataset in RAM.
    """

    def __init__(self, sources, protocol, *, seed, role, variants=2, fixed=False):
        from itertools import product
        from collections import OrderedDict
        from .synthetic_anomaly import NativeTinyDefectGenerator

        if role not in {"train", "dev"} or variants < 1 or not sources:
            raise ValueError("G1 requires nonempty TRAIN/DEV good sources and variants >= 1")
        self.generator = NativeTinyDefectGenerator(protocol)
        self.sources = tuple(sources)
        expected = "train" if role == "train" else "validation"
        if any(r.split != expected or r.defect_type != "good" for r in sources):
            raise ValueError(f"G1 {role} accepts only {expected}/good images")
        bins = protocol["train_core_bins"] if role == "train" else protocol["dev_mixed_bins"]
        self.combinations = list(product(protocol["defect_types"], protocol["placements"], bins))
        self.seed, self.role, self.fixed, self.epoch = int(seed), role, fixed, 0
        self.plan = [(i, v) for i in range(len(sources)) for v in range(variants + 1)]
        self.variants = variants
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.plan)

    def set_epoch(self, epoch):
        self.epoch = 0 if self.fixed else int(epoch)
        self.cache.clear()

    def __getitem__(self, index):
        import hashlib
        index = int(index)
        key = (self.epoch, index)
        if key not in self.cache:
            source_index, variant = self.plan[index]
            record = self.sources[source_index]
            original = load_rgb_native(record.image_path)
            combo_index = source_index * self.variants + max(variant - 1, 0)
            if not self.fixed:
                combo_index += self.epoch * len(self.sources) * self.variants
            kind, placement, size_bin = self.combinations[combo_index % len(self.combinations)]
            seed_text = f"{self.role}|{self.seed}|{self.epoch}|{index}".encode()
            sample_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
            sample = self.generator(original, seed=sample_seed, defect_type=kind,
                                    size_bin=size_bin, placement=placement, normal=variant == 0)
            sample_id = f"{self.role}_{source_index:05d}_{variant:02d}"
            self.cache[key] = dict(original=original, image=sample.image, mask=sample.mask,
                                   meta=dict(sample_id=sample_id, source=record.image_path,
                                             source_split=record.split, original_hw=list(original.shape[-2:]),
                                             synthetic=sample.metadata))
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


class G1TileDataset(Dataset):
    """512px local tiles with exact mask crops and coordinate metadata."""

    def __init__(self, native, *, tile_size=512, overlap=128, overfit16=False):
        from .tiling import generate_tile_records
        self.native, self.tile_size = native, tile_size
        self.records = []
        if overfit16 and len(native.sources) < 8:
            raise ValueError("Overfit-16 needs at least eight distinct TRAIN/good sources")
        if overfit16:
            native.fixed = True
        source_tiles = {}
        for i, record in enumerate(native.sources):
            with Image.open(record.image_path) as image:
                source_tiles[i] = generate_tile_records(image.height, image.width, tile_size, overlap,
                                                        context_size=tile_size)
        for index, (source_index, variant) in enumerate(native.plan):
            tiles = source_tiles[source_index]
            if overfit16:
                if source_index >= 8 or variant > 1:
                    continue
                if variant:
                    sample = native[index]
                    from .tiling import crop_with_padding
                    tile = max(tiles, key=lambda r: float(crop_with_padding(
                        sample["mask"], r.local_xyxy, pad_mode="constant").sum()))
                else:
                    tile = tiles[source_index % len(tiles)]
                self.records.append((index, tile))
            else:
                self.records.extend((index, tile) for tile in tiles)
        native.cache.clear()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from dataclasses import asdict
        from .tiling import crop_with_padding
        native_index, record = self.records[index]
        sample = self.native[native_index]
        image = crop_with_padding(sample["image"], record.local_xyxy)
        mask = crop_with_padding(sample["mask"], record.local_xyxy, pad_mode="constant")
        return dict(image=normalize_dinov3(image), mask=mask,
                    meta=dict(sample["meta"], tile=asdict(record)))


def g1_tile_collate(batch):
    return dict(image=torch.stack([r["image"] for r in batch]),
                mask=torch.stack([r["mask"] for r in batch]), meta=[r["meta"] for r in batch])
