
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


def scan_mvtec_ad2(data_root: str | Path) -> List[SampleRecord]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    all_files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    images = []
    masks = []

    for p in all_files:
        rel = p.relative_to(root)
        parts = list(rel.parts)

        split = None
        for part in parts:
            s = _canonical_split(part)
            if s is not None:
                split = s
                break

        if split is None:
            continue

        category = parts[0] if parts else "unknown"

        item = {
            "path": str(p),
            "category": category,
            "split": split,
            "stem": _canonical_stem(p.stem),
        }

        if _is_mask_path(p):
            masks.append(item)
        else:
            images.append(item)

    # Pair mask conservatively by category + split + canonical stem.
    mask_index: Dict[Tuple[str, str, str], List[str]] = {}
    for m in masks:
        key = (m["category"], m["split"], m["stem"])
        mask_index.setdefault(key, []).append(m["path"])

    # Fallback index ignores split, useful when ground-truth is stored
    # under a parallel directory layout.
    mask_fallback: Dict[Tuple[str, str], List[str]] = {}
    for m in masks:
        key = (m["category"], m["stem"])
        mask_fallback.setdefault(key, []).append(m["path"])

    records = []
    for im in images:
        key = (im["category"], im["split"], im["stem"])
        candidates = mask_index.get(key, [])

        if not candidates:
            candidates = mask_fallback.get((im["category"], im["stem"]), [])

        mask_path = candidates[0] if len(candidates) == 1 else None

        records.append(
            SampleRecord(
                image_path=im["path"],
                category=im["category"],
                split=im["split"],
                mask_path=mask_path,
            )
        )

    return records


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
    ):
        self.data_root = Path(data_root)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        records = scan_mvtec_ad2(self.data_root)

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

        mask = None
        if rec.mask_path is not None:
            mask = load_mask_native(rec.mask_path, target_hw=(h, w))
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
            },
        }


def native_collate_fn(batch):
    # Native-resolution images may have different H,W.
    # Preserve as a Python list instead of torch.stack().
    return batch
