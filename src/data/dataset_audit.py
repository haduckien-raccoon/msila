"""Read-only dataset inventory, verified source hashes and cached-mask padding checks."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image

from .loader import EXPECTED_SPLITS, _records_from_inventory, _scan_inventory

PRIVATE_SPLITS = {"test_private", "test_private_mixed"}
MVTEC_AD2_CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")


def _role_family(role):
    if role in {"train", "training", "train_core"}:
        return "train"
    if role in {"dev", "validation", "dev_synthetic", "dev_tiny", "dev_mixed"}:
        return "dev"
    if role in {"test", "test_public", "test_private", "test_private_mixed"}:
        return "test"
    return None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source_manifest(root, inventory, manifest):
    """Only byte-verified raw-image hashes establish duplicate-content evidence."""
    result = dict(status="NOT_CHECKED", source_disjoint_status="NOT_CHECKED",
                  verified_sources=0, missing_sources=[],
                  duplicate_content=[], errors=[])
    if manifest is None:
        return result
    errors, seen, verified, roles, declared_roles = result["errors"], set(), {}, {}, {}
    sources = {p.relative_to(root).as_posix(): key for p, key in inventory}
    try:
        with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"relative_path", "file_size", "sha256"}.issubset(reader.fieldnames or []):
                raise ValueError("manifest needs relative_path,file_size,sha256 columns")
            for line, row in enumerate(reader, 2):
                if any(row.get(field) is None for field in ("relative_path", "file_size", "sha256")):
                    errors.append(dict(error="manifest_invalid_entry", line=line))
                    continue
                relative = row["relative_path"].replace("\\", "/")
                pure = PurePosixPath(relative)
                path = root / pure
                if (not relative or pure.is_absolute() or ".." in pure.parts
                        or not path.resolve().is_relative_to(root.resolve())):
                    errors.append(dict(path=relative, error="manifest_unsafe_path", line=line))
                    continue
                relative = pure.as_posix()
                if relative in seen:
                    errors.append(dict(path=relative, error="manifest_duplicate_path", line=line))
                    continue
                seen.add(relative)
                digest = row["sha256"].lower()
                try:
                    size = int(row["file_size"])
                    if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError("invalid size/hash")
                    # Recompute: stale/unverified CSV entries are never source evidence.
                    actual_hash = _sha256(path)
                    if path.stat().st_size != size or actual_hash != digest:
                        errors.append(dict(path=relative, error="manifest_hash_mismatch"))
                        continue
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(dict(path=relative, error="manifest_invalid_entry", detail=str(exc)))
                    continue
                if relative in sources:
                    key = sources[relative]
                    if (row.get("category") and row["category"] != key[0]) or (
                            row.get("split") and row["split"] != key[1]):
                        errors.append(dict(path=relative, error="manifest_identity_mismatch"))
                        continue
                    verified[relative] = digest
                    roles[relative] = (row.get("source_role") or key[1]).strip().lower()
                    declared_roles[relative] = (row.get("source_role") or "").strip().lower()
                    if key[1] in PRIVATE_SPLITS | {"test_public"} and _role_family(declared_roles[relative]) in {"train", "dev"}:
                        errors.append(dict(path=relative, error="test_source_for_train_dev"))
    except (OSError, ValueError, csv.Error) as exc:
        errors.append(dict(error="manifest_unreadable", detail=str(exc)))
    result["missing_sources"] = sorted(set(sources) - seen)
    errors.extend(dict(path=p, error="manifest_missing_source") for p in result["missing_sources"])
    by_hash = defaultdict(list)
    for path, digest in sorted(verified.items()):
        by_hash[digest].append(path)
    for digest, paths in sorted(by_hash.items()):
        if len(paths) < 2:
            continue
        splits = sorted({sources[p][1] for p in paths})
        source_roles = sorted({roles[p] for p in paths})
        # Copies between private/private_mixed can be intentional; report them,
        # but only TRAIN overlap with another declared role is leakage here.
        families = {_role_family(role) for role in source_roles}
        overlap = "train" in families and bool(families & {"dev", "test"})
        result["duplicate_content"].append(dict(sha256=digest, paths=paths, splits=splits,
                                                source_roles=source_roles, cross_split=len(splits) > 1,
                                                train_overlap=overlap))
        if overlap:
            errors.append(dict(error="duplicate_train_source", paths=paths))
    result.update(status="FAIL" if errors else "PASS", verified_sources=len(verified),
                  verification="FILE_SHA256_RECOMPUTED", manifest=str(Path(manifest).resolve()))
    declared_families = {_role_family(role) for role in declared_roles.values()}
    result["source_disjoint_status"] = (
        "FAIL" if any(d["train_overlap"] for d in result["duplicate_content"]) or
        any(e["error"] == "test_source_for_train_dev" for e in errors) else
        "NOT_VERIFIED" if errors else
        "PASS" if None not in declared_families and {"train", "dev"}.issubset(declared_families) else "NOT_CHECKED")
    return result


def audit_inventory(data_root, *, expected_categories=None, expected_splits=None, source_manifest=None):
    root = Path(data_root)
    images, masks, unrecognized = _scan_inventory(root)
    records = _records_from_inventory(images, masks)
    relative = lambda p: Path(p).relative_to(root).as_posix()
    rows, counts, errors, warnings = [], {}, [], []
    image_keys = defaultdict(list)
    for path, key in images:
        image_keys[key].append(relative(path))
    orphan_gt = sorted(relative(p) for key, paths in masks.items() for p in paths if key not in image_keys)
    duplicate_gt = [dict(category=k[0], split=k[1], defect_type=k[2], image_id=list(k[3]),
                         paths=[relative(p) for p in paths])
                    for k, paths in sorted(masks.items()) if len(paths) > 1]
    duplicate_images = [dict(category=k[0], split=k[1], defect_type=k[2], image_id=list(k[3]), paths=paths)
                        for k, paths in sorted(image_keys.items()) if len(paths) > 1]
    ignored = [dict(path=relative(p), kind="mask" if is_mask else "image", error=reason)
               for p, is_mask, reason in unrecognized]
    errors.extend(ignored)
    errors.extend(dict(path=p, error="orphan_gt") for p in orphan_gt)
    errors.extend(dict(error="duplicate_gt", **row) for row in duplicate_gt)
    errors.extend(dict(error="duplicate_image_identity", **row) for row in duplicate_images)
    inventory_errors = len(errors)
    for record in records:
        key = f"{record.category}/{record.split}"
        count = counts.setdefault(key, dict(normal=0, abnormal=0, normal_zero=0, matched=0,
                                            missing=0, ambiguous=0, eligible=0, private_hidden=0))
        count["normal" if record.is_normal else "abnormal"] += 1
        count[record.gt_status] += 1
        row = dict(image=relative(record.image_path), category=record.category, split=record.split,
                   defect_type=record.defect_type, gt_status=record.gt_status,
                   mask=relative(record.mask_path) if record.mask_path else None,
                   candidates=[relative(p) for p in record.mask_candidates], native_hw=None,
                   gt_availability=record.gt_status.upper(), pixel_evaluation_eligible=False)
        valid = True
        try:
            with Image.open(record.image_path) as im:
                im.load()
                row["native_hw"] = [im.height, im.width]
        except (OSError, ValueError) as exc:
            errors.append(dict(image=row["image"], error="image_unreadable", detail=str(exc)))
            valid = False
        hidden = record.gt_status == "missing" and record.split in PRIVATE_SPLITS
        if hidden:
            row["gt_availability"] = "PRIVATE_HIDDEN"
            count["private_hidden"] += 1
            warnings.append(dict(image=row["image"], warning="private_gt_unavailable"))
            valid = False
        elif record.gt_status in {"missing", "ambiguous"}:
            errors.append(dict(image=row["image"], error=record.gt_status))
            valid = False
        gt_paths = record.mask_candidates if record.is_normal else (record.mask_path,) if record.mask_path else ()
        for gt_path in gt_paths:
            try:
                with Image.open(gt_path) as im:
                    row["mask_hw"] = [im.height, im.width]
                    row["mask_requires_nearest_resize"] = row["mask_hw"] != row["native_hw"]
                    nonzero = bool(np.asarray(im.convert("L")).any())
                    if nonzero == record.is_normal:
                        errors.append(dict(image=row["image"], error="normal_gt_nonzero" if record.is_normal else "abnormal GT is empty"))
                        valid = False
            except (OSError, ValueError) as exc:
                errors.append(dict(image=row["image"], error="mask_unreadable", detail=str(exc)))
                valid = False
        row["pixel_evaluation_eligible"] = valid
        count["eligible"] += int(valid)
        rows.append(row)
    categories = sorted({r.category for r in records})
    splits = sorted({r.split for r in records})
    expected_categories = sorted(set(expected_categories or []))
    expected_splits = sorted(set(expected_splits or []))
    if not set(expected_splits).issubset(EXPECTED_SPLITS):
        raise ValueError(f"unknown expected splits: {sorted(set(expected_splits) - EXPECTED_SPLITS)}")
    missing_categories = sorted(set(expected_categories) - set(categories))
    missing_splits = sorted(set(expected_splits) - set(splits))
    missing_pairs = sorted(f"{cat}/{split}" for cat in (expected_categories or categories)
                           for split in expected_splits if f"{cat}/{split}" not in counts)
    errors.extend(dict(error="missing_category", category=cat) for cat in missing_categories)
    errors.extend(dict(error="missing_split", split=split) for split in missing_splits)
    errors.extend(dict(error="missing_category_split", path=pair) for pair in missing_pairs)
    if not records:
        errors.append(dict(error="no_recognized_images"))
    source_integrity = audit_source_manifest(root, images, source_manifest)
    errors.extend(source_integrity["errors"])
    coverage = dict(status=("FAIL" if missing_categories or missing_splits or missing_pairs else "PASS")
                    if expected_categories or expected_splits else "NOT_REQUESTED",
                    observed_categories=categories, observed_splits=splits,
                    expected_categories=expected_categories, expected_splits=expected_splits,
                    missing_categories=missing_categories, missing_splits=missing_splits,
                    missing_category_splits=missing_pairs)
    hidden_count = sum(c["private_hidden"] for c in counts.values())
    eligible = sum(r["pixel_evaluation_eligible"] for r in rows)
    pixel_errors = [e for e in errors[inventory_errors:] if e.get("error") in
                    {"missing", "ambiguous", "image_unreadable", "mask_unreadable", "abnormal GT is empty", "normal_gt_nonzero"}]
    inventory = dict(image_files=len(images) + sum(not m for _, m, _ in unrecognized),
                     mask_files=sum(map(len, masks.values())) + sum(m for _, m, _ in unrecognized),
                     recognized_images=len(images), recognized_masks=sum(map(len, masks.values())),
                     unrecognized=ignored, orphan_gt=orphan_gt, duplicate_gt=duplicate_gt,
                     duplicate_images=duplicate_images, status="FAIL" if errors else "PASS")
    return dict(schema="msila.dataset_audit.v3", root=str(root.resolve()), counts=counts, images=rows,
                inventory=inventory, coverage=coverage, source_integrity=source_integrity,
                pixel_evaluation=dict(status="FAIL" if pixel_errors or not records else
                                      "PARTIAL" if hidden_count else "PASS",
                                      eligible=eligible, ineligible=len(rows) - eligible,
                                      private_hidden=hidden_count), errors=errors, warnings=warnings,
                status="FAIL" if errors else "PASS")


def audit_cached_tile_padding(manifest, *, mask_root=None):
    """Check saved local-mask pixels outside native source bounds; never rewrite caches.

    JSON list or {samples: [...]}: image_id, mask_path, source_hw=[H,W],
    local_box=[x0,y0,x1,y1] (or geometry.local_box). Source geometry is declared
    evidence; this check alone does not verify cache producer provenance.
    """
    path = Path(manifest)
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("samples", []) if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("tile manifest must be a JSON list or an object with a samples list of objects")
    root = Path(mask_root) if mask_root else path.parent
    tiles, errors, blockers = [], [], []
    for index, row in enumerate(rows):
        identity = dict(index=index, image_id=row.get("image_id"))
        geometry = row.get("geometry", {})
        if not isinstance(geometry, dict):
            raise ValueError(f"tile {index}: geometry must be an object")
        box = row.get("local_box", geometry.get("local_box"))
        hw = row.get("source_hw")
        mask_path = row.get("mask_path")
        if not identity["image_id"] or hw is None or box is None or not mask_path:
            blockers.append(dict(**identity, error="missing_source_mapping"))
            continue
        try:
            if (len(hw) != 2 or len(box) != 4 or
                    any(type(x) is not int for x in [*hw, *box])):
                raise ValueError("source_hw/local_box must contain integer coordinates")
            height, width = hw
            x0, y0, x1, y1 = box
            if min(height, width) <= 0 or x1 <= x0 or y1 <= y0:
                raise ValueError("invalid source bounds/local_box")
            local = Path(mask_path)
            if not local.is_absolute():
                local = root / local
            if local.suffix.lower() == ".npy":
                mask = np.load(local, allow_pickle=False)
            else:
                with Image.open(local) as im:
                    mask = np.asarray(im.convert("L"))
            if mask.ndim != 2 or mask.shape != (y1 - y0, x1 - x0):
                raise ValueError("cached mask must be 2D with the unresized local_box dimensions")
            if not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
                raise ValueError("cached mask must be finite binary (0/1 or 0/255)")
            foreground = mask > 0
            valid = np.zeros(mask.shape, dtype=bool)
            # Intersection projected into local coordinates, including negative boxes.
            left, right = max(0, x0), min(width, x1)
            top, bottom = max(0, y0), min(height, y1)
            if left < right and top < bottom:
                valid[top - y0:bottom - y0, left - x0:right - x0] = True
            contaminated = int(np.count_nonzero(foreground & ~valid))
            tiles.append(dict(**identity, mask_path=str(local.resolve()), source_hw=hw,
                              local_box=box, padded_pixels=int(np.count_nonzero(~valid)),
                              padded_foreground_pixels=contaminated))
            if contaminated:
                errors.append(dict(**identity, error="foreground_in_padding", pixels=contaminated))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(dict(**identity, error="invalid_cached_mask", detail=str(exc)))
    if not rows:
        blockers.append(dict(error="empty_tile_manifest"))
    return dict(schema="msila.cached_tile_padding_audit.v1", checked=len(tiles), tiles=tiles,
                errors=errors, blockers=blockers,
                verification_scope="SAVED_MASK_PADDING_FROM_DECLARED_SOURCE_GEOMETRY",
                status="FAIL" if errors else "BLOCKED" if blockers else "PASS")
