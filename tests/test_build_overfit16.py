from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.tools.build_overfit16 import (
    ANOMALY_SAMPLES,
    NORMAL_SAMPLES,
    TOTAL_SAMPLES,
    SourcePoolError,
    build_overfit16,
    verify_overfit16,
)


def _make_normal_pool(root: Path, n: int = 24) -> None:
    root.mkdir(parents=True, exist_ok=True)
    h, w = 72, 88
    yy, xx = np.mgrid[0:h, 0:w]

    for i in range(n):
        # Deterministic non-constant RGB images. PNG keeps exact decoded pixels.
        r = (xx * 3 + i * 17) % 256
        g = (yy * 5 + i * 11) % 256
        b = ((xx + yy) * 2 + i * 7) % 256
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(root / f"normal_{i:03d}.png")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8).copy()


def _manifest_without_root_specific_state(path: Path) -> dict:
    # Manifest deliberately stores no absolute output/source root, so it should
    # be byte-equivalent across two deterministic builds from the same pool.
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_build_exact_locked_composition_and_verify(tmp_path: Path):
    normal_root = tmp_path / "normal"
    out = tmp_path / "overfit16"
    _make_normal_pool(normal_root)

    manifest = build_overfit16(
        normal_root,
        out,
        base_seed=2026,
        category="fabric",
    )

    assert len(manifest["samples"]) == TOTAL_SAMPLES
    assert sum(not x["is_anomaly"] for x in manifest["samples"]) == NORMAL_SAMPLES
    assert sum(x["is_anomaly"] for x in manifest["samples"]) == ANOMALY_SAMPLES
    assert len({x["source_relpath"] for x in manifest["samples"]}) == TOTAL_SAMPLES
    assert len({x["sample_id"] for x in manifest["samples"]}) == TOTAL_SAMPLES

    report = verify_overfit16(out)
    assert report["status"] == "PASS"
    assert report["n_samples"] == 16
    assert report["n_normal"] == 8
    assert report["n_anomaly"] == 8


def test_same_pool_and_seed_reproduce_exact_pixels_and_manifest(tmp_path: Path):
    normal_root = tmp_path / "normal"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _make_normal_pool(normal_root)

    build_overfit16(normal_root, out_a, base_seed=77, selection_seed=91)
    build_overfit16(normal_root, out_b, base_seed=77, selection_seed=91)

    ma = _manifest_without_root_specific_state(out_a)
    mb = _manifest_without_root_specific_state(out_b)
    assert ma == mb

    for ra, rb in zip(ma["samples"], mb["samples"]):
        assert np.array_equal(
            _load_rgb(out_a / ra["image_path"]),
            _load_rgb(out_b / rb["image_path"]),
        )
        assert np.array_equal(
            _load_mask(out_a / ra["mask_path"]),
            _load_mask(out_b / rb["mask_path"]),
        )


def test_normal_and_anomaly_persisted_alignment_contract(tmp_path: Path):
    normal_root = tmp_path / "normal"
    out = tmp_path / "overfit16"
    _make_normal_pool(normal_root)
    manifest = build_overfit16(normal_root, out, base_seed=314)

    for record in manifest["samples"]:
        source = _load_rgb(normal_root / record["source_relpath"])
        image = _load_rgb(out / record["image_path"])
        mask = _load_mask(out / record["mask_path"])
        mask_bool = mask > 0

        assert image.shape == source.shape
        assert mask.shape == source.shape[:2]
        assert set(np.unique(mask)).issubset({0, 255})

        if record["is_anomaly"]:
            assert mask_bool.any()
            assert np.array_equal(image[~mask_bool], source[~mask_bool])
            assert not np.array_equal(image[mask_bool], source[mask_bool])
            assert record["synthetic"]["is_anomaly"] is True
        else:
            assert not mask_bool.any()
            assert np.array_equal(image, source)
            assert record["synthetic"]["is_anomaly"] is False


def test_requires_at_least_16_distinct_normal_sources(tmp_path: Path):
    normal_root = tmp_path / "normal"
    _make_normal_pool(normal_root, n=15)

    with pytest.raises(SourcePoolError, match="at least 16"):
        build_overfit16(normal_root, tmp_path / "out")


def test_existing_output_requires_explicit_overwrite(tmp_path: Path):
    normal_root = tmp_path / "normal"
    out = tmp_path / "out"
    _make_normal_pool(normal_root)
    build_overfit16(normal_root, out, base_seed=1)

    with pytest.raises(FileExistsError):
        build_overfit16(normal_root, out, base_seed=1)

    # Explicit overwrite is allowed and remains valid.
    build_overfit16(normal_root, out, base_seed=1, overwrite=True)
    assert verify_overfit16(out)["status"] == "PASS"
