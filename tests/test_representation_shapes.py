"""Real-batch Day-05 representation shape gate.

Project-relative path:
    msila/tests/test_representation_shapes.py

This is intentionally an integration/acceptance test, not a random-tensor unit
test. It reads ONE REAL MVTec AD 2 training image, builds the project's locked
Local/Context views, runs the REAL frozen DINOv3 backbone, performs Context->Local
alignment for R2, applies the existing 1x1 feature projection, selects R0/R1/R2,
and runs the same parameter-free MeanFusion.

Expected output artifact:
    msila/tests/artifacts/representation_shape_report.json

Required external asset:
    DINOV3_WEIGHTS=/path/to/the/checkpoint.pth

Optional overrides:
    DINOV3_REPO=/content/dinov3_repo
    MSILA_MVTEC_AD2_ROOT=/path/to/mvtec_ad_2
    MSILA_TEST_CATEGORY=fabric

The test deliberately FAILS (rather than silently skipping) when real data or
DINOv3 weights are unavailable. A skipped/random test would not satisfy the
Day-05 scientific gate requested for one real training batch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # .../msila
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multiview_transform import (  # noqa: E402
    MultiViewConfig,
    NestedMultiViewTransform,
)
from src.geometry.view_meta import build_view_meta_from_transform_meta  # noqa: E402
from src.models.context_alignment import ContextToLocalAligner  # noqa: E402
from src.models.dinov3_extractor import build_online_extractor  # noqa: E402
from src.models.feature_projection import SixFeatureProjection  # noqa: E402
from src.models.feature_selector import FeatureSelector  # noqa: E402
from src.models.mean_fusion import MeanFusion  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DAY05_CONFIG = ROOT / "configs" / "day05_representation.yaml"
REPORT_PATH = ROOT / "tests" / "artifacts" / "representation_shape_report.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        pytest.fail(f"Missing required project config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        pytest.fail(f"YAML root must be a mapping: {path}")
    return data


def _resolve_data_root(default_cfg: dict[str, Any]) -> Path:
    override = os.getenv("MSILA_MVTEC_AD2_ROOT")
    raw = override or default_cfg.get("data", {}).get("mvtec_ad2")
    if not raw:
        pytest.fail(
            "No MVTec AD 2 root. Set data.mvtec_ad2 in configs/default.yaml "
            "or MSILA_MVTEC_AD2_ROOT."
        )
    root = Path(str(raw)).expanduser()
    if not root.is_dir():
        pytest.fail(f"MVTec AD 2 root does not exist: {root}")
    return root


def _resolve_category_dir(data_root: Path, category: str) -> Path:
    direct = data_root / category
    if direct.is_dir():
        return direct
    matches = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower() == category.lower()]
    if len(matches) == 1:
        return matches[0]
    pytest.fail(f"Cannot resolve category {category!r} under {data_root}")


def _first_real_train_image(category_dir: Path, context_size: int) -> Path:
    train_root = category_dir / "train"
    if not train_root.is_dir():
        pytest.fail(f"Expected real training directory: {train_root}")

    candidates = sorted(
        p for p in train_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        pytest.fail(f"No real training images found under {train_root}")

    # The locked Context crop is 768x768. Choose the first image that can
    # support the real Local/Context geometry without hidden upscaling.
    for path in candidates:
        with Image.open(path) as im:
            w, h = im.size
        if h >= context_size and w >= context_size:
            return path

    pytest.fail(
        f"Found {len(candidates)} training images but none is >= "
        f"{context_size}x{context_size}; refusing hidden resize/upscale."
    )


def _resolve_dinov3_repo() -> Path:
    candidates: list[Path] = []
    if os.getenv("DINOV3_REPO"):
        candidates.append(Path(os.environ["DINOV3_REPO"]))
    candidates.extend([Path("/content/dinov3_repo"), Path("/content/dinov3")])

    for path in candidates:
        path = path.expanduser()
        if (path / "hubconf.py").is_file():
            return path

    pytest.fail(
        "Cannot find official DINOv3 repo. Set DINOV3_REPO to a checkout "
        "containing hubconf.py."
    )


def _resolve_dinov3_weights(data_root: Path, model_name: str) -> Path:
    explicit = os.getenv("DINOV3_WEIGHTS")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            pytest.fail(f"DINOV3_WEIGHTS does not exist: {path}")
        return path

    # Conservative local discovery only. Never pick arbitrarily when multiple
    # checkpoints are available because that would destroy reproducibility.
    search_roots = [ROOT / "weights", data_root.parent / "weights", Path("/content/checkpoints")]
    patterns = [f"*{model_name}*.pth", "*dinov3*vitb16*.pth"]
    found: list[Path] = []
    for base in search_roots:
        if not base.is_dir():
            continue
        for pattern in patterns:
            found.extend(base.rglob(pattern))
    unique = sorted({p.resolve() for p in found if p.is_file()})

    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        pytest.fail(
            "Multiple DINOv3 checkpoints found; set DINOV3_WEIGHTS explicitly:\n"
            + "\n".join(str(p) for p in unique[:20])
        )
    pytest.fail(
        "Cannot find DINOv3 weights. Set DINOV3_WEIGHTS=/path/to/checkpoint.pth."
    )


def _shape(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.shape]


def _finite_stats(xs: list[torch.Tensor]) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for x in xs:
        nan_count += int(torch.isnan(x).sum().item())
        inf_count += int(torch.isinf(x).sum().item())
    return nan_count, inf_count


@pytest.mark.integration
def test_representations_on_one_real_train_batch() -> None:
    default_cfg = _load_yaml(DEFAULT_CONFIG)
    day05_cfg = _load_yaml(DAY05_CONFIG)

    input_cfg = default_cfg["input"]
    backbone_cfg = default_cfg["backbone"]
    tile_size = int(input_cfg["tile_size"])
    context_size = int(input_cfg["context_size"])
    blocks = tuple(int(v) for v in backbone_cfg["feature_blocks"])
    model_name = str(backbone_cfg["name"])

    assert tile_size == 512
    assert context_size == 768
    assert blocks == (4, 8, 12)
    assert bool(backbone_cfg["frozen"]) is True

    projection_cfg = day05_cfg["locked"]["projection"]
    fusion_dim = int(projection_cfg["fusion_dim"])
    share_across_views = bool(projection_cfg["share_across_views"])

    data_root = _resolve_data_root(default_cfg)
    category = os.getenv("MSILA_TEST_CATEGORY", "fabric")
    category_dir = _resolve_category_dir(data_root, category)
    image_path = _first_real_train_image(category_dir, context_size)

    with Image.open(image_path) as im:
        image = im.convert("RGB")
        source_wh = [int(image.width), int(image.height)]

    # Deterministic debug crop from a REAL training image. This does not change
    # the locked FOV sizes; it only removes random crop location from the gate.
    transform = NestedMultiViewTransform(
        MultiViewConfig(
            local_size=tile_size,
            context_size=context_size,
            input_size=tile_size,
            sampling="center",
            normalize=True,
            validate_output=True,
        )
    )
    view = transform(image)
    x_local = view["x_local"].unsqueeze(0)
    x_context = view["x_context"].unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_local = x_local.to(device)
    x_context = x_context.to(device)

    repo = _resolve_dinov3_repo()
    weights = _resolve_dinov3_weights(data_root, model_name)
    extractor = build_online_extractor(
        repo_dir=repo,
        weights=weights,
        device=device,
        model_name=model_name,
        blocks=blocks,
        norm=True,
        check_finite=True,
    )
    assert extractor.backbone.training is False
    assert all(p.requires_grad is False for p in extractor.backbone.parameters())

    # REAL backbone inference; no random/synthetic feature tensor is used.
    online = extractor.extract_local_context(
        x_local,
        x_context,
        strategy="concat",
    )
    expected_online = {"L4", "L8", "L12", "C4", "C8", "C12"}
    assert set(online) == expected_online
    assert all(bool(torch.isfinite(v).all()) for v in online.values())

    local = {f"L{b}": online[f"L{b}"] for b in blocks}
    context = {f"C{b}": online[f"C{b}"] for b in blocks}
    raw_h, raw_w = local["L4"].shape[-2:]

    geometry = build_view_meta_from_transform_meta(
        view["meta"],
        local_input_hw=(tile_size, tile_size),
        context_input_hw=(tile_size, tile_size),
        validate=True,
    )
    aligner = ContextToLocalAligner(
        check_finite=True,
        check_bounds=True,
    ).to(device)
    aligned_context = aligner(
        context,
        geometry,
        target_hw=(int(raw_h), int(raw_w)),
    )

    # Explicit scientific gate: aligned Context must lie on the Local lattice.
    for block in blocks:
        assert aligned_context[f"C{block}_to_L"].shape[0] == local[f"L{block}"].shape[0]
        assert aligned_context[f"C{block}_to_L"].shape[-2:] == local[f"L{block}"].shape[-2:]
        assert bool(torch.isfinite(aligned_context[f"C{block}_to_L"].all()))

    in_channels = int(local["L4"].shape[1])
    assert all(int(local[f"L{b}"].shape[1]) == in_channels for b in blocks)
    assert all(int(context[f"C{b}"].shape[1]) == in_channels for b in blocks)

    # ONE shared projection object and ONE shared fusion object are used for all
    # three candidates. Only source_keys changes.
    projection = SixFeatureProjection(
        in_channels=in_channels,
        fusion_dim=fusion_dim,
        blocks=blocks,
        share_across_views=share_across_views,
        check_finite=True,
    ).to(device)
    fusion = MeanFusion(validate=True, return_weights=True).to(device)

    report: dict[str, Any] = {
        "gate": "MS-ILA Day05 real-batch representation shape gate",
        "status": "PASS",
        "real_train_image": str(image_path),
        "category": category,
        "source_image_wh": source_wh,
        "device": str(device),
        "default_config": "msila/configs/default.yaml",
        "day05_config": "msila/configs/day05_representation.yaml",
        "backbone": {
            "name": model_name,
            "frozen": True,
            "blocks": list(blocks),
            "repo": str(repo),
            "weights": str(weights),
        },
        "input": {
            "local_source_fov": [tile_size, tile_size],
            "context_source_fov": [context_size, context_size],
            "local_model_input": _shape(x_local),
            "context_model_input": _shape(x_context),
        },
        "raw_backbone_shapes": {k: _shape(v) for k, v in online.items()},
        "aligned_context_shapes": {k: _shape(v) for k, v in aligned_context.items()},
        "projection": {
            "in_channels": in_channels,
            "fusion_dim": fusion_dim,
            "share_across_views": share_across_views,
        },
        "pipeline_order": [
            "real_train_image",
            "NestedMultiViewTransform",
            "frozen_DINOv3",
            "ContextToLocalAligner_for_R2",
            "SixFeatureProjection.project_sources",
            "FeatureSelector",
            "MeanFusion",
        ],
        "candidates": {},
    }

    expected_counts = {"R0": 1, "R1": 3, "R2": 6}
    for candidate_id, expected_count in expected_counts.items():
        selector = FeatureSelector.from_config(day05_cfg, candidate_id)

        projected = projection.project_sources(
            local,
            aligned_context if candidate_id == "R2" else None,
            source_keys=selector.source_keys,
        )
        selected = selector(projected)
        fused, weights_out = fusion(selected)

        assert len(selected) == expected_count
        assert len(projected) == expected_count
        assert tuple(projected.keys()) == selector.source_keys
        assert all(x.ndim == 4 for x in selected)

        reference_shape = selected[0].shape
        assert int(reference_shape[1]) == fusion_dim
        assert all(x.shape == reference_shape for x in selected)
        assert all(x.dtype == selected[0].dtype for x in selected)
        assert all(x.device == selected[0].device for x in selected)
        assert fused.shape == reference_shape
        assert weights_out.shape == (expected_count,)
        assert torch.allclose(
            weights_out,
            torch.full_like(weights_out, 1.0 / expected_count),
        )

        nan_count, inf_count = _finite_stats(selected + [fused])
        assert nan_count == 0
        assert inf_count == 0

        if candidate_id == "R0":
            # Exact scientific requirement: one source means identity fusion.
            assert fused is selected[0]
            assert float(weights_out[0]) == pytest.approx(1.0)

        report["candidates"][candidate_id] = {
            "mode": selector.mode,
            "sources": list(selector.source_keys),
            "num_sources": len(selected),
            "source_shapes": {
                key: _shape(tensor)
                for key, tensor in zip(selector.source_keys, selected)
            },
            "channel_d_identical": len({int(x.shape[1]) for x in selected}) == 1,
            "spatial_hw_identical": len({tuple(x.shape[-2:]) for x in selected}) == 1,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "fused_shape": _shape(fused),
            "mean_weights": [float(v) for v in weights_out.detach().cpu()],
            "context_spatial_alignment": (
                "PASS" if candidate_id == "R2" else "N/A"
            ),
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REPORT_PATH)

    assert REPORT_PATH.is_file()
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"
    assert saved["candidates"]["R0"]["num_sources"] == 1
    assert saved["candidates"]["R1"]["num_sources"] == 3
    assert saved["candidates"]["R2"]["num_sources"] == 6
