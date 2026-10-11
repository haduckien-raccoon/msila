"""D7 geometry QA only; CPU fixtures never stand in for eight real categories.

Run tests with pytest. Export QA and examples with:
    python -m tests.test_g2_tv2_context_geometry --output-root outputs/G2/D7
Optionally pass --data-root (or MVTEC_AD2_ROOT) for real TRAIN/good sources.
No model, training, metric or synthetic protocol is implemented here.
"""
from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch
import torch.nn.functional as F

from src.data.multiview_transform import MultiViewConfig, NestedMultiViewTransform
from src.data.tiling import crop_with_padding, extract_local_context, generate_tile_records, stitch_tiles_hann
from src.geometry.view_meta import build_padded_view_meta, build_view_meta, transform_points_xy
from src.models.context_alignment import ContextToLocalAligner

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
LIMITS = dict(point_error_px=1e-9, affine_alignment_error_px=1e-8, mask_error=0.,
              local_image_error=0., context_bilinear_error=2e-6)
CASES = {
    "interior": (1024, 1200),
    "top_left": (777, 931),
    "bottom_right": (777, 931),
    "portrait_boundary": (931, 777),
    "small_image": (73, 101),
    "one_pixel": (1, 1),
}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def make_fixture(name):
    h, w = CASES[name]
    records = generate_tile_records(h, w)
    if name == "interior":
        record = next(r for r in records if r.context_xyxy[0] >= 0 and r.context_xyxy[1] >= 0
                      and r.context_xyxy[2] <= w and r.context_xyxy[3] <= h)
    else:
        record = records[0] if name == "top_left" else records[-1]
    x0, y0, x1, y1 = record.local_xyxy
    # One native anomaly, with independently known pixel support.
    if name in {"bottom_right", "portrait_boundary", "small_image", "one_pixel"}:
        ax, ay = min(w, x1)-min(2, w), min(h, y1)-min(2, h)
    elif name == "top_left":
        ax, ay = x0, y0
    else:
        ax, ay = x0+123, y0+211
    mask = torch.zeros(1, h, w)
    mask[:, ay:ay+2, ax:ax+2] = 1
    image = torch.full((3, h, w), .2)
    image[:, mask[0].bool()] = .8
    return image, mask, record


def exact_local_mask(mask, record):
    """Independent native-coordinate oracle, with zero labels beyond native bounds."""
    h, w = mask.shape[-2:]
    x0, y0, x1, y1 = record.local_xyxy
    expected = torch.zeros(1, 512, 512)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    expected[:, ya-y0:yb-y0, xa-x0:xb-x0] = mask[:, ya:yb, xa:xb]
    return expected


def coordinate_field(box, hw):
    """Feature cell centers measured in native pixel-edge coordinates."""
    x0, y0, x1, y1 = box
    h, w = hw
    x = x0+(torch.arange(w, dtype=torch.float64)+.5)*(x1-x0)/w
    y = y0+(torch.arange(h, dtype=torch.float64)+.5)*(y1-y0)/h
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy))[None]


def analytic_context_pixels(image, context, box):
    """Independent four-neighbor oracle for Context centers inside native bounds.

    Outside-native padding is tested separately against known reflected/replicated
    pixel arrays. This oracle uses pixel-center arithmetic, never interpolate().
    """
    h, w = image.shape[-2:]
    x0, y0, _, _ = box
    x = x0+1.5*(torch.arange(512, dtype=torch.float64)+.5)-.5
    y = y0+1.5*(torch.arange(512, dtype=torch.float64)+.5)-.5
    j, i = torch.where((x >= 0) & (x <= w-1))[0], torch.where((y >= 0) & (y <= h-1))[0]
    if not len(i) or not len(j):
        return dict(max_error=0., checked_centers=0)
    x, y = x[j], y[i]
    xa, ya = x.floor().long(), y.floor().long()
    xb, yb = (xa+1).clamp_max(w-1), (ya+1).clamp_max(h-1)
    dx, dy = (x-xa)[None, :], (y-ya)[:, None]
    expected = (image[:, ya[:, None], xa].double()*(1-dx)*(1-dy)
                + image[:, ya[:, None], xb].double()*dx*(1-dy)
                + image[:, yb[:, None], xa].double()*(1-dx)*dy
                + image[:, yb[:, None], xb].double()*dx*dy)
    actual = context[:, i[:, None], j]
    return dict(max_error=float((actual-expected).abs().max()), checked_centers=len(i)*len(j))


def quantify_pair(image, mask, record, *, feature_hw=(32, 32), source_meta=None):
    tf = NestedMultiViewTransform(MultiViewConfig(normalize=False))
    sample = tf.from_tile(image, record, mask=mask, source_meta=source_meta or {"sample_id": "one_native_source"})
    geometry = sample["view_meta"]["geometry"]
    expected_mask = exact_local_mask(mask, record)
    mask_error = float((sample["mask"]-expected_mask).abs().max())
    assert mask_error <= LIMITS["mask_error"]
    assert sample["image"].shape == sample["context"].shape == (3, 512, 512)
    assert sample["view_meta"]["local_sample_id"] == sample["view_meta"]["context_sample_id"]
    assert torch.isfinite(sample["image"]).all() and torch.isfinite(sample["context"]).all()
    h, w = image.shape[-2:]
    x0, y0, x1, y1 = record.local_xyxy
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    local_error = float((sample["image"][:, ya-y0:yb-y0, xa-x0:xb-x0]-image[:, ya:yb, xa:xb]).abs().max())
    context_pixels = analytic_context_pixels(image, sample["context"], record.context_xyxy)
    assert local_error <= LIMITS["local_image_error"]
    assert context_pixels["max_error"] < LIMITS["context_bilinear_error"]
    # Source points include crop pixel centers and actual anomalous pixel centers.
    x0, y0, x1, y1 = record.local_xyxy
    cx0, cy0, _, _ = record.context_xyxy
    points = torch.tensor([[x0+.5, y0+.5], [x1-.5, y1-.5], [x0+123.5, y0+211.5]], dtype=torch.float64)
    yy, xx = torch.where(mask[0] != 0)
    if len(xx):
        points = torch.cat((points, torch.stack((xx.double()+.5, yy.double()+.5), dim=1)))
    expected_local = points-torch.tensor([x0, y0])
    expected_context = (points-torch.tensor([cx0, cy0]))*(2/3)
    local = transform_points_xy(points, geometry["native_to_local"])
    context = transform_points_xy(points, geometry["native_to_context"])
    back = transform_points_xy(context, geometry["context_to_native"])
    point_error = max(float((local-expected_local).abs().max()),
                      float((context-expected_context).abs().max()), float((back-points).abs().max()))
    assert point_error < LIMITS["point_error_px"]
    # Independent native-coordinate oracle, rather than deriving the expectation
    # from the aligner's own sampling grid.
    field = coordinate_field(record.context_xyxy, feature_hw)
    expected = coordinate_field(record.local_xyxy, feature_hw)
    aligner = ContextToLocalAligner(check_finite=True)
    aligned = aligner({"C4": field, "C8": field+1., "C12": field-2.}, geometry)
    alignment_error = max(float((aligned[key]-(expected+offset)).abs().max())
                          for key, offset in (("C4_to_L", 0.), ("C8_to_L", 1.), ("C12_to_L", -2.)))
    assert alignment_error < LIMITS["affine_alignment_error_px"]
    assert aligner.align_corners is False
    return sample, dict(status="PASS", native_hw=list(image.shape[-2:]),
                        local_native_xyxy=list(record.local_xyxy), context_native_xyxy=list(record.context_xyxy),
                        padding_ltrb=geometry["padding_ltrb"].tolist(), point_error_px=point_error,
                        affine_alignment_error_px=alignment_error, local_mask_error=mask_error,
                        local_image_error=local_error, context_bilinear_error=context_pixels["max_error"],
                        checked_context_centers=context_pixels["checked_centers"],
                        local_positive_pixels=int(sample["mask"].sum()),
                        native_positive_pixels=int(mask.sum()), feature_hw=list(feature_hw))


@pytest.mark.parametrize("name", CASES)
def test_known_native_points_mask_and_three_context_levels(name):
    image, mask, record = make_fixture(name)
    _, result = quantify_pair(image, mask, record)
    assert result["local_positive_pixels"] == (1 if name == "one_pixel" else 4)


@pytest.mark.parametrize("hw", [(16, 24), (31, 47), (64, 64)])
def test_alignment_on_different_feature_grids(hw):
    image, mask, record = make_fixture("bottom_right")
    quantify_pair(image, mask, record, feature_hw=hw)


def test_float32_feature_alignment_against_native_coordinate_oracle():
    image, mask, record = make_fixture("portrait_boundary")
    sample = NestedMultiViewTransform(MultiViewConfig(normalize=False)).from_tile(image, record, mask=mask)
    field = coordinate_field(record.context_xyxy, (32, 32)).float()
    expected = coordinate_field(record.local_xyxy, (32, 32)).float()
    aligned = ContextToLocalAligner()({"C4": field, "C8": field+1., "C12": field-2.},
                                    sample["view_meta"]["geometry"])
    for key, offset in (("C4_to_L", 0.), ("C8_to_L", 1.), ("C12_to_L", -2.)):
        torch.testing.assert_close(aligned[key], expected+offset, atol=2e-4, rtol=0)


def test_half_pixel_context_resize_matches_analytic_bilinear_weights():
    image, mask, record = make_fixture("interior")
    tf = NestedMultiViewTransform(MultiViewConfig(normalize=False))
    sample = tf.from_tile(image, record, mask=mask)
    y, x = torch.where(mask[0] != 0)
    ax, ay = int(x.min()), int(y.min())
    cx0, cy0, _, _ = record.context_xyxy
    center_x, center_y = (ax+1-cx0)*(2/3), (ay+1-cy0)*(2/3)
    for i in range(int(center_y)-2, int(center_y)+3):
        for j in range(int(center_x)-2, int(center_x)+3):
            # align_corners=False maps output centers to native pixel indices.
            native_x = cx0+1.5*(j+.5)-.5
            native_y = cy0+1.5*(i+.5)-.5
            wx = sum(max(0., 1-abs(native_x-p)) for p in (ax, ax+1))
            wy = sum(max(0., 1-abs(native_y-p)) for p in (ay, ay+1))
            assert float(sample["context"][0, i, j]) == pytest.approx(.2+.6*wx*wy, abs=1e-6)
    # Swapping in align_corners=True must violate that known coordinate oracle.
    source = crop_with_padding(image, record.context_xyxy)
    wrong = F.interpolate(source[None], size=(512, 512), mode="bilinear", align_corners=True)[0]
    assert float((wrong-sample["context"]).abs().max()) > .01


def test_reflection_and_tiny_image_replicate_padding_have_known_pixels():
    image = torch.arange(6).reshape(1, 2, 3).float()
    expected = torch.tensor([[[4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                              [4, 3, 4, 5, 4], [1, 0, 1, 2, 1]]]).float()
    assert torch.equal(crop_with_padding(image, (-1, -1, 4, 3)), expected)
    pixel = torch.tensor([[[.7]]])
    assert torch.equal(crop_with_padding(pixel, (-128, -128, 640, 640)), torch.full((1, 768, 768), .7))


def test_mask_padding_has_no_reflected_or_replicated_positive_labels():
    image, mask, record = make_fixture("small_image")
    sample, _ = quantify_pair(image, mask, record)
    assert sample["mask"][0, 71:73, 99:101].sum() == 4
    assert not sample["mask"][0, 73:].any() and not sample["mask"][0, :, 101:].any()
    all_records = generate_tile_records(*mask.shape[-2:])
    restored = stitch_tiles_hann([NestedMultiViewTransform(MultiViewConfig(normalize=False))
                                 .from_tile(image, r, mask=mask)["mask"][0] for r in all_records],
                                all_records, tuple(mask.shape[-2:]))
    torch.testing.assert_close(restored, mask[0], atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", ["top_left", "bottom_right", "portrait_boundary"])
def test_tiny_mask_stitches_back_to_exact_native_support(name):
    image, mask, _ = make_fixture(name)
    records = generate_tile_records(*mask.shape[-2:])
    tf = NestedMultiViewTransform(MultiViewConfig(normalize=False))
    tiles = [tf.from_tile(image, record, mask=mask)["mask"][0] for record in records]
    restored = stitch_tiles_hann(tiles, records, tuple(mask.shape[-2:]))
    torch.testing.assert_close(restored, mask[0], atol=1e-6, rtol=0)


def test_g2_normalization_and_inference_without_mask():
    image, mask, record = make_fixture("top_left")
    raw = NestedMultiViewTransform(MultiViewConfig(normalize=False)).from_tile(image, record, mask=mask)
    normalized = NestedMultiViewTransform().from_tile(image, record)
    mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
    for key in ("image", "context"):
        torch.testing.assert_close(normalized[key], (raw[key]-mean)/std)
    assert "mask" not in normalized


@pytest.mark.parametrize("case", ["nonbinary_mask", "mask_shape", "nan_mask", "offcenter", "center_metadata"])
def test_invalid_pair_fails_fast(case):
    image, mask, record = make_fixture("top_left")
    if case == "nonbinary_mask":
        mask[0, 0, 0] = .3
    elif case == "nan_mask":
        mask[0, 0, 0] = float("nan")
    elif case == "mask_shape":
        mask = mask[:, :-1]
    elif case == "offcenter":
        c = record.context_xyxy
        record = replace(record, context_xyxy=(c[0]+1, c[1], c[2]+1, c[3]))
    else:
        record = replace(record, center_xy=(0., 0.))
    with pytest.raises(ValueError, match="mask|concentric"):
        NestedMultiViewTransform().from_tile(image, record, mask=mask)


def test_strict_geometry_and_align_corners_true_stay_rejected():
    image, _, record = make_fixture("top_left")
    with pytest.raises(ValueError, match="outside source"):
        build_view_meta(source_hw=image.shape[-2:], local_box_xyxy=record.local_xyxy,
                        context_box_xyxy=record.context_xyxy)
    with pytest.raises(ValueError, match="intersect"):
        build_padded_view_meta(source_hw=(100, 100), local_box_xyxy=(200, 200, 712, 712),
                               context_box_xyxy=(72, 72, 840, 840))
    with pytest.raises(ValueError, match="align_corners must be False"):
        ContextToLocalAligner(align_corners=True)


def save_examples(sample, directory):
    directory.mkdir(parents=True, exist_ok=True)
    def rgb(tensor):
        return Image.fromarray((tensor.clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8))
    local, context = rgb(sample["image"]), rgb(sample["context"])
    mask = Image.fromarray((sample["mask"][0].numpy()*255).astype(np.uint8))
    local.save(directory/"local.png")
    context.save(directory/"context.png")
    mask.save(directory/"mask.png")
    context_overlay = context.copy()
    box = sample["view_meta"]["geometry"]["local_box_in_context_input_xyxy"].tolist()
    ImageDraw.Draw(context_overlay).rectangle(box, outline="red", width=2)
    y, x = torch.where(sample["mask"][0] != 0)
    left, top = max(0, int(x.float().mean())-8), max(0, int(y.float().mean())-8)
    zoom = local.crop((left, top, min(512, left+17), min(512, top+17)))
    zoom = zoom.resize((512, 512), resample=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (2048, 542), "white")
    for index, (image, label) in enumerate(((local, "Local 512"), (context_overlay, "Context 768 -> 512; Local FOV in red"),
                                         (mask.convert("RGB"), "Exact Local mask; padding labels = 0"),
                                         (zoom, f"Tiny Local zoom; crop origin ({left},{top}); nearest"))):
        canvas.paste(image, (index*512, 30))
        ImageDraw.Draw(canvas).text((index*512+8, 8), label, fill="black")
    canvas.save(directory/"comparison.png")
    return [str(directory/name) for name in ("local.png", "context.png", "mask.png", "comparison.png")]


def run_context_qa(output_root, *, data_root=None, seed=42, images_per_category=1):
    """Use the same numerical checks as tests; report each real category explicitly."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fixture_cases = {}
    for name in CASES:
        try:
            sample, result = quantify_pair(*make_fixture(name))
            if name in {"top_left", "small_image", "interior"}:
                result["examples"] = save_examples(sample, output/"fixtures"/name)
        except (AssertionError, ValueError, RuntimeError) as exc:
            result = dict(status="FAIL", reason=str(exc))
        fixture_cases[name] = result
    categories = {}
    for category in CATEGORIES:
        record = dict(status="NOT RUN", reason="No dataset configured; fixture QA is separate", samples=[])
        if data_root is not None:
            try:
                # Reuse existing public data/synthetic APIs. No loader or anomaly
                # algorithm is reimplemented by the geometry QA.
                from src.data.loader import scan_mvtec_ad2
                from src.data.synthetic_anomaly import NativeTinyDefectGenerator
                from tests.test_full_scale_contracts import protocol
                sources = [r for r in scan_mvtec_ad2(data_root, split="train", categories=[category])
                           if r.defect_type == "good"]
                if not sources:
                    record.update(status="BLOCKED", reason=f"Missing {category}/TRAIN/good under {data_root}")
                else:
                    generator = NativeTinyDefectGenerator(protocol())
                    for source_index, source in enumerate(sources[:images_per_category]):
                        path = Path(source.image_path)
                        with Image.open(path) as opened:
                            image = torch.from_numpy(np.array(opened.convert("RGB"), copy=True)).permute(2, 0, 1).float()/255
                        for placement_index, placement in enumerate(("interior", "image_boundary")):
                            # Exactly one native anomaly per QA sample. Both views
                            # consume that sample; nothing is generated per view.
                            sample_seed = seed+source_index*2+placement_index
                            native = generator(image, seed=sample_seed, defect_type="pinhole",
                                               size_bin=protocol()["dev_tiny_bins"][0], placement=placement)
                            records = generate_tile_records(*image.shape[-2:])
                            positive = [r for r in records if exact_local_mask(native.mask, r).any()]
                            if not positive:
                                raise AssertionError("Native tiny anomaly has no supervised Local tile")
                            pair, metrics = quantify_pair(native.image, native.mask, positive[0], source_meta={
                                "sample_id": f"{category}:{source_index}:{placement}:{sample_seed}",
                                "category": category, "source": str(path), "split": "train",
                                "synthetic": native.metadata})
                            metrics.update(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                           seed=sample_seed, synthetic=native.metadata, placement=placement,
                                           generation_scope="one anomaly per native sample; shared by Local/Context")
                            if source_index == 0:
                                metrics["examples"] = save_examples(pair, output/"dataset"/category/placement)
                            record["samples"].append(metrics)
                    record.update(status="PASS", reason="", checked_source_images=min(len(sources), images_per_category))
            except FileNotFoundError as exc:
                record.update(status="BLOCKED", reason=str(exc))
            except (OSError, AssertionError, ValueError, RuntimeError) as exc:
                record.update(status="FAIL", reason=str(exc))
        categories[category] = record
    fixture_pass = sum(r["status"] == "PASS" for r in fixture_cases.values())
    dataset_pass = sum(r["status"] == "PASS" for r in categories.values())
    implementation = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
        "src/data/tiling.py", "src/data/multiview_transform.py", "src/geometry/view_meta.py",
        "src/models/context_alignment.py", "tests/test_g2_tv2_context_geometry.py", "docs/G2_CONTRACT.md")}
    report = dict(task="D7-TV2", scope="Local/Context geometry only; no E4 training or AU-PRO acceptance",
                  implementation_sha256=implementation,
                  seed=seed, device="cpu", git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  limits=LIMITS, geometry_fixture_precision="float64 coordinate field; RGB/resize float32",
                  contract=dict(local_size=512, context_size=768, input_size=512,
                  align_corners=False, mask_padding="constant_zero", geometry_source_frame="padded_native"),
                  fixture_qa=dict(status="PASS" if fixture_pass == len(CASES) else "FAIL",
                                  passed=fixture_pass, total=len(CASES), cases=fixture_cases),
                  dataset_qa=dict(status="NOT RUN" if data_root is None else
                                  ("FAIL" if any(r["status"] == "FAIL" for r in categories.values()) else
                                   ("PASS" if dataset_pass == 8 else "BLOCKED")),
                                  data_root=str(data_root) if data_root is not None else None,
                                  passed_categories=dataset_pass, expected_categories=8, complete=dataset_pass == 8,
                                  images_per_category=images_per_category, categories=categories))
    (output/"context_qa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return report


def test_report_separates_fixture_pass_from_all8_not_run(tmp_path):
    report = run_context_qa(tmp_path)
    assert report["fixture_qa"]["status"] == "PASS" and report["fixture_qa"]["passed"] == len(CASES)
    assert report["dataset_qa"]["status"] == "NOT RUN" and report["dataset_qa"]["passed_categories"] == 0
    assert not report["dataset_qa"]["complete"]
    assert list(report["dataset_qa"]["categories"]) == list(CATEGORIES)
    assert all(r["status"] == "NOT RUN" for r in report["dataset_qa"]["categories"].values())
    assert len(list(tmp_path.glob("fixtures/*/comparison.png"))) == 3
    assert json.loads((tmp_path/"context_qa.json").read_text()) == report


def test_real_source_qa_api_one_native_generation_per_pair_and_partial_coverage(tmp_path, monkeypatch):
    """Temporary source fixture validates the optional dataset path, not real coverage."""
    from types import SimpleNamespace
    from src.data import loader
    from src.data.synthetic_anomaly import NativeTinyDefectGenerator
    source = tmp_path/"data/rice/TRAIN/good/unit.png"
    source.parent.mkdir(parents=True)
    Image.fromarray(np.full((73, 101, 3), 100, dtype=np.uint8)).save(source)
    calls, generations = [], []
    def scan(root, *, split, categories):
        calls.append((split, categories))
        return [SimpleNamespace(image_path=str(source), defect_type="good")] if categories == ["rice"] else []
    monkeypatch.setattr(loader, "scan_mvtec_ad2", scan)
    original = NativeTinyDefectGenerator.__call__
    def count(self, image, **kwargs):
        generations.append(kwargs)
        return original(self, image, **kwargs)
    monkeypatch.setattr(NativeTinyDefectGenerator, "__call__", count)
    report = run_context_qa(tmp_path/"report", data_root=tmp_path/"data")
    assert len(generations) == 2  # interior and boundary; never twice for two views
    assert [r["placement"] for r in generations] == ["interior", "image_boundary"]
    assert calls == [("train", [category]) for category in CATEGORIES]
    assert report["dataset_qa"]["passed_categories"] == 1 and not report["dataset_qa"]["complete"]
    assert report["dataset_qa"]["status"] == "BLOCKED"
    assert report["dataset_qa"]["categories"]["rice"]["status"] == "PASS"
    samples = report["dataset_qa"]["categories"]["rice"]["samples"]
    assert len(samples) == 2 and all(4 <= s["native_positive_pixels"] <= 32 for s in samples)
    assert all(s["local_positive_pixels"] == s["native_positive_pixels"] for s in samples)
    assert all(s["status"] == "BLOCKED" for c,s in report["dataset_qa"]["categories"].items() if c != "rice")


def test_missing_dataset_reports_all8_blocked_without_fabricated_samples(tmp_path):
    report = run_context_qa(tmp_path/"report", data_root=tmp_path/"missing")
    assert report["fixture_qa"]["status"] == "PASS"
    assert report["dataset_qa"]["passed_categories"] == 0 and report["dataset_qa"]["status"] == "BLOCKED"
    assert all(c["status"] == "BLOCKED" and c["samples"] == [] for c in report["dataset_qa"]["categories"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("MVTEC_AD2_ROOT"))
    parser.add_argument("--output-root", default="outputs/G2/D7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-per-category", type=int, default=1)
    args = parser.parse_args()
    if args.images_per_category < 1:
        parser.error("--images-per-category must be positive")
    torch.set_num_threads(1)
    report = run_context_qa(args.output_root, data_root=args.data_root, seed=args.seed,
                            images_per_category=args.images_per_category)
    print(json.dumps({k: v for k, v in report.items() if k not in {"fixture_qa", "dataset_qa"}}, indent=2))
    print("Fixture:", report["fixture_qa"]["status"], report["fixture_qa"]["passed"], "/", report["fixture_qa"]["total"])
    print("Dataset:", report["dataset_qa"]["status"], report["dataset_qa"]["passed_categories"], "/8")
    return 1 if "FAIL" in {report["fixture_qa"]["status"], report["dataset_qa"]["status"]} else (
        2 if args.data_root and not report["dataset_qa"]["complete"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())

