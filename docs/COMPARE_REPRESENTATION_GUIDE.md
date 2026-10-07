# TV2 — Diagnostic comparison: `src/analysis/compare_representation.py`

## 1. File mới, tự copy/paste

Bạn **tự tạo file mới `src/analysis/compare_representation.py`** rồi dán toàn bộ code ở cuối tài liệu. Repo đặt package dưới `src/`, nên đây là vị trí tương ứng với `analysis/compare_representation.py` trong task.

Không chèn vào hoặc thay file evaluator, tiny, boundary hay region_stats. Bản bàn giao chỉ tạo artifact ở `/tmp/msila_tv2_compare_delivery/`; không tự sửa/tạo/xóa file trong project.

Code reuse helper đã có:

- `boundary_analysis.STRUCTURE`, `boundary_zone`, `validate_inputs`;
- `region_stats.read_json`, `validate_tiny_protocol`;
- `tiny_analysis.binary_mask`, `load_array`.

Cần các module Task 4–6 ở đúng package, cùng NumPy, SciPy, Pillow, Matplotlib. Các dependency đang có trong môi trường kiểm tra; không cần thay evaluator.

## 2. Output chính và range chung

Mỗi figure là **một sample duy nhất** với layout:

```text
Image | GT | R0 | R1 | R2
```

Ba anomaly maps dùng:

```python
shared_norm = Normalize(vmin=0.0, vmax=1.0)
```

Cùng object normalization, cùng colormap `magma`, cùng colorbar và cùng range `[0,1]`. Range này giữ nguyên giữa candidate, sample và bản full/zoom.

Không có per-candidate/per-image min-max normalization, scaling theo min/max ảnh, sigmoid lại, clipping để sửa score lỗi hay threshold anomaly map. Map phải là probability 2-D, finite, `[0,1]`, native H×W. Logits hoặc score ngoài range bị từ chối.

GT dùng black/white với range `[0,1]`. Image là ảnh gốc display-ready; không đưa ảnh đã normalize DINO vào panel Image.

Threshold prediction/tolerance trong boundary protocol chỉ được validator kiểm tra tính nhất quán với các task trước; visualization **không dùng chúng để binarize heatmap**.

## 3. Sample selection dựa hoàn toàn vào GT

Script đọc GT/boundary zones của danh sách sample trước, tính connected components bằng 8-connectivity và lấy flags từ cùng protocol đã khóa:

```text
tiny: area_native_px <= locked tiny_area_px
boundary: component giao boundary zone ít nhất 1 pixel
```

Trước khi load bất kỳ anomaly map nào, script chọn sample theo từng category:

1. Ảnh có cả tiny và boundary defect.
2. Xen kẽ tiny-only và boundary-only, bắt đầu tiny-only.
3. Các ảnh còn lại khi chưa đủ số lượng.

Trong mỗi nhóm, tie-break bằng `image_id` tăng dần. Mặc định tối đa ba sample/category; đổi bằng `--samples-per-category`. Selection không phụ thuộc thứ tự đầu vào manifest, score, metric hoặc candidate thắng.

“Ảnh có cả tiny và boundary” có thể là hai region khác nhau; `n_tiny_boundary_regions` ghi riêng số component đồng thời thỏa cả hai flags. Nếu có nhiều ảnh cả hai nhóm, chúng được ưu tiên trước các nhóm còn lại.

Script không đọc/ranking metric từ `region_stats.csv`, không chọn hình đẹp theo prediction, không xếp hạng winner. Giữ `comparison_manifest.json` để lưu lại sample IDs/rule/crop đã dùng và báo đầy đủ diagnostic theo kế hoạch, không chỉ hình có lợi cho một candidate.

## 4. GT zoom chung để thấy defect nhỏ

Figure full giữ toàn ảnh. Figure zoom dùng **cùng crop native pixel cho cả năm panel**.

Focus region chọn từ GT bằng thứ tự: component vừa tiny vừa boundary → tiny → boundary → còn lại; sau đó area nhỏ hơn và region ID thấp hơn. Không chọn crop dựa trên anomaly heatmap.

Crop là bounding box của focus region cộng padding chung `--zoom-padding-px` (mặc định 32 pixel), clip tại mép ảnh. Đây là tham số hiển thị, không thay định nghĩa tiny/boundary hay metric. Crop metadata theo `[y0,y1,x0,x1]`, điểm dừng exclusive theo slicing NumPy.

Nếu GT không có defect thì không tạo zoom giả. Nếu crop đã bằng toàn ảnh thì không xuất figure trùng. Zoom dùng dữ liệu slice trực tiếp, không nội suy score hoặc scale riêng.

## 5. Manifest: thêm `image_path`

Dùng manifest prediction/GT của các task trước và **tự thêm đường dẫn ảnh gốc** cho từng sample:

```json
"image_path": "../data/dev_synthetic/fabric/sample_001.png"
```

`image_id` là định danh, không được tự suy đoán là đường dẫn file. Script không scan/sort thư mục để ghép ảnh và prediction.

Các field được dùng:

```text
split
categories
seg_f1_threshold
normalization_by_candidate: R0/R1/R2 khai báo giống nhau
samples[]:
    image_id
    category
    original_hw: [H,W]
    image_path
    gt_mask
    maps: R0/R1/R2
    boundary_zone_mask: nếu mode provided_zone_mask
```

Đường dẫn tương đối tính từ **thư mục chứa manifest**. Với DEV synthetic, `image_path` phải là ảnh sample synthetic thực tạo ra GT/maps, không phải ảnh normal nguồn trước khi tạo anomaly.

GT/zone binary boolean, `{0,1}` hoặc `{0,255}`, đúng native H×W. GT chỉ đổi encoding sang boolean sau kiểm tra; không threshold tùy ý. Image/GT/R0/R1/R2 phải có cùng kích thước native và cùng sample identity. Shape equality không thay thế việc kiểm tra semantic registration upstream.

Image loader nhận file ảnh grayscale/RGB/RGBA (bỏ alpha để hiển thị); palette/CMYK chuyển sang RGB. Unsigned integer image được chia cho dtype maximum, ví dụ uint8/255 hoặc uint16/65535; float image phải đã nằm trong `[0,1]`. Đây là chuyển ảnh gốc sang display range, **không phải normalization anomaly maps**. Không auto min-max cả ảnh gốc. File ảnh signed integer chưa có display protocol rõ ràng sẽ bị từ chối.

Code chỉ load image/anomaly maps của sample đã chọn. Nó đọc toàn bộ GT/zones để chọn sample nhưng không thay thế full artifact validation/evaluator cho candidate maps không được visualize.

GT content hash được ghi lúc selection và đối chiếu khi load sample để tránh GT thay đổi giữa selection và rendering. Normalization declarations giống nhau vẫn cần đối chiếu config/provenance upstream; visualization không chứng minh pipeline thực sự giống nhau.

## 6. Protocol dùng lại, không đặt threshold mới

Dùng đúng `tiny_protocol.lock.json` và `boundary_protocol.lock.json` từ Task 4/5. Mode boundary (mép ảnh hay zone mask tham chiếu) giữ như thí nghiệm đã định nghĩa; không đổi riêng cho visualization đẹp hơn.

Các `.example.json` kèm bàn giao chứa placeholder/null/false, chưa chạy được. Nếu đã có bản khóa thật, dùng lại bản đó thay vì điền lại ngưỡng theo candidate results.

## 7. Cách chạy

Sau khi bạn tự tạo file mới, thêm `image_path` và chuẩn bị protocol đã khóa, chạy tại root project:

```bash
python -m src.analysis.compare_representation \
  --manifest outputs/day05_eval_input.json \
  --tiny-protocol configs/tiny_protocol.lock.json \
  --boundary-protocol configs/boundary_protocol.lock.json \
  --output-dir outputs/day05_comparisons_run01 \
  --samples-per-category 3 \
  --zoom-padding-px 32
```

Lệnh do bạn chạy tạo output mới:

```text
outputs/day05_comparisons_run01/
├── comparison_<category>_<id-hash>.png
├── comparison_<category>_<id-hash>_zoom.png
└── comparison_manifest.json
```

Mỗi tên chứa category và hash image ID để tránh ký tự path trong tên file. Caption/manifest chứa ID thật. Output directory đã tồn tại bị từ chối để không ghi đè.

`comparison_manifest.json` ghi purpose `diagnostic_only`, layout/range/colormap, source paths, số GT regions theo flags, focus region/crop, selection rule, sample limit/padding và SHA của input manifest/two protocols. Không ghi winner/rank theo hình.

Manifest output chỉ ghi sau khi render tất cả selected samples xong. Nếu lỗi ở sample sau, có thể còn PNG của sample trước nhưng chưa có `comparison_manifest.json`; script không tự xóa artifacts này. Sửa input và chọn output directory mới.

## 8. Kiểm tra bàn giao và demo

20 tests PASS ngoài project, gồm:

- selection GT-only hoàn tất trước khi đọc maps;
- ưu tiên và interleaving tiny/boundary, tie-break ổn định khi đảo thứ tự manifest;
- đúng năm panel Image/GT/R0/R1/R2;
- ba heatmap dùng cùng Normalize object và range `[0,1]`, giá trị score gốc được giữ;
- cùng crop cho năm panel full/zoom, clipping tại image boundary;
- PNG/manifest đúng, không ghi đè;
- uint8/uint16 image scaling theo dtype range, không image min-max;
- GT thay đổi sau selection bị từ chối;
- normal fallback không tạo focus/zoom giả;
- từ chối image/GT/map shape/range sai, thiếu map/path, normalization khác, protocol chưa khóa và duplicate IDs.

Đã render và xem full/zoom figure để kiểm tra layout và khả năng nhìn defect nhỏ. Hai hình dưới là **dữ liệu giả lập kiểm tra code**, không phải so sánh R0/R1/R2 thực nghiệm:

![Synthetic full comparison](comparison_demo.synthetic.png)

![Synthetic GT zoom comparison](comparison_demo.synthetic_zoom.png)

Chưa có image/GT/maps thực để tạo figure nghiên cứu. Hình diagnostic giúp xem localization/smoothing/false positives; winner phải dựa vào metric/protocol đã khóa, không dựa vào hình đẹp.

## 9. Code đầy đủ để copy/paste

Tự tạo **`src/analysis/compare_representation.py`** và dán code này; giữ nguyên các file task trước:

```python
"""GT-selected diagnostics: Image | GT | R0 | R1 | R2."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label

from src.eval.boundary_analysis import STRUCTURE, boundary_zone, validate_inputs
from src.eval.region_stats import read_json, validate_tiny_protocol
from src.eval.tiny_analysis import binary_mask, load_array

CANDIDATES = ("R0", "R1", "R2")


def sample_plan(sample, base, tiny, boundary, padding):
    gt = binary_mask(load_array(base, sample["gt_mask"]))
    hw = sample["original_hw"]
    if len(hw) != 2 or any(type(x) is not int or x < 1 for x in hw) or gt.shape != tuple(hw):
        raise ValueError(f"GT must match original_hw: {sample['image_id']}")
    if not isinstance(sample.get("image_path"), str) or not sample["image_path"].strip():
        raise ValueError(f"Provide the original image_path: {sample['image_id']}")
    zone = boundary_zone(sample, gt.shape, base, boundary)
    labels, count = label(gt, structure=STRUCTURE)
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    overlaps = np.bincount(labels[zone], minlength=count + 1)[1:]
    is_tiny, is_boundary = areas <= tiny["tiny_area_px"], overlaps > 0
    focus_id, crop = None, None
    if count:
        focus_index = min(
            range(count),
            key=lambda i: (
                0 if is_tiny[i] and is_boundary[i] else 1 if is_tiny[i] else 2 if is_boundary[i] else 3,
                int(areas[i]), i,
            ),
        )
        ys, xs = find_objects(labels)[focus_index]
        focus_id = focus_index + 1
        crop = [max(0, ys.start - padding), min(hw[0], ys.stop + padding),
                max(0, xs.start - padding), min(hw[1], xs.stop + padding)]
    return {
        "sample": sample, "n_regions": int(count),
        "n_tiny_regions": int(is_tiny.sum()), "n_boundary_regions": int(is_boundary.sum()),
        "n_tiny_boundary_regions": int((is_tiny & is_boundary).sum()),
        "focus_region_id": focus_id, "crop_y0_y1_x0_x1": crop,
        "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(),
    }


def select_plans(plans, limit):
    """Selection is independent of candidate maps and scores."""
    selected = []
    for category in sorted({p["sample"]["category"] for p in plans}):
        groups = {"both": [], "tiny": [], "boundary": [], "other": []}
        for plan in sorted(plans, key=lambda p: p["sample"]["image_id"]):
            if plan["sample"]["category"] != category:
                continue
            t, b = plan["n_tiny_regions"] > 0, plan["n_boundary_regions"] > 0
            key = "both" if t and b else "tiny" if t else "boundary" if b else "other"
            groups[key].append(plan)
        alternating = [p for pair in zip_longest(groups["tiny"], groups["boundary"]) for p in pair if p is not None]
        selected.extend((groups["both"] + alternating + groups["other"])[:limit])
    return selected


def load_display_image(base, value):
    path = Path(value)
    path = path if path.is_absolute() else base / path
    with Image.open(path) as image:
        arr = np.array(image.convert("RGB") if image.mode in {"P", "CMYK"} else image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("Original image must be grayscale, RGB or RGBA.")
    arr = arr[..., :3]
    if arr.dtype.kind == "u":
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    elif arr.dtype.kind != "f":
        raise ValueError("Use unsigned-integer or display-ready float original images.")
    if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Original image must have a valid display range.")
    return arr


def load_sample(plan, base):
    sample = plan["sample"]
    gt = binary_mask(load_array(base, sample["gt_mask"]))
    if gt.shape != tuple(sample["original_hw"]):
        raise ValueError("GT shape changed after sample selection.")
    if hashlib.sha256(gt.tobytes()).hexdigest() != plan["gt_sha256"]:
        raise ValueError("GT changed after sample selection.")
    image = load_display_image(base, sample["image_path"])
    if image.shape[:2] != gt.shape:
        raise ValueError(f"Original image and GT shapes differ: {sample['image_id']}")
    scores = {}
    for candidate in CANDIDATES:
        score = load_array(base, sample["maps"][candidate])
        if score.shape != gt.shape or score.dtype.kind not in "uif" or not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
            raise ValueError(f"Invalid probability map: {candidate}/{sample['image_id']}")
        scores[candidate] = score
    return image, gt, scores


def render_comparison(path, image, gt, scores, plan, crop=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if crop is not None:
        y0, y1, x0, x1 = crop
        image, gt = image[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        scores = {c: s[y0:y1, x0:x1] for c, s in scores.items()}
    fig = plt.figure(figsize=(18, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.04])
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    axes[0].imshow(image, interpolation="nearest")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    shared_norm = Normalize(vmin=0.0, vmax=1.0)
    for axis, candidate in zip(axes[2:], CANDIDATES):
        heatmap = axis.imshow(scores[candidate], cmap="magma", norm=shared_norm, interpolation="nearest")
    for axis, title in zip(axes, ["Image", "GT", *CANDIDATES]):
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(heatmap, cax=fig.add_subplot(grid[0, 5]), label="Probability [0,1]")
    view = "full image" if crop is None else f"GT zoom region {plan['focus_region_id']}, crop={crop}"
    fig.suptitle(
        f"{plan['sample']['image_id']}\n{view} | tiny={plan['n_tiny_regions']} | boundary={plan['n_boundary_regions']} | diagnostic only",
        fontsize=10,
    )
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def compare(manifest_path, tiny_protocol_path, boundary_protocol_path, output_dir,
            samples_per_category=3, zoom_padding_px=32):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing figures are not overwritten.")
    if type(samples_per_category) is not int or samples_per_category < 1:
        raise ValueError("samples_per_category must be a positive integer.")
    if type(zoom_padding_px) is not int or zoom_padding_px < 0:
        raise ValueError("zoom_padding_px must be a non-negative integer.")
    manifest, manifest_sha = read_json(manifest_path)
    tiny, tiny_sha = read_json(tiny_protocol_path)
    boundary, boundary_sha = read_json(boundary_protocol_path)
    validate_tiny_protocol(tiny)
    validate_inputs(manifest, boundary)
    base = manifest_path.parent
    # Finish GT-based selection and crop planning before loading any anomaly map.
    plans = [sample_plan(s, base, tiny, boundary, zoom_padding_px) for s in manifest["samples"]]
    selected = select_plans(plans, samples_per_category)
    output_ready, records = False, []
    for plan in selected:
        image, gt, scores = load_sample(plan, base)
        if not output_ready:
            output_dir.mkdir(parents=True, exist_ok=False)
            output_ready = True
        sample = plan["sample"]
        token = hashlib.sha256(sample["image_id"].encode()).hexdigest()[:16]
        name = f"comparison_{sample['category']}_{token}"
        full_path = output_dir / f"{name}.png"
        render_comparison(full_path, image, gt, scores, plan)
        zoom_filename = None
        crop = plan["crop_y0_y1_x0_x1"]
        if crop is not None and crop != [0, gt.shape[0], 0, gt.shape[1]]:
            zoom_filename = f"{name}_zoom.png"
            render_comparison(output_dir / zoom_filename, image, gt, scores, plan, crop)
        records.append({
            **{key: value for key, value in plan.items() if key != "sample"},
            "image_id": sample["image_id"], "category": sample["category"],
            "image_path": sample["image_path"], "gt_mask": sample["gt_mask"], "maps": sample["maps"],
            "full_figure": full_path.name, "zoom_figure": zoom_filename,
        })
    report = {
        "purpose": "diagnostic_only", "layout": ["Image", "GT", *CANDIDATES],
        "anomaly_map_range": [0.0, 1.0], "colormap": "magma",
        "normalization_protocol": manifest["normalization_by_candidate"]["R0"],
        "selection_rule": "GT only: both groups, alternating tiny/boundary, other; ID tie-break",
        "samples_per_category": samples_per_category, "zoom_padding_px": zoom_padding_px,
        "manifest_sha256": manifest_sha, "tiny_protocol_sha256": tiny_sha,
        "boundary_protocol_sha256": boundary_sha, "samples": records,
    }
    with (output_dir / "comparison_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiny-protocol", type=Path, required=True)
    parser.add_argument("--boundary-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=3)
    parser.add_argument("--zoom-padding-px", type=int, default=32)
    args = parser.parse_args()
    report = compare(args.manifest, args.tiny_protocol, args.boundary_protocol,
                     args.output_dir, args.samples_per_category, args.zoom_padding_px)
    print(f"Rendered {len(report['samples'])} GT-selected samples to {args.output_dir}")


if __name__ == "__main__":
    main()

```
