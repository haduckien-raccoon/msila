# TV2 — Boundary diagnostic: `src/eval/boundary_analysis.py`

## 1. File mới và phạm vi bàn giao

Bạn **tự tạo `src/eval/boundary_analysis.py`** rồi dán toàn bộ code ở cuối tài liệu. Không chèn vào `evaluator.py`, không thay evaluator hoặc `tiny_analysis.py`.

Code reuse ba helper đã có trong task trước: `binary_mask`, `load_array`, `region_aupro` từ `src.eval.tiny_analysis`. File tiny hiện tại đã có các helper này. Nếu workspace khác chưa áp dụng task tiny, cần đưa file tiny đã bàn giao vào đúng package trước. Code không fallback sang implementation khác khi thiếu module.

Bản bàn giao chỉ tạo artifact trong `/tmp/msila_tv2_boundary_delivery/`; không tạo/sửa/xóa file trong project. Thư viện dùng: NumPy, SciPy, Pillow, Matplotlib; chúng đang có trong môi trường kiểm tra.

## 2. Hai khái niệm cần phân biệt

- **Boundary zone**: vùng không gian dùng để chọn nhóm defect, ví dụ dải sát mép ảnh hoặc band quanh biên vật thể/ROI/tile seams.
- **Defect contour**: đường biên của segmentation defect dùng để đánh giá khả năng giữ biên.

Binary GT defect mask chỉ cho biết defect contour, không tự xác định biên vật thể. Nếu coi contour của mọi defect là boundary zone thì mọi component đều được chọn; như vậy không còn phân nhóm boundary/non-boundary có ý nghĩa.

Repo chưa khóa cụ thể boundary zone cho task này, nên code **không tự chọn một cách hiểu**. Protocol bắt buộc chọn một trong hai mode:

### `image_border_band`

Band sát bốn mép ảnh gốc, rộng đúng `band_width_px` pixel. Với width=w, zone gồm hàng 0..w-1 và H-w..H-1, cột 0..w-1 và W-w..W-1.

```text
zone(y,x) = y<w OR y>=H-w OR x<w OR x>=W-w
```

Đây là diagnostic defect gần mép ảnh; không tự suy ra đó là defect gần biên vật thể. Width quá lớn có thể phủ toàn ảnh; visualization giúp phát hiện định nghĩa không còn phân biệt interior.

### `provided_zone_mask`

Bạn cung cấp `boundary_zone_mask` cho mỗi sample trong manifest, native H×W, binary. Mask này có thể là band quanh biên vật thể/ROI hoặc tile seams, được tạo từ geometry/reference độc lập với candidate. Với mode này `band_width_px` phải là `null`, vì zone đã có trong mask.

Rule/band producer của zone phải được khóa và ghi vào `rule_basis`, cùng cho mọi sample/category theo protocol đã thống nhất. Không tạo zone từ heatmap hoặc prediction R0/R1/R2. Không dùng chính defect contour làm zone để chọn tất cả region.

## 3. Rule phân nhóm GT duy nhất

Connected components dùng **8-connectivity** trên GT binary ở độ phân giải gốc. Pixel chạm góc thuộc cùng component.

```text
region thuộc nhóm boundary <=> region có ít nhất 1 pixel giao boundary zone
```

Khi component được chọn, **toàn bộ region** tham gia metric, không chỉ các pixel bên trong band. GT selection được tính một lần và dùng chung cho R0/R1/R2. Các region còn lại là non-boundary.

Hàm `select_regions()` trả cả mask nhóm và thông tin từng region: ID, area theo native pixel, số pixel giao zone, `is_boundary`. Không có overlap threshold theo candidate hay component selection dựa trên kết quả candidate.

## 4. Hai metric diagnostic

### Boundary-region AU-PRO_0.05

Metric localization tính PRO trên **toàn bộ GT region được chọn**. Mỗi region có trọng số bằng nhau. FPR dùng pixel `GT==0` của tất cả ảnh trong category, gồm ảnh normal và background của ảnh anomalous.

Non-boundary GT **không được đổi thành background**. Nhóm này bị bỏ khỏi PRO boundary và vẫn bị bỏ khỏi FPR. Helper `region_aupro()` gọi lại AU-PRO hiện có bằng score packing, giống task tiny; không sửa công thức hay FPR range.

AU-PRO này đánh giá localization trong nhóm boundary, không trực tiếp đo độ khớp contour. AU-PRO toàn bộ GT từ evaluator E2 vẫn là metric chính.

### Boundary-F1 trên contour của nhóm boundary

Để trả lời “candidate nào giữ biên defect tốt hơn”, code báo thêm `boundary_f1`.

1. Prediction binary dùng một `prediction_threshold` chung, phải bằng `seg_f1_threshold` đã khóa trong manifest E2.
2. Prediction được tách components và chọn bằng **cùng boundary zone và cùng rule giao ít nhất 1 pixel**. Vì vậy false-positive component chạm zone cũng được chọn và bị tính vào precision; không dùng GT để cherry-pick component dự đoán.
3. Lấy contour phía trong bằng `mask & ~erosion(mask)`, erosion dùng kernel 3×3 full connectivity. Phần ngoài ảnh coi là background; contour tại frame ảnh và contour quanh hole đều được tính.
4. Predicted contour pixel được match nếu cách GT contour không quá `tolerance_px`; GT contour pixel được match nếu cách predicted contour không quá tolerance. Khoảng cách dùng Euclidean distance trên pixel grid gốc, inclusive tại tolerance.
5. Cộng số contour pixels/matches qua ảnh trong category rồi tính precision, recall và F1.

```text
P = matched_pred_contour_pixels / pred_contour_pixels
R = matched_gt_contour_pixels / gt_contour_pixels
F1 = 2PR / (P+R)
```

Đây là diagnostic pixel-based contour matching, không phải instance matching 1-1. F1 được tính từ counts gộp, không average F1 từng ảnh hay region. Nếu GT nhóm boundary có nhưng prediction không có, F1=0. Khi category không có GT boundary regions, F1 undefined: ô trống, `NO_GT_BOUNDARY_REGIONS`.

Rule prediction-group selection là một phần của diagnostic: nếu candidate dự đoán component dịch hoàn toàn ra ngoài zone thì component đó không được chọn và GT vẫn có thể bị miss. False positive wholly outside zone không đi vào Boundary-F1 của nhóm này, nhưng background false positives vẫn đi vào FPR của AU-PRO. Đây không phải Boundary-F1 toàn ảnh; cần ghi đúng scope khi báo cáo.

## 5. Khóa protocol trước khi xem candidate results

Mẫu này chưa chạy được vì các giá trị chưa khóa cố ý để `null`/`false`. Bạn tự điền protocol đã thống nhất, ví dụ `configs/boundary_protocol.lock.json`:

```json
{
  "boundary_mode": null,
  "band_width_px": null,
  "connectivity": 8,
  "max_fpr": 0.05,
  "prediction_threshold": null,
  "tolerance_px": null,
  "locked_before_candidate_results": false,
  "rule_basis": ""
}
```

Điền mode, width nếu dùng image-border mode, prediction threshold đã khóa, tolerance native pixel, và căn cứ rule. Chỉ đặt `locked_before_candidate_results=true` khi thực sự đã khóa trước khi xem kết quả.

Code chỉ nhận đúng tám field trong mẫu; không nhận field riêng cho R0/R1/R2. Nó kiểm tra khai báo và SHA protocol được ghi vào mọi dòng CSV; không thể tự chứng minh lịch sử bạn đã xem kết quả hay chưa. Lưu protocol/zone provenance vào experiment log hoặc commit tại thời điểm khóa.

Không tune band, tolerance hay score threshold để candidate thắng. Không chọn mode khác nhau riêng theo candidate. Không đánh giá các mode rồi chỉ báo mode có lợi; nếu có nhiều diagnostic được định trước, báo từng protocol riêng.

## 6. Manifest đầu vào

Có thể dùng manifest của TV2-E2, gồm:

- `split`: một split có GT local; hidden-GT splits bị từ chối;
- `categories`: tập canonical đã khóa;
- `seg_f1_threshold`: threshold chung đã khóa;
- `normalization_by_candidate`: đúng ba khai báo R0/R1/R2 giống nhau;
- `samples`: toàn bộ sample của split/category đã chọn, gồm ảnh normal theo protocol;
- mỗi sample: `image_id`, `category`, `original_hw`, `gt_mask`, `maps` R0/R1/R2;
- với mode `provided_zone_mask`, thêm `boundary_zone_mask`.

Ví dụ thêm field cho một sample:

```json
"boundary_zone_mask": "../data/boundary_zones/fabric/sample_001.npy"
```

Đường dẫn tương đối tính từ **thư mục chứa manifest**. Không ghép GT/prediction bằng sort folder hoặc đoán index filename. Bạn cần đối chiếu sample list với data manifest để bảo đảm đầy đủ; code chỉ đảm bảo mỗi dòng có đúng ba map.

GT/zone phải là binary boolean, `{0,1}` hoặc `{0,255}`. Zone supplied phải non-empty và cùng kích thước gốc. Ảnh normal cần GT zero rõ ràng; không coi thiếu GT là normal.

Prediction phải là probability 2-D, finite, `[0,1]`, native H×W. Code không sigmoid lại, không clip, min-max normalize hay resize dữ liệu metric. Map round-off nằm ngoài `[0,1]` bị từ chối. H/W bằng nhau chưa chứng minh semantic registration: upstream stitching/alignment và overlay QA cần đúng.

## 7. Cách chạy

Sau khi tự tạo file và điền protocol/manifest, chạy từ root project:

```bash
python -m src.eval.boundary_analysis \
  --manifest outputs/day05_eval_input.json \
  --boundary-protocol configs/boundary_protocol.lock.json \
  --output-dir outputs/day05_boundary_run01 \
  --visualizations-per-category 3
```

Output do lệnh này tạo:

```text
outputs/day05_boundary_run01/
├── boundary_region_metrics.csv
├── boundary_protocol.json
└── mask_qa/
    ├── <category>_<id-hash>.png
    └── <category>_<id-hash>_native_gt.png
```

Một dòng CSV cho mỗi candidate × category. Không tự tính macro với category thiếu boundary.

Script từ chối output directory đã tồn tại. CSV chỉ được ghi sau khi mọi category/visualization hoàn tất. Nếu lỗi ở category sau hoặc lỗi rendering, có thể còn thư mục chứa PNG/protocol của category trước nhưng **chưa có result CSV**; script không tự xóa artifacts này. Khắc phục đầu vào rồi chọn thư mục output mới.

Code xử lý category lần lượt và lưu QA từng category để không giữ các map native của tất cả category trong RAM. Category nhiều ảnh native vẫn cần đủ bộ nhớ cho GT/background score arrays.

## 8. Visualization kiểm tra mask

QA sample chọn từ **GT trước khi đọc candidate maps**: ưu tiên sample có GT boundary region, sau đó sort ID; số sample tối đa là `--visualizations-per-category`. Không chọn hình đẹp theo score candidate. Tăng tham số nếu cần kiểm tra thêm.

PNG panel:

- góc trái trên: toàn bộ GT components; **xanh dương** là zone, **magenta** là boundary GT region, **xám** là non-boundary GT;
- góc trái dưới: zone binary riêng;
- hàng trên R0/R1/R2: probability với cùng colormap và thang cố định `[0,1]`;
- hàng dưới R0/R1/R2: **xanh lá** là GT contour, **đỏ** là predicted contour của nhóm boundary, **vàng** là giao contour đúng pixel.

Màu vàng hiển thị **exact overlap**, không phải toàn bộ match trong tolerance. Boundary-F1 có thể cao khi contour lệch trong tolerance dù còn màu đỏ/xanh lá.

File `_native_gt.png` giữ đúng native H×W, không resize; có thể zoom để kiểm tra region nhỏ và zone selection. Figure panel là preview có thể làm region rất nhỏ khó nhìn; hãy đối chiếu native mask khi cần.

File `boundary_mask_qa.synthetic.png` trong bàn giao là **demo giả lập kiểm tra code**, không phải kết quả R0/R1/R2 thật:

![Synthetic mask QA demo](boundary_mask_qa.synthetic.png)

## 9. Diễn giải kết quả

- So sánh Boundary-F1 để nhận định candidate nào giữ contour tốt hơn với **cùng group, threshold, tolerance**.
- So sánh boundary AU-PRO để nhận định localization tốt hơn trong nhóm đã định nghĩa.
- Hai metric đo khía cạnh khác nhau; AU-PRO tăng chưa bảo đảm Boundary-F1 tăng.
- Một category/seed không chứng minh mọi category hay seed đều tốt hơn. Training/inference fairness và same-seed pairing vẫn cần giữ khóa.
- Nếu inference R2 chỉ sao chép Local sang Context, metric boundary không biến nó thành thử nghiệm Context thật; cần bảo đảm upstream Context đúng FOV/alignment.

Chưa có dữ liệu thực để kết luận R0, R1 hay R2 thắng. Không đưa số giả lập vào bảng nghiên cứu.

## 10. Kiểm tra bàn giao

22 tests PASS: 8-connectivity, whole-component selection, pixel tại rìa band, contour tại frame/hole, Euclidean tolerance, missing/empty predictions, GT group invariant, non-boundary GT không là background, normal-image false positive, supplied boundary zone, CSV/undefined metric/no overwrite, và các input/protocol sai.

Visualization đã được render và xem để sửa title trùng, bổ sung colorbar chung, kiểm tra native mask colors. Sau chỉnh layout, test end-to-end CSV/visualization chạy lại PASS.

## 11. Code đầy đủ để copy/paste

Tự tạo **file mới** `src/eval/boundary_analysis.py`; giữ nguyên evaluator và tiny_analysis:

```python
"""Boundary-defect diagnostics. New file: src/eval/boundary_analysis.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from src.eval.evaluator import MVTEC_AD2_CATEGORIES, MVTEC_AD2_SPLITS
from src.eval.tiny_analysis import binary_mask, load_array, region_aupro

CANDIDATES = ("R0", "R1", "R2")
STRUCTURE = np.ones((3, 3), dtype=bool)


def boundary_zone(sample, shape, base, protocol):
    if protocol["boundary_mode"] == "provided_zone_mask":
        zone = binary_mask(load_array(base, sample["boundary_zone_mask"]))
        if zone.shape != shape or not zone.any():
            raise ValueError("Boundary zone must be non-empty and match original_hw.")
        return zone
    width = protocol["band_width_px"]
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (y < width) | (y >= shape[0] - width) | (x < width) | (x >= shape[1] - width)


def select_regions(mask, zone, include_indices=True):
    """A component is selected iff at least one of its pixels intersects zone."""
    labels, count = label(mask, structure=STRUCTURE)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    overlaps = np.bincount(labels[zone], minlength=count + 1)
    selected_ids = np.flatnonzero(overlaps[1:] > 0) + 1
    selected = np.isin(labels, selected_ids)
    regions = [
        {
            "region_id": i, "area_px": int(areas[i]),
            "boundary_overlap_px": int(overlaps[i]), "is_boundary": bool(overlaps[i]),
            "indices": np.flatnonzero(labels.ravel() == i)
            if include_indices and overlaps[i] else None,
        }
        for i in range(1, count + 1)
    ]
    return selected, regions


def contour(mask):
    # Inner contour; image exterior is treated as background.
    return mask & ~binary_erosion(mask, structure=STRUCTURE, border_value=0)


def contour_counts(gt_selected, pred_selected, tolerance_px):
    gt_edge, pred_edge = contour(gt_selected), contour(pred_selected)
    matched_pred = pred_edge & (distance_transform_edt(~gt_edge) <= tolerance_px) if gt_edge.any() else np.zeros_like(pred_edge)
    matched_gt = gt_edge & (distance_transform_edt(~pred_edge) <= tolerance_px) if pred_edge.any() else np.zeros_like(gt_edge)
    return np.array([pred_edge.sum(), matched_pred.sum(), gt_edge.sum(), matched_gt.sum()], dtype=np.int64)


def contour_metrics(counts):
    n_pred, matched_pred, n_gt, matched_gt = map(int, counts)
    if n_gt == 0:
        return None, None, None, "NO_GT_BOUNDARY_REGIONS"
    precision = matched_pred / n_pred if n_pred else 0.0
    recall = matched_gt / n_gt
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, "OK"


def save_visualization(path, sample, gt, zone, selected, scores, predictions, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Save native-resolution GT classification before plotting a preview.
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[zone] = (30, 80, 180)         # Blue: boundary zone.
    rgb[gt & ~selected] = (160, 160, 160)  # Gray: non-boundary defect.
    rgb[selected] = (255, 0, 180)    # Magenta: selected full GT component.
    Image.fromarray(rgb).save(path.with_name(path.stem + "_native_gt.png"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(rgb, interpolation="nearest")
    axes[0, 0].set_title("GT component selection\nblue=zone, magenta=boundary, gray=interior", fontsize=10)
    axes[1, 0].imshow(zone, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title("Boundary zone only\nwhite=zone, black=outside", fontsize=10)
    gt_edge = contour(selected)
    for column, candidate in enumerate(CANDIDATES, start=1):
        probability_plot = axes[0, column].imshow(scores[candidate], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[0, column].set_title(f"{candidate}: probability [0,1]")
        pred_edge = contour(predictions[candidate])
        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        overlay[zone] = (20, 35, 70)
        overlay[gt_edge] = (0, 255, 0)
        overlay[pred_edge] = (255, 0, 0)
        overlay[gt_edge & pred_edge] = (255, 255, 0)
        axes[1, column].imshow(overlay, interpolation="nearest")
        axes[1, column].set_title(f"{candidate}: contour comparison\ngreen=GT, red=pred, yellow=exact overlap", fontsize=10)
    fig.colorbar(probability_plot, ax=list(axes[0, 1:]), shrink=0.7, label="probability (fixed 0-1)")
    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(f"{sample['image_id']}\n{protocol['boundary_mode']} | threshold={protocol['prediction_threshold']} | tolerance={protocol['tolerance_px']}px", fontsize=11)
    try:
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def validate_inputs(manifest, protocol):
    fields = {"boundary_mode", "band_width_px", "connectivity", "max_fpr",
              "prediction_threshold", "tolerance_px", "locked_before_candidate_results", "rule_basis"}
    if set(protocol) != fields:
        raise ValueError("Use one global boundary protocol; candidate-specific settings are forbidden.")
    mode = protocol["boundary_mode"]
    width, threshold, tolerance = protocol["band_width_px"], protocol["prediction_threshold"], protocol["tolerance_px"]
    if mode not in {"image_border_band", "provided_zone_mask"}:
        raise ValueError("Choose explicitly: image_border_band or provided_zone_mask.")
    if mode == "image_border_band" and (type(width) is not int or width < 1):
        raise ValueError("Lock a positive integer band_width_px.")
    if mode == "provided_zone_mask" and width is not None:
        raise ValueError("For provided_zone_mask, use band_width_px=null; the supplied mask defines the zone.")
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Lock one finite probability threshold in [0,1].")
    if threshold != manifest.get("seg_f1_threshold"):
        raise ValueError("Reuse the locked seg_f1_threshold from the evaluation manifest.")
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("Lock one non-negative integer contour tolerance in native pixels.")
    if protocol["connectivity"] != 8 or protocol["max_fpr"] != 0.05:
        raise ValueError("Keep 8-connectivity and AU-PRO max_fpr=0.05.")
    if protocol["locked_before_candidate_results"] is not True or not isinstance(protocol["rule_basis"], str) or not protocol["rule_basis"].strip():
        raise ValueError("Record the rule basis and its lock before candidate results.")
    categories, samples = manifest["categories"], manifest["samples"]
    if not categories or len(set(categories)) != len(categories) or not samples:
        raise ValueError("Provide unique categories and a non-empty common sample list.")
    if set(categories) - set(MVTEC_AD2_CATEGORIES) or {s["category"] for s in samples} != set(categories):
        raise ValueError("Use the exact declared canonical category set.")
    if manifest["split"] not in MVTEC_AD2_SPLITS or manifest["split"] in {"test_private", "test_private_mixed"}:
        raise ValueError("Use one locked split with local GT.")
    ids = [s["image_id"] for s in samples]
    if not all(isinstance(i, str) and i and i == i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample image IDs must be unique/non-empty.")
    if any(set(s["maps"]) != set(CANDIDATES) for s in samples):
        raise ValueError("Every sample requires exactly R0/R1/R2 maps.")
    norms = manifest["normalization_by_candidate"]
    if set(norms) != set(CANDIDATES) or not all(isinstance(v, str) and v.strip() for v in norms.values()) or len(set(norms.values())) != 1:
        raise ValueError("Declare the same normalization protocol for R0/R1/R2.")


def analyze(manifest_path, protocol_path, output_dir, visualizations_per_category=3):
    manifest_path, protocol_path, output_dir = Path(manifest_path).resolve(), Path(protocol_path).resolve(), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing output is not overwritten.")
    if type(visualizations_per_category) is not int or visualizations_per_category < 1:
        raise ValueError("visualizations_per_category must be a positive integer.")
    manifest_raw, protocol_raw = manifest_path.read_bytes(), protocol_path.read_bytes()
    manifest, protocol = json.loads(manifest_raw), json.loads(protocol_raw)
    validate_inputs(manifest, protocol)
    rows, output_ready = [], False
    base = manifest_path.parent
    threshold, tolerance = protocol["prediction_threshold"], protocol["tolerance_px"]

    for category in manifest["categories"]:
        plans = []
        for sample in (s for s in manifest["samples"] if s["category"] == category):
            gt = binary_mask(load_array(base, sample["gt_mask"]))
            hw = sample["original_hw"]
            if len(hw) != 2 or any(type(x) is not int or x < 1 for x in hw) or gt.shape != tuple(hw):
                raise ValueError(f"GT must match native original_hw: {sample['image_id']}")
            zone = boundary_zone(sample, gt.shape, base, protocol)
            selected, regions = select_regions(gt, zone)
            plans.append((sample, gt, zone, selected, regions))
        # Select QA samples using GT only, before reading candidate maps.
        qa_ids = {p[0]["image_id"] for p in sorted(plans, key=lambda p: (not p[3].any(), p[0]["image_id"]))[:visualizations_per_category]}
        qa = {i: {"scores": {}, "predictions": {}} for i in qa_ids}
        n_regions = sum(sum(r["is_boundary"] for r in p[4]) for p in plans)

        for candidate in CANDIDATES:
            normal_parts, boundary_parts = [], []
            counts = np.zeros(4, dtype=np.int64)
            n_pred_regions = 0
            for sample, gt, zone, selected, regions in plans:
                score = load_array(base, sample["maps"][candidate])
                if score.shape != gt.shape or score.dtype.kind not in "uif" or not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
                    raise ValueError(f"Invalid probability map: {candidate}/{sample['image_id']}")
                normal_parts.append(score[~gt])  # Non-boundary GT is NOT background.
                boundary_parts.extend(score.ravel()[r["indices"]] for r in regions if r["is_boundary"])
                pred_selected, pred_regions = select_regions(score >= threshold, zone, include_indices=False)
                n_pred_regions += sum(r["is_boundary"] for r in pred_regions)
                counts += contour_counts(selected, pred_selected, tolerance)
                if sample["image_id"] in qa_ids:
                    qa[sample["image_id"]]["scores"][candidate] = score
                    qa[sample["image_id"]]["predictions"][candidate] = pred_selected
            value, status = region_aupro(normal_parts, boundary_parts)
            precision, recall, f1, f1_status = contour_metrics(counts)
            rows.append({
                "candidate": candidate, "category": category, "split": manifest["split"],
                "boundary_mode": protocol["boundary_mode"], "band_width_px": protocol["band_width_px"],
                "selection_rule": "component intersects zone", "connectivity": 8,
                "prediction_threshold": threshold, "tolerance_px": tolerance, "max_fpr": 0.05,
                "n_images": len(plans), "n_gt_boundary_regions": n_regions,
                "n_pred_boundary_regions": int(n_pred_regions),
                "n_normal_pixels": sum(p.size for p in normal_parts),
                "boundary_aupro_0.05": value, "aupro_status": status,
                "boundary_precision": precision, "boundary_recall": recall,
                "boundary_f1": f1, "boundary_f1_status": f1_status,
                "n_pred_contour_pixels": int(counts[0]), "n_matched_pred_contour_pixels": int(counts[1]),
                "n_gt_contour_pixels": int(counts[2]), "n_matched_gt_contour_pixels": int(counts[3]),
                "normalization_protocol": manifest["normalization_by_candidate"][candidate],
                "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            })
        # Save QA per category to avoid retaining native maps across categories.
        if not output_ready:
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "mask_qa").mkdir()
            (output_dir / "boundary_protocol.json").write_bytes(protocol_raw)
            output_ready = True
        for sample, gt, zone, selected, _ in plans:
            if sample["image_id"] in qa_ids:
                token = hashlib.sha256(sample["image_id"].encode()).hexdigest()[:16]
                path = output_dir / "mask_qa" / f"{sample['category']}_{token}.png"
                item = qa[sample["image_id"]]
                save_visualization(path, sample, gt, zone, selected, item["scores"], item["predictions"], protocol)

    # Write the result CSV only after all categories finish successfully.
    with (output_dir / "boundary_region_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--boundary-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visualizations-per-category", type=int, default=3)
    args = parser.parse_args()
    rows = analyze(args.manifest, args.boundary_protocol, args.output_dir, args.visualizations_per_category)
    print(f"Wrote {len(rows)} rows and mask QA to {args.output_dir}")


if __name__ == "__main__":
    main()

```
