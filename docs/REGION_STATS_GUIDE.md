# TV2 — Per-region statistics: `src/eval/region_stats.py`

## 1. Đây là file mới

Bạn **tự tạo `src/eval/region_stats.py`**, dán toàn bộ code ở cuối tài liệu. Không chèn vào hoặc thay `evaluator.py`, `tiny_analysis.py`, `boundary_analysis.py`.

Bản bàn giao chỉ tạo artifact trong `/tmp/msila_tv2_region_stats_delivery/`. Không có file trong project được sửa/tạo/xóa bởi bước bàn giao. Các thay đổi đã có của bạn được giữ nguyên.

Code gọi lại các helper của task trước:

- `tiny_analysis.binary_mask`, `tiny_analysis.load_array`;
- `boundary_analysis.STRUCTURE`, `boundary_analysis.boundary_zone`, `boundary_analysis.validate_inputs`.

Hai file tiny/boundary cần ở đúng `src/eval/` và có các helper này. Môi trường kiểm tra có NumPy, SciPy và Pillow; task này không render visualization và không cần Matplotlib.

## 2. Granularity đúng yêu cầu

Một dòng CSV tương ứng **một GT connected component**. Cùng dòng đó chứa metric/score của R0, R1, R2. Không có dataset-mean row, không tạo ba dòng candidate riêng cho cùng region.

GT được label đúng một lần trên mỗi ảnh bằng **8-connectivity**. Diagonal-touch pixels thuộc cùng region. Region không được tạo từ prediction.

Khóa định danh một dòng là `(image_id, region_id)`. `region_id` bắt đầu từ 1 và là ID local trong ảnh; ID 0 là background nên không được xuất. Label/ID được tính từ GT, dùng chung cả ba candidate. Thay GT có thể thay region ID; không coi ID là định danh bất biến giữa các phiên bản GT.

## 3. Area và flags giữ nguyên protocol

```text
area = số foreground pixels của component ở kích thước ảnh gốc
is_tiny = area <= tiny_protocol.tiny_area_px
is_boundary = component có ít nhất 1 pixel giao boundary zone
boundary_flag = is_boundary
```

Code xuất cả `is_boundary` và `boundary_flag` là cùng một giá trị để đáp ứng hai cách đặt tên trong yêu cầu. Không tạo hai định nghĩa boundary khác nhau.

Dùng lại **đúng file protocol đã khóa** từ Task 4 và Task 5. Task 6 không tự đặt threshold tiny, band width hay score threshold mới. Các flags có thể đồng thời đúng: một region có thể vừa tiny vừa boundary.

Boundary mode được lấy từ Task 5:

- `image_border_band`: band sát bốn mép ảnh gốc;
- `provided_zone_mask`: zone binary chung cho ba candidate, đường dẫn `boundary_zone_mask` trong sample.

Flag dựa trên giao với zone; metric luôn tính trên **toàn bộ region**, không cắt region theo band. Với provided-zone mode, zone phải đúng native H/W và non-empty. Biên vật thể cần zone/reference đúng của vật thể; không tự suy ra từ defect GT.

## 4. Metric/score mỗi candidate

`R0_metric`, `R1_metric`, `R2_metric` là **region overlap tại threshold prediction đã khóa**:

```text
pred_pixel = anomaly_probability >= locked_threshold
TP_region = số pixel trong GT region có pred_pixel=True
FN_region = area - TP_region
region_overlap = TP_region / area
```

Đây là pixel recall của region, hay PRO tại một threshold. **Không phải AU-PRO, IoU hay F1.** `metric_name` trong CSV là `region_overlap_at_locked_threshold` để tránh hiểu sai.

Threshold lấy từ boundary protocol và phải bằng `seg_f1_threshold` trong manifest đã khóa. Cùng một threshold áp dụng cho R0/R1/R2. Pixel có score đúng bằng threshold được tính là detected.

Các cột kèm theo cho mỗi candidate:

| Cột | Ý nghĩa |
|---|---|
| `<candidate>_metric` | Region overlap/recall, trong `[0,1]` |
| `<candidate>_mean_score` | Trung bình anomaly probability trên toàn GT region |
| `<candidate>_max_score` | Probability lớn nhất trên region |
| `<candidate>_tp` | Số pixel GT region được phát hiện |
| `<candidate>_fn` | Số pixel GT region bị bỏ sót |

Mean/max score là thống kê mô tả, không tự chứng minh localization tốt hay xác suất được calibration tốt.

**Region overlap không phạt false positives ngoài GT region**: prediction high-score toàn ảnh có thể có overlap=1. Vì vậy dùng bảng này để xem region nào được/mất khi đổi representation, cùng AU-PRO chính từ E2 và boundary diagnostic. Không dùng mean của `R*_metric` để thay AU-PRO_0.05.

Không xuất per-region IoU/F1 ngầm vì cần khóa thêm cách gán predicted components và false positives cho từng region. Task này báo metric/score đã định nghĩa rõ, không tự thêm instance-matching protocol.

## 5. Đầu vào

Dùng một manifest prediction/GT chung như các task trước:

```text
split
categories
seg_f1_threshold
normalization_by_candidate: R0/R1/R2 phải khai báo giống nhau
samples[]:
    image_id
    category
    original_hw: [H,W]
    gt_mask
    maps: đúng ba đường dẫn R0/R1/R2
    boundary_zone_mask: cần nếu dùng provided_zone_mask
```

Đường dẫn tương đối được giải từ **thư mục chứa manifest**. Điền toàn bộ sample của split/category đã khóa, gồm normal images theo protocol. Code kiểm tra ba map trên từng sample nhưng bạn vẫn phải đối chiếu danh sách với data manifest để bảo đảm không cùng bỏ sót sample.

GT phải binary boolean, `{0,1}` hoặc `{0,255}`. Prediction phải finite probability 2-D `[0,1]`, cùng kích thước GT và `original_hw`. Code không sigmoid lại, normalize, resize, threshold GT grayscale hoặc tự sửa dữ liệu lỗi.

Cùng H/W chỉ bảo đảm tương thích shape, không chứng minh GT/prediction đã đăng ký đúng pixel; inference stitching/alignment phải đúng. Normalization declarations giống nhau cần được đối chiếu provenance upstream.

Hidden-GT splits `test_private`/`test_private_mixed` bị từ chối. DEV synthetic cần GT của sample synthetic thực, không dùng zero GT của ảnh normal nguồn.

Các file `.example.json` trong bàn giao là skeleton; placeholder/null/false **không phải protocol chạy được hay giá trị đã khóa**. Nếu đã có protocol từ Task 4/5, dùng lại bản đó, không điền threshold mới dựa trên kết quả candidate.

## 6. Cách chạy

Sau khi tự tạo file mới và chuẩn bị các đầu vào đã khóa, chạy từ root project:

```bash
python -m src.eval.region_stats \
  --manifest outputs/day05_eval_input.json \
  --tiny-protocol configs/tiny_protocol.lock.json \
  --boundary-protocol configs/boundary_protocol.lock.json \
  --output outputs/day05_region_stats/region_stats.csv
```

Lệnh do bạn chạy tạo thư mục output nếu cần và ghi file CSV mới. Có file CSV cùng tên thì script từ chối ghi đè. Khi dữ liệu/protocol không hợp lệ, script dừng trước khi ghi CSV. Nó không tự sửa/xóa input hay output cũ.

Hàm `analyze()` có thể gọi trong notebook: trả `list[dict]`, không ghi file. Bộ nhớ được dùng theo từng ảnh cho label/map, không giữ cả dataset anomaly maps; list row có kích thước theo số region.

Ảnh normal không có defect component nên không tạo row giả. Prediction của ảnh đó vẫn được load/validate để kiểm tra artifact completeness. Nếu toàn bộ GT không có defect, CSV chỉ có header và log `0 GT defect-region rows`; không chèn mean/normal pseudo-region.

## 7. Schema và output

Một phần header dễ đối chiếu yêu cầu:

```text
image_id,category,split,region_id,area,is_tiny,is_boundary,boundary_flag,
R0_metric,R1_metric,R2_metric
```

CSV thực còn có area/boundary overlap, score mean/max, TP/FN, metric name, threshold và provenance. SHA256 của manifest/tiny protocol/boundary protocol được ghi trong từng row. Các SHA này nhận diện nội dung file metadata; chúng không thay thế checksum riêng của GT/map hay chứng nhận lock lịch sử.

Các group cờ chung được xuất một lần; không có `R0_is_tiny`, `R1_is_tiny`, `R2_is_tiny` để tránh phân nhóm theo candidate.

File `region_stats.synthetic.csv` kèm bàn giao là **kết quả test giả lập**, không phải số nghiên cứu thật. Test có hai GT regions, nên đúng hai row; cùng row có đủ ba candidate.

## 8. Cách dùng bảng cho câu hỏi nghiên cứu

Có thể tính delta per-region:

```text
delta_multi_layer = R1_metric - R0_metric
delta_context     = R2_metric - R1_metric
```

Lọc `is_tiny` hoặc `is_boundary` để xem region nào cải thiện/suy giảm, đối chiếu area và score. Bảng giữ các row gốc nên không mất thông tin khi mean tăng nhưng một số region giảm.

Các vùng cùng ảnh có thể phụ thuộc nhau; không coi mỗi row là một replicate độc lập khi suy luận thống kê. Same-seed pairing và các khóa train/inference vẫn cần giữ. Metric per-region không khắc phục việc Context upstream không được tạo/căn chỉnh đúng.

Chưa có GT/maps thật để xuất `region_stats.csv` nghiên cứu. Các số kiểm tra chỉ kiểm chứng code.

## 9. Kiểm tra bàn giao

22 tests PASS ngoài project, gồm:

- mỗi GT component đúng một wide row và đủ R0/R1/R2;
- metric, mean/max score, TP/FN khớp oracle tính trực tiếp từng region;
- 8-connectivity, cutoff tiny inclusive, native area và boundary whole-region flags;
- mode supplied-zone dùng đúng rule Task 5;
- region ID local, khóa `(image_id, region_id)` không trùng;
- normal GT không có pseudo-region nhưng map lỗi vẫn bị từ chối;
- header-only CSV khi không có defect;
- từ chối protocol/threshold riêng candidate, normalization khác, shape/GT/score sai, duplicate ID, thiếu R2, hidden-GT split;
- CLI xuất CSV đúng row count và từ chối ghi đè.

## 10. Code đầy đủ để copy/paste

Đích mới do bạn tự tạo: **`src/eval/region_stats.py`**.

```python
"""Per-GT-region statistics. New file: src/eval/region_stats.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from src.eval.boundary_analysis import STRUCTURE, boundary_zone, validate_inputs
from src.eval.tiny_analysis import binary_mask, load_array

CANDIDATES = ("R0", "R1", "R2")
FIELDS = [
    "image_id", "category", "split", "region_id", "area",
    "is_tiny", "is_boundary", "boundary_flag", "boundary_overlap_px",
    "metric_name", "prediction_threshold", "tiny_area_px", "connectivity",
    "boundary_mode", "band_width_px", "normalization_protocol",
    "manifest_sha256", "tiny_protocol_sha256", "boundary_protocol_sha256",
]
for candidate in CANDIDATES:
    FIELDS.extend(f"{candidate}_{name}" for name in ("metric", "mean_score", "max_score", "tp", "fn"))


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_tiny_protocol(protocol):
    fields = {
        "tiny_area_px", "area_unit", "connectivity", "max_fpr",
        "locked_before_candidate_results", "threshold_basis",
    }
    if set(protocol) != fields:
        raise ValueError("Use the single shared tiny protocol from Task 4.")
    if type(protocol["tiny_area_px"]) is not int or protocol["tiny_area_px"] < 1:
        raise ValueError("tiny_area_px must be a locked positive integer.")
    if (
        protocol["area_unit"] != "original_image_pixels"
        or protocol["connectivity"] != 8
        or protocol["max_fpr"] != 0.05
    ):
        raise ValueError("Keep the existing tiny protocol: native pixels, connectivity=8, max_fpr=0.05.")
    if (
        protocol["locked_before_candidate_results"] is not True
        or not isinstance(protocol["threshold_basis"], str)
        or not protocol["threshold_basis"].strip()
    ):
        raise ValueError("Record the tiny threshold basis and its lock before candidate results.")


def component_score_stats(score, labels, areas, threshold):
    """Return overlap, mean/max score and TP/FN for every GT component."""
    if (
        score.shape != labels.shape
        or score.dtype.kind not in "uif"
        or not np.isfinite(score).all()
        or np.any(score < 0)
        or np.any(score > 1)
    ):
        raise ValueError("Prediction must be a finite native-resolution probability map in [0,1].")
    n = len(areas)
    score = score.astype(np.float64, copy=False)
    flat_labels = labels.ravel()
    tp = np.bincount(labels[score >= threshold], minlength=n)
    sums = np.bincount(flat_labels, weights=score.ravel(), minlength=n)
    maxima = np.full(n, -np.inf, dtype=np.float64)
    np.maximum.at(maxima, flat_labels, score.ravel())
    # Index 0 is background and is never exported as a defect region.
    return {
        region_id: {
            "metric": float(tp[region_id] / areas[region_id]),
            "mean_score": float(sums[region_id] / areas[region_id]),
            "max_score": float(maxima[region_id]),
            "tp": int(tp[region_id]),
            "fn": int(areas[region_id] - tp[region_id]),
        }
        for region_id in range(1, n)
    }


def analyze(manifest_path, tiny_protocol_path, boundary_protocol_path):
    manifest_path = Path(manifest_path).resolve()
    manifest, manifest_sha = read_json(manifest_path)
    tiny, tiny_sha = read_json(tiny_protocol_path)
    boundary, boundary_sha = read_json(boundary_protocol_path)
    validate_tiny_protocol(tiny)
    validate_inputs(manifest, boundary)
    threshold = boundary["prediction_threshold"]
    base, rows = manifest_path.parent, []

    for sample in manifest["samples"]:
        gt = binary_mask(load_array(base, sample["gt_mask"]))
        hw = sample["original_hw"]
        if (
            len(hw) != 2
            or any(type(x) is not int or x < 1 for x in hw)
            or gt.shape != tuple(hw)
        ):
            raise ValueError(f"GT must match original_hw: {sample['image_id']}")
        zone = boundary_zone(sample, gt.shape, base, boundary)
        # One GT labeling determines the identity/area/flags for all candidates.
        labels, count = label(gt, structure=STRUCTURE)
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        overlaps = np.bincount(labels[zone], minlength=count + 1)
        candidate_stats = {}
        for candidate in CANDIDATES:
            score = load_array(base, sample["maps"][candidate])
            try:
                candidate_stats[candidate] = component_score_stats(score, labels, areas, threshold)
            except ValueError as exc:
                raise ValueError(f"{candidate}/{sample['image_id']}: {exc}") from exc

        for region_id in range(1, count + 1):
            is_boundary = bool(overlaps[region_id] > 0)
            row = {
                "image_id": sample["image_id"], "category": sample["category"],
                "split": manifest["split"], "region_id": region_id,
                "area": int(areas[region_id]),
                "is_tiny": bool(areas[region_id] <= tiny["tiny_area_px"]),
                "is_boundary": is_boundary, "boundary_flag": is_boundary,
                "boundary_overlap_px": int(overlaps[region_id]),
                "metric_name": "region_overlap_at_locked_threshold",
                "prediction_threshold": threshold, "tiny_area_px": tiny["tiny_area_px"],
                "connectivity": 8, "boundary_mode": boundary["boundary_mode"],
                "band_width_px": boundary["band_width_px"],
                "normalization_protocol": manifest["normalization_by_candidate"]["R0"],
                "manifest_sha256": manifest_sha, "tiny_protocol_sha256": tiny_sha,
                "boundary_protocol_sha256": boundary_sha,
            }
            for candidate in CANDIDATES:
                for name, value in candidate_stats[candidate][region_id].items():
                    row[f"{candidate}_{name}"] = value
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiny-protocol", type=Path, required=True)
    parser.add_argument("--boundary-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new CSV path; existing output is not overwritten.")
    rows = analyze(args.manifest, args.tiny_protocol, args.boundary_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} GT defect-region rows to {args.output}")


if __name__ == "__main__":
    main()

```
