# TV2 — Tiny diagnostic: `src/eval/tiny_analysis.py`

## 1. Đây là file mới, không phải thay evaluator

Bạn **tự tạo file mới `src/eval/tiny_analysis.py`** và dán toàn bộ code ở cuối tài liệu. Không chèn code này vào `src/eval/evaluator.py`; không thay file evaluator gốc.

Bản bàn giao này chỉ tạo artifact trong `/tmp/msila_tv2_tiny_delivery/`. Không có file trong project được sửa, xóa hoặc tạo bởi bước bàn giao.

Đích `src/eval/` tương ứng với `eval/` trong mô tả task vì repo hiện đặt evaluator tại `src/eval/evaluator.py`.

## 2. Định nghĩa tiny phải khóa trước

Mỗi GT mask 2-D được tách bằng connected-component analysis với **8-connectivity**, giống convention AU-PRO hiện có. Pixel chạm góc thuộc cùng region. Area là số foreground pixel của region ở **độ phân giải gốc**, không phải diện tích bounding box, số pixel tile hoặc diện tích sau resize.

```text
area_px = số pixel của connected component
is_tiny = area_px <= tiny_area_px
non_tiny = area_px > tiny_area_px
```

`tiny_area_px` là một số nguyên dương **chung cho tất cả category và R0/R1/R2**. Không có threshold riêng theo candidate. Region đúng bằng threshold được xếp vào tiny.

Repo chưa có threshold tiny đã khóa, nên bản bàn giao **không tự đặt số**. Bạn cần điền giá trị đã thống nhất vào một protocol file duy nhất trước khi xem kết quả R0/R1/R2. Căn cứ có thể là định nghĩa nghiên cứu đã thống nhất hoặc thống kê GT trên TRAIN; không chọn threshold vì nó làm R1/R2 thắng trên kết quả đang đánh giá.

Mẫu `tiny_protocol.lock.example.json` cố ý chứa `null`, `false` và chuỗi rỗng để không bị nhầm là protocol đã được khóa. Bạn tự lưu bản đầy đủ, ví dụ `configs/tiny_protocol.lock.json`:

```json
{
  "tiny_area_px": null,
  "area_unit": "original_image_pixels",
  "connectivity": 8,
  "max_fpr": 0.05,
  "locked_before_candidate_results": false,
  "threshold_basis": ""
}
```

Điền `tiny_area_px`, ghi căn cứ thật vào `threshold_basis`, và chỉ đặt `locked_before_candidate_results = true` khi điều đó thực sự đúng. Lưu protocol vào experiment log/commit cùng thời điểm khóa. Code kiểm tra khai báo, không thể tự xác minh lịch sử bạn đã xem kết quả hay chưa.

Protocol chỉ chấp nhận đúng sáu trường trong mẫu; thêm threshold theo candidate sẽ bị từ chối. Script đọc protocol một lần trước khi đọc prediction và dùng cùng giá trị cho ba candidate. SHA256 protocol được ghi vào từng dòng CSV để đối chiếu.

Không có tham số threshold tiny riêng cho R0/R1/R2 và không có mặc định ngầm. Việc quét các threshold score để tính AU-PRO không phải thay threshold **diện tích tiny**.

## 3. Metric localization được chọn

Task yêu cầu localization metric cho tiny nhưng chưa nêu metric cụ thể. Bản triển khai này dùng **tiny AU-PRO_0.05** để diagnostic phù hợp với metric chính, thêm non-tiny AU-PRO_0.05 làm đối chiếu. Đây là diagnostic của nhóm region, không thay AU-PRO toàn bộ GT trong evaluator E2.

Với tập tiny region T và N normal pixels của **GT đầy đủ**, tại threshold score t:

```text
PRO_tiny(t) = mean_{region thuộc T} [số pixel region có score >= t / area(region)]
FPR(t)      = số pixel GT==0 có score >= t / N
Tiny AU-PRO_0.05 = integral_0^0.05 PRO_tiny(FPR) dFPR / 0.05
```

Mỗi region có trọng số bằng nhau, không lấy pixel recall tổng rồi gọi là PRO. Normal pixels lấy từ tất cả sample của category, gồm ảnh normal và background của ảnh anomalous. Không chỉ chọn ảnh chứa tiny.

**Không chuyển non-tiny GT thành background.** Khi tính tiny metric, non-tiny foreground bị bỏ khỏi PRO và cũng bị bỏ khỏi FPR. Khi tính non-tiny metric, tiny foreground được xử lý tương tự. Mẫu số background không thay giữa hai diagnostic hay giữa ba candidate.

Code gọi lại hàm public `src.metrics.aupro.aupro()` hiện có để giữ xử lý score ties, tích phân và nội suy tại FPR=0.05. Không sửa file metric. Để hàm này nhận đúng nhóm region, helper `region_aupro()` đóng gói:

- pixel của background gốc vào các mảng toàn background;
- pixel của mỗi region được chọn vào một mảng riêng toàn foreground, tương ứng đúng một component.

Đây là cách truyền **tập score** vào metric, không phải resize/biến đổi ảnh thật. Nó bảo toàn score, số normal pixels, area và số component, nên giữ nguyên PRO/FPR mong muốn. Hàm AU-PRO tự cân đều các region. Không đưa các mảng đóng gói này vào evaluator kiểm tra kích thước ảnh gốc.

CSV báo riêng từng category, không tính macro với category thiếu tiny. Nếu tự làm macro sau đó, phải khóa cách xử lý nhóm thiếu và báo số category hợp lệ; không thay giá trị undefined bằng 0 hay 1.

## 4. Manifest đầu vào dùng chung

Có thể dùng cùng schema manifest prediction/GT của TV2-E2. Script này sử dụng các trường:

```text
split
categories
normalization_by_candidate: R0/R1/R2 phải khai báo cùng protocol
samples[]:
    image_id
    category
    original_hw: [H, W]
    gt_mask: đường dẫn GT
    maps: ba đường dẫn prediction R0/R1/R2
```

`evaluation_manifest.example.json` trong bàn giao là skeleton; phải thay đường dẫn/ID/kích thước mẫu và điền **toàn bộ sample của split/category đã khóa**. Không ghép GT và prediction bằng cách sort hai folder độc lập. Code bảo đảm đủ ba map trên mỗi sample; bạn cần đối chiếu sample list với data manifest để tránh bỏ sót cùng một sample ở cả ba candidate.

Đường dẫn tương đối được giải từ **thư mục chứa manifest**. `gt_mask` có thể là `.npy` hoặc PNG/TIFF 2-D, binary `{0,1}`, `{0,255}` hay boolean. Code chỉ đổi encoding binary sang boolean sau khi kiểm tra; không threshold GT grayscale tùy ý.

GT ảnh normal phải được cung cấp rõ là zero mask theo protocol đã khóa. Không coi thiếu GT là ảnh normal.

Prediction là probability map 2-D, finite, trong `[0,1]`, cùng H/W với GT và ảnh gốc; không sigmoid lại, min-max normalize, threshold, clip hay resize. Map có round-off ra ngoài `[0,1]` sẽ bị từ chối ở diagnostic này. Cần dùng artifact probability hợp lệ từ upstream.

Chỉ dùng một split có local GT. `test_private` và `test_private_mixed` bị từ chối. `dev_synthetic` cần GT của chính sample synthetic, không phải GT của ảnh normal nguồn.

Khai báo normalization giống nhau không chứng minh upstream giống nhau: phải đối chiếu log/config inference. Tương tự, matching H/W không chứng minh map đã đăng ký đúng pixel; stitching/alignment và overlay QA vẫn phải đúng.

Các trường bổ sung từ manifest E2, như `seg_f1_threshold` và `evaluator_sha256`, không được dùng để thay đổi tiny metric. CSV ghi SHA của source AU-PRO hiện đang chạy và kiểm tra nó không thay đổi trong analysis; SHA này không tự chứng nhận khớp bản lịch sử Day 4. Nếu cần chứng nhận bản Day 4, đối chiếu baseline đã lưu trước khi chạy.

## 5. Cách chạy sau khi bạn tự tạo file

Chạy tại root project:

```bash
python -m src.eval.tiny_analysis \
  --manifest outputs/day05_eval_input.json \
  --tiny-protocol configs/tiny_protocol.lock.json \
  --output outputs/day05_tiny/tiny_region_metrics.csv
```

Lệnh do bạn chạy sẽ tạo thư mục output nếu cần và ghi CSV mới. Nếu CSV đã tồn tại, script từ chối ghi đè. Khi lỗi dữ liệu, script dừng trước khi ghi bảng kết quả.

Không dùng `python src/eval/tiny_analysis.py` từ thư mục tùy ý; cách `python -m ...` từ root giúp Python tìm đúng package `src`.

## 6. Output `tiny_region_metrics.csv`

Mỗi dòng là **candidate × category**, tức `3 × số category` dòng dữ liệu. Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| candidate/category/split | Phạm vi kết quả |
| tiny_area_px/tiny_rule/connectivity | Cùng định nghĩa tiny cho ba candidate |
| n_images | Số ảnh của category, gồm normal |
| n_tiny_regions/n_non_tiny_regions | Số GT components trong hai nhóm |
| n_normal_pixels | Background của GT đầy đủ dùng cho FPR |
| tiny_aupro_0.05 | Metric localization trên tiny regions |
| non_tiny_aupro_0.05 | Diagnostic đối chiếu trên non-tiny |
| tiny_status/non_tiny_status | OK, NO_REGIONS hoặc NO_NORMAL_PIXELS |
| normalization_protocol | Khai báo score protocol |
| protocol_sha256/manifest_sha256/aupro_source_sha256 | Provenance của đầu vào/protocol/source metric |

Không có tiny region: `tiny_aupro_0.05` là **ô trống**, `tiny_status=NO_REGIONS`. Không có background: ô trống và `NO_NORMAL_PIXELS`. Không xuất score 0/1 hoặc NaN để che metric undefined.

Hàm `defect_regions()` trả area và phân nhóm từng component. Nếu cần xem area để kiểm tra GT, có thể gọi API này trên GT, độc lập với prediction:

```python
from src.eval.tiny_analysis import binary_mask, defect_regions

regions = defect_regions(binary_mask(gt_hw), tiny_area_px=locked_threshold)
for region in regions:
    print(region['region_id'], region['area_px'], region['is_tiny'])
```

Area và phân nhóm được tính từ GT **một lần cho mỗi category** rồi dùng cùng danh sách cho R0/R1/R2. Không đổi tiny definition theo kết quả candidate.

## 7. Diễn giải và kiểm tra

So sánh `tiny_aupro_0.05(R1) - tiny_aupro_0.05(R0)` và `tiny_aupro_0.05(R2) - tiny_aupro_0.05(R1)` trên cùng category/GT/threshold. Delta dương cho thấy candidate sau tốt hơn trên tiny trong điều kiện đang xét, không chứng minh cải thiện mọi category hay mọi seed.

Dùng cùng tập seed và các khóa model/training/inference để có ablation hợp lệ. Script diagnostic không chứng nhận fairness upstream. Việc inference hiện gán feature Local cho Context vẫn ảnh hưởng cách diễn giải R1→R2; tiny metric không khắc phục điều đó.

Chưa có prediction/GT thực để báo tiny score nghiên cứu. Không sử dụng số giả lập trong tests làm kết quả thực nghiệm.

18 tests đã PASS ngoài project, bao gồm:

- cutoff inclusive và area ở native pixel;
- diagonal 8-connectivity;
- equal region weighting;
- non-tiny foreground không thành background;
- ảnh normal được tính trong FPR;
- đối chiếu oracle độc lập với nhiều ảnh và score ties;
- không có tiny/background thì metric undefined;
- từ chối threshold theo candidate, threshold chưa khóa, normalization khác, shape/GT/score lỗi, duplicate ID, thiếu map, hidden-GT split;
- CSV đúng candidate và không ghi đè file có sẵn.

## 8. Code đầy đủ — tự tạo `src/eval/tiny_analysis.py`

```python
"""Tiny-defect diagnostics. New file: src/eval/tiny_analysis.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from src.eval.evaluator import MVTEC_AD2_CATEGORIES, MVTEC_AD2_SPLITS
from src.metrics.aupro import aupro

CANDIDATES = ("R0", "R1", "R2")
MAX_FPR = 0.05
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)  # Fixed 8-connectivity.


def load_array(base: Path, value: str) -> np.ndarray:
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def binary_mask(value: np.ndarray) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("GT must be a non-empty 2-D mask.")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
        raise ValueError("GT must contain finite binary values.")
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("GT encoding must be {0,1} or {0,255}.")
    return mask != 0


def defect_regions(gt: np.ndarray, tiny_area_px: int) -> list[dict]:
    """Classify GT components once; no prediction-dependent region selection."""
    if isinstance(tiny_area_px, bool) or not isinstance(tiny_area_px, int) or tiny_area_px < 1:
        raise ValueError("tiny_area_px must be a positive integer.")
    labels, count = label(binary_mask(gt), structure=CONNECTIVITY)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    return [
        {
            "region_id": region_id,
            "area_px": int(areas[region_id]),
            "is_tiny": bool(areas[region_id] <= tiny_area_px),
            "indices": np.flatnonzero(labels.ravel() == region_id),
        }
        for region_id in range(1, count + 1)
    ]


def region_aupro(normal_parts: list, region_parts: list) -> tuple[float | None, str]:
    """Reuse the existing AU-PRO with original background and selected regions."""
    if not region_parts:
        return None, "NO_REGIONS"
    normal_parts = [part for part in normal_parts if part.size]
    if not normal_parts:
        return None, "NO_NORMAL_PIXELS"
    # Score packing preserves the exact pixel sets and region weights.
    # Each selected GT component is one all-foreground array (one component).
    # Original background arrays contain only actual GT==0 pixels.
    maps = [part.reshape(1, -1) for part in normal_parts + region_parts]
    masks = [np.zeros((1, part.size), dtype=bool) for part in normal_parts]
    masks += [np.ones((1, part.size), dtype=bool) for part in region_parts]
    return float(aupro(maps, masks, max_fpr=MAX_FPR)["aupro"]), "OK"


def analyze(manifest_path: Path, protocol_path: Path) -> list[dict]:
    manifest_path, protocol_path = Path(manifest_path).resolve(), Path(protocol_path).resolve()
    manifest_raw, protocol_raw = manifest_path.read_bytes(), protocol_path.read_bytes()
    manifest, protocol = json.loads(manifest_raw), json.loads(protocol_raw)
    fields = {"tiny_area_px", "area_unit", "connectivity", "max_fpr",
              "locked_before_candidate_results", "threshold_basis"}
    if set(protocol) != fields:
        raise ValueError("Use exactly one global tiny protocol; candidate-specific settings are not allowed.")
    area = protocol["tiny_area_px"]
    if isinstance(area, bool) or not isinstance(area, int) or area < 1:
        raise ValueError("Lock a positive integer tiny_area_px before evaluation.")
    if protocol.get("area_unit") != "original_image_pixels" or protocol.get("connectivity") != 8:
        raise ValueError("Protocol requires original_image_pixels and 8-connectivity.")
    if protocol.get("max_fpr") != MAX_FPR:
        raise ValueError("Protocol max_fpr must be 0.05.")
    if protocol.get("locked_before_candidate_results") is not True:
        raise ValueError("Confirm the area threshold was locked before candidate results.")
    if not isinstance(protocol.get("threshold_basis"), str) or not protocol["threshold_basis"].strip():
        raise ValueError("Record the prediction-independent threshold selection basis.")

    categories, samples, split = manifest["categories"], manifest["samples"], manifest["split"]
    if not categories or len(categories) != len(set(categories)) or not samples:
        raise ValueError("Provide unique categories and a non-empty common sample list.")
    if set(categories) - set(MVTEC_AD2_CATEGORIES):
        raise ValueError("Use canonical MVTec AD 2 category names.")
    if split not in MVTEC_AD2_SPLITS or split in {"test_private", "test_private_mixed"}:
        raise ValueError("Use one locked split with available local GT.")
    if {s["category"] for s in samples} != set(categories):
        raise ValueError("Sample categories differ from the declared category set.")
    ids = [s["image_id"] for s in samples]
    if not all(isinstance(i, str) and i and i == i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Sample image_id values must be non-empty and unique.")
    if any(set(s["maps"]) != set(CANDIDATES) for s in samples):
        raise ValueError("Each sample must have exactly R0/R1/R2 prediction paths.")
    norms = manifest["normalization_by_candidate"]
    if set(norms) != set(CANDIDATES) or not all(isinstance(v, str) and v.strip() for v in norms.values()):
        raise ValueError("Declare the existing normalization protocol for R0/R1/R2.")
    if len(set(norms.values())) != 1:
        raise ValueError("R0/R1/R2 must use the same normalization protocol.")

    rows = []
    metric_source = Path(aupro.__code__.co_filename)
    metric_sha = hashlib.sha256(metric_source.read_bytes()).hexdigest()
    for category in categories:
        plans = []
        for sample in (s for s in samples if s["category"] == category):
            gt = binary_mask(load_array(manifest_path.parent, sample["gt_mask"]))
            hw = sample["original_hw"]
            if len(hw) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in hw):
                raise ValueError(f"Invalid original_hw: {sample['image_id']}")
            if gt.shape != tuple(hw):
                raise ValueError(f"GT must match original_hw: {sample['image_id']}")
            plans.append((sample, gt, defect_regions(gt, area)))
        n_tiny = sum(r["is_tiny"] for _, _, regions in plans for r in regions)
        n_non_tiny = sum(not r["is_tiny"] for _, _, regions in plans for r in regions)
        n_normal = sum(int((~gt).sum()) for _, gt, _ in plans)

        for candidate in CANDIDATES:
            normal_parts, tiny_parts, non_tiny_parts = [], [], []
            for sample, gt, regions in plans:
                score = load_array(manifest_path.parent, sample["maps"][candidate])
                if score.shape != gt.shape or score.dtype.kind not in "uif":
                    raise ValueError(f"Invalid probability map: {candidate}/{sample['image_id']}")
                if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
                    raise ValueError(f"Probability map must be finite in [0,1]: {candidate}/{sample['image_id']}")
                normal_parts.append(score[~gt])  # Non-tiny defects NEVER become background.
                for region in regions:
                    parts = tiny_parts if region["is_tiny"] else non_tiny_parts
                    parts.append(score.ravel()[region["indices"]])
            tiny_value, tiny_status = region_aupro(normal_parts, tiny_parts)
            non_tiny_value, non_tiny_status = region_aupro(normal_parts, non_tiny_parts)
            rows.append({
                "candidate": candidate, "category": category, "split": split,
                "tiny_area_px": area, "tiny_rule": "area_px <= tiny_area_px",
                "connectivity": 8, "max_fpr": MAX_FPR,
                "n_images": len(plans), "n_tiny_regions": n_tiny,
                "n_non_tiny_regions": n_non_tiny, "n_normal_pixels": n_normal,
                "tiny_aupro_0.05": tiny_value, "tiny_status": tiny_status,
                "non_tiny_aupro_0.05": non_tiny_value, "non_tiny_status": non_tiny_status,
                "normalization_protocol": norms[candidate],
                "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "aupro_source_sha256": metric_sha,
            })
    if metric_sha != hashlib.sha256(metric_source.read_bytes()).hexdigest():
        raise ValueError("AU-PRO source changed during analysis.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiny-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new CSV path; existing output is not overwritten.")
    rows = analyze(args.manifest, args.tiny_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

```
