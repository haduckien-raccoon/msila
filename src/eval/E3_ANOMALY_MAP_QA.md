# E3 — Anomaly-map QA cho MS-ILA

## 1. Mục tiêu

E3 chạy **trước khi tính AU-PRO/SegF1** và chỉ trả lời một câu hỏi:

> anomaly map có thực sự hợp lệ ở cùng pixel grid với GT và ở đúng kích thước ảnh gốc hay không?

File triển khai:

```text
src/eval/evaluator.py
```

E3 không sửa công thức AU-PRO, không tune threshold, không đo params/latency/VRAM và không chọn model.

---

## 2. Cơ sở khoa học của contract

Trong `docs/data_card.md` của project, pipeline đã khóa:

```text
- giữ nguyên aspect ratio;
- mask dùng nearest-neighbor;
- inference dùng overlapping tiles;
- final prediction phải được stitch về original image resolution.
```

Vì AU-PRO/SegF1 là pixel-level metrics, anomaly map và GT phải nằm trên cùng pixel grid trước khi đếm TP/FP/FN hoặc overlap theo connected region.

MVTec AD 2 cũng cung cấp pixel-precise anomaly annotations cho public test; do đó một score map lệch kích thước/toạ độ so với GT làm metric không còn diễn giải đúng theo pixel localization.

Repo hiện tại còn có `src/utils/TASK11_VISUALIZATION.md` khóa H/W của image-mask-logits phải giống nhau trong training QA. E3 đưa cùng nguyên tắc hình học sang evaluation sau stitching.

---

## 3. Contract đầu vào mới của E3

Mỗi record vẫn dùng contract E2, nhưng phải có thêm kích thước ảnh gốc:

```python
{
    "anomaly_map": probability_hw,   # [H,W], finite, [0,1]
    "gt_mask": gt_hw,                # [H,W], binary
    "meta": {
        "image_id": "...",
        "category": "fabric",
        "split": "test_public",

        # Một trong hai cách dưới đây:
        "original_hw": (H, W),
        # hoặc loader hiện tại đã có:
        "H": H,
        "W": W,

        # optional provenance:
        "coordinate_space": "original_image",
    },
}
```

`original_hw` được ưu tiên. Vì `src/data/loader.py` hiện đã trả `meta.H` và `meta.W` tại native resolution, E3 chấp nhận cặp này để không phá contract hiện tại.

Nếu cả `original_hw` và `H/W` cùng tồn tại nhưng khác nhau, sample bị FAIL thay vì đoán giá trị nào đúng.

---

## 4. Các kiểm tra hard-gate

Một sample chỉ PASS khi toàn bộ điều kiện sau đúng:

```text
1. anomaly_map là mảng 2-D [H,W]
2. anomaly_map numeric, finite, không NaN/Inf
3. anomaly_map nằm trong [0,1]
4. gt_mask là 2-D và binary {0,1} hoặc {0,255}
5. anomaly_map.shape == gt_mask.shape
6. anomaly_map.shape == original_hw
7. gt_mask.shape == original_hw
```

Điều kiện hình học cốt lõi là:

\[
(H_s,W_s)=(H_{gt},W_{gt})=(H_{orig},W_{orig})
\]

trong đó:

- \((H_s,W_s)\): anomaly-map size;
- \((H_{gt},W_{gt})\): GT-mask size;
- \((H_{orig},W_{orig})\): native/original image size.

Nếu một sample FAIL, evaluator **không tính metric**.

---

## 5. Alignment được kiểm tra đến mức nào?

E3 kiểm tra **geometric compatibility**:

```text
anomaly map H/W == GT H/W == original H/W
```

Điều này loại được các lỗi phổ biến:

```text
- quên stitch tile về full resolution;
- score map còn ở feature resolution;
- resize sai H/W;
- map và mask khác resolution;
- metadata H/W bị sai.
```

Nhưng cần phân biệt rõ:

> Hai mảng có cùng H/W không chứng minh tuyệt đối rằng chúng cùng semantic pixel registration.

Ví dụ, một heatmap bị `flip left-right` vẫn có H/W giống GT. Vì vậy E3 hỗ trợ provenance metadata tùy chọn:

```python
"coordinate_space": "original_image"
```

hoặc:

```python
"anomaly_map_space": "original_image"
"gt_mask_space": "original_image"
```

Nếu metadata này được khai báo thì E3 yêu cầu cả hai cùng ở `original_image`. Nếu không khai báo, sample vẫn có thể PASS theo contract hình học, nhưng `qa_report.json` ghi rõ là coordinate-space provenance chưa được khai báo.

Muốn phát hiện flip/translation cùng kích thước cần thêm transformation provenance hoặc overlay định tính RGB/GT/heatmap; không thể suy ra chắc chắn chỉ từ hai ma trận cùng shape.

---

## 6. API QA riêng

Có thể chạy E3 mà chưa tính metric:

```python
from src.eval.evaluator import build_anomaly_map_qa_report

qa = build_anomaly_map_qa_report(
    records,
    output_path="outputs/day04/qa_report.json",
)

assert qa["summary"]["status"] == "PASS"
assert qa["summary"]["valid_fraction"] == 1.0
```

Đây là cách phù hợp để kiểm tra artifact anomaly-map ngay sau stitching.

---

## 7. E3 được tích hợp vào evaluator thế nào?

`evaluate_segmentation_records()` chạy theo thứ tự:

```text
records
  ↓
E3 anomaly-map QA
  ↓ only if 100% PASS
general E2 contract validation
  ↓
AU-PRO0.05 + SegF1
  ↓
metrics.json
```

Ví dụ:

```python
metrics = evaluate_segmentation_records(
    records,
    seg_f1_threshold=0.5,
    expected_split="test_public",
    expected_categories=("fabric", "vial", "wallplugs"),
    output_path="outputs/day04/metrics.json",
)
```

Khi có `output_path`, nếu không chỉ định `qa_output_path`, evaluator tự ghi:

```text
outputs/day04/metrics.json
outputs/day04/qa_report.json
```

Nếu QA FAIL:

```text
qa_report.json vẫn được ghi
metrics.json không được tạo mới bởi lần evaluate đó
metric computation không chạy
```

Có thể chỉ định path riêng:

```python
evaluate_segmentation_records(
    ...,
    output_path="outputs/day04/metrics.json",
    qa_output_path="outputs/day04/map_qa.json",
)
```

---

## 8. Schema `qa_report.json`

Các trường chính:

```json
{
  "schema_version": "msila-anomaly-map-qa-v1",
  "dataset": "mvtec_ad2",
  "summary": {
    "status": "PASS",
    "n_samples": 120,
    "n_pass": 120,
    "n_fail": 0,
    "valid_fraction": 1.0
  },
  "per_sample": [
    {
      "image_id": "...",
      "status": "PASS",
      "anomaly_map_hw": [H, W],
      "gt_mask_hw": [H, W],
      "original_hw": [H, W],
      "score_min": 0.0,
      "score_max": 1.0,
      "checks": {
        "probability_map_valid": true,
        "binary_gt_valid": true,
        "map_mask_same_hw": true,
        "map_matches_original_hw": true,
        "mask_matches_original_hw": true,
        "declared_coordinate_space_valid": true
      },
      "issues": []
    }
  ]
}
```

Report chỉ lưu metadata/diagnostics, không dump toàn bộ anomaly map nên nhỏ và dễ audit.

---

## 9. Điều kiện E3 PASS

Hard PASS cho một run:

\[
\text{valid\_fraction}=1.0
\]

hay tương đương:

```text
n_fail = 0
n_pass = n_samples
```

Không dùng rule kiểu "95% sample hợp lệ" vì chỉ một map sai kích thước/alignment cũng có thể làm metric pixel-level của run bị sai.

---

## 10. Kiểm chứng đã thực hiện

Bản code này đã được smoke-test với các trường hợp:

```text
[PASS] 3 synthetic samples hợp lệ -> QA PASS 3/3
[PASS] perfect maps vẫn cho AU-PRO0.05 = 1.0
[PASS] perfect maps vẫn cho SegF1 = 1.0
[PASS] map/GT khác H/W -> reject
[PASS] map+GT cùng H/W nhưng khác original_hw -> reject
[PASS] anomaly_map chứa NaN -> reject
[PASS] thiếu original H/W -> reject
[PASS] declared coordinate_space="tile" -> reject
[PASS] loader-native meta.H/meta.W không khai báo coordinate_space -> vẫn hợp lệ
```

Không thêm `tests/test_evaluator.py` ở task này vì regression-test suite đã được dành riêng cho E10. Smoke test ở đây chỉ dùng để xác nhận implementation E3 hiện tại hoạt động đúng trước khi bàn giao.

---

## 11. Phạm vi chưa làm

E3 không thực hiện:

```text
E4 params
E5 latency
E6 peak VRAM
E7 candidate aggregation
E8 model selection
E9 fairness audit
E10 regression suite
```

E3 chỉ khóa:

```text
anomaly-map validity + GT validity + same H/W + original-size consistency
```

trước khi cho phép metric chạy.

---

## 12. Nguồn đối chiếu

Trong repository:

```text
docs/data_card.md
src/data/loader.py
src/utils/TASK11_VISUALIZATION.md
docs/metric_protocol_v1.md
```

Nguồn benchmark chính thức:

```text
MVTec AD 2 — official dataset page
The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection
```

Trang chính thức xác nhận public test có pixel-precise annotations và private GT không công khai; do đó local pixel-level QA/evaluation phải được thực hiện trên dữ liệu có GT hợp lệ.
