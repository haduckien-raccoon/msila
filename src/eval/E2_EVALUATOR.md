# E2 — Unified evaluator for MS-ILA

## 1. Mục tiêu

`src/eval/evaluator.py` là lớp **orchestration** cho đánh giá segmentation. Module này **không cài lại công thức metric**; nó kiểm tra contract đầu vào rồi gọi trực tiếp:

- `src.metrics.aupro.aggregate_aupro`
- `src.metrics.segf1.aggregate_seg_f1`

Như vậy AU-PRO chỉ có **một source of truth** trong `src/metrics/aupro.py`, đúng với E1.

## 2. Contract đầu vào

Mỗi sample:

```python
{
    "anomaly_map": probability_hw,   # [H,W], finite, [0,1]
    "gt_mask": mask_hw,              # [H,W], binary
    "meta": {
        "image_id": "...",          # hoặc "path"
        "category": "fabric",
        "split": "test_public",
    },
}
```

Evaluator kiểm tra trước khi tính metric:

1. `anomaly_map` và `gt_mask` đều là mảng 2-D;
2. hai mảng có **cùng H×W**;
3. anomaly map hữu hạn và nằm trong `[0,1]`;
4. GT là binary `{0,1}` hoặc `{0,255}`;
5. category thuộc 8 category MVTec AD 2;
6. split hợp lệ và toàn bộ một lần evaluate chỉ chứa **một split**;
7. sample ID không trùng;
8. mỗi category phải có anomalous region và normal pixel để AU-PRO có nghĩa.

`test_private` và `test_private_mixed` bị từ chối cho local pixel-level evaluation vì GT của hai split này là hidden theo data card của project; kết quả chính thức phải lấy từ benchmark server.

## 3. Vì sao anomaly map phải là probability `[0,1]`?

AU-PRO bản thân là metric dựa trên thứ hạng/threshold nên có thể chạy trên score hữu hạn bất kỳ. Tuy nhiên **unified evaluator còn tính SegF1 bằng một threshold cố định**. Vì thế E2 khóa contract ở probability map `[0,1]` để threshold có cùng ý nghĩa giữa các run/candidate.

Nếu decoder trả logits `z`, chuyển trước evaluator:

```python
probability = torch.sigmoid(z)
```

Evaluator không tự sigmoid vì tự suy đoán representation sẽ che lỗi pipeline.

## 4. AU-PRO không được average theo từng ảnh

Evaluator gom toàn bộ image của **một category**, gọi `aggregate_aupro`, rồi mới lấy macro average giữa category. Đây là đúng granularity đã khóa trong `docs/metric_protocol_v1.md`:

```text
per-category AU-PRO -> unweighted macro across categories
```

Không tính `mean(AU-PRO từng ảnh)`.

AU-PRO luôn bị khóa:

```text
max_fpr = 0.05
```

Caller không có tham số đổi giá trị này trong evaluator.

## 5. SegF1: không tune threshold trên tập đang đánh giá

Evaluator bắt buộc caller truyền:

```python
seg_f1_threshold=<fixed threshold>
```

Nó **không gọi `best_threshold_f1()`**. Threshold phải được khóa trước từ calibration/development protocol. Điều này ngăn việc tối ưu threshold trực tiếp trên `test_public` rồi báo chính score đó như final result.

## 6. Cách dùng

```python
from src.eval.evaluator import evaluate_segmentation_records

records = [
    {
        "anomaly_map": score_hw,
        "gt_mask": gt_hw,
        "meta": {
            "image_id": image_id,
            "category": "fabric",
            "split": "test_public",
        },
    },
    # ...
]

metrics = evaluate_segmentation_records(
    records,
    seg_f1_threshold=0.5,              # phải là threshold đã khóa
    expected_split="test_public",
    expected_categories=("fabric", "vial", "wallplugs"),
    output_path="outputs/day04/metrics.json",
)
```

`expected_categories` nên truyền khi chạy pilot/full benchmark để evaluator phát hiện category thiếu hoặc thừa thay vì âm thầm tính macro trên tập không đầy đủ.

## 7. Schema `metrics.json`

```json
{
  "schema_version": "msila-evaluator-v1",
  "metric_protocol_version": "1.0",
  "dataset": "mvtec_ad2",
  "split": "test_public",
  "n_samples": 120,
  "n_categories": 3,
  "categories": ["fabric", "vial", "wallplugs"],
  "settings": {
    "aupro_max_fpr": 0.05,
    "seg_f1_threshold": 0.5
  },
  "metrics": {
    "aupro_0.05": {
      "per_category": {},
      "macro": 0.0
    },
    "seg_f1": {
      "per_category": {},
      "macro": 0.0
    }
  },
  "counts": {
    "per_category": {}
  },
  "validation": {
    "status": "PASS"
  }
}
```

File được ghi **atomically** (`.tmp -> replace`) và dùng `allow_nan=False`, vì `NaN/Inf` trong JSON sẽ làm artifact khó tái lập và không phải JSON chuẩn.

## 8. Điều kiện E2 PASS

E2 PASS khi một run hợp lệ tạo được `metrics.json` và evaluator chủ động reject các lỗi sau:

```text
shape mismatch
score ngoài [0,1]
NaN/Inf
GT không binary
split sai / trộn nhiều split
category ngoài MVTec AD 2
category thiếu so với expected_categories
duplicate sample ID
category không có anomalous region hoặc normal pixel
```

## 9. Những gì cố ý chưa làm ở E2

Không mở rộng sang các task kế tiếp:

- chưa kiểm tra `original_hw`, coordinate alignment, seam/stitching QA — **E3**;
- chưa đo params/VRAM/latency — **E4–E6**;
- chưa loop candidate/seed và không ranking model — **E7–E8**;
- chưa làm fairness audit/regression suite tổng thể — **E9–E10**.

E2 chỉ có một trách nhiệm: **validated records -> locked metrics -> deterministic `metrics.json`**.
