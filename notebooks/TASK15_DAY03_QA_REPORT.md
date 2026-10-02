# Day-3 — Task 15: Day-03 QA Report trên Colab

## 1. Mục tiêu

Notebook:

```text
notebooks/Day03_QA_Report.ipynb
```

dùng để tạo bằng chứng QA cuối Day-3, đặc biệt xác nhận gate còn thiếu:

```text
Cached feature
→ model
→ loss
→ backward
→ optimizer.step
→ PASS
```

Notebook không đo AU-PRO\(_{0.05}\), không đánh giá generalization và không phải
benchmark khoa học cuối cùng.

## 2. Quy trình Colab

Notebook thực hiện theo thứ tự:

```text
clone GitHub repository
→ ghi commit SHA
→ nếu Task 14 chưa có trên branch: upload/overlay ZIP Task 14
→ cài dependency tối thiểu
→ chạy controlled cached integration pytest
→ chạy toàn bộ nhóm test Day-3
→ tùy chọn chạy one-step trên cache thật
→ sinh JSON + Markdown report
```

## 3. Hai mức QA

### Controlled cached integration — bắt buộc

Chạy:

```text
tests/test_day03_cached_train_step.py::test_cached_feature_to_optimizer_step_end_to_end
```

Test tạo cache thông qua `FeatureCacheWriter`, đọc lại bằng
`CachedFeatureDataset`, rồi thực hiện:

```text
Adapter
→ Alignment
→ Projection
→ Fusion
→ Decoder
→ BCE+Dice
→ backward
→ AdamW.step
```

và kiểm tra parameter thực sự thay đổi.

### Real cache — khuyến nghị / có thể bắt buộc

Cấu hình:

```python
FEATURE_CACHE_DIR = "..."
FEATURE_TRAIN_INDEX = "..."
FEATURE_MASK_ROOT = None
FEATURE_FUSION_DIM = 64
REQUIRE_REAL_CACHE = True
```

Notebook gọi test external của Task 14 trên cache do project thực tạo.

Nếu chưa cấu hình cache thật, trạng thái phải là:

```text
SKIPPED
```

chứ không được ghi thành PASS.

## 4. Overlay Task 14

Nếu Task 14 chưa được commit vào GitHub branch, đặt:

```python
AUTO_UPLOAD_TASK14_IF_MISSING = True
```

Notebook sẽ yêu cầu upload:

```text
day3_task14_cached.zip
```

và giải nén lên repo clone.

Nếu đã commit Task 14 vào branch thì không cần upload.

## 5. Output

Notebook xuất:

```text
reports/day03_qa_report.json
reports/day03_qa_report.md
```

Report chứa:

```text
Git commit SHA
Python / PyTorch / CUDA
controlled cached integration status
Day-3 test status
real-cache status
overall PASS/FAIL
scientific scope
```

## 6. Quy tắc PASS

Nếu:

```python
REQUIRE_REAL_CACHE = False
```

overall PASS yêu cầu controlled Task-14 gate và Day-3 tests PASS; real-cache có
thể là `SKIPPED`.

Nếu:

```python
REQUIRE_REAL_CACHE = True
```

real-cache one-step cũng bắt buộc PASS.

Đối với bàn giao Day-3 của project, nên đặt `REQUIRE_REAL_CACHE=True` khi cache
thật đã sẵn sàng.

## 7. Kết luận khoa học hợp lệ

Day-03 QA PASS chỉ hỗ trợ kết luận:

> Pipeline training từ cached frozen features tới optimizer update hoạt động
> nhất quán theo tensor/gradient contract đã khóa.

Không được suy ra:

```text
MS-ILA generalize tốt
AU-PRO0.05 cao
MS-ILA tốt hơn baseline
synthetic anomaly đại diện defect thật
```
