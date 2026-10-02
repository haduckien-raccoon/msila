# E8 — Adapter Candidate Analysis & Selection

**File:** `src/analysis/select_adapter.py`  
**Input:** E7 `day04_category_runs.json` + `day04_completeness.json`  
**Output:** `selection_report.json`, `candidate_summary.csv`, `selection_report.md`

E8 chỉ làm:

```text
mean ± std
per-category delta
efficiency summary
pre-locked selection rule
```

Không làm E9 fairness audit và không làm E10 regression tests.

---

## 1. Đơn vị thống kê đúng

E7 có một dòng cho mỗi:

\[
(candidate,\ category,\ seed).
\]

Với candidate \(c\), category \(k\), seed \(s\), ký hiệu:

\[
A_{c,k,s}=AU\text{-}PRO_{0.05}.
\]

Để giữ đúng macro-category protocol, E8 **không** gộp pixel hoặc số ảnh giữa
các category.

Đầu tiên tính macro AU-PRO trong từng seed:

\[
M_{c,s}
=
\frac{1}{K}
\sum_{k=1}^{K} A_{c,k,s}.
\]

Sau đó mới tổng hợp các replication seed:

\[
\bar M_c
=
\frac{1}{S}
\sum_{s=1}^{S}M_{c,s}.
\]

Standard deviation dùng **sample standard deviation**:

\[
s_c
=
\sqrt{
\frac{1}{S-1}
\sum_s(M_{c,s}-\bar M_c)^2
}.
\]

Tức là `ddof=1`, phù hợp khi các seed quan sát được xem như các replication của
experiment chứ không phải toàn bộ population.

Nếu chỉ có một seed, sample standard deviation không xác định. E8 vì vậy yêu cầu
`min_seeds >= 2`; với protocol hiện tại nên khóa 3 seed nếu Day-04 thực sự dùng
`42, 17, 2026`.

---

## 2. Per-category delta phải paired theo seed

Nếu `reference_candidate = r`, delta không được tính bằng cách bỏ matching seed.

E8 tính:

\[
\Delta_{c,k,s}
=
A_{c,k,s}-A_{r,k,s}.
\]

Sau đó:

\[
\bar\Delta_{c,k}
=
\frac{1}{S}\sum_s\Delta_{c,k,s}.
\]

Đây là phép so sánh paired tự nhiên vì candidate và reference được chạy trên
cùng category và cùng seed.

Report lưu cả:

```text
mean delta
sample std của delta
số category có mean delta > epsilon đã khóa
```

---

## 3. Primary metric

E8 v1 khóa primary model-selection metric là:

```text
mean macro AU-PRO_0.05 across seeds
```

Tức:

\[
\boxed{\bar M_c}
\]

và hướng là `maximize`.

SegF1 vẫn được tổng hợp mean±std nhưng không âm thầm thay AU-PRO thành primary
metric.

Điều này bám đúng `docs/metric_protocol_v1.md` của repo: AU-PRO\(_{0.05}\) là
primary metric, còn SegF1 là secondary threshold-dependent metric.

---

## 4. Efficiency được dùng thế nào?

E8 báo cáo:

```text
total/trainable params
mean±std của median latency từ E7 observations
mean±std của p95 latency
mean±std peak allocated VRAM
```

Parameter count phải cố định cho cùng một candidate.

Latency/VRAM chỉ được so sánh khi E7 đã xác nhận cùng
`benchmark_scope_fingerprint`.

Efficiency có thể được dùng theo hai cách **nếu đã khóa trong rule trước**:

```text
1. eligibility constraint
2. tie-breaker
```

Ví dụ:

```json
"eligibility": {
  "max_trainable_parameters_ratio_vs_reference": 1.25,
  "max_latency_median_ratio_vs_reference": 1.15,
  "max_peak_vram_ratio_vs_reference": 1.10
}
```

Nếu không khóa budget như vậy từ trước, E8 chỉ báo cáo efficiency và không được
tự tạo constraint sau khi nhìn kết quả.

---

## 5. Selection rule phải tồn tại trước khi xem kết quả

E8 không hard-code một rule dựa trên kết quả.

Input bắt buộc:

```text
configs/day04_selection_rule.json
```

với:

```json
{
  "schema_version": "msila.e8.selection_rule.v1",
  "rule_id": "adapter_day04_v1",
  "locked_before_results": true,
  "locked_at_utc": "2026-10-02T05:00:00Z",
  "expected_split": "dev_synthetic",
  "expected_e7_plan_sha256": "...",
  "reference_candidate": "candidate_01",
  "min_seeds": 3
}
```

E8 tính SHA-256 của toàn bộ rule và ghi hash đó vào report.

### Giới hạn khoa học cần nói rõ

```text
locked_before_results=true
+ timestamp
+ SHA-256
```

là **audit trail**, không tự chứng minh bằng mật mã rằng rule thật sự tồn tại
trước khi xem kết quả.

Muốn provenance mạnh hơn, rule phải được commit/tag hoặc lưu vào hệ thống có
timestamp **trước khi chạy/đọc result**.

E8 không giả vờ rằng một boolean có thể chứng minh preregistration.

---

## 6. Split hiện tại cần được nhóm khóa rõ

Repo hiện có tài liệu mô tả vai trò split không hoàn toàn đồng nhất:

```text
docs/research_protocol_v1.md
    VALIDATION -> model validation/model selection/calibration

docs/data_card.md
    DEV-synthetic -> architecture/hyperparameter selection
    VALIDATION -> normal-only calibration
```

E8 **không tự sửa hoặc chọn thay nhóm**.

Rule bắt buộc có:

```json
"expected_split": "..."
```

và E8 reject nếu E7 result dùng split khác.

Trước experiment chính thức, nhóm cần thống nhất lại protocol version.

---

## 7. Eligibility

Các constraint sau là optional nhưng nếu dùng phải khóa trước:

```text
min_macro_aupro_delta_vs_reference
min_categories_improved_vs_reference
category_improvement_epsilon

max_trainable_parameters
max_trainable_parameters_ratio_vs_reference

max_latency_median_ms
max_latency_median_ratio_vs_reference

max_peak_vram_mib
max_peak_vram_ratio_vs_reference
```

Ví dụ practical criterion kiểu:

\[
\Delta Macro\ AU\text{-}PRO_{0.05}\ge0.01
\]

và tốt hơn reference ở ít nhất `6/8` category chỉ nên bật nếu đó thực sự là rule
đã được chốt cho comparison tương ứng.

E8 không tự áp điều kiện `+1 percentage point, 6/8` cho mọi bài toán vì criterion
đó trong research protocol hiện gắn với RQ1 adapter-vs-frozen comparison, không
tự động đồng nghĩa với mọi hyperparameter-selection experiment.

---

## 8. Tie handling

Primary shortlist:

```text
giữ candidate có macro_aupro_mean
nằm trong primary_tolerance so với best
```

Nếu còn nhiều candidate, E8 chạy các tie-breaker **theo đúng thứ tự trong rule**.

Ví dụ:

```json
"tie_breakers": [
  {
    "metric": "trainable_parameters",
    "direction": "min",
    "tolerance": 0
  },
  {
    "metric": "latency_median_ms_mean",
    "direction": "min",
    "tolerance": 0.05
  },
  {
    "metric": "peak_vram_mib_mean",
    "direction": "min",
    "tolerance": 1.0
  }
]
```

Nếu sau toàn bộ tie-breaker vẫn còn >1 candidate:

```text
selection.status = AMBIGUOUS_TIE
selected_candidate = null
```

E8 **không dùng alphabetic candidate ID làm tie-break khoa học giả tạo**.

---

## 9. Output

### `selection_report.json`

Machine-readable:

```text
rule hash
E7 plan hash
grid
aggregation definition
candidate summaries
per-category paired deltas
eligibility
selection trace
selected candidate / ambiguous / no eligible candidate
```

### `candidate_summary.csv`

Một dòng/candidate:

```text
macro AU-PRO mean±std
delta vs reference mean±std
category improved count
SegF1 mean±std
params
latency mean±std
VRAM mean±std
eligibility
```

### `selection_report.md`

Bản ngắn để nhóm đọc/checkpoint.

---

## 10. Cách chạy

```bash
python src/analysis/select_adapter.py \
  --runs outputs/day04/table/day04_category_runs.json \
  --completeness outputs/day04/table/day04_completeness.json \
  --rule configs/day04_selection_rule.json \
  --output-dir outputs/day04/selection
```

---

## 11. PASS

E8 analysis PASS khi:

```text
E7 completeness = PASS
E7 grid đầy đủ
rule schema đúng
locked_before_results = true
rule split khớp E7
rule E7-plan hash khớp nếu đã khóa
reference candidate tồn tại
>= min_seeds
candidate/category/seed grid cân bằng
cùng efficiency scope
parameter signature ổn định
mean±std tính đúng theo seed
paired per-category delta tính đúng
selection chỉ dùng rule đã khóa
```

`analysis_status=PASS` không bắt buộc phải có winner.

Ba trạng thái selection hợp lệ:

```text
SELECTED
AMBIGUOUS_TIE
NO_ELIGIBLE_CANDIDATE
```

`AMBIGUOUS_TIE` khoa học hơn việc tự chế thêm tie-breaker sau khi thấy kết quả.

---

## 12. Giới hạn của E8

E8 chưa xác minh:

```text
config của các candidate chỉ khác r,d
```

Đó chính xác là E9.

Vì vậy mọi `selected_candidate` từ E8 phải được hiểu là:

> selected by the pre-locked E8 rule, conditional on passing E9 fairness audit.
