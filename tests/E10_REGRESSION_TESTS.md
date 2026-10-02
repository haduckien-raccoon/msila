# E10 — Regression Tests for Evaluator & Efficiency

**Files**

```text
tests/test_evaluator.py
tests/test_efficiency.py
```

E10 chỉ kiểm tra regression cho E1–E6 utilities. Không thêm metric mới, không
thay evaluator, không thay benchmark protocol.

## 1. Mục tiêu khoa học

Regression test không chứng minh mô hình đạt accuracy cao. Nó chứng minh rằng
các **software contracts đã khóa** không bị thay đổi âm thầm khi code tiếp tục
được chỉnh sửa.

E10 tập trung vào hai nhóm:

```text
Evaluator:
E1 AU-PRO + SegF1
E2 unified evaluation contract
E3 anomaly-map QA/original-size gate

Efficiency:
E4 parameter counting
E5 latency protocol
E6 peak-VRAM protocol
```

---

## 2. `tests/test_evaluator.py`

### 2.1 Perfect-reference case

Anomaly pixels được cho score cao hơn mọi normal pixel và threshold SegF1 phân
loại hoàn hảo.

Expected:

\[
AU\text{-}PRO_{0.05}=1,\qquad SegF1=1.
\]

Test chạy **actual evaluator + actual metric implementation**, không mock metric
ở case này.

### 2.2 Hand-counted SegF1

Synthetic 2×2:

```text
GT:
1 0
0 0

score:
0.9 0.8
0.2 0.1
```

threshold \(T=0.5\):

\[
TP=1,\quad FP=1,\quad FN=0.
\]

Do đó:

\[
Precision=\frac12,\qquad Recall=1
\]

và:

\[
F1=\frac{2TP}{2TP+FP+FN}=\frac23.
\]

Test yêu cầu implementation trả đúng các giá trị này.

### 2.3 E3 must run before metric computation

Một anomaly map sai shape được đưa vào evaluator.

Test monkeypatch AU-PRO/SegF1 bằng hàm sẽ crash nếu bị gọi. Kết quả mong đợi:

```text
E3 QA raises first
AU-PRO calls = 0
SegF1 calls = 0
```

Điều này khóa pipeline:

```text
QA -> metric
```

không phải:

```text
metric -> QA
```

### 2.4 Contract rejection

Regression tests khóa các trường hợp phải reject:

```text
NaN/Inf
score ngoài [0,1]
GT không binary
map/mask khác shape
map+mask cùng shape nhưng khác original H/W
duplicate sample identity
mixed split
private split có hidden GT
expected category thiếu
invalid SegF1 threshold
coordinate space = tile
```

### 2.5 Artifact writing

Test kiểm tra:

```text
metrics.json
qa_report.json
```

được ghi thành công, `.tmp` không còn sau atomic replace, và nội dung JSON đọc
lại đúng với object evaluator trả về.

---

## 3. `tests/test_efficiency.py`

### 3.1 E4 manual parameter count

Toy model:

```text
Conv2d(3 -> 4, 3×3, bias)
= 4×3×3×3 + 4
= 112

Linear(4 -> 2, bias)
= 2×4 + 2
= 10
```

Linear bị freeze:

\[
N_{total}=122,\quad
N_{trainable}=112,\quad
N_{frozen}=10.
\]

Test cũng khóa trường hợp shared/tied `nn.Parameter`: cùng Parameter object phải
được đếm **một lần**.

### 3.2 MS-ILA closed-form regression

Với test configuration:

```text
C = 384
r = 4 -> bottleneck = 96
d = 128
k = 3
```

manual formula đã khóa ở E4 phải tiếp tục cho:

```text
adapters   = 225,507
projection = 147,840
fusion     = 134
decoder    = 73,857
--------------------------------
total      = 447,338
```

Nếu architecture implementation/formula thay đổi ngoài chủ ý, test báo regression.

---

## 4. E5 latency test phải deterministic

Không nên dùng real wall-clock để kiểm tra chính xác `mean=...` trong unit test
vì OS scheduling tạo noise.

E10 monkeypatch:

```python
time.perf_counter_ns()
```

để mỗi measured inference chính xác 2 ms.

Như vậy test được **protocol logic**, không benchmark hardware:

```text
warm-up count
timed iteration count
round count
mean
median
p95
std
CV(round medians)
inference_mode
```

Expected:

\[
mean=median=p95=2\;ms
\]

và:

\[
CV=0.
\]

Một test riêng tạo round medians `1, 2, 4 ms` để chắc chắn stability gate chuyển
sang `FAIL`.

Đây là regression test; số latency thật vẫn phải đo trên GPU Day-04 bằng E5.

---

## 5. E6 peak-VRAM regression không cần GPU vật lý

CI thường có thể không có CUDA. Nếu E10 yêu cầu GPU thật thì regression suite
sẽ không portable.

Do đó có hai test khác nhau:

### CPU rejection

```text
benchmark_peak_vram(... device="cpu")
```

phải raise `CUDAUnavailableError`.

### Simulated CUDA protocol

Monkeypatch CUDA allocator API để kiểm tra thứ tự logic:

```text
warm-up
sync
reset_peak_memory_stats
read baseline
inference
sync
query peak
```

Synthetic allocator values:

```text
baseline allocated = 100 bytes
peak allocated     = 180 bytes
```

Expected:

\[
incremental=180-100=80\ bytes.
\]

Test cũng khóa:

```text
empty_cache_before_measurement = False
```

đúng E6 protocol.

Đây là unit regression của **measurement algorithm**, không được diễn giải là
kết quả VRAM thực nghiệm.

---

## 6. Scope fingerprint regression

Hai benchmark scope có cùng nội dung nhưng `extra` dict khác insertion order
phải có cùng SHA-256.

Thay batch size:

```text
batch=1 -> batch=2
```

phải đổi fingerprint.

Điều này bảo vệ fairness của E5/E6 candidate comparison.

---

## 7. Cách chạy

Sau khi E1–E6 files đã được đặt đúng trong repo:

```bash
pytest -q \
  tests/test_evaluator.py \
  tests/test_efficiency.py
```

Muốn xem chi tiết:

```bash
pytest -vv \
  tests/test_evaluator.py \
  tests/test_efficiency.py
```

---

## 8. Hard PASS

E10 PASS khi:

```text
tests/test_evaluator.py   -> all PASS
tests/test_efficiency.py  -> all PASS
```

Không chấp nhận:

```text
xfail
skip
NaN result
CUDA-only test bị bỏ toàn bộ
```

E6 logic vẫn được unit-test bằng simulated CUDA ngay cả khi CI không có GPU.

---

## 9. Phạm vi kết luận

E10 PASS cho phép kết luận:

> Metric/evaluator contracts E1–E3 và efficiency measurement utilities E4–E6
> vẫn hành xử đúng với các reference/negative cases đã khóa.

E10 PASS **không** cho phép kết luận:

```text
AU-PRO implementation giống tuyệt đối official private server
candidate nào tốt nhất
GPU latency/VRAM thực tế là bao nhiêu
```

Các kết luận đó thuộc protocol/benchmark tương ứng, không phải regression test.
