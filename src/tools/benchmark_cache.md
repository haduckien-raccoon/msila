# `benchmark_cache.py` — benchmark Online DINOv3 vs Feature Cache

## Mục tiêu

Task TV1:

```text
Online extraction                     Cached training feature acquisition

x_local/x_context                     FeatureCacheReader
      ↓                                      ↓
Frozen DINOv3                        6 cached tensors
      ↓                                      ↓
6 tensors                            CPU materialize / H2D
      └──────────── timing ────────────────┘
```

Benchmark chỉ được báo kết quả sau khi **correctness gate** PASS:

$$
e_{\max}(F)
=
\max_j
\left|
F_{\text{online},j}
-
F_{\text{cache},j}
\right|
<10^{-5}
$$

cho cả 6 nguồn:

```text
local_b4, local_b8, local_b12,
context_b4, context_b8, context_b12
```

Đây là **sai số tuyệt đối cực đại**, không dùng `torch.allclose` để tránh việc `rtol` làm acceptance rule mơ hồ.

---

## Benchmark đo cái gì?

### `online_compute`

```text
x_local/x_context đã ở device
→ DINOv3
→ 6 features
```

Đây là chi phí backbone thuần.

### `online_e2e`

```text
CPU preprocessed tensors
→ CPU→GPU nếu dùng CUDA
→ DINOv3
→ 6 features
```

### `cached_e2e`

```text
FeatureCacheReader.get()
→ đọc/materialize đủ 6 tensor
→ CPU→GPU nếu dùng CUDA
```

**So sánh chính:**

$$
S
=
\frac{
\operatorname{median}(T_{\text{online,e2e}})
}{
\operatorname{median}(T_{\text{cached,e2e}})
}
$$

- \(S>1\): cached path nhanh hơn.
- \(S=1\): tương đương.
- \(S<1\): cached path chậm hơn trên storage/hardware đang đo.

Không đặt điều kiện “cache bắt buộc phải nhanh hơn X lần”. Speedup là **kết quả thực nghiệm phụ thuộc NVMe/HDD/network storage/GPU**, không phải hằng số khoa học.

---

## Vì sao dùng median?

Timing có jitter do OS scheduling, filesystem cache, CUDA allocator và các tác vụ nền.

PyTorch `torch.utils.benchmark` nhấn mạnh:

- warm-up;
- nhiều replicate;
- median để giảm ảnh hưởng outlier;
- accelerator synchronization.

Nguồn chính thức:

- PyTorch Benchmark Utils: https://docs.pytorch.org/docs/stable/benchmark_utils.html
- PyTorch Benchmark recipe: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html

Tool này tự giữ samples để xuất JSON chi tiết, nhưng dùng cùng nguyên tắc: **warm-up + repeated measurements + median/p95**.

---

## Vì sao `perf_counter_ns()` + `torch.cuda.synchronize()`?

CUDA chạy bất đồng bộ. Nếu chỉ:

```python
t0 = time.perf_counter()
model(x)
t1 = time.perf_counter()
```

thì có thể chỉ đo thời gian **launch kernel**, không phải thời gian GPU hoàn thành.

PyTorch CUDA semantics khuyến nghị synchronize trước/sau timing hoặc dùng CUDA Events.

Nguồn:

- CUDA semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- `torch.cuda.synchronize()`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html
- `torch.cuda.Event`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html

Ở đây không dùng CUDA Event làm timer chính vì `cached_e2e` có cả:

```text
filesystem / mmap + CPU code + H2D
```

CUDA Event chỉ đo device timeline, không đại diện toàn bộ wall-clock của cache pipeline. Vì vậy:

```text
perf_counter_ns + CUDA synchronize
```

là phù hợp hơn cho so sánh end-to-end này.

---

## Vì sao `cached_e2e` phải materialize đủ tensor?

`FeatureCacheReader` dùng `torch.load(..., mmap=True)`. `mmap` có thể trì hoãn đọc tensor storage cho tới lúc tensor thực sự được truy cập.

Nếu benchmark chỉ:

```python
reader.get(...)
```

thì kết quả có thể quá lạc quan vì chưa chắc toàn bộ feature bytes đã được tiêu thụ.

Do đó:

```text
CPU benchmark → clone 6 tensors
CUDA benchmark → chuyển 6 tensors lên GPU
```

để buộc toàn bộ payload được dùng.

---

## Correctness trước performance

Benchmark không được dùng cache sai chỉ vì cache nhanh.

Tool kiểm tra độc lập:

```text
same real preprocessed sample
     ├── fresh online DINOv3
     └── prebuilt Feature Cache
```

Cache phải được build **trước** benchmark bằng cache builder thật.

Tool **không** lấy `F_online` vừa sinh rồi ghi xuống cache để so lại với chính nó.

---

## Input `REAL_SAMPLE_PT`

Có thể chứa một sample:

```python
{
    "image_id": "fabric/...",
    "category": "fabric",
    "x_local": Tensor[1,3,512,512],
    "x_context": Tensor[1,3,512,512],
    "geometry": {
        "local_box": ...,
        "context_box": ...,
        "context_to_local": ...,
        ...
    }
}
```

hoặc:

```python
{
    "samples": [
        {...},
        {...},
    ]
}
```

Nên benchmark nhiều sample thuộc nhiều ảnh/category thay vì lặp mãi một sample, vì một sample duy nhất dễ phản ánh filesystem/page cache “ấm” hơn thực tế.

---

## Cách chạy

```bash
python tools/benchmark_cache.py \
  --sample-pt artifacts/cache_benchmark_samples.pt \
  --cache-dir artifacts/feature_cache \
  --dinov3-repo third_party/dinov3 \
  --dinov3-weights checkpoints/dinov3_vits16_pretrain_lvd1689m.pth \
  --device cuda \
  --warmup 5 \
  --iterations 30 \
  --tolerance 1e-5 \
  --json-out artifacts/benchmark_cache.json
```

Hoặc dùng environment variables:

```bash
export REAL_SAMPLE_PT=artifacts/cache_benchmark_samples.pt
export FEATURE_CACHE_DIR=artifacts/feature_cache
export DINOV3_REPO=third_party/dinov3
export DINOV3_WEIGHTS=checkpoints/dinov3_vits16_pretrain_lvd1689m.pth

python tools/benchmark_cache.py \
  --device cuda \
  --json-out artifacts/benchmark_cache.json
```

---

## Output nên đưa vào checkpoint/report

```text
online_compute      median=  XX.XXX ms
online_e2e          median=  XX.XXX ms
cached_e2e          median=   X.XXX ms

PRIMARY speedup:
online_e2e / cached_e2e = X.XXXx

correctness = PASS
max_abs_error < 1e-5
```

Nên báo thêm:

```text
device
torch version
sample count
warmup
iterations
median
p95
```

Không chỉ báo một lần timing duy nhất.

---

## PASS/FAIL

**PASS** khi:

```text
[ ] cache có đủ sample benchmark
[ ] đúng 6 feature keys
[ ] shape online == cached
[ ] không NaN/Inf
[ ] geometry khớp preprocessing
[ ] max_abs_error < 1e-5 cho mọi feature
[ ] benchmark có warm-up
[ ] CUDA timing có synchronization
[ ] báo median + p95
[ ] cached_e2e materialize đủ 6 feature
[ ] JSON report lưu hardware/software context
```

Không nên đặt `cached faster than online` làm correctness criterion. Nếu cache chậm hơn, đó là **kết quả cần điều tra storage/I/O**, không phải lý do sửa số liệu.


### Lưu ý khoa học

`speedup`, `median latency`, `p95` là **measurement**, không phải novelty của mô hình.

Feature caching là optimization hợp lệ vì DINOv3 backbone đã frozen và preprocessing cho cache phải deterministic. Kiến trúc trainable của nghiên cứu vẫn nằm ở:

```text
Adapter → Fusion → Decoder
```

không nằm ở benchmark/cache.
