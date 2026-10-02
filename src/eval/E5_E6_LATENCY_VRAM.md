# E5–E6 — Latency & Peak VRAM Protocol

**Implementation:** `src/eval/efficiency.py`  
**Scope of this file:** E5 and E6 only, while preserving E4 functions already implemented.  
**Not implemented:** E7–E10.

---

## 1. Why E5 needs warm-up and CUDA synchronization

CUDA kernels are launched asynchronously relative to the CPU. Therefore:

```python
start = time.perf_counter()
y = model(x)
end = time.perf_counter()
```

is not a valid CUDA inference latency measurement by itself: the CPU may stop
the timer before GPU work has completed.

E5 uses:

```text
warm-up
CUDA synchronize
start timer
inference
CUDA synchronize
stop timer
repeat
```

The project already follows this same pattern in
`src/tools/benchmark_cache.py`.

PyTorch defines `torch.cuda.synchronize(device)` as waiting for all kernels in
all streams on the selected CUDA device to complete. This is why the timer is
synchronized on both sides.

---

## 2. E5 measurement

For each measured sample:

\[
t_i =
\frac{
t^{\mathrm{end}}_i-t^{\mathrm{start}}_i
}{10^6}
\quad\text{ms}.
\]

The report outputs:

\[
\bar t=\frac{1}{N}\sum_i t_i
\]

plus:

```text
mean
median
p95
std
min
max
```

Median is important because one transient system stall should not dominate the
central latency estimate; p95 exposes the slow tail.

### Repeated-run stability

E5 runs several independent timing rounds. Let \(m_j\) be the median latency of
round \(j\). Stability is summarized with

\[
CV_m =
\frac{
\sigma(m_1,\ldots,m_R)
}{
\mu(m_1,\ldots,m_R)
}.
\]

Default project QA rule:

\[
CV_m \le 0.10.
\]

This **10% is a project QA convention, not a universal scientific constant**.
If the team wants a stricter value, lock it before inspecting candidate
results.

Default:

```text
warm-up/round = 10
iterations/round = 50
rounds = 3
stability CV threshold = 0.10
```

For final paper measurements, use enough iterations for the target hardware and
keep the exact same protocol for every candidate.

---

## 3. Inference mode

The benchmark runs inside:

```python
torch.inference_mode()
```

by default because this is inference benchmarking, not training.

However:

```text
torch.inference_mode() != model.eval()
```

PyTorch explicitly notes that inference mode does not automatically switch a
model to evaluation mode.

Therefore call:

```python
model.eval()
```

**before** creating the benchmark callable.

---

## 4. E6 Peak VRAM definition

E6 primary metric is:

```python
torch.cuda.max_memory_allocated(device)
```

after:

```python
warm-up
synchronize
torch.cuda.reset_peak_memory_stats(device)
inference
synchronize
max_memory_allocated(device)
```

PyTorch defines `max_memory_allocated()` as the maximum GPU memory occupied by
tensors and states that `reset_peak_memory_stats()` resets the starting point
used to track this peak.

The report stores bytes plus binary units:

```text
MiB = bytes / 2^20
GiB = bytes / 2^30
```

The code intentionally uses **MiB/GiB**, not ambiguous labels MB/GB.

---

## 5. Absolute peak vs incremental peak

After warm-up and immediately after resetting peak statistics:

\[
M_0 = \text{memory\_allocated()}.
\]

During inference:

\[
M_{\mathrm{peak}}
=
\text{max\_memory\_allocated()}.
\]

The report contains both:

\[
\Delta M_{\mathrm{peak}}
=
M_{\mathrm{peak}}-M_0.
\]

Interpretation:

```text
peak_allocated
    = primary E6 result;
      includes model/input tensors already resident at measurement start.

incremental_peak_allocated
    = additional tensor memory required above that baseline during inference.
```

For Day-04 candidate comparison, report `peak_allocated` as the primary value
and keep `incremental_peak_allocated` as diagnostic evidence.

---

## 6. Why allocated and reserved are different

PyTorch uses a CUDA caching allocator.

Therefore:

```text
memory allocated
    = memory occupied by live PyTorch tensors

memory reserved
    = memory managed/reserved by the PyTorch caching allocator
```

The number shown by `nvidia-smi` may be larger because CUDA context and cached
allocator memory are not identical to live tensor memory.

E6 therefore reports:

```text
peak_allocated   <- PRIMARY
peak_reserved    <- diagnostic
```

Do not replace the primary result with an `nvidia-smi` screenshot.

---

## 7. Why `empty_cache()` is not called inside the benchmark

E6 deliberately does not do:

```python
torch.cuda.empty_cache()
```

between warm-up and measurement.

Reason: the comparison should measure every candidate after the same warm-up
state. Artificially modifying allocator state immediately before only some
measurements would change the benchmark condition.

If the experiment chooses a different cache policy, apply it identically to
every candidate and version the benchmark protocol.

---

## 8. Locking the benchmark scope

Latency and peak VRAM have no scientific meaning without a precise scope.

Example Day-04 cached-feature scope:

```python
from src.eval.efficiency import make_benchmark_scope

scope = make_benchmark_scope(
    scope_name="day04_cached_head",
    device="cuda:0",
    batch_size=1,
    precision="fp32",
    input_signature="6 cached DINO maps: [1,C,Hf,Wf]",
    pipeline_stages=(
        "adapter",
        "context_alignment",
        "projection",
        "attention_fusion",
        "decoder",
    ),
    extra={
        "output_size": [512, 512],
    },
)
```

This explicitly **excludes**:

```text
image loading
tiling
DINOv3 backbone
Hann stitching
```

Therefore this timing must be called:

> cached-feature trainable-pipeline latency

and **not**

> full image inference latency.

A deployment benchmark needs a separate scope such as:

```text
image
-> tiling
-> DINOv3
-> adapter
-> alignment
-> projection
-> fusion
-> decoder
-> Hann stitching
```

---

## 9. Scope fingerprint

`make_benchmark_scope()` writes a SHA-256 fingerprint over:

```text
scope name
device
batch size
precision
input signature
pipeline stages
extra benchmark metadata
```

Candidate-specific hyperparameters are intentionally not part of this
fingerprint.

Direct comparison is valid only when candidates have the same:

```text
scope_fingerprint_sha256
latency warm-up/iterations/rounds
stability threshold
VRAM warm-up/iterations
inference-mode setting
```

This prevents comparing, for example:

```text
Candidate A: cached head, batch=1, fp16
Candidate B: full DINO pipeline, batch=4, fp32
```

as though they were the same benchmark.

---

## 10. Recommended Day-04 usage

```python
import torch

from src.eval.efficiency import (
    make_benchmark_scope,
    benchmark_inference_efficiency,
)

model = model.to("cuda:0")
model.eval()

# Keep inputs resident on the benchmark device if H2D transfer is NOT
# part of the declared benchmark scope.
batch = move_batch_to_cuda(batch)

def inference_once():
    return model(batch)

scope = make_benchmark_scope(
    scope_name="day04_cached_head",
    device="cuda:0",
    batch_size=1,
    precision="fp32",
    input_signature="cached six-source DINO features",
    pipeline_stages=(
        "adapter",
        "context_alignment",
        "projection",
        "attention_fusion",
        "decoder",
    ),
    extra={
        "output_size": [512, 512],
    },
)

report = benchmark_inference_efficiency(
    inference_once,
    candidate_id="r4_d128",
    scope=scope,
    output_path="outputs/day04/r4_d128/efficiency.json",
    latency_warmup=10,
    latency_iterations=50,
    latency_rounds=3,
    stability_cv_threshold=0.10,
    vram_warmup=10,
    vram_iterations=1,
)
```

---

## 11. Expected output

Relevant fields:

```json
{
  "candidate_id": "r4_d128",
  "status": "PASS",
  "scope_fingerprint_sha256": "...",
  "latency": {
    "latency_ms": {
      "mean": 0.0,
      "median": 0.0,
      "p95": 0.0
    },
    "stability": {
      "round_median_cv": 0.0,
      "status": "PASS"
    }
  },
  "peak_vram": {
    "memory": {
      "peak_allocated": {
        "bytes": 0,
        "MiB": 0.0,
        "GiB": 0.0
      }
    }
  }
}
```

Zeros above are schema illustration only, **not experimental results**.

---

## 12. PASS criteria

### E5 PASS

```text
warm-up performed
same device/scope
CUDA sync before and after every CUDA timed call
>= 2 timing rounds
mean/median/p95 finite
CV(round medians) <= pre-locked threshold
```

If CV fails, the candidate result is not discarded scientifically; the timing
environment should be stabilized and re-measured under the same protocol.

### E6 PASS

```text
CUDA device available
same callable/scope as E5
warm-up completed
peak statistics reset after warm-up
inference completed
device synchronized before querying
peak_memory_allocated >= baseline_memory_allocated
```

E6 cannot be validly measured on a CPU-only runtime.

---

## 13. Evidence / references

PyTorch documentation:

- `torch.cuda.synchronize`: waits for all kernels in all streams on the selected
  CUDA device to complete.
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize

- `torch.cuda.memory.max_memory_allocated`: maximum GPU memory occupied by
  tensors; peak tracking can be reset with `reset_peak_memory_stats`.
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html

- CUDA memory management: PyTorch uses a caching allocator and distinguishes
  allocated tensor memory from reserved allocator memory.
  https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management

- `torch.inference_mode`: removes additional autograd overhead for inference but
  does not call `model.eval()`.
  https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html

Project precedent:

- `src/tools/benchmark_cache.py` already uses warm-up,
  `torch.cuda.synchronize(device)`, repeated `perf_counter_ns()` measurements,
  and median/p95 reporting. E5 keeps the same measurement logic for consistency.

---

## 14. Scientific claim that E5/E6 support

Valid:

> Under the locked Day-04 benchmark scope, candidate A has median latency X ms,
> p95 Y ms, and PyTorch peak allocated tensor memory Z MiB.

Not valid from E5/E6 alone:

```text
candidate A is more accurate
candidate A has fewer parameters
candidate A has lower end-to-end deployment latency
```

unless those quantities were measured under their corresponding protocols.
