# Day-3 — Task 13: Resume

## 1. Mục tiêu

Task 13 thực hiện phần đối xứng của Task 12:

> Đọc một checkpoint hợp lệ và khôi phục trạng thái training để quá trình tối ưu có thể tiếp tục từ đúng vị trí đã lưu.

Task này chỉ triển khai **resume state**. Không viết Integration QA đầy đủ của Task 14 và không sinh Day-03 report.

File mới:

```text
src/utils/resume.py
tests/test_resume.py
```

`resume.py` phụ thuộc trực tiếp vào `src/utils/checkpoint.py` của Task 12.

---

## 2. Những gì được restore

Task 13 khôi phục:

```text
model.state_dict()
optimizer.state_dict()

epoch
global_step

optional scheduler / GradScaler state

Python RNG
NumPy RNG
PyTorch CPU RNG
PyTorch CUDA RNG
named torch.Generator state
```

Kết quả trả về:

```python
ResumeResult(
    epoch=...,
    global_step=...,
    config=...,
    metadata=...,
    sha256_verified=...,
    rng_restored=...,
)
```

---

## 3. Tại sao chỉ load weights là chưa đủ?

Với Adam/AdamW, optimizer giữ các biến trạng thái nội bộ như moment bậc một/bậc hai.

Nếu chỉ:

```python
model.load_state_dict(...)
```

nhưng tạo optimizer mới từ đầu, trajectory optimization sau resume không còn là continuation đúng của run trước.

Tương tự, nếu không restore RNG/DataLoader generator, thứ tự sample và các phép ngẫu nhiên có thể đổi.

Vì vậy resume đầy đủ yêu cầu:

\[
S_t =
(\theta_t,\ O_t,\ R_t,\ t)
\]

trong đó:

- \(\theta_t\): model state;
- \(O_t\): optimizer/stateful optimizer state;
- \(R_t\): RNG state;
- \(t\): epoch/global step.

---

## 4. Integrity gate trước khi load

Mặc định Task 13 yêu cầu sidecar:

```text
checkpoint.pt.sha256
```

và kiểm tra SHA-256 **trước khi `torch.load`**.

Nếu digest không khớp:

```text
FAIL
```

Điều này phát hiện file bị thay đổi/hỏng sau khi Task 12 ghi.

SHA-256 là integrity check, không phải chữ ký số và không xác thực nguồn gốc checkpoint.

---

## 5. Schema gate

Checkpoint phải đúng:

```text
schema_name = msila_training_checkpoint
schema_version = 1
```

và phải có đủ:

```text
training_state
model_state
optimizer_state
config
rng_state
stateful_states
model_contract
runtime
metadata
```

Checkpoint schema khác không được âm thầm suy đoán.

---

## 6. Model contract gate

Trước khi mutate model, code kiểm tra mặc định:

```text
model class name
trainable parameter names
frozen parameter names
```

Ví dụ nếu checkpoint cũ có DINO frozen nhưng current model vô tình bật:

```python
requires_grad=True
```

Task 13 từ chối resume.

Mục tiêu là tránh tiếp tục một experiment với computational/training contract đã thay đổi.

---

## 7. Optimizer topology gate

Task 13 kiểm tra trước:

```text
số optimizer parameter groups
số parameter trong từng group
```

phải tương thích với checkpoint.

Sau đó mới:

```python
optimizer.load_state_dict(...)
```

Task 9 vẫn là nơi định nghĩa optimizer parameter set hợp lệ:

```text
Adapter + Projection + Fusion + Decoder
DINO ∩ optimizer = ∅
```

Task 13 chỉ bảo đảm state optimizer được nối lại đúng topology đã có.

---

## 8. Config compatibility

Có thể khóa experiment config bằng:

```python
expected_config=...
```

Ví dụ:

```python
result = resume_training_checkpoint(
    "outputs/day03/checkpoints/epoch_010.pt",
    model=model,
    optimizer=optimizer,
    expected_config={
        "seed": 2026,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "loss": "BCE+Dice",
        "dataset": "outputs/day03/overfit16",
    },
)
```

Nếu config khác checkpoint:

```text
FAIL
```

Không nên resume một experiment rồi âm thầm đổi `fusion_dim`, loss, dataset hoặc optimizer hyperparameter nhưng vẫn coi đó là cùng một run.

---

## 9. RNG được restore cuối cùng

Thứ tự code:

```text
verify file
→ load payload
→ validate schema/config/contracts
→ load model
→ load optimizer
→ load scheduler/scaler
→ restore RNG LAST
```

RNG được restore cuối để các thao tác validation/loading trước đó không làm tiêu thụ sequence ngẫu nhiên vừa được khôi phục.

---

## 10. DataLoader generator

Task 10 dùng `torch.Generator` riêng để cố định shuffle.

Task 12 có thể lưu:

```python
generators={
    "dataloader": loader.generator,
}
```

Task 13 cần truyền một generator tương ứng:

```python
resume_generator = torch.Generator()

result = resume_training_checkpoint(
    checkpoint_path,
    model=model,
    optimizer=optimizer,
    generators={
        "dataloader": resume_generator,
    },
)
```

Sau resume, tạo/tiếp tục DataLoader bằng generator đã restore theo pipeline thực tế của nhóm.

Tên generator được kiểm tra strict mặc định để tránh bỏ sót RNG stream đã được lưu.

---

## 11. Scheduler / scaler

Nếu Task 12 lưu:

```python
stateful_objects={
    "scheduler": scheduler,
    "scaler": scaler,
}
```

thì Task 13 phải truyền object cùng tên:

```python
resume_training_checkpoint(
    ...,
    stateful_objects={
        "scheduler": scheduler,
        "scaler": scaler,
    },
)
```

Tên không khớp → FAIL mặc định.

---

## 12. API chính

```python
from src.utils.resume import resume_training_checkpoint

result = resume_training_checkpoint(
    "outputs/day03/checkpoints/epoch_010.pt",
    model=model,
    optimizer=optimizer,
    expected_config=config,
    generators={
        "dataloader": loader_generator,
    },
    stateful_objects={
        "scheduler": scheduler,
    },
    map_location="cpu",
)

start_epoch = result.epoch
global_step = result.global_step
```

Nếu training trên GPU, có thể khởi tạo model/optimizer đúng device và chọn `map_location` phù hợp với training script.

---

## 13. `epoch` tiếp tục như thế nào?

Checkpoint lưu **epoch hiện tại đã được caller xác định**.

Task 13 không tự cộng:

```python
epoch + 1
```

vì ý nghĩa `epoch` phụ thuộc thời điểm save:

- save cuối epoch;
- save giữa epoch;
- save theo global step.

Training script phải quyết định điểm tiếp tục dựa trên convention đã khóa.

Với Day-3 Overfit-16, nên save ở **cuối epoch** nếu muốn resume đơn giản.

---

## 14. Hard PASS

Chạy:

```bash
pytest -q tests/test_resume.py
```

PASS khi:

```text
[PASS] model state restore chính xác
[PASS] optimizer state restore chính xác
[PASS] epoch/global_step restore chính xác
[PASS] Python RNG restore
[PASS] NumPy RNG restore
[PASS] PyTorch RNG restore
[PASS] named DataLoader generator restore
[PASS] scheduler state restore
[PASS] corrupted SHA-256 bị reject
[PASS] schema mismatch bị reject
[PASS] config mismatch bị reject
[PASS] frozen/trainable contract mismatch bị reject
```

Test quan trọng nhất là:

```text
uninterrupted training
        vs
save → restart → resume → continue
```

trên một controlled CPU experiment phải cho cùng future losses, model state và optimizer state.

---

## 15. Ý nghĩa khoa học

Resume không cải thiện accuracy.

Vai trò khoa học/kỹ thuật là bảo đảm:

> Việc gián đoạn tiến trình không tự tạo ra một experiment mới do mất optimizer state, RNG state hoặc training position.

Điều này làm experiment dễ audit và tái lập hơn.

---

## 16. Giới hạn

Ngay cả khi RNG được restore, bitwise reproducibility trên GPU vẫn có thể bị ảnh hưởng bởi:

```text
CUDA/cuDNN kernels
driver
hardware
nondeterministic operations
DataLoader multiprocessing
```

Vì vậy test exact trajectory trong Task 13 được thực hiện trên CPU controlled setup.

---

## 17. Security

Code dùng:

```python
torch.load(..., weights_only=False)
```

vì Task-12 checkpoint chứa Python/NumPy RNG objects.

Do đó:

> Chỉ load checkpoint do chính project tạo hoặc từ nguồn tin cậy.

Không dùng Task 13 để mở file `.pt` không rõ nguồn gốc.

---

## 18. Handoff sang Task 14

Sau Task 13, pipeline infrastructure đã có:

```text
Task 7  Overfit-16
Task 8  Loss
Task 9  Optimizer
Task 10 Trainer
Task 11 Visualization
Task 12 Checkpoint
Task 13 Resume
```

Task 14 mới kiểm tra **real integrated training step** trên pipeline MS-ILA thật:

```text
cached/online feature path
→ Adapter
→ Alignment
→ Projection
→ Fusion
→ Decoder
→ Loss
→ backward
→ optimizer.step
```

Task 13 không thực hiện gate đó.
