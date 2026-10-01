# Day-3 — Task 12: Checkpoint

## 1. Mục tiêu

Task 12 chỉ làm một việc:

> Lưu đầy đủ trạng thái cần thiết của một training run tại một thời điểm xác định.

Không thực hiện restore/resume. Việc nạp checkpoint trở lại model/optimizer thuộc **Task 13**.

File:

```text
src/utils/checkpoint.py
tests/test_checkpoint.py
```

---

## 2. Checkpoint lưu những gì?

Payload có schema cố định:

```text
schema_name
schema_version

training_state
├── epoch
└── global_step

model_state
optimizer_state
config
rng_state
stateful_states
model_contract
runtime
metadata
```

### Model state

```python
model.state_dict()
```

Lưu toàn bộ parameter/buffer của model hiện tại.

### Optimizer state

```python
optimizer.state_dict()
```

Điều này cần thiết vì Adam/AdamW có các moment nội tại. Chỉ lưu model weights là chưa đủ để tiếp tục optimization đúng trạng thái.

### Training position

```text
epoch
global_step
```

Hai giá trị được lưu riêng, không suy đoán từ filename.

---

## 3. RNG state

Checkpoint lưu:

```text
Python random
NumPy RNG
PyTorch CPU RNG
PyTorch CUDA RNG states
named torch.Generator states
```

Lý do: tái lập training không chỉ phụ thuộc model và optimizer mà còn phụ thuộc thứ tự shuffle và các phép lấy mẫu ngẫu nhiên.

Task 10 tạo DataLoader bằng một `torch.Generator` riêng. Nếu muốn Task 13 sau này tái lập đúng shuffle state, truyền generator này vào Task 12:

```python
save_training_checkpoint(
    ...,
    generators={
        "dataloader": loader.generator,
    },
)
```

Nếu `loader.generator is None`, không truyền mục này.

---

## 4. Optional stateful objects

Module hỗ trợ các object có:

```python
state_dict()
```

Ví dụ:

```text
learning-rate scheduler
AMP GradScaler
```

Dùng:

```python
stateful_objects={
    "scheduler": scheduler,
    "scaler": scaler,
}
```

Day-3 hiện chưa bắt buộc scheduler/scaler; cơ chế này chỉ bảo đảm checkpoint không thiếu state nếu chúng được dùng sau này.

---

## 5. Model contract

Checkpoint ghi thêm:

```text
model class
trainable parameter names
frozen parameter names
```

Mục đích là audit.

Ví dụ có thể xác nhận DINO nằm trong nhóm frozen còn Adapter / Projection / Fusion / Decoder nằm trong nhóm trainable theo cấu hình model thực tế.

Đây không thay thế hard gate optimizer của Task 9.

---

## 6. Atomic write

Checkpoint không ghi trực tiếp đè lên file đích.

Quy trình:

```text
payload
  ↓
temporary file cùng thư mục
  ↓
flush + fsync
  ↓
os.replace()
  ↓
checkpoint chính thức
```

Nếu quá trình save lỗi trước `os.replace`, checkpoint cũ không bị thay bằng một file ghi dở.

Điều này quan trọng hơn việc chỉ gọi:

```python
torch.save(..., "latest.pt")
```

trực tiếp.

---

## 7. SHA-256 integrity

Mặc định tạo:

```text
checkpoint.pt
checkpoint.pt.sha256
```

Sidecar chứa SHA-256 của file checkpoint.

SHA-256 chỉ kiểm tra:

> bytes của checkpoint có thay đổi/hỏng hay không.

Nó **không** chứng minh kết quả nghiên cứu đúng và không phải cơ chế bảo mật/chữ ký số.

---

## 8. Cách dùng

```python
from src.utils.checkpoint import save_training_checkpoint

result = save_training_checkpoint(
    "outputs/day03/checkpoints/epoch_010.pt",
    model=model,
    optimizer=optimizer,
    epoch=10,
    global_step=40,
    config={
        "seed": 2026,
        "dataset": "outputs/day03/overfit16",
        "batch_size": 4,
        "learning_rate": 1e-3,
        "loss": "BCE+Dice",
    },
    generators={
        "dataloader": loader.generator,
    },
    metadata={
        "purpose": "architecture_qa_only",
    },
)

print(result.path)
print(result.sha256)
print(result.size_bytes)
```

Nếu không có generator riêng:

```python
generators=None
```

---

## 9. Tại sao lưu config?

Weights không đủ để mô tả một experiment.

Checkpoint phải đi kèm các thông tin như:

```text
seed
dataset/version
batch size
learning rate
loss
fusion_dim
adapter settings
synthetic anomaly settings
```

Task 12 yêu cầu config là dữ liệu JSON-compatible để tránh nhét object Python tùy ý vào experiment record.

`pathlib.Path`, dataclass và NumPy scalar được chuyển về representation ổn định.

---

## 10. Runtime metadata

Checkpoint ghi các thông tin nhẹ:

```text
Python version
PyTorch version
NumPy version
CUDA version
cuDNN version
CUDA availability/device count
```

Mục đích là hỗ trợ truy vết môi trường khi QA.

Runtime metadata không bảo đảm bitwise reproducibility; GPU kernels, driver và backend vẫn có thể gây nondeterminism.

---

## 11. Hard PASS

Chạy:

```bash
pytest -q tests/test_checkpoint.py
```

Task 12 PASS khi:

```text
[PASS] model state được lưu
[PASS] optimizer state được lưu
[PASS] epoch/global_step được lưu
[PASS] config được lưu
[PASS] Python/NumPy/Torch RNG được lưu
[PASS] DataLoader generator state có thể được lưu
[PASS] scheduler/scaler state có thể được lưu
[PASS] SHA-256 khớp file
[PASS] overwrite dùng atomic replacement
[PASS] save không làm thay đổi model/optimizer
[PASS] invalid training position/config bị reject
```

---

## 12. Ý nghĩa khoa học

Checkpoint không cải thiện accuracy.

Vai trò của nó là bảo đảm trạng thái thí nghiệm có thể được lưu tại một điểm xác định, giúp:

```text
audit
reproduce
debug
resume ở Task 13
```

Nếu thiếu optimizer/RNG/training position, một file chỉ chứa weights không phải là checkpoint training đầy đủ cho mục tiêu tái lập quá trình optimization.

---

## 13. Giới hạn của Task 12

Task 12 **không**:

```text
load model weights
load optimizer
restore RNG
khởi động lại epoch
tiếp tục training
so sánh resumed run với uninterrupted run
```

Toàn bộ các thao tác trên thuộc Task 13 — Resume.
