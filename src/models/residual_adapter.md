# Residual Adapter md
## 1. Mục tiêu

Module:

```text
models/residual_adapter.py
```

thực hiện:

\[
\Delta x = W_{up}\left(\mathrm{GELU}\left(\mathrm{DWConv}(W_{down}(x))\right)\right)
\]

\[
y = x + \gamma \Delta x
\]

Trong đó:

- `down_proj`: Conv \(1\times1\), giảm số kênh.
- `DWConv`: Depthwise Conv \(3\times3\), học thông tin không gian cục bộ.
- `GELU`: hàm kích hoạt.
- `up_proj`: Conv \(1\times1\), khôi phục số kênh.
- `gamma`: hệ số residual học được, khởi tạo `0`.

Khi khởi tạo:

\[
\gamma=0 \Rightarrow y=x
\]

Do đó adapter ban đầu không làm thay đổi feature của backbone.


## 2. Chạy trên Google Colab

### Bước 1 - Tạo notebook

### Bước 2 - Clone repository

### Bước 3 — Kiểm tra cấu trúc

Repository tối thiểu cần:

```text
project/
├── models/
│   ├── __init__.py
│   └── residual_adapter.py
│
└── tests/
    └── test_residual_adapter_identity.py
```

Kiểm tra:

```bash
!ls models
!ls tests
```

---

### Bước 4 - Cài dependency

Colab thường đã có PyTorch. Kiểm tra:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

Nếu repository có `requirements.txt`:

```bash
!pip install -q -r requirements.txt
```

Nếu chỉ test adapter:

```bash
!pip install -q pytest
```

---

## 4. Chạy unit test

```bash
!pytest -q tests/test_residual_adapter_identity.py
```

Kỳ vọng:

```text
4 passed
```

Các test cần xác nhận:

1. Output giữ nguyên shape.
2. `gamma = 0` ⇒ output bằng input.
3. `gamma` có gradient.
4. `gamma != 0` ⇒ adapter có khả năng thay đổi feature.

---

## 5. Smoke test trực tiếp

```python
import torch
from models.residual_adapter import ResidualAdapter2d

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResidualAdapter2d(
    in_channels=384,
    reduction=4,
    kernel_size=3,
    gamma_init=0.0,
).to(device)

x = torch.randn(2, 384, 16, 16, device=device)

with torch.no_grad():
    y = model(x)

print("device :", device)
print("input  :", x.shape)
print("output :", y.shape)
print("gamma  :", model.gamma.item())
print("max |y-x|:", (y - x).abs().max().item())
```

Kỳ vọng lúc khởi tạo:

```text
input  : torch.Size([2, 384, 16, 16])
output : torch.Size([2, 384, 16, 16])
gamma  : 0.0
max |y-x|: 0.0
```
Chạy test: ```pytest -s -q tests/test_dino_features.py```
---

## 6. Kiểm tra backward

```python
model.train()

x = torch.randn(
    2, 384, 16, 16,
    device=device,
    requires_grad=True,
)

target = torch.randn_like(x)

y = model(x)

loss = torch.nn.functional.mse_loss(y, target)
loss.backward()

print("loss:", loss.item())
print("gamma grad:", model.gamma.grad)
```

Điều kiện PASS:

```text
model.gamma.grad != None
```

và gradient phải hữu hạn.

---

## 7. Quy ước trước khi tích hợp

Không tích hợp adapter vào DINOv3/Fusion/Decoder nếu chưa đạt:

```text
[PASS] import module
[PASS] forward
[PASS] shape invariant
[PASS] gamma=0 identity
[PASS] backward
[PASS] pytest
```

Sau khi PASS mới nối:

```text
Frozen Backbone
      ↓
feature [B,C,H,W]
      ↓
ResidualAdapter2d
      ↓
adapted feature
      ↓
Fusion / Decoder
```

### Cấu hình V1 đề xuất

```python
ResidualAdapter2d(
    in_channels=C,
    reduction=4,
    kernel_size=3,
    gamma_init=0.0,
)
```

Giữ kiến trúc V1 đơn giản để ablation khoa học; chưa thêm attention, SE/CBAM hoặc normalization nếu chưa có bằng chứng thực nghiệm.