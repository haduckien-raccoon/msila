# Day-3 — Task 9: Optimizer Setup

## 1. Phạm vi

Task 9 chỉ xây dựng **optimizer contract** cho kiến trúc MS-ILA đã khóa ở Day-2.

Không thực hiện:

- loss;
- Overfit trainer;
- scheduler;
- visualization;
- checkpoint/resume;
- integration training QA.

Đầu ra của task:

```text
src/train/optimizer.py
tests/test_optimizer_setup.py
```

## 2. Contract cần bảo vệ

Optimizer Day-3 chỉ được cập nhật:

```text
Adapter
Projection
Fusion
Decoder
```

DINOv3 phải thỏa cả hai điều kiện:

```text
requires_grad = False
DINO parameter IDs ∩ optimizer parameter IDs = ∅
```

Điều kiện thứ hai quan trọng vì chỉ kiểm tra `requires_grad=False` chưa đủ để chứng minh optimizer được cấu hình đúng.

## 3. Vì sao bốn module trên được train?

Pipeline hiện tại có dạng:

```text
Frozen DINO features
      ↓
Adapter
      ↓
Alignment
      ↓
Projection
      ↓
Attention Fusion
      ↓
Decoder
      ↓
anomaly logits
```

Trong đó:

- **Adapter** chứa các phép chiếu và spatial branch có tham số học.
- **Projection** dùng Conv2d `1×1`, nên có trọng số học.
- **Attention Fusion** có `score_proj` và `source_bias`.
- **Decoder** có các convolution tạo anomaly logits.
- **DINOv3** là backbone frozen.

Task 9 không thay đổi kiến trúc; chỉ xác định chính xác parameter set mà optimizer được phép cập nhật.

## 4. API chính

```python
from src.train.optimizer import build_day3_msila_optimizer

optimizer, report = build_day3_msila_optimizer(
    adapters=adapters,
    projection=projection,
    fusion=fusion,
    decoder=decoder,
    dino=dino,
    optimizer_name="adamw",
    learning_rate=1e-3,
    weight_decay=0.0,
)
```

`adapters` có thể là một adapter hoặc `nn.ModuleDict` chứa nhiều adapter.

Kết quả:

```text
optimizer
report
```

`report` ghi:

```text
optimizer_name
learning_rate
weight_decay
trainable_parameter_tensors
trainable_parameter_elements
frozen_parameter_tensors
frozen_parameter_elements
trainable_groups
frozen_groups
```

Có thể lưu báo cáo:

```python
import json

with open("optimizer_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

## 5. Lựa chọn optimizer

Default của code:

```text
AdamW
learning_rate = 1e-3
weight_decay = 0
```

Đây chỉ là **Day-3 architecture-QA default**, không phải kết luận rằng AdamW hoặc `lr=1e-3` là tối ưu khoa học.

`weight_decay=0` được dùng mặc định vì mục tiêu Overfit-16 kế tiếp là kiểm tra khả năng **memorize một tập rất nhỏ**, không phải đánh giá generalization.

Code vẫn hỗ trợ:

```text
adamw
adam
sgd
```

để việc thay optimizer sau này là cấu hình rõ ràng, không sửa logic lựa chọn parameter.

Day-3 cũng chưa dùng differential learning rate giữa Adapter/Projection/Fusion/Decoder. Tất cả dùng cùng `lr` và `weight_decay` để tránh thêm một yếu tố gây nhiễu khi debug architecture.

## 6. Hard gates

### Gate A — DINO phải frozen

Nếu có parameter của DINO:

```python
p.requires_grad == True
```

builder dừng ngay bằng `OptimizerContractError`.

Không tự động freeze DINO trong optimizer code, vì tự sửa im lặng có thể che lỗi cấu hình ở pipeline phía trước.

### Gate B — DINO tuyệt đối không nằm trong optimizer

Sau khi tạo optimizer:

```text
IDs(optimizer) ∩ IDs(DINO) = ∅
```

Nếu có overlap → FAIL.

### Gate C — đúng toàn bộ trainable parameter

Optimizer phải chứa **chính xác** union:

```text
Adapter ∪ Projection ∪ Fusion ∪ Decoder
```

Không được:

```text
thiếu parameter
thừa parameter
trùng parameter
```

### Gate D — target module không được vô tình frozen

Nếu một parameter trong Adapter/Projection/Fusion/Decoder có:

```text
requires_grad=False
```

Task 9 FAIL thay vì âm thầm bỏ parameter đó khỏi optimizer.

### Gate E — parameter không được xuất hiện ở hai logical groups

Ví dụ cùng một module bị truyền vừa làm `projection` vừa làm `fusion` sẽ bị reject. Điều này tránh optimizer cập nhật cùng một Parameter nhiều lần.

## 7. Kiểm thử bắt buộc

Chạy:

```bash
pytest -q tests/test_optimizer_setup.py
```

PASS khi:

```text
[PASS] optimizer chứa đúng Adapter + Projection + Fusion + Decoder
[PASS] DINO không nằm trong optimizer
[PASS] DINO phải requires_grad=False
[PASS] target module không bị frozen ngoài ý muốn
[PASS] không có parameter duplicate giữa groups
[PASS] optimizer.step() làm thay đổi trainable parameters
[PASS] optimizer.step() không làm thay đổi DINO
[PASS] DINO không có gradient
[PASS] invalid hyperparameters bị reject
```

## 8. Ý nghĩa khoa học

Task 9 **không chứng minh hiệu năng anomaly detection**.

Nó chỉ chứng minh một invariant của thí nghiệm:

> Các cập nhật gradient của Day-3 được giới hạn đúng vào Adapter, Projection, Fusion và Decoder, trong khi backbone DINOv3 giữ nguyên.

Điều này cần thiết để sau này khi model thay đổi sau training, ta có thể quy thay đổi cho các module trainable thay vì vô tình fine-tune backbone.

## 9. Điều chưa được kết luận

Không được dùng Task 9 để kết luận:

```text
AdamW tốt hơn Adam/SGD
lr=1e-3 là tối ưu
weight_decay=0 là tối ưu
một module cần learning rate cao hơn module khác
```

Những câu hỏi đó cần experiment/ablation riêng.

## 10. Handoff sang Task 10

Task 10 chỉ cần nhận:

```text
model
criterion
optimizer
Overfit-16 dataloader
```

và thực hiện:

```text
zero_grad
→ forward
→ loss
→ backward
→ optimizer.step
```

Task 9 không triển khai loop này.
