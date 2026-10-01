# Day-3 — Task 10: Overfit-16 Trainer

## 1. Mục tiêu

Task 10 chỉ trả lời một câu hỏi kỹ thuật:

> Với đúng 16 mẫu cố định của Task 7, computational graph hiện tại có thể được tối ưu để học/memorize bài toán segmentation hay không?

Đây là **architecture QA**, không phải benchmark và không đo generalization.

Không triển khai trong task này:

```text
Task 11 — Visualization
Task 12 — Checkpoint
Task 13 — Resume
Task 14 — Full integration QA
Task 15 — Day-03 report
```

## 2. File

```text
src/train/overfit16.py
tests/test_overfit16_trainer.py
```

Task 10 sử dụng lại:

```text
Task 7: Overfit-16 dataset
Task 8: AnomalySegmentationLoss
Task 9: optimizer đã khóa đúng trainable parameters
```

Không định nghĩa lại ba thành phần trên.

## 3. Pipeline

Training step được khóa:

```text
Overfit-16 batch
      ↓
model / project feature pipeline
      ↓
raw anomaly logits [B,1,H,W]
      ↓
Task-8 criterion
      ↓
loss
      ↓
zero_grad()
backward()
optimizer.step()
```

Trainer không tự sigmoid trước loss. Task-8 nhận **raw logits**.

## 4. Dataset contract

`Overfit16Dataset` đọc trực tiếp output của Task 7:

```text
overfit16/
├── manifest.json
├── images/
└── masks/
```

Mỗi sample:

```python
{
    "image": Tensor[3,H,W],   # float32, [0,1]
    "mask": Tensor[1,H,W],    # {0,1}
    "sample_id": str,
    "index": int,
    "is_anomaly": bool,
}
```

Hard gate:

```text
schema_name == msila_overfit16
schema_version == 1
purpose == architecture_qa_only
N == 16
image/mask spatial size giống nhau
mask binary
manifest label == mask
```

### Không resize âm thầm

Trainer không tự resize ảnh/mask.

Nếu 16 ảnh có H/W khác nhau:

```text
batch_size = 1
```

hoặc truyền `sample_transform` có biến đổi **đồng bộ image + mask**.

Điều này tránh tạo sai lệch pixel-level GT.

## 5. Model input

Default:

```python
logits = model(batch["image"])
```

Tuy nhiên MS-ILA thực có thể cần:

```text
image
→ Local/Context
→ frozen DINO/cache
→ Adapter
→ Alignment
→ Projection
→ Fusion
→ Decoder
```

Do đó trainer cho phép inject:

```python
model_forward(model, batch)
```

Ví dụ:

```python
def model_forward(model, batch):
    # xây Local/Context hoặc đọc cache theo pipeline đã khóa
    return model(...)
```

Trainer chỉ yêu cầu output cuối cùng là:

```text
[B,1,H,W]
```

và phải đúng spatial size của GT mask.

## 6. Cách dùng

```python
from src.train.overfit16 import (
    Overfit16Dataset,
    Overfit16Trainer,
    make_overfit16_loader,
    seed_everything,
)

from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.train.optimizer import build_day3_msila_optimizer

seed_everything(2026)

dataset = Overfit16Dataset(
    "outputs/day03/overfit16",
)

loader = make_overfit16_loader(
    dataset,
    batch_size=4,
    shuffle=True,
    seed=2026,
)

criterion = AnomalySegmentationLoss()

optimizer, optimizer_report = build_day3_msila_optimizer(
    adapters=adapters,
    projection=projection,
    fusion=fusion,
    decoder=decoder,
    dino=dino,
    learning_rate=1e-3,
    weight_decay=0.0,
)

trainer = Overfit16Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device="cuda",
    frozen_modules={"dino": dino},
)

result = trainer.fit(
    loader,
    epochs=100,
    log_every=10,
)
```

## 7. Diagnostics được theo dõi

Mỗi logged step có:

```text
step
epoch
loss
BCE
DiceLoss
pixel Dice
IoU
precision
recall
normal FPR
gradient L2 norm
learning rate
positive samples
```

Đây là **training diagnostics**, không phải final evaluation metrics.

### Gradient norm

Trước `optimizer.step()`, trainer tính:

\[
\lVert g\rVert_2
=
\sqrt{\sum_j \lVert g_j\rVert_2^2}
\]

Nếu không có parameter nào nhận gradient hoặc gradient chứa NaN/Inf → FAIL.

## 8. Metrics

Với threshold mặc định:

\[
p \ge 0.5
\]

trainer tính confusion counts trên pixel.

### Pixel Dice

\[
Dice =
\frac{2TP}{2TP+FP+FN}
\]

### IoU

\[
IoU =
\frac{TP}{TP+FP+FN}
\]

### Precision

\[
Precision =
\frac{TP}{TP+FP}
\]

### Recall

\[
Recall =
\frac{TP}{TP+FN}
\]

### Normal FPR

Chỉ tính trên các sample có GT mask rỗng:

\[
FPR_{normal}
=
\frac{\text{predicted anomaly pixels on normal images}}
{\text{all pixels of normal images}}
\]

Mục tiêu của metric này là phát hiện trường hợp model học anomaly nhưng đồng thời tô anomaly lên ảnh normal.

## 9. Initial vs Final

`fit()` đánh giá trên **chính Overfit-16** trước và sau training:

```python
result.initial
result.final
result.loss_ratio
```

với:

\[
loss\_ratio =
\frac{L_{final}}{L_{initial}}
\]

Kỳ vọng của architecture QA:

```text
final loss << initial loss
pixel Dice tăng mạnh
IoU tăng
normal FPR giảm/thấp
```

Không nên chọn một threshold cứng duy nhất rồi tuyên bố đó là tiêu chuẩn khoa học.

Ví dụ `loss_ratio < 0.1` hoặc Dice gần 1 chỉ nên dùng như **debug target** nếu nhóm muốn khóa gate nội bộ.

## 10. DINO frozen

Khi truyền:

```python
frozen_modules={"dino": dino}
```

trainer kiểm tra:

```text
mọi DINO parameter requires_grad=False
```

và sau mỗi `model.train()` sẽ gọi lại:

```python
dino.eval()
```

Điều này quan trọng vì `requires_grad=False` chỉ khóa gradient; nó không tự đảm bảo module luôn ở evaluation mode nếu parent model bị gọi `.train()`.

Task 9 vẫn là nơi chịu trách nhiệm bảo đảm DINO **không nằm trong optimizer**.

## 11. Reproducibility

Trước training:

```python
seed_everything(2026)
```

seed:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
```

`make_overfit16_loader(..., seed=2026)` dùng một `torch.Generator` riêng để cố định shuffle order.

Reproducibility tuyệt đối trên GPU còn phụ thuộc CUDA/kernel/backend và không được suy ra chỉ từ việc đặt seed.

## 12. PASS cho Task 10

Task 10 chỉ nên PASS khi quan sát được đồng thời:

```text
[PASS] forward → loss → backward → optimizer.step chạy
[PASS] loss finite
[PASS] gradient finite và khác 0
[PASS] final loss thấp rõ rệt so với initial loss
[PASS] prediction học được GT mask trên chính 16 mẫu
[PASS] normal FPR không bùng nổ
[PASS] DINO/frozen modules không nhận gradient
```

`tests/test_overfit16_trainer.py` có một controlled 16-sample problem mà model nhỏ phải memorize được; test này xác nhận trainer hoạt động đúng, không chứng minh MS-ILA thật đã PASS.

## 13. Ý nghĩa khoa học

Nếu MS-ILA thật overfit được 16 mẫu, kết luận hợp lệ là:

> Kiến trúc và đường truyền gradient hiện tại có đủ khả năng biểu diễn/tối ưu để fit một tập segmentation nhỏ có kiểm soát.

Không được kết luận:

```text
model generalize tốt
AU-PRO_0.05 cao
synthetic anomaly giống defect thật
kiến trúc tốt hơn baseline
```

Các kết luận đó cần train/validation/test và ablation riêng.

## 14. Nếu Overfit-16 thất bại

Ưu tiên debug theo thứ tự:

```text
1. logits/GT shape
2. loss finite
3. gradient tồn tại
4. optimizer parameter contract
5. Adapter gamma có rời 0 hay không
6. Projection/Fusion/Decoder có gradient không
7. image-mask synchronization
8. Local/Context alignment và cache contract
```

Không nên tăng model size hoặc đổi loss ngay khi chưa xác định lỗi computational graph.
