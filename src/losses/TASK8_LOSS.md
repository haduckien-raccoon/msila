# Day 3 — Task 8: Loss

## 1. Mục tiêu

Task 8 chỉ xây **hàm mất mát cho anomaly localization** để phục vụ Overfit-16 Architecture QA.

Đầu vào:

```text
anomaly_logits : [B,1,H,W]   — raw output từ Decoder
binary_mask    : [B,1,H,W]   — ground-truth mask {0,1}
```

Đầu ra:

```text
loss            scalar
bce             scalar
Dice loss       scalar
positive_samples số sample có anomaly trong batch
```

Không resize, threshold hay hậu xử lý bên trong loss. Nếu `logits` và `mask` khác shape thì xem là lỗi pipeline.

---

## 2. Loss được chọn

Dùng objective tối thiểu:

\[
\mathcal L
= \lambda_{BCE}\,\mathcal L_{BCE}
+ \lambda_D\,\mathcal L_{Dice}.
\]

Mặc định:

\[
\lambda_{BCE}=1,\qquad \lambda_D=1.
\]

### BCEWithLogits

Với logit \(z_i\), target \(y_i\in\{0,1\}\):

\[
\mathcal L_{BCE}
= -\frac{1}{N}\sum_i
\left[y_i\log\sigma(z_i)
+(1-y_i)\log(1-\sigma(z_i))\right].
\]

Implementation dùng trực tiếp:

```python
F.binary_cross_entropy_with_logits(logits, target)
```

Không gọi `sigmoid()` trước `BCEWithLogits`, vì hàm PyTorch đã kết hợp sigmoid + BCE theo dạng ổn định số học.

### Dice loss

Với xác suất \(p_i=\sigma(z_i)\):

\[
Dice
= \frac{2\sum_i p_i y_i+\epsilon}
{\sum_i p_i+\sum_i y_i+\epsilon},
\]

\[
\mathcal L_{Dice}=1-Dice.
\]

Trong implementation này, Dice **chỉ tính trên sample có ít nhất một pixel anomaly**.

Lý do: với normal sample, target rỗng hoàn toàn. Khi đó khái niệm overlap vùng anomaly không có vùng dương để so sánh; BCE đã cung cấp supervision background đầy đủ. Cách xử lý này tránh để hệ số smoothing của Dice tự quyết định hành vi trên empty mask.

---

## 3. Vai trò của hai thành phần

`BCEWithLogits` cung cấp supervision theo từng pixel cho cả anomaly và background. `Dice` bổ sung tín hiệu overlap vùng dương, hữu ích khi diện tích anomaly nhỏ hơn background nhiều.

Task 8 **không khẳng định BCE + Dice là loss tối ưu cuối cùng cho MVTec AD 2**. Đây là objective đơn giản, minh bạch để kiểm tra kiến trúc có học được Overfit-16 hay không.

---

## 4. File

```text
src/losses/
├── __init__.py
└── anomaly_loss.py

tests/
└── test_anomaly_loss.py
```

Class chính:

```python
from src.losses.anomaly_loss import AnomalySegmentationLoss

criterion = AnomalySegmentationLoss(
    bce_weight=1.0,
    dice_weight=1.0,
)

out = criterion(logits, mask)
loss = out["loss"]
loss.backward()
```

Log được:

```python
out["bce"]
out["dice"]
out["positive_samples"]
```

---

## 5. PASS criteria

Task 8 PASS khi:

```text
[PASS] logits/mask đúng [B,1,H,W]
[PASS] target chỉ có {0,1}
[PASS] total loss finite
[PASS] BCE finite
[PASS] Dice finite
[PASS] loss.backward() chạy được
[PASS] gradient finite và khác 0 trên logits
[PASS] prediction đúng có loss thấp hơn prediction đảo ngược
[PASS] normal-only batch vẫn train được bằng BCE
[PASS] shape mismatch bị reject thay vì tự resize âm thầm
[PASS] non-binary target bị reject
```

---

## 6. Giới hạn kết luận khoa học

Sau Task 8 chỉ được kết luận:

> Loss implementation hợp lệ về tensor contract, ổn định số học và truyền gradient được cho bài toán binary anomaly localization trong Architecture QA.

Chưa được kết luận:

- BCE + Dice tốt nhất cho MS-ILA;
- loss này cải thiện AU-PRO\(_{0.05}\);
- loss này tốt hơn focal/Tversky/boundary-aware loss.

Các kết luận đó cần ablation trên validation/test protocol ở giai đoạn thí nghiệm sau.
