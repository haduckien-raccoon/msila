# Day-3 — Task 11: Training Visualization

## 1. Mục tiêu

Task 11 chuyển visualization thử nghiệm trong notebook thành một module dùng được trong training QA.

Phạm vi chỉ gồm:

```text
raw image
GT mask
raw anomaly logits
↓
qualitative maps + training curves
```

Không triển khai:

```text
Task 12 — checkpoint
Task 13 — resume
Task 14 — integration QA
Task 15 — Day-03 report
```

File:

```text
src/utils/visualize.py
tests/test_training_visualization.py
```

---

## 2. Contract

Đầu vào định tính:

```text
image  : [B,3,H,W] hoặc [3,H,W]
mask   : [B,1,H,W] hoặc [1,H,W]
logits : [B,1,H,W] hoặc [1,H,W]
```

Yêu cầu:

```text
image  = RGB float trong [0,1]
mask   = binary {0,1}
logits = raw decoder logits, finite
H/W của image-mask-logits giống nhau
```

Module **không nhận trực tiếp DINO-normalized RGB** để hiển thị. Hãy giữ ảnh RGB trước normalization hoặc denormalize đúng thông số preprocessing trước khi gọi visualization.

---

## 3. Xử lý score map

Probability được tính đúng bằng:

\[
p(x,y)=\sigma(z(x,y))
\]

với \(z\) là raw anomaly logit.

Code:

```python
probability = torch.sigmoid(logits)
```

Không thực hiện:

```text
per-image min-max normalization
histogram equalization
contrast stretching
```

Lý do: min-max từng ảnh sẽ làm một score map yếu và một score map mạnh đều bị kéo về `[0,1]`, làm mất khả năng so sánh trực tiếp giữa step/epoch/sample.

Visualization probability luôn khóa:

\[
v_{min}=0,\qquad v_{max}=1
\]

---

## 4. Sáu panel định tính

`plot_training_sample()` tạo:

```text
1. Input RGB
2. Exact GT mask
3. Anomaly probability = sigmoid(logit)
4. Binary prediction tại threshold
5. Probability overlay trên RGB
6. Signed error map
```

### Signed error

Định nghĩa:

\[
E=\hat y-y
\]

với \(\hat y\in\{0,1\}\), \(y\in\{0,1\}\).

Do đó:

```text
E = -1  → False Negative
E =  0  → Correct
E = +1  → False Positive
```

Scale luôn khóa:

\[
[-1,1]
\]

không scale lại theo từng ảnh.

---

## 5. Metrics trên hình

Mỗi sample có:

```text
Dice
IoU
Precision
Recall
```

Các số này chỉ phục vụ **qualitative training QA** cho đúng sample được vẽ.

Không dùng các metrics này thay cho benchmark chính của nghiên cứu.

---

## 6. Cách dùng với Task 10

Sau khi lấy một batch và logits:

```python
from src.utils.visualize import save_training_sample

path, vis = save_training_sample(
    image=batch["image"],
    mask=batch["mask"],
    logits=logits,
    index=0,
    threshold=0.5,
    sample_id=batch["sample_id"][0],
    epoch=epoch,
    step=global_step,
    output_path=f"outputs/vis/step_{global_step:05d}.png",
)
```

Task 11 không sửa trainer. Việc chọn thời điểm gọi visualization thuộc training script, ví dụ:

```text
step 0
step 10
step 50
final
```

Không nên save mọi step nếu điều đó làm tăng I/O không cần thiết.

---

## 7. Training curves

Task 10 đã log:

```text
loss
pixel_dice
iou
normal_fpr
grad_norm
...
```

Task 11 cung cấp:

```python
from src.utils.visualize import save_training_curves

save_training_curves(
    result.history,
    "outputs/day03/curves",
)
```

Mặc định tạo các file độc lập:

```text
loss.png
pixel_dice.png
iou.png
normal_fpr.png
grad_norm.png
```

Không ghép các đại lượng khác đơn vị vào cùng một trục.

---

## 8. Ý nghĩa từng curve

### Loss

Dùng để kiểm tra optimizer có giảm objective trên Overfit-16 hay không.

Kỳ vọng architecture QA:

\[
L_{final}\ll L_{initial}
\]

Nhưng loss thấp không tự chứng minh localization đúng.

### Pixel Dice / IoU

Cho biết binary prediction có khớp GT mask trên **chính tập Overfit-16** hay không.

Nếu loss giảm nhưng Dice/IoU vẫn thấp, cần kiểm tra:

```text
class imbalance
threshold
decoder output
mask alignment
```

### Normal FPR

Phát hiện trường hợp model dự đoán anomaly tràn lên các ảnh normal.

### Gradient norm

Giúp phát hiện:

```text
gradient = 0
gradient explosion
NaN/Inf
```

Đây là diagnostic, không phải metric chất lượng anomaly detection.

---

## 9. Tại sao không dùng auto-normalization cho heatmap?

Giả sử hai epoch có score range:

```text
epoch A: 0.01 → 0.08
epoch B: 0.10 → 0.95
```

Nếu min-max riêng:

```text
cả hai đều bị kéo thành 0 → 1
```

thì hình có thể trông “mạnh” tương tự nhau dù confidence khác xa.

Do đó Task 11 giữ probability scale cố định `[0,1]`.

---

## 10. Hard PASS

Chạy:

```bash
pytest -q tests/test_training_visualization.py
```

Task 11 PASS khi:

```text
[PASS] probability = sigmoid(raw logits) chính xác
[PASS] không min-max normalization
[PASS] image/mask/logits contract được kiểm tra
[PASS] binary mask sai bị reject
[PASS] DINO-normalized image không bị hiển thị nhầm như RGB
[PASS] signed error phân biệt FP/FN đúng
[PASS] qualitative PNG được lưu
[PASS] training curves được lưu thành từng figure độc lập
```

---

## 11. Kết luận khoa học hợp lệ

Task 11 chỉ cung cấp bằng chứng trực quan để kiểm tra:

> model đang học vùng nào, bỏ sót vùng nào, dự đoán sai ở đâu và diễn biến optimization có nhất quán với loss/metrics hay không.

Không được dùng visualization để tự kết luận:

```text
model generalize tốt
synthetic anomaly realistic
AU-PRO_0.05 cao
MS-ILA tốt hơn baseline
```

Các kết luận đó cần evaluation định lượng trên dữ liệu thích hợp.

---

## 12. Handoff sang Task 12

Task 11 không lưu state model.

Task 12 mới chịu trách nhiệm lưu:

```text
model state
optimizer state
epoch
global_step
config
RNG state
```

Visualization chỉ lưu ảnh/curve phục vụ QA.
