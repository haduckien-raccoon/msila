# Day-3 — Task 14: Integration QA — Cached Feature → Real Train Step

## 1. Kết luận trước khi code

Đúng: pipeline hiện tại **chưa có gate đầy đủ**

```text
Cached feature
→ model
→ loss
→ backward
→ optimizer.step
→ PASS
```

Repo đã có các phần riêng lẻ:

```text
cache read/write
forward/backward TV1→TV2
loss
optimizer
trainer
```

nhưng test `tests/test_tv1_integration.py` trước đó dùng một frozen backbone nhỏ trong test và dừng ở:

```text
forward
→ MSE loss
→ backward
```

Nó **không** đi qua `CachedFeatureDataset`, **không** dùng Task-8 BCE+Dice loss, và **không** gọi optimizer của Task 9 để kiểm tra parameter thực sự được update.

Task 14 bổ sung đúng gate còn thiếu này.

---

## 2. File mới

```text
src/models/cached_training.py
tests/test_day03_cached_train_step.py
TASK14_INTEGRATION_QA.md
```

Task 15 được cung cấp riêng trong cùng gói dưới dạng `notebooks/Day03_QA_Report.ipynb` để chạy trên Colab.

---

## 3. Một lỗi contract quan trọng cần tránh

Feature cache v1 đang dùng keys:

```text
local_b4
local_b8
local_b12
context_b4
context_b8
context_b12
```

Nhưng các tensor này là **raw frozen-DINO features**.

Trong khi `MSILADay2Head` cũng dùng đúng sáu tên trên cho:

```text
aligned + projected features
```

Do đó không được làm:

```text
CachedFeatureDataset
→ MSILADay2Head
```

trực tiếp.

Hai mapping có cùng tên nhưng **khác semantic stage**.

Task 14 khóa pipeline đúng:

```text
cache raw feature
      ↓
Adapter
      ↓
Context→Local Alignment
      ↓
Projection
      ↓
six aligned/projected features
      ↓
Attention Fusion
      ↓
Decoder
```

---

## 4. Pipeline integration cuối cùng

Gate bắt buộc:

```text
FeatureCacheWriter / real cache
        ↓
FeatureCacheReader
        ↓
CachedFeatureDataset
        ↓
cached_collate_fn / DataLoader
        ↓
CachedFeatureTrainingModel
        ↓
Adapter
        ↓
Context→Local Alignment
        ↓
SixFeatureProjection
        ↓
AttentionFusion
        ↓
BasicDecoder
        ↓
raw anomaly logits [B,1,H,W]
        ↓
AnomalySegmentationLoss
        ↓
BCEWithLogits + Dice
        ↓
loss.backward()
        ↓
AdamW optimizer.step()
        ↓
assert parameters changed
```

Đây mới là **end-to-end training step trên cached features**.

---

## 5. `CachedFeatureTrainingModel`

File:

```text
src/models/cached_training.py
```

Đây chỉ là integration glue, không phải module novelty mới.

Nó tái sử dụng:

```text
ResidualAdapter2d
ContextToLocalAligner
SixFeatureProjection
MSILADay2Head
```

DINOv3 **không được instantiate** trong cached-feature training vì feature đã được sinh trước từ frozen backbone.

---

## 6. Adapter trên Local và Context

Với mỗi block:

\[
k \in \{4,8,12\}
\]

Task 14 dùng cùng adapter cho hai view:

\[
\tilde L_k=A_k(L_k)
\]

\[
\tilde C_k=A_k(C_k)
\]

Lý do: Local và Context được lấy từ cùng frozen DINO block, nên cùng nằm trong một pretrained channel basis.

Task 14 không thêm hai bộ adapter riêng cho Local/Context.

---

## 7. Alignment phải nằm trước Projection

Context raw/adapted feature không được fusion trực tiếp với Local.

Đúng:

\[
C_k
\rightarrow
A_k
\rightarrow
Align_{C\rightarrow L}
\rightarrow
P_k
\]

và:

\[
L_k
\rightarrow
A_k
\rightarrow
P_k
\]

sau đó mới fusion.

`ContextToLocalAligner` là parameter-free; gradient vẫn đi qua phép `grid_sample`.

---

## 8. Bridge geometry cache ↔ aligner

Có một contract khác cần nối rõ.

Feature cache v1 bắt buộc:

```text
context_to_local
```

trong khi `ContextToLocalAligner` sử dụng:

```text
local_to_context
local_input_hw
context_input_hw
```

Task 14 không giả định hai contract tự giống nhau.

Nếu cache đã có:

```text
local_to_context
```

thì dùng trực tiếp.

Nếu cache chỉ có:

```text
context_to_local
```

thì tính:

\[
T_{L\rightarrow C}
=
T_{C\rightarrow L}^{-1}
\]

Với kích thước input:

```text
local_input_hw
context_input_hw
```

được ưu tiên; cache-v1 cũ có thể fallback sang:

```text
local_hw
context_hw
```

Nếu thiếu cả hai dạng → FAIL, không đoán kích thước.

---

## 9. Real train step

Test chính:

```python
test_cached_feature_to_optimizer_step_end_to_end()
```

thực hiện thật:

```python
optimizer.zero_grad(set_to_none=True)

logits = model(batch)

loss_out = criterion(
    logits,
    batch["mask"],
)

loss = loss_out["loss"]

loss.backward()

optimizer.step()
```

Không dùng MSE giả như test integration cũ.

Loss là đúng Task 8:

\[
\mathcal L
=
\mathcal L_{BCEWithLogits}
+
\mathcal L_{Dice}
\]

---

## 10. Optimizer contract

Optimizer dùng đúng Task 9:

```text
Adapter
Projection
Fusion
Decoder
```

Cached training không chứa DINO:

```text
DINO params = ∅
```

Do đó:

```python
build_day3_msila_optimizer(
    adapters=model.adapters,
    projection=model.projection,
    fusion=model.fusion,
    decoder=model.decoder,
    dino=None,
)
```

---

## 11. Gate gradient

Sau:

```python
loss.backward()
```

phải có finite non-zero gradient ở ít nhất một parameter của từng logical module:

```text
Adapter
Projection
Fusion
Decoder
```

### Lưu ý `gamma_init = 0`

Residual Adapter:

\[
y=x+\gamma f(x)
\]

với:

\[
\gamma_0=0
\]

Ở step đầu tiên, không được yêu cầu **mọi** branch parameter đều có non-zero gradient.

Vì:

\[
\frac{\partial L}{\partial \theta_f}
\propto
\gamma
\]

nên khi \(\gamma=0\), gradient của branch bên trong có thể bằng 0.

Gate đúng là:

```text
Adapter có ít nhất một finite non-zero gradient
```

đặc biệt `gamma` có thể nhận gradient ở step đầu.

---

## 12. Gate optimizer.step

Chỉ kiểm tra gradient chưa đủ.

Task 14 snapshot parameter trước step và yêu cầu sau:

```python
optimizer.step()
```

ít nhất một parameter phải thay đổi trong từng nhóm:

```text
Adapter       changed
Projection    changed
Fusion        changed
Decoder       changed
```

Đây là khác biệt chính giữa:

```text
forward/backward QA
```

và:

```text
real training-step QA
```

---

## 13. Cached tensor không phải trainable parameter

Feature đọc từ cache phải:

```text
requires_grad = False
grad = None
```

Điều này là đúng.

Ta không tối ưu frozen feature tensor; gradient chỉ tối ưu các module downstream.

---

## 14. Controlled integration test

Test mặc định tự tạo một feature cache thật bằng:

```python
FeatureCacheWriter
```

sau đó đọc lại bằng:

```python
CachedFeatureDataset
make_cached_dataloader
```

Tức là không bypass storage layer bằng một dictionary giả.

Feature values trong unit test là controlled random frozen tensors, vì mục tiêu là kiểm tra integration contract chứ không benchmark DINO.

---

## 15. Real external-cache acceptance test

Có thêm:

```python
@pytest.mark.integration
def test_external_real_cache_one_training_step()
```

Để chạy trên cache thật:

```bash
export FEATURE_CACHE_DIR=/path/to/cache
export FEATURE_TRAIN_INDEX=/path/to/train_index.json
export FEATURE_MASK_ROOT=/path/to/mask/root
export FEATURE_FUSION_DIM=64

pytest -q \
  tests/test_day03_cached_train_step.py \
  -m integration -s
```

Hai biến bắt buộc:

```text
FEATURE_CACHE_DIR
FEATURE_TRAIN_INDEX
```

Nếu chưa cung cấp, test real-cache được `SKIP`, không giả vờ PASS.

---

## 16. Chạy Task 14

Unit integration:

```bash
pytest -q tests/test_day03_cached_train_step.py
```

Expected:

```text
controlled cached-feature train step PASS
geometry bridge PASS
real external cache SKIP nếu chưa set env
```

Sau khi có cache thật, chạy thêm gate integration bên trên.

---

## 17. PASS criteria

Task 14 chỉ được chốt PASS khi:

```text
[PASS] cache được đọc qua CachedFeatureDataset
[PASS] six raw frozen features finite
[PASS] Adapter forward
[PASS] Context→Local Alignment forward
[PASS] Projection forward
[PASS] Attention Fusion forward
[PASS] Decoder logits [B,1,H,W]
[PASS] logits.shape == mask.shape
[PASS] Task-8 loss finite
[PASS] backward chạy
[PASS] Adapter có finite/non-zero gradient
[PASS] Projection có finite/non-zero gradient
[PASS] Fusion có finite/non-zero gradient
[PASS] Decoder có finite/non-zero gradient
[PASS] optimizer.step chạy
[PASS] parameter thực sự thay đổi
[PASS] cached tensors vẫn không trainable
[PASS] cached path không instantiate DINO
```

---

## 18. Ý nghĩa khoa học

Nếu Task 14 PASS, kết luận hợp lệ là:

> Cached frozen-backbone features có thể đi xuyên suốt computational graph của MS-ILA tới segmentation loss và tạo ra một optimizer update hợp lệ cho các module downstream trainable.

Không được kết luận:

```text
model đã hội tụ
Overfit-16 đã thành công
AU-PRO0.05 tốt
MS-ILA tốt hơn baseline
```

Đó là câu hỏi của training/evaluation, không phải một integration step.

---

## 19. Quan hệ với Task 10

Task 10 chứng minh trainer generic có thể:

```text
model → loss → backward → optimizer.step
```

Task 14 chứng minh **project-specific cached path** thực sự nối đúng:

```text
CachedFeatureDataset
→ Adapter
→ Alignment
→ Projection
→ Fusion
→ Decoder
→ Task-8 Loss
→ Task-9 Optimizer
```

Hai task không trùng nhau.

---

## 20. Task 15

Theo yêu cầu hiện tại, Task 15 **không được code trong gói Task 14**.

Khi thực hiện Task 15, định dạng phù hợp là:

```text
Day03_QA_Report.ipynb
```

để chạy trên Google Colab và thu:

```text
environment
repo commit
unit-test summary
real-cache integration result
Overfit-16 curves
qualitative prediction
checkpoint/resume QA
final Day-03 PASS/FAIL table
```


---

## 13. Trạng thái xác minh khi bàn giao

Tại thời điểm kiểm tra repository `main`, commit được quan sát là:

```text
e0034756be89f2e0febffe1886fd0b4e76a126d4
feat: day 3
```

Repo đã có các module Task 6–13 nhưng chưa có test bắt buộc:

```text
CachedFeatureDataset
→ model
→ Task-8 loss
→ backward
→ Task-9 optimizer.step
```

Hai file Task 14 trong gói này bổ sung đúng gate đó.

Trong môi trường tạo artifact, code Task 14 đã qua **static Python compile**.
Pytest tích hợp với repository thật được thiết kế để chạy trong notebook Task 15
sau khi clone repo và overlay/commit các file Task 14. Không nên ghi `PASS` cho
real-cache experiment trước khi cell pytest/real-cache trên Colab thực sự PASS.
