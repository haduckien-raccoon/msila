# E9 — Day-04 Fairness Audit

**File:** `tests/test_day04_fairness.py`  
**Mục tiêu:** chứng minh các candidate Day-04 giống nhau ở toàn bộ điều kiện
kiểm soát và **chỉ được phép khác `r` và `d`**.

E9 không làm E8 model selection và không làm E10 regression tests.

## 1. `r` và `d` trong implementation hiện tại

`src/models/cached_training.py` định nghĩa
`CachedFeatureTrainingModel.build_default(...)` với hai tham số trực tiếp:

```python
adapter_reduction
fusion_dim
```

Do đó Day-04 dùng:

```text
r = adapter reduction ratio
d = fusion / projection channel dimension
```

`adapter_reduction` được truyền vào `ResidualAdapter2d(... reduction=...)`.
`fusion_dim` được truyền vào `SixFeatureProjection(... fusion_dim=...)` và
`MSILADay2Head(... fusion_dim=...)`.

E9 không đoán vị trí hai biến này trong YAML/JSON. Manifest phải map semantic
`r,d` sang đúng dotted path của config thực tế.

## 2. Fairness criterion

Gọi config candidate \(i\) là \(C_i\), baseline là \(C_0\).

Hai path được phép khác:

\[
P=\{p_r,p_d\}.
\]

E9 tính exact recursive diff:

\[
D_i=Diff(C_0,C_i).
\]

Hard PASS yêu cầu:

\[
\boxed{D_i\subseteq P,\quad \forall i}
\]

Do đó bất kỳ thay đổi nào ngoài `r,d`, ví dụ:

```text
learning rate
seed
batch size
precision
kernel size
gamma_init
feature blocks
output size
augmentation
loss weights
optimizer
training steps
projection sharing
dataset/split
```

đều làm E9 FAIL nếu nằm trong subtree được audit.

Ý nghĩa khoa học: nếu ngoài `r,d` còn biến khác thay đổi, chênh lệch metric
không thể được quy riêng cho `r,d`.

## 3. Kiểm tra độc lập bằng canonical hash

Sau recursive diff, E9 tạo:

\[
\tilde C_i=C_i\setminus\{r,d\}.
\]

Rồi tính:

\[
h_i=SHA256(\tilde C_i).
\]

PASS tiếp tục yêu cầu:

\[
\boxed{h_1=h_2=\dots=h_n}.
\]

Vì vậy fairness được kiểm tra hai lớp:

```text
recursive diff -> giải thích chính xác field nào sai
canonical hash -> chứng minh toàn bộ controlled config giống nhau
```

## 4. Exact comparison

Config được so sánh exact:

```text
0.0001 != 0.0001001
1       != 1.0
true    != 1
```

Không dùng floating tolerance vì tolerance phù hợp với numerical output, không
phù hợp để xác nhận experiment configuration có giống nhau hay không.

Mapping order không ảnh hưởng. Sequence order có ảnh hưởng:

```text
[4, 8, 12] != [4, 12, 8]
```

## 5. `r,d` phải hợp lệ

Implementation hiện tại yêu cầu reduction/fusion dimension dương. E9 vì vậy
yêu cầu:

```text
r > 0
d > 0
```

và cả hai phải là integer.

Mỗi candidate cũng phải có cặp `(r,d)` duy nhất. Hai candidate có cùng `(r,d)`
không tạo ra một experimental condition mới nên bị reject.

## 6. Manifest

Tạo:

```text
configs/day04_fairness.json
```

Ví dụ:

```json
{
  "schema_version": "msila.e9.fairness_manifest.v1",
  "baseline_candidate": "r4_d128",
  "config_root": null,
  "allowed_differences": {
    "r": "adapter_reduction",
    "d": "fusion_dim"
  },
  "candidates": {
    "r4_d128": "configs/day04/r4_d128.yaml",
    "r8_d128": "configs/day04/r8_d128.yaml",
    "r4_d256": "configs/day04/r4_d256.yaml"
  }
}
```

Nếu config thực tế lồng sâu:

```yaml
model:
  adapter:
    reduction: 4
  fusion:
    dim: 128
```

manifest phải ghi:

```json
"allowed_differences": {
  "r": "model.adapter.reduction",
  "d": "model.fusion.dim"
}
```

Không sửa test để tự đoán alias.

## 7. `config_root`

Nếu file config có metadata không thuộc scientific config:

```yaml
candidate_id: r4_d128
created_at: ...
experiment:
  adapter_reduction: 4
  fusion_dim: 128
  training:
    learning_rate: 0.0001
```

không thêm `candidate_id` hay timestamp vào ignore list.

Dùng:

```json
"config_root": "experiment"
```

E9 chỉ audit subtree `experiment`.

Không có cơ chế `ignore_paths` tùy ý vì điều đó dễ vô tình bỏ qua learning rate,
precision hoặc augmentation.

## 8. Output `config_diff.json`

PASS:

```json
{
  "status": "PASS",
  "candidate_factors": {
    "r4_d128": {"r": 4, "d": 128},
    "r8_d128": {"r": 8, "d": 128}
  },
  "controlled_hashes_identical": true,
  "unexpected_differences": []
}
```

Nếu learning rate bị đổi:

```json
{
  "status": "FAIL",
  "unexpected_differences": [
    {
      "candidate_id": "r8_d128",
      "path": "training.learning_rate",
      "reference": 0.0001,
      "candidate": 0.0002
    }
  ]
}
```

Report được ghi trước khi raise `FairnessAuditError`, nên CI FAIL vẫn có diff để
debug.

## 9. Chạy trực tiếp

```bash
python tests/test_day04_fairness.py \
  --manifest configs/day04_fairness.json \
  --report outputs/day04/fairness/config_diff.json
```

PASS trả:

```text
[E9 PASS]
baseline=...
r_path=...
d_path=...
manifest_sha256=...
report=...
```

FAIL trả exit code `2`.

## 10. Chạy bằng pytest

Nếu file mặc định tồn tại:

```text
configs/day04_fairness.json
```

chạy:

```bash
pytest -q tests/test_day04_fairness.py
```

Hoặc explicit:

```bash
DAY04_FAIRNESS_MANIFEST=configs/day04_fairness.json \
pytest -q tests/test_day04_fairness.py
```

Nếu chưa có manifest thật, pytest trả **SKIP**.

Quan trọng:

```text
SKIP != E9 PASS
```

Test không tạo candidate config giả để biến project thành PASS giả.

## 11. Hard PASS

E9 PASS khi đồng thời:

```text
1. Có ít nhất 2 candidate configs.
2. r path tồn tại trong mọi config.
3. d path tồn tại trong mọi config.
4. r,d là positive integers.
5. Mọi (r,d) pair là unique.
6. Recursive diff chỉ chứa r_path hoặc d_path.
7. Không key nào thêm/xóa ngoài r,d.
8. Sequence ordering ngoài r,d giống nhau.
9. Type/value ngoài r,d giống nhau exact.
10. SHA-256 sau khi bỏ r,d giống nhau cho mọi candidate.
```

Đây là đúng phạm vi E9: audit fairness của candidate configuration.
