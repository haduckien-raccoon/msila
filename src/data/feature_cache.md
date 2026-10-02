# `feature_cache.py` — format cache chuẩn cho MS-ILA

## Chốt thiết kế

Dùng **`.pt` sharded cache + `manifest.json`**, thay vì một file `.pt` khổng lồ hoặc hàng chục nghìn file `.pt` theo từng ảnh.

```text
feature_cache/
├── manifest.json
└── shards/
    ├── shard-000000.pt
    ├── shard-000001.pt
    └── ...
```

Mặc định mỗi shard khoảng **512 MiB** tensor payload, tối đa **1024 samples/shard**. Đây là **engineering default**, không phải hằng số khoa học; benchmark I/O trên SSD/Drive thật rồi mới đổi.

## Schema v1 cố định

Mỗi record có đúng:

```python
{
    "image_id": str,
    "category": str,
    "local_b4": Tensor,
    "local_b8": Tensor,
    "local_b12": Tensor,
    "context_b4": Tensor,
    "context_b8": Tensor,
    "context_b12": Tensor,
    "geometry": {
        "local_box": [x0, y0, x1, y1],
        "context_box": [x0, y0, x1, y1],
        "context_to_local": [[...], [...], [...]],
        # optional inside geometry:
        "image_hw": [H, W],
        "local_hw": [H, W],
        "context_hw": [H, W],
    },
}
```

Tên schema:

```text
msila.feature_cache
```

Version:

```text
1
```

Không tự thêm top-level field vào schema v1. Muốn đổi cấu trúc thì tăng `SCHEMA_VERSION` và viết migration rõ ràng.

---

## Vì sao chọn `.pt` shard vào 09/2026?

Pipeline train hiện tại là PyTorch, nên `.pt` giữ trực tiếp `torch.Tensor`, shape và dtype mà không cần chuyển qua NumPy.

PyTorch 2.x hiện dùng ZIP64 cho `torch.save`; `torch.load(..., mmap=True)` hỗ trợ memory-map tensor storages, hữu ích khi cache lớn. Từ PyTorch 2.6, `weights_only=True` trở thành mặc định khi không truyền custom pickle module; code vẫn ghi **tường minh** `weights_only=True`.

Nguồn chính thức:

- PyTorch — Serialization semantics:  
  https://docs.pytorch.org/docs/stable/notes/serialization
- PyTorch — `torch.load`:  
  https://docs.pytorch.org/docs/stable/generated/torch.load.html

### Tại sao không chọn `.npz` làm mặc định?

`numpy.savez` / `savez_compressed` là chuẩn tốt khi cần NumPy/cross-framework. Nhưng ở đây training đọc trực tiếp PyTorch tensor, vì vậy `.npz` thêm bước NumPy → Torch. `savez_compressed` còn phải nén/giải nén DEFLATE, thường không đáng cho feature cache được đọc lặp lại nhiều epoch.

Nguồn:

- NumPy — `numpy.savez` / `numpy.savez_compressed`:  
  https://numpy.org/doc/stable/reference/generated/numpy.savez.html  
  https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html

---

## Tại sao phải shard?

Hai cực đoan đều không tốt:

```text
1 sample = 1 file
```

→ rất nhiều inode/open/close, đặc biệt khó chịu trên network drive/Google Drive.

```text
toàn bộ dataset = 1 file
```

→ file lớn, khó rebuild một phần, hỏng một file là ảnh hưởng toàn cache.

Shard là điểm cân bằng:

```text
samples → shard ~512 MiB → manifest index
```

Random lookup:

```text
(image_id, category)
    ↓ SHA-256
sample_key
    ↓ manifest.json
shard + record_key
    ↓
sample tensors
```

---

## Công thức duy nhất trong module

Ước lượng payload của một sample:

\[
B_{\text{sample}}
=
\sum_{i=1}^{6}
N_i\,s_i
\]

trong đó:

- \(N_i=\text{numel}(F_i)\)
- \(s_i=\text{element\_size}(F_i)\) byte
- 6 feature là `L4,L8,L12,C4,C8,C12`.

Công thức này chỉ dùng để quyết định lúc nào flush shard; nó **không phải loss hay kiến trúc AI**.

Ví dụ FP32:

\[
s_i=4\text{ bytes}.
\]

Nếu feature `float16`:

\[
s_i=2\text{ bytes}.
\]

Nguồn API PyTorch cho tensor storage/serialization:  
https://docs.pytorch.org/docs/stable/notes/serialization

---

## `producer_signature` để chống cache stale

Manifest lưu fingerprint của pipeline sinh feature:

```python
producer_signature = {
    "backbone": "dinov3_vits16",
    "checkpoint_sha256": "...",
    "logical_layers_1based": [4, 8, 12],
    "internal_indices_0based": [3, 7, 11],
    "preprocess_version": "msila_local_context_v1",
    "local_size": [512, 512],
    "context_size": [768, 768],
    "normalization": "...",
}
```

Hash:

\[
h = \mathrm{SHA256}(\mathrm{canonicalJSON(signature)})
\]

Nếu checkpoint/preprocessing/layer selection khác → `ProducerMismatchError`.

SHA-256 ở đây chỉ dùng làm **content/config fingerprint**, không phải thành phần của mô hình học sâu.

---

## Cách dùng

```python
from src.data.feature_cache import FeatureCacheWriter, FeatureCacheReader

signature = {
    "backbone": "dinov3_vits16",
    "checkpoint_sha256": "...",
    "logical_layers_1based": [4, 8, 12],
    "internal_indices_0based": [3, 7, 11],
    "preprocess_version": "msila_local_context_v1",
}

with FeatureCacheWriter(
    "artifacts/feature_cache",
    producer_signature=signature,
) as writer:
    for sample in extracted_samples:
        writer.add(sample)
```

Đọc trong Dataset:

```python
cache = FeatureCacheReader(
    "artifacts/feature_cache",
    expected_producer_signature=signature,
    mmap=True,
)

sample = cache.get(
    image_id="fabric/train/good/000.png",
    category="fabric",
)

x = sample["local_b8"]
```

Pre-training gate:

```python
from src.data.feature_cache import validate_cache

print(validate_cache("artifacts/feature_cache"))
```

---

## Quan hệ với Feature Cache Builder

Phân vai đúng:

```text
build_feature_cache.py
    = chạy frozen extractor + tạo sample

data/feature_cache.py
    = định nghĩa format + shard + manifest + save/load
```

Tức là file này **không gọi DINOv3**.

Pipeline:

```text
Image
  ↓
Frozen DINOv3 / extractor
  ↓
{L4,L8,L12,C4,C8,C12 + geometry}
  ↓
FeatureCacheWriter
  ↓
.pt shards + manifest.json
  ↓
FeatureCacheReader
  ↓
Adapter → Fusion → Decoder → Loss
```

---

## PASS trước khi giao cho training

```text
[ ] manifest schema_name == "msila.feature_cache"
[ ] schema_version == 1
[ ] đủ chính xác 6 feature keys
[ ] tensor floating-point, finite, non-empty
[ ] geometry có local_box/context_box/context_to_local
[ ] producer_signature khớp
[ ] không có duplicate (category, image_id)
[ ] validate_cache(...) PASS
[ ] test round-trip tensor PASS
[ ] test shard split PASS
[ ] test producer mismatch PASS
```

Nếu schema/version/signature không khớp: **fail loud, không train tiếp**.
