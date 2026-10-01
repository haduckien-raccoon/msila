# `cached_dataset.py` — Cache loader cho training MS-ILA

## Mục tiêu

Ngày train **không chạy DINOv3 nữa**.

```text
feature cache
    ↓
CachedFeatureDataset
    ↓
DataLoader
    ↓
Adapter → Fusion → Decoder → Loss
```

Mỗi `__getitem__()` trả:

```python
{
    "local_b4": Tensor,
    "local_b8": Tensor,
    "local_b12": Tensor,
    "context_b4": Tensor,
    "context_b8": Tensor,
    "context_b12": Tensor,
    "mask": Tensor[1, H, W],
    "meta": {...},
}
```

`cached_dataset.py` **không import DINOv3, không load ảnh RGB để extract feature, không gọi backbone**.

---

## 1. Quan hệ với `feature_cache.py`

Hai module tách trách nhiệm:

```text
data/feature_cache.py
    = schema + shard + manifest + save/load tensor

data/cached_dataset.py
    = PyTorch Dataset/DataLoader cho training
```

Loader dùng trực tiếp `FeatureCacheReader`.

Schema 6 feature vẫn là:

$$
\mathcal{F}_i =
\{
L_i^4,L_i^8,L_i^{12},
C_i^4,C_i^8,C_i^{12}
\}
$$

và một training sample được ký hiệu:

$$
D(i)=\left(\mathcal{F}_i,M_i,\mu_i\right)
$$

trong đó:

- \(M_i\): binary anomaly mask.
- \(\mu_i\): metadata (`image_id`, category, geometry, defect type, ...).

Đây chỉ là **ký hiệu dữ liệu của project**, không phải công thức/kiến trúc được lấy từ một paper.

---

## 2. Tại sao dùng map-style `Dataset`?

Feature cache đã có index:

```text
(image_id, category)
        ↓
manifest
        ↓
shard + record
```

nên một sample có thể truy cập theo chỉ số dataset.

PyTorch định nghĩa `Dataset` + `DataLoader` để hỗ trợ map-style datasets, batching, multiprocessing và automatic memory pinning.

Nguồn chính thức, kiểm tra đến 09/2026:

- PyTorch — `torch.utils.data`  
  https://docs.pytorch.org/docs/stable/data.html

---

## 3. Điểm quan trọng: bỏ batch dimension của lúc extract

Feature extraction thường chạy từng ảnh:

```text
[1, C, H, W]
```

Số `1` là **batch extraction**, không phải dimension feature của sample.

Nếu giữ nó rồi DataLoader stack tiếp:

```text
[B, 1, C, H, W]     # sai cho phần lớn Adapter/Fusion
```

Loader mặc định:

```python
squeeze_cached_batch_dim=True
```

đưa sample thành:

```text
[C, H, W]
```

sau đó DataLoader mới tạo:

```text
[B, C, H, W]
```

Tương tự token feature:

```text
[1, N, C] → [N, C] → [B, N, C]
```

Nếu cache của bạn ngay từ đầu đã lưu tensor **unbatched**, đặt:

```python
squeeze_cached_batch_dim=False
```

---

## 4. Mask: không resize ngầm trong Dataset

Loader đọc mask grayscale thành:

```text
float32 [1,H,W], giá trị {0,1}
```

quy tắc:

$$
M(x,y)=
\begin{cases}
1,&I_{\text{mask}}(x,y)>T\\
0,&\text{ngược lại}
\end{cases}
$$

mặc định \(T=0\), phù hợp mask công nghiệp phổ biến `0 / 255`.

Điểm thiết kế quan trọng: **loader không tự resize/crop mask**.

Lý do: nếu local/context pipeline đã chốt alignment, resize lại âm thầm trong loader có thể làm thay đổi biên defect và ảnh hưởng SegF1/AU-PRO. Mask phải được preprocessing đúng trước đó.

Có thể kiểm tra kích thước qua:

```python
mask_hw_source="record"   # mặc định
mask_hw_source="local"
mask_hw_source="context"
mask_hw_source="image"
```

Nếu shape sai → `MaskError`, không train tiếp.

### Normal image không có mask file

Record:

```python
{
    "image_id": "...",
    "category": "fabric",
    "mask_path": None,
    "is_anomaly": False,
    "mask_hw": [512, 512],
}
```

loader sinh zero-mask:

$$
M_i=\mathbf{0}_{H\times W}
$$

Chỉ làm vậy khi sample được ghi **tường minh** `is_anomaly=False`.

Anomaly sample thiếu mask → FAIL.

---

## 5. Multiprocessing: mỗi worker có cache reader riêng

Không nên tạo một mmap/shard reader rồi vô tình chia sẻ trạng thái reader giữa các worker.

Code dùng:

```python
os.getpid()
```

và lazy-create:

```text
worker 0 → FeatureCacheReader riêng
worker 1 → FeatureCacheReader riêng
worker 2 → FeatureCacheReader riêng
...
```

`__getstate__()` cũng bỏ reader trước khi Dataset được pickle/spawn.

Điều này phù hợp với mô hình multiprocessing của `DataLoader`.

Nguồn:

- PyTorch — `torch.utils.data.DataLoader` multiprocessing  
  https://docs.pytorch.org/docs/stable/data.html

---

## 6. `mmap=True`

`feature_cache.py` đọc shard bằng:

```python
torch.load(..., mmap=True, weights_only=True)
```

PyTorch mô tả `mmap=True` là map file và **lazy-load tensor storages khi được truy cập**, thay vì phải copy toàn bộ storage của checkpoint vào CPU memory ngay từ đầu.

Nguồn:

- PyTorch — `torch.load`  
  https://docs.pytorch.org/docs/stable/generated/torch.load.html
- PyTorch — Serialization semantics  
  https://docs.pytorch.org/docs/stable/notes/serialization

---

## 7. DataLoader setting nên dùng

Ví dụ:

```python
from data.cached_dataset import CachedFeatureDataset, make_cached_dataloader

dataset = CachedFeatureDataset(
    cache_dir="artifacts/feature_cache",
    records="train_cache_index.json",
    expected_producer_signature=signature,
    mask_root="dataset",
    mask_hw_source="record",
)

loader = make_cached_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)
```

Helper mặc định:

```text
pin_memory = torch.cuda.is_available()
persistent_workers = num_workers > 0
prefetch_factor = 2
```

PyTorch docs xác nhận:

- `persistent_workers=True` giữ worker Dataset instances sống qua các epoch.
- `pin_memory=True` cho DataLoader thực hiện memory pinning.
- `prefetch_factor` kiểm soát số batch mỗi worker preload.

Nguồn:

- https://docs.pytorch.org/docs/stable/data.html

Nếu train CUDA, vòng train nên dùng:

```python
x = batch["local_b4"].to(device, non_blocking=True)
mask = batch["mask"].to(device, non_blocking=True)
```

PyTorch tutorial hiện khuyến nghị để **DataLoader** thực hiện pinning thay vì gọi `.pin_memory()` thủ công trong main thread; `non_blocking=True` có thể cải thiện CPU→GPU transfer.

Nguồn:

- PyTorch — *A guide on good usage of non_blocking and pin_memory()*  
  https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

**Không có một `num_workers` tối ưu cho mọi máy.** NVMe, HDD, NFS và Google Drive có I/O khác nhau; phải benchmark throughput trên máy thật.

---

## 8. Training index đề xuất

`train_cache_index.json`:

```json
[
  {
    "image_id": "fabric/train/good/000.png",
    "category": "fabric",
    "mask_path": null,
    "is_anomaly": false,
    "mask_hw": [512, 512],
    "meta": {
      "split": "train",
      "defect_type": "good"
    }
  },
  {
    "image_id": "fabric/test/hole/001.png",
    "category": "fabric",
    "mask_path": "masks/fabric/test/hole/001.png",
    "is_anomaly": true,
    "mask_hw": [512, 512],
    "meta": {
      "split": "train",
      "defect_type": "hole"
    }
  }
]
```

Index chỉ chứa **annotation/meta nhẹ**. Sáu feature lớn vẫn nằm trong `.pt` shards.

---

## 9. `cached_collate_fn`

Collate tạo:

```text
local_b4   : [B, ...]
local_b8   : [B, ...]
local_b12  : [B, ...]
context_b4 : [B, ...]
context_b8 : [B, ...]
context_b12: [B, ...]
mask       : [B,1,H,W]
meta       : list[dict]
```

Nếu shape giữa samples khác nhau → FAIL.

Không tự pad/interpolate vì điều đó có thể che lỗi preprocessing/alignment.

---

## 10. PASS/FAIL trước khi train

**PASS khi:**

```text
[ ] training index không duplicate
[ ] mọi image_id/category tồn tại trong feature cache
[ ] producer_signature khớp
[ ] trả đủ L4,L8,L12,C4,C8,C12
[ ] feature finite, floating point
[ ] extraction batch dim đã được xử lý đúng
[ ] mask -> float32 [1,H,W] {0,1}
[ ] anomaly sample không thiếu mask
[ ] normal sample zero-mask có H,W xác định
[ ] batch collate ra đúng [B,...]
[ ] multi-worker không dùng chung stale reader
[ ] __getitem__ không chạy DINOv3
```

Nếu một invariant sai → **không train tiếp**.
