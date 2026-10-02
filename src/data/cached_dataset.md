# `data/cached_dataset.py` — Day 04 cache-only training

## Kết luận

File cũ **đã đúng kiến trúc chính** cho Day 04: loader chỉ đọc feature đã cache và không chứa DINO/DINOv3/extractor. Vì vậy **không cần viết lại kiến trúc Dataset**.

Bản này chỉ bổ sung phần cần thiết để screen Adapter công bằng và tái lập được.

## Pipeline

```text
frozen DINO feature cache (Day 03)
        ↓
CachedFeatureDataset
        ↓
DataLoader
        ↓
Adapter(r,d) → Fusion → Decoder → Loss
```

`cached_dataset.py` **không chạy lại DINO**. Adapter cũng không được đưa vào cache, vì `r,d` là biến cần thay đổi ở Day 04.

## Thay đổi so với file cũ

1. Thêm `seed_worker()` để seed Python/NumPy theo worker seed của PyTorch.
2. `make_cached_dataloader(..., seed=42)` tạo `torch.Generator` riêng để thứ tự shuffle có thể lặp lại giữa các candidate.
3. Cấm truyền đồng thời `seed` và `generator` để tránh hai nguồn RNG mơ hồ.
4. Giữ nguyên mặc định `feature_dtype=None`: loader không tự đổi dtype của feature cache.
5. Không thêm augmentation, DINO inference hay bất kỳ xử lý làm thay đổi feature.

## Dùng cho Day 04

```python
dataset = CachedFeatureDataset(
    cache_dir="cache/day03",
    records="splits/fabric_train.json",
    expected_producer_signature=DAY03_SIGNATURE,
    feature_dtype=None,
)

loader = make_cached_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    seed=42,
)
```

Với 9 candidate Adapter, giữ **y hệt**:

```text
cache_dir / producer signature
records + train/val split
batch_size
shuffle
seed
num_workers
mask policy
feature dtype
Fusion / Decoder / loss / optimizer / LR / epochs / augmentation
```

Chỉ thay:

```text
r = bottleneck_dim
d = projection_dim
```

## Cache ↔ online consistency

Loader này **không tự chạy online DINO để kiểm tra lại**, vì làm vậy trái mục tiêu “không chạy lại DINO”.

Consistency phải được xác nhận ở gate của Day 03 khi tạo cache. Day 04 chỉ:

- đọc đúng schema cache;
- kiểm tra `expected_producer_signature`;
- fail nếu sample index không có trong cache;
- fail nếu shape mask/alignment sai;
- trả feature cache mà không có phép biến đổi ngẫu nhiên.

Do đó `expected_producer_signature` của Day 03 phải được khóa và dùng giống nhau cho cả 9 run.

## Nguồn kỹ thuật

Không có công thức AI mới trong file Dataset này.

Phần reproducibility bám theo tài liệu chính thức PyTorch `DataLoader`: `generator` điều khiển `RandomSampler`/base seed của workers; mỗi worker có seed PyTorch riêng và `worker_init_fn` có thể dùng `torch.initial_seed()` để seed các thư viện khác.

- PyTorch — `torch.utils.data.DataLoader`, *Randomness in multi-process data loading*:
  https://docs.pytorch.org/docs/main/data.html

Lưu ý khoa học: cùng seed không đồng nghĩa toàn bộ training chắc chắn bitwise-deterministic trên mọi GPU/operator. Determinism của toàn experiment còn phải được khóa ở training runner; Dataset chỉ chịu trách nhiệm về dữ liệu/cache và thứ tự sampling.
