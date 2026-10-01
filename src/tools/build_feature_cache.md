# `build_feature_cache.py` — Feature Cache cho MS-ILA / Frozen DINOv3

**Mục tiêu:** DINOv3 đã frozen thì không chạy lại ở mỗi epoch. Mỗi ảnh được chạy backbone **một lần**, sau đó train `Adapter → Fusion → Decoder → Loss` trực tiếp từ cache.

## 1. Contract bắt buộc

Mỗi sample cache có **đúng 6 nguồn feature cốt lõi**:

```text
L4, L8, L12, C4, C8, C12
```

Trong code:

```python
local_b4, local_b8, local_b12
context_b4, context_b8, context_b12
```

và metadata alignment tối thiểu:

```python
{
    "image_id": "...",
    "category": "...",
    "geometry": {
        "local_box": [x0, y0, x1, y1],
        "context_box": [x0, y0, x1, y1],
        "context_to_local": [[...], [...], [...]]
    }
}
```

**Gate bắt buộc trước khi train:**

```python
assert torch.allclose(
    f_online["local_b8"].cpu(),
    f_cache["local_b8"],
    atol=1e-5,
)
```

File test kiểm tra **cả 6 tensor**, không chỉ `local_b8`.

---

## 2. Vì sao cache là đúng?

Đặt:

$$
F_s^{(k)} = E_k(T_s(I)), \qquad
s\in\{L,C\},\;k\in\{4,8,12\}
$$

- \(I\): ảnh.
- \(T_L,T_C\): preprocessing/crop **deterministic** cho local/context.
- \(E_k\): output của DINOv3 tại block \(k\).
- Backbone \(E\) frozen + `eval()`.

Với cùng ảnh, cùng checkpoint và cùng preprocessing, \(F_s^{(k)}\) không đổi; do đó có thể tính một lần và lưu lại.

**Điều kiện quan trọng:** nếu crop/augmentation trước DINO thay đổi ngẫu nhiên theo epoch thì feature **không còn cố định**. Khi đó không được cache một view duy nhất; phải cố định transform/seed hoặc cache từng view augmentation.

DINOv3 chính thức cũng dùng frozen backbone trong các evaluation head, và implementation có API `get_intermediate_layers(...)`.  
Nguồn:
- DINOv3 paper: https://arxiv.org/abs/2508.10104
- Official repo: https://github.com/facebookresearch/dinov3
- `vision_transformer.py`: `get_intermediate_layers`
- `eval/segmentation/models/__init__.py`: frozen backbone + intermediate layers

### Cực kỳ dễ sai: số block

Trong DINOv3 ViT, danh sách `n=[...]` của `get_intermediate_layers` dùng **index Python 0-based**.

Nếu project gọi tên theo block 1-based:

```text
B4, B8, B12
```

thì với ViT 12-block:

```python
internal_indices = [3, 7, 11]
```

Không được tự động truyền `[4, 8, 12]`.

Lưu ý: reference segmentation chính thức của DINOv3 chọn `[2, 5, 8, 11]` cho chế độ `FOUR_EVEN_INTERVALS` ở ViT-S/B 12 block. Đó là một lựa chọn khác; **không phải lý do để đổi thiết kế B4/B8/B12 của MS-ILA**. Cache builder chỉ lưu đúng output mà extractor ngày 2 đã chốt.

---

## 3. Alignment `Context → Local`

Code cung cấp:

```python
context_to_local_from_boxes(...)
```

Với box trong ảnh gốc:

```text
local   = [lx0, ly0, lx1, ly1]
context = [cx0, cy0, cx1, cy1]
```

và kích thước sau resize:

```text
local_hw   = (Hl, Wl)
context_hw = (Hc, Wc)
```

ta lưu ma trận affine đồng nhất:

$$
\begin{bmatrix}
u_L\\v_L\\1
\end{bmatrix}
=
\begin{bmatrix}
s_x&0&t_x\\
0&s_y&t_y\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
u_C\\v_C\\1
\end{bmatrix}
$$

với:

$$
s_x=\frac{W_L(c_{x1}-c_{x0})}{W_C(l_{x1}-l_{x0})},
\qquad
t_x=\frac{W_L(c_{x0}-l_{x0})}{l_{x1}-l_{x0}}
$$

và tương tự cho \(y\).

Đây là phép biến đổi affine chuẩn. Để học nền tảng:
- Richard Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., §2.1.1 (2D transformations), §3.6 (Geometric transformations).
- Bernd Jähne, *Digital Image Processing*, Ch. 10.4 (Geometric Transformations), 10.5 (Interpolation).

**Chú ý:** helper trên dùng *continuous edge coordinates*. Nếu pipeline crop hiện tại dùng quy ước pixel-center/grid riêng, hãy lưu chính ma trận mà pipeline đó sử dụng; đừng ép dùng helper này.

---

## 4. Vì sao dùng `safetensors` + JSON?

Mỗi sample tạo:

```text
samples/<shard>/<id>__<hash>.safetensors
samples/<shard>/<id>__<hash>.json
```

- `.safetensors`: 6 tensor dense.
- `.json`: `image_id`, category, geometry, shape/dtype, signature pipeline, SHA-256.
- File tensor có checksum để phát hiện cache hỏng/stale.
- Không dùng pickle cho feature payload.

Nguồn API:
- Safetensors Torch API: https://huggingface.co/docs/safetensors/api/torch

---

## 5. Vì sao `eval()` + `torch.inference_mode()`?

Builder:
- gọi `eval()` và `requires_grad_(False)` nếu extractor là `nn.Module`;
- chạy extraction trong `torch.inference_mode()`.

`inference_mode()` giảm overhead autograd khi chắc chắn không cần gradient trong feature extraction. Tuy nhiên nó **không tự gọi `eval()`**, nên phải dùng cả hai.

Nguồn:
- PyTorch `torch.inference_mode`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html

Sau khi load cache, Adapter/Fusion/Decoder vẫn train bình thường: gradient chạy qua các module trainable phía sau; chỉ không quay ngược vào DINOv3.

---

## 6. Dtype: chọn gì?

Mặc định code dùng:

```python
cache_dtype=torch.float32
```

Đây là lựa chọn đúng cho gate:

```python
torch.allclose(..., atol=1e-5)
```

`float16/bfloat16` tiết kiệm disk/I/O nhưng có thể làm sai gate \(10^{-5}\). Chỉ đổi dtype khi bạn **chủ động** thay acceptance criterion và benchmark thấy không ảnh hưởng AU-PRO.

---

## 7. Signature — tránh dùng nhầm cache

Nên đưa tối thiểu các trường sau vào `signature`:

```python
signature = {
    "backbone": "dinov3_vits16",
    "checkpoint_sha256": "...",
    "dinov3_commit": "...",
    "logical_layers_1based": [4, 8, 12],
    "internal_indices_0based": [3, 7, 11],
    "preprocess_version": "msila_local_context_v1",
    "local_size": [512, 512],
    "context_size": [768, 768],
    "normalization": "...",
}
```

Bất kỳ thay đổi nào ở checkpoint, layer, resize/crop, normalization, local/context geometry đều phải làm **cache invalid** và build lại.

---

## 8. Cách nối với code project

Tạo một factory nhỏ trong repo, ví dụ `project/cache_job.py`:

```python
from tools.build_feature_cache import CacheJob
from project.data import build_train_samples
from project.features import build_frozen_extractor

def create_job():
    extractor = build_frozen_extractor()   # output 6 tensor + geometry
    samples = build_train_samples()

    signature = {
        "backbone": "dinov3_vits16",
        "checkpoint_sha256": "...",
        "logical_layers_1based": [4, 8, 12],
        "internal_indices_0based": [3, 7, 11],
        "preprocess_version": "msila_local_context_v1",
    }
    return CacheJob(samples=samples, extractor=extractor, signature=signature)
```

Chạy:

```bash
pip install safetensors pytest

python tools/build_feature_cache.py \
  --factory project.cache_job:create_job \
  --cache-dir artifacts/feature_cache
```

Test:

```bash
pytest -q tests/test_build_feature_cache.py
```

---

## 9. PASS / FAIL trước ngày 3

**PASS chỉ khi đồng thời:**
1. Cache đủ `L4,L8,L12,C4,C8,C12`.
2. Shape online == cache cho cả 6 tensor.
3. `torch.allclose(..., atol=1e-5, rtol=1e-5)` PASS cho cả 6.
4. `local_box`, `context_box`, `context_to_local` đọc lại được.
5. Test `Context → Local` tái lập đúng.
6. Signature đúng checkpoint/preprocess/layer hiện tại.
7. Không có NaN/Inf.
8. DINOv3 không chạy lại trong epoch train Adapter/Fusion/Decoder.

**FAIL → không train tiếp** nếu một trong các mục trên sai.
