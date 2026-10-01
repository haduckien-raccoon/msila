# `multiview_transform.py` — Local 512 + Context 768→512

**Cập nhật:** 30/09/2026.  
**Mục tiêu:** sinh hai view có **cùng input 512×512 nhưng khác FOV nguồn** cho nhánh multi-scale/local–context của đề tài anomaly localization.

## Contract

- Local: crop nguồn `512×512` → `x_local [3,512,512]`.
- Context: crop nguồn `768×768` → resize `512×512` → `x_context [3,512,512]`.
- Local nằm **đồng tâm** trong Context; batch qua `DataLoader` thành `[B,3,512,512]`.
- Tọa độ dùng half-open box `[x0,y0,x1,y1)` để khớp slicing Python/PyTorch.

Với `S_local=512`, `S_context=768`:

$$
m=\frac{S_{context}-S_{local}}{2}=128\text{ px}
$$

Context được đưa về input bằng hệ số:

$$
s=\frac{S_{input}}{S_{context}}=\frac{512}{768}=\frac{2}{3}.
$$

Do đó vùng Local 512 px chỉ chiếm khoảng `512×2/3 = 341.33 px` trong tensor Context sau resize. **Hai tensor cùng shape nhưng khác spatial scale**, nên không được fuse point-to-point nếu chưa alignment; metadata `local_box_in_context_input_xyxy` được trả ra để phục vụ bước đó.

Normalization mặc định theo DINOv3 LVD-1689M:

$$
x'_c=\frac{x_c-\mu_c}{\sigma_c},
$$

với `mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`.

## Vì sao code chọn cách này

- **Nested same-center** là thiết kế của đề tài để quan hệ hình học Local↔Context cố định, dễ audit và dễ alignment; không tuyên bố đây là công thức của DINO.
- DINO chứng minh giá trị của multi-crop/global–local views; DINOv3 tiếp tục dùng global/local crops và trong cấu hình high-resolution chính thức có crop size gồm `512` và `768`.
- Context `768→512` là downsampling, vì vậy dùng **bicubic + antialias**. DINO/DINOv3 augmentation chính thức dùng bicubic; torchvision khuyến nghị antialias khi bilinear/bicubic resize.
- Không resize âm thầm ảnh `<768`; nếu cần upscale/tile phải quyết định ở data pipeline và ghi trong ablation, tránh thay đổi ngầm FOV vật lý.
- Không nhét ColorJitter/blur/rotation vào module này để geometry ảnh–mask của anomaly localization không bị khó kiểm soát.

## Dùng

```python
from data.multiview_transform import MultiViewConfig, NestedMultiViewTransform

train_tf = NestedMultiViewTransform(MultiViewConfig(sampling="random"))
eval_tf  = NestedMultiViewTransform(MultiViewConfig(sampling="center"))

sample = train_tf(image)
print(sample["x_local"].shape)    # [3,512,512]
print(sample["x_context"].shape)  # [3,512,512]
```

Chạy test:

```bash
pytest -q tests/test_multiview_transform.py
```

## PASS

- `x_local`, `x_context`: đúng shape, finite.
- crop Local khớp **đúng pixel nguồn** khi `normalize=False`.
- Local nằm hoàn toàn trong Context và margin mặc định đúng `128 px`.
- Context FOV gốc đúng `768×768`, scale `768→512 = 2/3`.
- batch đúng `[B,3,512,512]`.
- random crop reproducible khi khóa PyTorch seed.
- PIL / NumPy / Tensor đều hỗ trợ; ảnh quá nhỏ hoặc float ngoài `[0,1]` fail rõ ràng.
