# `context_alignment.py` — Context → Local Alignment

## Mục tiêu

DINOv3 trả về:

- Local: `L4, L8, L12`
- Context: `C4, C8, C12`

Cả hai có thể cùng shape `32×32`, nhưng **không cùng FOV**. Vì vậy không được fuse trực tiếp theo cùng chỉ số `(i,j)`.

Module này dùng metadata từ `geometry/view_meta.py` để warp Context về hệ tọa độ Local:

```text
C4,C8,C12 + local_to_context
        ↓
 grid_sample (bilinear, align_corners=False)
        ↓
C4_to_L, C8_to_L, C12_to_L
```

## Công thức

Với tâm cell Local `(i,j)` trên feature map `Hl×Wl`, tọa độ pixel-edge trong input Local là:

$$
x_L=(j+\tfrac12)\frac{W_L}{W_l},\qquad
y_L=(i+\tfrac12)\frac{H_L}{H_l}.
$$

Metadata đã lưu ma trận đồng nhất:

$$
\tilde p_C=T_{L\rightarrow C}\tilde p_L.
$$

Sau phép chia homogeneous, chuẩn hóa sang grid PyTorch:

$$
g_x=2\frac{x_C}{W_C}-1,\qquad
g_y=2\frac{y_C}{H_C}-1.
$$

Sau đó dùng bilinear sampling:

$$
F_C^{L}(i,j)=\operatorname{BilinearSample}(F_C,g_x,g_y).
$$

`align_corners=False` là lựa chọn bắt buộc trong module vì metadata đang dùng **pixel-edge coordinates**. PyTorch cũng mô tả lựa chọn này là resolution-agnostic hơn so với `align_corners=True`.

## Tại sao không crop feature map bằng index nguyên?

Với Local `512` nằm trong Context nguồn `768`, Context được resize `768→512`. Biên Local trong Context-input rơi vào tọa độ phân số (`~85.33...` đến `~426.67...`). Trên feature grid DINOv3 `/16`, vị trí cũng là phân số. Crop nguyên chỉ số sẽ tạo sai lệch không gian; bilinear sampling giữ mapping liên tục và tránh quantization.

## Tối ưu implementation

`C4/C8/C12` có cùng grid nên code nối chúng theo channel và gọi **một lần** `grid_sample`, sau đó split lại. Bilinear sampling độc lập theo channel nên kết quả tương đương chạy 3 lần nhưng giảm overhead.

Module **không** dùng `torch.no_grad()` để nếu sau này đặt trainable adapter trước alignment thì gradient vẫn truyền qua sampler.

## Nguồn nên đọc

1. Jaderberg et al., **Spatial Transformer Networks**, NeurIPS 2015 — differentiable spatial sampler / bilinear sampling.  
   https://arxiv.org/abs/1506.02025
2. PyTorch `torch.nn.functional.grid_sample` — normalized grid, bilinear interpolation, `align_corners=False`.  
   https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
3. Richard Szeliski, **Computer Vision: Algorithms and Applications, 2nd ed.** — Sections 2.1 và 3.6 về homogeneous coordinates và geometric transformations.

## PASS

- `C4_to_L`, `C8_to_L`, `C12_to_L` đúng spatial correspondence.
- Synthetic affine grid có numerical error dưới tolerance.
- Local sampling grid phải nằm trong Context FOV.
