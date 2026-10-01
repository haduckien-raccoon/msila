# `feature_projection.py` — Projection 6 nguồn feature về cùng dimension `d`

## Input sau alignment

```text
Local:              L4, L8, L12
Context aligned:    C4_to_L, C8_to_L, C12_to_L
```

Tất cả phải có cùng `B/H/W` trước projection.

## Projection

Mỗi level dùng `1×1 Conv`:

$$
F'_k(p)=W_kF_k(p)+b_k,
$$

với:

$$
W_k\in\mathbb{R}^{d\times C_{DINO}}.
$$

`kernel_size=1`, `stride=1`, `padding=0` nên không đổi `H,W`; chỉ học phép chiếu channel:

$$
[B,C_{DINO},h,w]\rightarrow[B,d,h,w].
$$

Output khóa:

```python
{
    "local_b4":   ...,   # [B,d,h,w]
    "local_b8":   ...,
    "local_b12":  ...,
    "context_b4": ...,   # đã aligned về Local
    "context_b8": ...,
    "context_b12":...
}
```

## Vì sao mặc định share projection Local/Context theo cùng block?

`L4` và `C4_to_L` đều đến từ **cùng block 4 của cùng frozen DINOv3 backbone**, nên chúng có cùng channel basis. Dùng cùng `proj_b4` giữ hai view trong cùng learned projection space và giảm một nửa số tham số projection. Tương tự cho block 8 và 12.

Đây là **design choice của đề tài**, không phải DINOv3 bắt buộc. Code cho phép `share_across_views=False` để ablation nếu muốn kiểm tra projector riêng cho từng view.

## Vì sao 1×1 Conv?

Đây là cách chuẩn để đổi channel dimension mà không trộn không gian. FPN dùng lateral `1×1 convolution` để đưa các feature level về cùng số channel trước fusion. Với bài toán này, cùng nguyên lý được dùng để khóa 6 nguồn về cùng dimension `d`.

Không thêm activation/normalization ở đây để module chỉ làm đúng một nhiệm vụ: **channel projection**. Adapter/Fusion phía sau mới quyết định nonlinear transformation.

## PASS

Sau `SixFeatureProjection` phải có đúng 6 tensor, tất cả:

$$
F_i\in\mathbb{R}^{B\times d\times h\times w}
$$

và finite.
