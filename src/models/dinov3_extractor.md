# `dinov3_extractor.py` — Local/Context DINOv3 features

**Mục tiêu TV1:** dùng **cùng một DINOv3 frozen backbone** để lấy 3 tầng đặc trưng cho hai view:

```text
Local   [B,3,512,512] -> L4, L8, L12
Context [B,3,512,512] -> C4, C8, C12
```

Với `dinov3_vits16` (patch size `P=16`, embedding `C=384`):

```text
L4,L8,L12,C4,C8,C12: [B,384,32,32]
32 = 512 / 16
```

## 1. Block 4/8/12

Trong báo cáo ta dùng số block theo cách đọc của con người (1-based):

```text
block 4  -> API index 3
block 8  -> API index 7
block 12 -> API index 11
```

`get_intermediate_layers(..., n=(3,7,11), reshape=True, norm=True)` là API chính thức của DINOv3.

**Lưu ý khoa học:** chọn 4/8/12 là thiết kế multi-level của đề tài (lấy early/mid/deep features tương đối đều trong ViT-S 12 blocks), **không phải DINOv3 paper tuyên bố 4/8/12 là bộ tối ưu**. Phải ablation nếu dùng làm novelty/claim.

## 2. Kích thước feature map

Với patch size `P`, ảnh `H x W` sinh lưới patch:

$$
H_f = H/P, \qquad W_f = W/P.
$$

Nên `512/16 = 32`, tức mỗi block trả `[B,C,32,32]` cho ViT-S/16.

Nguồn: implementation chính thức `VisionTransformer.get_intermediate_layers()` của DINOv3; `reshape=True` trả dense patch map.

## 3. Local và Context dùng cùng backbone

Hai tensor cùng `512x512` nhưng **không cùng FOV nguồn**:

- Local: crop nguồn 512.
- Context: crop nguồn 768 rồi downsample về 512.

Do đó L4/C4 (tương tự L8/C8, L12/C12) cùng shape nhưng mỗi ô patch đại diện spatial scale khác nhau. Việc alignment/fusion phía sau phải dùng metadata hình học từ `geometry/view_meta.py`, không coi chúng pixel-aligned mặc định.

## 4. Tối ưu throughput

Mặc định `extract_local_context(..., strategy="concat")`:

```text
[B,3,512,512] Local
[B,3,512,512] Context
       |
       +-- cat batch --> [2B,3,512,512]
                           |
                       DINOv3 x1
                           |
                     split Local/Context
```

Ưu điểm: chỉ **1 backbone call**, phù hợp throughput. FLOPs lý thuyết không giảm vì vẫn xử lý `2B` ảnh; peak memory tăng. Nếu GPU thiếu VRAM, dùng `strategy="sequential"`.

## 5. Vì sao dùng `torch.no_grad()` chứ không ép `inference_mode()`?

Backbone bị freeze (`requires_grad=False`, `eval()`), nhưng output còn được đưa vào Adapter/Fusion/Decoder **có train**. PyTorch mô tả tensor tạo trong `inference_mode()` có hạn chế khi tham gia computation được autograd ghi lại sau đó; `no_grad()` phù hợp và an toàn hơn cho pipeline frozen-backbone -> trainable-head.

## 6. Preprocessing

Extractor giả định đầu vào đã chuẩn hóa từ `data/multiview_transform.py`. Với DINOv3 LVD-1689M, implementation chính thức dùng ImageNet normalization:

$$
x'_c=\frac{x_c-\mu_c}{\sigma_c}
$$

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Không normalize lại trong extractor để tránh preprocessing hai lần.

## 7. PASS criteria

```text
keys = L4,L8,L12,C4,C8,C12
shape hợp lệ
finite khi bật check_finite=True
backbone frozen + eval
block mapping 4/8/12 -> 3/7/11
Local/Context split đúng
concat = 1 backbone call
sequential = 2 backbone calls
```
