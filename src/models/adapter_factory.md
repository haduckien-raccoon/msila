# `models/adapter_factory.py` — Day 04 candidate factory

## Mục tiêu

File này **không sửa lại kiến trúc Adapter**. Nó dùng `ResidualAdapter2d` hiện có và chỉ tạo candidate theo hai biến:

- `r = bottleneck_dim`
- `d = projection_dim`

Các tham số còn lại của Adapter được giữ trong một `AdapterFactoryConfig` bất biến (`frozen dataclass`): `in_dim`, `kernel_size`, `gamma_init`, `bias`.

Vì vậy API screening là:

```python
fixed = AdapterFactoryConfig(
    in_dim=384,
    kernel_size=3,
    gamma_init=0.0,
    bias=True,
)

factory = ResidualAdapterFactory(fixed)
build = factory.build_rd(r=64, d=256)

model = build.model
print(build.run_name)        # adapter_r64_d256
print(build.trainable_params)
```

Nếu backbone hiện tại là **DINOv3 ViT-S/16**, `in_dim=384` đúng với implementation/model card chính thức. Nếu đổi backbone thì phải đổi `in_dim` **một lần trước toàn bộ screen**, không đổi giữa các candidate.

## Kiến trúc được factory tạo

Kiến trúc nằm trong `residual_adapter.py`:

\[
F\in\mathbb{R}^{B\times C\times H\times W}
\]

\[
Z_r=P_{down}(F),\qquad Z_r\in\mathbb{R}^{B\times r\times H\times W}
\]

\[
Z_s=\mathrm{GELU}(\mathrm{DWConv}(Z_r))
\]

\[
Z_d=\mathrm{GELU}(P_{mid}(Z_s)),\qquad
Z_d\in\mathbb{R}^{B\times d\times H\times W}
\]

\[
\Delta F=P_{out}(Z_d),\qquad
F'=F+\gamma\Delta F.
\]

Chuỗi `C -> r -> DWConv -> d -> C` là **thiết kế hybrid của project để ablation**, không phải kiến trúc được chép nguyên xi từ một paper.

### Nguồn nên đọc

1. **He et al., Deep Residual Learning for Image Recognition, CVPR 2016** — nguồn cho residual connection `F' = F + R(F)`.  
   https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

2. **Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019** — nền tảng bottleneck Adapter và frozen pretrained network.  
   https://proceedings.mlr.press/v97/houlsby19a.html

3. **Chen et al., AdaptFormer, NeurIPS 2022** — lightweight residual adapter cho pretrained Vision Transformer.  
   https://proceedings.neurips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html

4. **Jie et al., Convolutional Bypasses Are Better Vision Transformer Adapters** — convolutional bypass cho ViT. Bản đầu trên arXiv năm 2022; phiên bản peer-reviewed xuất hiện tại **ECAI 2024**.  
   https://arxiv.org/abs/2207.07039  
   https://doi.org/10.3233/FAIA240489

5. **DINOv3, Meta AI, 2025** — DINOv3 được thiết kế để tạo dense frozen features mạnh và Meta mô tả việc dùng lightweight adapters/readouts trên backbone frozen.  
   https://ai.meta.com/research/dinov3/

> Không có nguồn trên chứng minh `r=64, d=256` hay bất kỳ cặp `(r,d)` nào là tối ưu cho MVTec AD 2. Grid `3×3` là **coarse screening design của thí nghiệm**, phải chọn theo validation, không được tuyên bố là giá trị lý thuyết tối ưu.

## Vì sao cần `adapter_factory.py`?

Nếu code train tự tạo Adapter nhiều nơi, rất dễ xảy ra candidate A dùng `kernel_size=3` nhưng candidate B vô tình dùng `kernel_size=5`. Khi đó kết quả không còn là ablation `r,d`.

Factory khóa:

```text
in_dim       SAME
kernel_size  SAME
gamma_init   SAME
bias         SAME
```

và candidate chỉ chứa:

```text
r = bottleneck_dim
d = projection_dim
```

`AdapterCandidate.from_mapping()` cũng từ chối key lạ, vì vậy không thể lén truyền `kernel_size`, `dropout`, `reduction`, ... vào từng candidate.

## Grid Day 04

Grid đang dùng:

```text
r: 32, 64, 128
d: 128, 256, 384
```

=> `3 × 3 = 9` candidate:

```text
adapter_r32_d128
adapter_r32_d256
adapter_r32_d384
adapter_r64_d128
adapter_r64_d256
adapter_r64_d384
adapter_r128_d128
adapter_r128_d256
adapter_r128_d384
```

**Grid không hard-code trong factory.** Danh sách này phải nằm ở config/runner. Factory chỉ nhận candidate và tạo đúng model tương ứng.

## Audit số tham số

Với `C=in_dim`, kernel `k`, `r=bottleneck_dim`, `d=projection_dim`, số weight của nhánh là:

\[
Cr + rk^2 + rd + dC.
\]

Nếu `bias=True`, bias là:

\[
r+r+d+C=2r+d+C.
\]

Cộng thêm một scalar `gamma`:

\[
N_{param}=Cr+rk^2+rd+dC+(2r+d+C)+1.
\]

Công thức này **suy ra trực tiếp từ kích thước tensor của các `Conv2d` trong implementation**, không phải công thức trích từ paper. Factory kiểm tra số tham số thực tế với công thức của `ResidualAdapter2d`; sai là fail ngay.

## Điều factory không thể khóa

Yêu cầu Day 04 là **chỉ thay `r,d` trên toàn thí nghiệm**. `adapter_factory.py` chỉ kiểm soát phần Adapter, nên runner vẫn bắt buộc khóa:

```text
DINO checkpoint
DINO frozen
block 4/8/12
Local/Context
alignment
feature cache
Fusion
Decoder
loss
optimizer
learning rate
epochs
batch size
seed
train/val split
augmentation
```

Nên runner lưu full resolved config vào mỗi run và so sánh với base config trước khi train. Đó là nhiệm vụ của `train/screen_adapter.py`, **không nên nhét logic training vào model factory**.

## Test

```bash
pytest -q tests/test_adapter_factory.py
```

Test kiểm tra:

- đủ 9 candidate build được;
- run name deterministic;
- `r,d` đi đúng vào layer tương ứng;
- mọi fixed adapter setting giữ nguyên;
- forward giữ shape;
- parameter-count audit đúng;
- candidate ngoài grid vẫn build được → chứng minh factory không hard-code `32/64/128` hay `128/256/384`;
- key ngoài `r,d` bị từ chối;
- duplicate candidate bị từ chối.
