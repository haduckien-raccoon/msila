# `residual_adapter.py` — Day 04: screen `r × d`

## Mục tiêu

Thay Adapter cũ bằng module **configurable** để ablation đúng hai biến:

- `r = bottleneck_dim`: độ rộng bottleneck.
- `d = projection_dim`: độ rộng projection/hidden.

Không hard-code `r=64`, `d=256` trong model. Grid được khóa riêng tại `configs/day04_adapter_grid.yaml`.

> **Thuật ngữ:** code không gọi `r` là *matrix rank* vì module này không factorize ma trận như LoRA; `r` ở đây chỉ là **bottleneck width**.

## Kiến trúc và công thức

Với `F ∈ R^(B×C×H×W)`:

\[
Z_r=P_{down}(F), \qquad
Z_s=\phi(\mathrm{DWConv}_k(Z_r)),
\]

\[
Z_d=\phi(P_{mid}(Z_s)), \qquad
\Delta F=P_{out}(Z_d),
\]

\[
\boxed{F'=F+\gamma\Delta F}
\]

trong đó `P_down: C→r`, `DWConv: r→r`, `P_mid: r→d`, `P_out: d→C`, và `φ = GELU`.

`gamma_init=0` ⇒ `F'=F` chính xác lúc khởi tạo. Lưu ý: ở backward đầu tiên, gradient nhánh adapter bị nhân bởi `γ=0`; `γ` vẫn có thể nhận gradient. Sau khi optimizer làm `γ≠0`, các weight nhánh bắt đầu nhận gradient.

**Chuỗi `C→r→DWConv→d→C` là thiết kế hybrid của đề tài để screen `r,d`, không phải kiến trúc được một paper đề xuất nguyên xi.** Các thành phần có căn cứ riêng bên dưới.

## Grid Day 04 đã khóa

`configs/day04_adapter_grid.yaml` có đúng 9 candidate:

```text
r ∈ {32, 64, 128}
d ∈ {128, 256, 384}
```

Tên run deterministic:

```text
adapter_r32_d128   adapter_r32_d256   adapter_r32_d384
adapter_r64_d128   adapter_r64_d256   adapter_r64_d384
adapter_r128_d128  adapter_r128_d256  adapter_r128_d384
```

Với DINOv3 ViT-S/16 hiện tại, `C=384`; official DINOv3 khai báo `embed_dim=384`. Nếu backbone thay đổi, sửa `adapter_defaults.in_dim` **một lần trước khi bắt đầu toàn bộ screen**, không sửa giữa các run.

## Cách dùng

```python
from src.models.residual_adapter import ResidualAdapter2d

adapter = ResidualAdapter2d(
    in_dim=384,
    bottleneck_dim=64,   # r
    projection_dim=256, # d
    kernel_size=3,
    gamma_init=0.0,
)

y = adapter(x)          # x, y: [B, 384, H, W]
```

Nếu DINOv3 trả token `[B,N,C]`, phải reshape patch token về `[B,C,H,W]` ở pipeline feature trước Adapter; module này cố ý chỉ nhận spatial feature map 4-D.

## Điều kiện PASS

```bash
pytest -q tests/test_residual_adapter_screening.py
pytest -q tests/test_adapter_candidates.py
```

Test kiểm tra: YAML đúng 9 candidate cố định; tên run đúng; cả 9 candidate forward được; shape/identity đúng; `r,d` thật sự đi vào đúng layer; gradient hữu hạn; parameter count đúng công thức; config lỗi bị chặn.

Với `bias=True`, số tham số trainable của một Adapter là:

\[
N=Cr+rk^2+rd+dC+(2r+d+C)+1,
\]

trong đó `+1` là scalar `γ`. Công thức này giúp báo cáo trade-off AU-PRO0.05 ↔ params/runtime/VRAM khi screen.

## Nguồn nên học

1. **He et al., CVPR 2016 — Deep Residual Learning for Image Recognition.** Nguồn residual mapping `F + R(F)`.  
   https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

2. **Houlsby et al., ICML 2019 — Parameter-Efficient Transfer Learning for NLP.** Bottleneck adapter và freeze pretrained backbone.  
   https://proceedings.mlr.press/v97/houlsby19a.html

3. **Chen et al., NeurIPS 2022 — AdaptFormer.** Lightweight residual adapter cho ViT.  
   https://arxiv.org/abs/2205.13535

4. **Jie & Deng, ECCV 2022 — ConvPass.** Convolutional bypass đưa local spatial inductive bias vào ViT adapter; paper mô tả adapter bottleneck và convolutional bypass cho visual tasks.  
   https://arxiv.org/abs/2207.07039

5. **Howard et al., 2017 — MobileNets.** Nền tảng depthwise convolution / depthwise separable convolution.  
   https://arxiv.org/abs/1704.04861

6. **Bachlechner et al., UAI 2021 — ReZero.** Zero-initialized scalar residual gate.  
   https://arxiv.org/abs/2003.04887

7. **DINOv3, 2025.** Frozen visual backbone + strong dense features; official ViT-S/16 dùng `embed_dim=384`.  
   https://ai.meta.com/research/dinov3/  
   https://github.com/facebookresearch/dinov3

8. **Zhang et al., 2025 — META: Memory Efficient Transformer Adapter for Dense Predictions.** Bằng chứng gần hơn với dense prediction rằng lightweight convolutional branch vẫn là hướng hợp lý.  
   https://arxiv.org/abs/2502.01962

9. **Lüddecke et al., CVPR 2026 — LiDeRe.** Lightweight dense readout trên frozen backbones, gồm DINOv3; nhắc rằng không nên mặc định phải fine-tune backbone lớn.  
   https://openaccess.thecvf.com/content/CVPR2026/html/Luddecke_LiDeRe_A_Lightweight_Readout_for_Fast_and_Data-Efficient_Dense_Prediction_CVPR_2026_paper.html

## Quy tắc screen khoa học

Ngày 04 chỉ thay `(r,d)`. Giữ cố định backbone/checkpoint, feature layer, fusion/decoder, loss, optimizer, LR, epoch/steps, augmentation, seed và evaluation protocol. Đánh giá ít nhất `AU-PRO0.05` + trainable params + runtime + VRAM. **Không có paper nào chứng minh `(64,256)` hay bất kỳ cặp nào là tối ưu phổ quát; winner phải được chọn bằng validation/ablation của chính đề tài.**
