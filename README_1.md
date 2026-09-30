# MS-ILA — Day 1 Architecture QA

## Mục tiêu

Ngày 1 **không train mô hình** và **không đánh giá AU-PRO\(_{0.05}\)**. Mục tiêu duy nhất là chứng minh đường truyền kiến trúc chạy đúng:

```text
RGB [B,3,H,W]
    ↓
Frozen DINOv3 — blocks 4 / 8 / 12
    ↓
3 × Residual Adapter, gamma = 0
    ↓
Mean Fusion
    ↓
Basic Decoder
    ↓
Anomaly logits [B,1,H,W]
```

Tensor nội bộ được khóa ở dạng **BCHW**. Mean Fusion chỉ nhận ba feature cùng shape; Day 1 không resize/projection để che lỗi contract.

## Nhiệm vụ được thực hiện

| File | Trách nhiệm |
|---|---|
| `src/models/msila.py` | Ghép DINOv3 → Adapter → Mean Fusion → Decoder; kiểm tra tensor contract ở boundary. |
| `tests/test_full_forward.py` | Dummy image `[2,3,512,512]` chạy end-to-end; kiểm tra output `[2,1,512,512]`, finite, DINO frozen, gamma=0 và adapter identity. |
| `scripts/day01_smoke_test.py` | Chạy Architecture QA thực với DINOv3 + checkpoint và sinh `day01_report.json`. |

Các module đã có và được tái sử dụng: `dinov3_extractor.py`, `residual_adapter.py`, `contracts.py`, `mean_fusion.py`, `basic_decoder.py`.

## Chạy

Mặc định theo layout Colab hiện tại:

```bash
pytest -q tests/test_residual_adapter_identity.py
pytest -q tests/test_dino_features.py
pytest -q tests/test_full_forward.py

python scripts/day01_smoke_test.py \
  --dinov3-repo /content/dinov3 \
  --checkpoint /content/checkpoints/dinov3_vits16_pretrain_lvd1689m.pth \
  --output day01_report.json
```

Có thể đổi đường dẫn cho pytest bằng biến môi trường:

```bash
export DINOV3_REPO=/path/to/dinov3
export DINOV3_CHECKPOINT=/path/to/dinov3_vits16_pretrain_lvd1689m.pth
```

> `test_full_forward.py` sẽ **SKIP** nếu thiếu repo/checkpoint để CI không báo lỗi giả. Tuy nhiên **Day-1 chỉ được tuyên bố PASS khi `day01_smoke_test.py` chạy thật và `day01_report.json` có `"status": "PASS"`.**

## Điều kiện Day-1 PASS

```text
[PASS] DINOv3 load được
[PASS] Extract đúng block 4 / 8 / 12
[PASS] DINO backbone frozen
[PASS] gamma initialize = 0
[PASS] Adapter(gamma=0) = Identity
[PASS] Mean Fusion + Basic Decoder chạy được
[PASS] Full forward trả [B,1,H,W], không NaN/Inf
```

Không thuộc Day 1: training, AU-PRO, Overfit-16, Local/Context, projection/alignment, Attention Fusion, illumination loss, tiny/boundary-defect analysis.

## Cơ sở đối chiếu paper / repository

1. **DINOv3 — Siméoni et al., 2025**  
   Paper: https://arxiv.org/abs/2508.10104  
   Official repo: https://github.com/facebookresearch/dinov3  
   Dùng để đối chiếu API backbone, checkpoint và `get_intermediate_layers()`.

2. **AdaptFormer — Chen et al., NeurIPS 2022**  
   Paper: https://arxiv.org/abs/2205.13535  
   Official repo: https://github.com/ShoufaChen/AdaptFormer  
   Đối chiếu tư tưởng parameter-efficient adapter trên pretrained ViT với backbone giữ frozen.

3. **ConvPass — Jie & Deng, 2022**  
   Paper: https://arxiv.org/abs/2207.07039  
   Code collection: https://github.com/JieShibo/PETL-ViT  
   Đối chiếu việc đưa convolutional spatial inductive bias vào adapter cho ViT.

4. **Deep Residual Learning — He et al., CVPR 2016**  
   Paper: https://arxiv.org/abs/1512.03385  
   Cơ sở cho residual form `x + residual(x)`.

5. **ReZero — Bachlechner et al., 2020**  
   Paper: https://arxiv.org/abs/2003.04887  
   Đối chiếu zero-initialized learnable residual gate. Trong MS-ILA Day 1: `y = x + gamma * delta`, `gamma=0` ⇒ identity lúc khởi tạo.

6. **MobileNets — Howard et al., 2017**  
   Paper: https://arxiv.org/abs/1704.04861  
   Đối chiếu depthwise convolution dùng trong spatial adapter.

## Ghi chú khoa học

Mean Fusion và Basic Decoder ở Day 1 là **control / QA components**, không phải novelty claim. Adapter hiện tại là thiết kế tổng hợp có căn cứ từ residual learning + parameter-efficient adapters + convolutional bypass + zero-initialized residual gate; cần ablation thực nghiệm ở các ngày sau trước khi khẳng định đóng góp khoa học.
