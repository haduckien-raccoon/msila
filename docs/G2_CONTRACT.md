# G2 contract v2 — D5-TV1 → TV2

Manifest chung: `configs/g2_experiments.yaml`. E1/E2 đã có trong
`src.models.msila`; E3–E5 dành cho TV2, chưa có implementation trong task này.
Năm fusion candidate: `mean`, `concat`, `weighted_sum`, `gated`, `attention`;
đây là danh mục cho TV2, không phải năm method đã được triển khai/chạy.

## Model và batch

Giao diện E1–E5: `forward(image, *, return_trace=False)` nhận tensor hoặc batch,
trả raw logits
`[B,1,512,512]`; `return_trace=True` trả `(logits, trace: dict)` với
`trace["decoder_feature"]` là tensor đi vào Decoder. E1/E2 hỗ trợ ngay; TV2 giữ
cùng chữ ký cho E3–E5. `model(image)` vẫn nhận tensor như runner G1; kích thước
khác 512 chỉ thuộc đường tương thích G1, batch G2 luôn khóa 512.

| Batch key | Contract |
| --- | --- |
| `image` | Local RGB float `[B,3,512,512]`, đã normalize DINOv3 bằng mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` |
| `mask` | Train: float binary `[B,1,512,512]`, 1=anomaly; cùng device với image |
| `meta` | Train: list B dict; giữ source/split/native size và tile coordinates từ loader G1 |
| `context` | Tùy chọn cho TV2: cùng shape/device/dtype với image; FOV 768 được resize về 512, cùng tâm Local |
| `view_meta` | Tùy chọn: list B dict geometry từ pipeline hiện có; TV2 truyền cho alignment khi cần |

Inference có thể bỏ `mask/meta`. Dùng `validate_g2_batch(batch)` trước train;
model tự kiểm tra các key có mặt. E1/E2 chỉ trích feature từ `image`, không chạy
Context. Không sửa Data/Evaluator; adaptation key nếu cần nằm ở boundary TV2.

## E2 và gradient

`DINOv3(feature_mode="deepest", frozen/eval) → 1 ResidualAdapter2d → BasicDecoder`.
Lấy duy nhất `b{extractor.depth}: [B,C,32,32]`, với `C=extractor.out_channels`;
ViT-S/16 C=384, ViT-B/16 C=768, cùng block cuối b12. Không Context/Projection/Fusion.
E2 kế thừa đường extract/Decoder của E1, tạo Adapter bằng `ResidualAdapterFactory`:
`C→r→DWConv→d→C`, `F'=F+gamma*ΔF`. `r=adapter_bottleneck_dim`,
`d=adapter_projection_dim`; `d` không đổi C của Decoder.

`gamma=0` phải identity chính xác; nếu Decoder có cùng state, logits E2 bằng E1.
Backward đầu: gamma/Decoder có gradient hữu hạn; các Conv của Adapter có gradient
bằng 0 hợp lệ vì gate=0. Sau gate update, nhánh Conv phải nhận gradient khác 0.
DINO không có gradient và không đổi state sau optimizer step. Optimizer chỉ lấy
`p.requires_grad`; E2 train đúng Adapter+Decoder, E1 chỉ Decoder.
Adapter params: `C*r+r*k²+r*d+d*C+(2*r+d+C nếu bias)+1`.

## Đổi backbone ở một chỗ

YAML mặc định `backbone.name: dinov3_vitb16`; chỉ đổi name để chọn checkpoint
trong `backbone.checkpoints` (cần file thật đúng path). Không khai báo tay
`channels/deepest_block/feature_blocks/patch_size`: `resolve_g2_config` suy ra từ
registry hiện có, extractor kiểm tra lại kiến trúc và state_dict lúc load.
Nếu file ở chỗ khác, đổi mapping hoặc đặt `backbone.weights` làm override.

`adapter.r/d: null` tự tính `C*r_ratio`, `C*d_ratio`, làm tròn lên bội `round_to=8`.
Đặt số nguyên dương để cố định từng width khi muốn cùng ngân sách Adapter.

| Backbone | C | Block cuối | r/d tự tính (C/6, 2C/3) |
| --- | --- | --- | --- |
| `dinov3_vits16`, `dinov3_vits16plus` | 384 | 12 | 64/256 |
| `dinov3_vitb16` | 768 | 12 | 128/512 |
| `dinov3_vitl16` | 1024 | 24 | 176/688 |
| `dinov3_vith16plus` | 1280 | 32 | 216/856 |

`fusion.dim=64`, `decoder.hidden_channels=64`, output 512 và Context FOV 768 px
là cấu hình độc lập với C; không tăng theo backbone. `build_g2_model` chỉ dựng
E1/E2; cùng một config dùng được cho cả hai khi so sánh ablation.

```python
import torch, yaml
from src.models.msila import build_g2_model, resolve_g2_config

with open("configs/g2_experiments.yaml") as stream:
    cfg = yaml.safe_load(stream)
model = build_g2_model(cfg, experiment="E2")  # Chọn E1 nếu cần baseline.
resolved = resolve_g2_config(cfg)  # Lưu config đã suy ra cùng run/checkpoint.
logits = model(batch)  # Không sigmoid trước BCEWithLogits.
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=1e-3, weight_decay=1e-4)
```

## Boundary sáu feature → Fusion → Decoder

Với ViT-S/S+/B (depth 12), giữ thứ tự `MULTIVIEW_FEATURE_KEYS`:
`local_b4, local_b8, local_b12, context_b4, context_b8, context_b12`.
Không dùng insertion order của mapping để stack; đọc theo tuple trên.
Sáu tensor đã align/project vào hệ tọa độ Local, cùng shape/device/dtype và
hữu hạn `[B,C,H,W]`; ở boundary này `C=fusion_dim=64`, `H=W=32` cho tile 512.
`fusion_dim` độc lập với Adapter `d` và backbone C. Boundary sáu feature hiện có
dùng các key b4/b8/b12; khi làm multiview cho L/H, TV2 cần mở rộng chọn block/key
trong pipeline. E1/E2 không dùng boundary này và đã suy ra block cuối tự động.

Mọi fusion candidate đưa **một tensor** `[B,C,H,W]` cho Decoder, giữ C/H/W;
`concat` phải có projection `6C→C` bên trong. `mean` trung bình đều;
`weighted_sum` dùng trọng số nguồn được học; `gated` dùng gate theo input;
`attention` có thể dùng `AttentionFusion` hiện có. Module attention cũ trả dict
`{feature, attention, logits}`: wrapper TV2 lấy `feature` để nối Decoder và giữ
diagnostics trong trace. Kiểm tra bằng `validate_multiview_features` và
`validate_g2_fused_feature`; không đổi API các module Day-2 đang tồn tại.
Decoder dùng `BasicDecoder(C,64)` với `output_size=(512,512)`.

## Protocol và kiểm tra

Giữ 8 category, checkpoint frozen, seed 2026/dev seed 17017, 20 epochs,
batch 4, AdamW lr 1e-3/weight decay 1e-4 và BCE+positive-mask Dice như G1.
Synthetic chỉ sinh từ TRAIN/good (đổi theo epoch); DEV từ VALIDATION/good cố định,
không trùng source. Chọn checkpoint bằng synthetic DEV AU-PRO@0.05, tie lấy epoch
sớm nhất; TEST_PUBLIC chỉ dùng chấm cuối. Mỗi category/experiment có model và
optimizer mới; giữ split, synthetic protocol, seed/order và update budget chung.
Tỷ lệ `r/C=1/6,d/C=2/3` là default khai báo, chưa phải kết quả chọn thực nghiệm.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -rs tests/test_g2_tv1_model.py tests/test_g2_tv1_contract.py
```

Fixture chỉ xác minh kiến trúc/API/autograd. Test integration tự tìm asset local
hoặc nhận `MVTEC_AD2_ROOT`, `DINOV3_REPO`, `DINOV3_WEIGHTS` (tùy chọn
`G2_CATEGORY`, `DINOV3_MODEL`), chạy một batch TRAIN/good qua loader/synthetic/loss
hiện có và một optimizer step. Thiếu asset: SKIP/**NOT RUN**; đường explicit sai:
FAIL. Không chạy full training trong D5-TV1.
