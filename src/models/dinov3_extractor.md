# DINOv3 Frozen Feature Extractor — Colab Guide

## 1. Mục tiêu

Module:

```text
models/dinov3_extractor.py
```

Load **DINOv3 frozen** và trả đúng 3 dense feature maps:

```text
Block 4  → F4
Block 8  → F8
Block 12 → F12
```

Contract:

```text
Input : [B, 3, H, W]
Output: (F4, F8, F12)

DINOv3:
requires_grad = False
eval() = True
```

Với `dinov3_vits16`:

```text
Fi: [B, 384, H/16, W/16]
```

> Block 4/8/12 là **ba độ sâu semantic**, không phải ba spatial scales.

---

## 2. Cơ sở khoa học

Dùng API chính thức của DINOv3:

```python
backbone.get_intermediate_layers(...)
```

Quy đổi block 1-based sang index 0-based:

```text
Block 4  → index 3
Block 8  → index 7
Block 12 → index 11
```

Backbone được khóa:

```python
backbone.requires_grad_(False)
backbone.eval()
```

Forward dùng:

```python
with torch.no_grad():
```

vì DINOv3 không train, nhưng Adapter/Fusion/Decoder phía sau vẫn cần train.

**Nguồn học**

- Siméoni et al., *DINOv3*, 2025  
  https://arxiv.org/abs/2508.10104
- Official implementation  
  https://github.com/facebookresearch/dinov3

---

# 3. Trước khi extract lại: kiểm tra cache cũ

Chỉ tái sử dụng cache nếu **đồng thời** thỏa:

```text
□ đúng checkpoint DINOv3
□ đúng model: dinov3_vits16
□ đúng preprocessing
□ đúng Block 4 / 8 / 12
□ đúng norm=True
□ lưu full spatial feature [C,H/16,W/16]
□ không chỉ lưu sampled patch vectors
□ không phải anomaly score map
```

Nếu cache cũ chỉ chứa:

```text
[N_patches, C]
```

hoặc:

```text
[H, W] anomaly score map
```

thì **không thay thế được** F4/F8/F12 cho Adapter + Fusion + Decoder.

---

# 4. Chạy trên Google Colab

GPU **không bắt buộc cho smoke test**. CPU chạy được nhưng extract toàn dataset sẽ chậm.

## Bước 1 — Clone project

```bash
!git clone <PROJECT_REPO_URL>
%cd <PROJECT_REPO>
```

Kiểm tra:

```bash
!ls models
!ls tests
```

Tối thiểu:

```text
models/
└── dinov3_extractor.py

tests/
└── test_dinov3_extractor.py
```

---

## Bước 2 — Clone DINOv3 chính thức

```bash
!git clone https://github.com/facebookresearch/dinov3.git /content/dinov3
```

### Khóa phiên bản source

Để thí nghiệm reproducible, nên checkout đúng commit đã dùng:

```bash
%cd /content/dinov3
!git checkout <DINOV3_COMMIT>
!git rev-parse HEAD
```

Sau đó quay lại project:

```bash
%cd /content/<PROJECT_REPO>
```

> Không nên để source DINOv3 tự thay đổi giữa các lần chạy thí nghiệm.

---

## Bước 3 — Chuẩn bị checkpoint

Ví dụ:

```text
dinov3_vits16_pretrain_lvd1689m.pth
```

Có thể mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Ví dụ checkpoint:

```text
/content/drive/MyDrive/checkpoints/
dinov3_vits16_pretrain_lvd1689m.pth
```

---

## Bước 4 — Cài dependency

```bash
!pip install -q pytest
```

Kiểm tra môi trường:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
```

---

# 5. Preprocessing bắt buộc

DINOv3 pretrained dùng RGB image đã normalize.

```python
from torchvision.transforms import v2
import torch

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])
```

> Preprocessing phải được khóa giống nhau giữa train/validation/test và khi tạo cache.

---

# 6. Smoke test cấu trúc

Smoke test này chỉ kiểm tra:

```text
load model
shape
block selection
freeze
forward
```

Không dùng để đánh giá chất lượng feature.

```python
import torch

from models.dinov3_extractor import DINOv3FeatureExtractor

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

extractor = DINOv3FeatureExtractor(
    repo_dir="/content/dinov3",
    weights="/content/drive/MyDrive/checkpoints/"
            "dinov3_vits16_pretrain_lvd1689m.pth",
    model_name="dinov3_vits16",
).to(device)

x = torch.randn(
    1, 3, 224, 224,
    device=device,
)

f4, f8, f12 = extractor(x)

print("F4    :", f4.shape)
print("F8    :", f8.shape)
print("F12   :", f12.shape)
print("Frozen:", extractor.backbone_is_frozen())
```

Kỳ vọng:

```text
F4     : [1, 384, 14, 14]
F8     : [1, 384, 14, 14]
F12    : [1, 384, 14, 14]
Frozen : True
```

vì:

\[
224/16=14.
\]

---

# 7. Unit test

```bash
!pytest -q tests/test_dinov3_extractor.py
```

Task chỉ PASS khi:

```text
[PASS] load DINOv3
[PASS] đúng 3 feature
[PASS] đúng Block 4/8/12
[PASS] đúng shape
[PASS] requires_grad=False
[PASS] backbone vẫn eval khi parent model.train()
```

---

# 8. Cache F4/F8/F12 để không chạy DINO mỗi epoch

Không ghi cache trực tiếp trong:

```text
models/dinov3_extractor.py
```

Nên tạo riêng:

```text
scripts/cache_dinov3_features.py
```

Pipeline:

```text
Dataset
   ↓
DINOv3 Frozen
   ↓
F4 / F8 / F12
   ↓
Google Drive
════════════════════════
Training:
Drive
   ↓
Residual Adapter
   ↓
Fusion
   ↓
Decoder
```

Mỗi sample nên lưu full spatial tensor:

```text
sample_xxx/
├── f4.pt
├── f8.pt
└── f12.pt
```

Ví dụ với ảnh/tile `512×512`:

```text
f4  : [384, 32, 32]
f8  : [384, 32, 32]
f12 : [384, 32, 32]
```

Không flatten thành:

```text
[N, 384]
```

vì Adapter có `DWConv` cần giữ cấu trúc không gian.

---

# 9. Reproducibility metadata

Mỗi lần tạo cache nên lưu:

```text
model_name
checkpoint filename
checkpoint SHA256
DINOv3 git commit
blocks = [4,8,12]
indices = [3,7,11]
norm = True
patch_size = 16
image/tile size
preprocessing mean/std
dataset split
```

Ví dụ:

```json
{
  "model_name": "dinov3_vits16",
  "blocks": [4, 8, 12],
  "block_indices": [3, 7, 11],
  "norm": true,
  "patch_size": 16
}
```

---

# 10. Pipeline sau khi PASS

```text
RGB image
   ↓
DINOv3 ViT-S/16
(FROZEN)
   ├── Block 4  → F4
   ├── Block 8  → F8
   └── Block 12 → F12
          ↓
Residual Adapter
          ↓
Fusion
          ↓
Decoder
          ↓
Anomaly map
```

Không fine-tune DINOv3 trong baseline V1.

Không thêm projection/fusion vào extractor; giữ module này chỉ làm nhiệm vụ:

```text
image → frozen DINOv3 → F4/F8/F12
```

để ablation rõ ràng.

---

## Checklist bàn giao

```text
□ Clone project thành công
□ Clone official DINOv3
□ Khóa DINOv3 commit
□ Có checkpoint hợp lệ
□ Preprocessing đúng
□ Smoke test PASS
□ Trả đúng F4/F8/F12
□ DINOv3 frozen hoàn toàn
□ pytest PASS
□ Cache full spatial feature
□ Metadata reproducibility được lưu
```
