# MS-ILA — Day 03 Architecture QA Report & Week 04 Handoff

## 1. Mục tiêu Day-03

Day-03 xác nhận toàn bộ pipeline training từ cached DINOv3 feature đến loss/backward/checkpoint hoạt động đúng trước khi chuyển sang Week-04.

```text
Frozen DINOv3
    ↓
Cached L4/L8/L12 + C4/C8/C12
    ↓
Adapter
    ↓
Context → Local Alignment
    ↓
Projection
    ↓
Attention Fusion
    ↓
Decoder
    ↓
BCEWithLogits + Dice
    ↓
Backward + Optimizer
    ↓
Checkpoint / Resume
```

> Overfit-16 chỉ là architecture/training sanity check, không phải kết quả generalization hay AU-PRO0.05 benchmark.

## 2. Kết quả cuối cùng

| Hạng mục | Kết quả |
|---|---:|
| Cached samples | 16 |
| DINO feature channels | 384 |
| 6 cached feature sources | PASS |
| Cache schema | PASS |
| Alignment metadata | PASS |
| NaN / Inf | 0 / 0 |
| Architecture QA | **PASS** |

### Overfit-16 trajectory

| Epoch | Loss | Loss ratio | Pixel Dice | IoU |
|---:|---:|---:|---:|---:|
| Initial | 1.027628 | 1.0000 | 0.0000 | 0.0000 |
| 1 | 0.697685 | 0.6789 | 0.0000 | 0.0000 |
| 5 | 0.666992 | 0.6491 | 0.0000 | 0.0000 |
| 10 | 0.504765 | 0.4912 | 0.3091 | 0.1828 |
| 15 | 0.244136 | 0.2376 | 0.8136 | 0.6858 |
| 20 | 0.143923 | 0.1401 | 0.8534 | 0.7443 |

Early-stop tại epoch 20.

```text
Initial loss : 1.027628
Final loss   : 0.143923
Loss ratio   : 0.1401
Loss giảm    : PASS
Convergence  : PASS
```

## 3. Gradient QA

| Module | Kết quả |
|---|---:|
| Adapter | PASS |
| Projection | PASS |
| Attention Fusion | PASS |
| Decoder | PASS |
| DINOv3 | NONE — PASS |

DINOv3 không nằm trong optimizer và không có gradient.

## 4. Checkpoint / Resume QA

| Hạng mục | Kết quả |
|---|---:|
| Save | PASS |
| Load | PASS |
| Model restore | PASS |
| Optimizer restore | PASS |
| Resume next step | PASS |
| Visualization | PASS |

## 5. Baseline khóa cho Week-04

### Data contract

```python
{
    "local_b4": Tensor,
    "local_b8": Tensor,
    "local_b12": Tensor,
    "context_b4": Tensor,
    "context_b8": Tensor,
    "context_b12": Tensor,
    "mask": Tensor[B, 1, 512, 512],
    "meta": [...]
}
```

Feature dimension: `384`.

DINO blocks: `[4, 8, 12]`.

### Locked preprocessing

```text
Local source FOV   = 512 × 512
Context source FOV = 768 × 768
Context input      = resize → 512 × 512
Local input        = 512 × 512
```

### Locked training pipeline

```text
cached DINO features
    ↓
Residual Adapter
    ↓
Context-to-Local Alignment
    ↓
SixFeatureProjection
    ↓
AttentionFusion
    ↓
BasicDecoder
    ↓
raw anomaly logits
    ↓
BCEWithLogits + Dice
```

### Trainable

```text
Adapter
Projection
Attention Fusion
Decoder
```

### Frozen

```text
DINOv3
```

## 6. Day-03 baseline configuration

```yaml
seed: 42
batch_size: 4

backbone:
  model: dinov3_vits16
  frozen: true
  blocks: [4, 8, 12]
  feature_dim: 384

adapter:
  reduction: 4
  kernel_size: 3
  gamma_init: 0.0

fusion:
  fusion_dim: 64

projection:
  share_local_context: true

decoder:
  output_size: [512, 512]

loss:
  bce_weight: 1.0
  dice_weight: 1.0

optimizer:
  type: AdamW
  lr: 1.0e-3
  weight_decay: 0.0

overfit16:
  max_epochs: 150
  min_epochs: 20
  early_stop_loss_ratio: 0.20
```

## 7. Artifacts hiện có

### Feature cache

```text
/content/msila_runtime/day03_fabric_seed42_cache_v1/
```

Persistent archive:

```text
/content/drive/MyDrive/[Q3-4] 2026/[S7] Computer Vision/CV-Nhóm 9/
└── msila_day03/
    └── archives/
        └── day03_fabric_seed42_cache_v1.tar.gz
```

### Reports

```text
.../msila_day03/reports/
├── day03_fabric_seed42_final_qa.json
└── day03_fabric_seed42_final_qa.md
```

### Training run archive

```text
.../msila_day03/runs/
└── day03_fabric_seed42_final_qa.tar.gz
```

## 8. Quy tắc cho Week-04

Không cần chạy lại DINO hoặc build lại cache nếu không thay đổi:

```text
DINO checkpoint
DINO model
DINO blocks
Local/Context crop policy
input resolution
normalization
preprocessing contract
```

Nếu chỉ thay:

```text
Adapter
Projection
Fusion
Decoder
Loss
Optimizer
LR schedule
```

thì reuse cache hiện tại.

## 9. Những thành phần có thể custom trong Week-04

- Adapter: reduction ratio, kernel size, dilation, gamma, shared/layer-specific.
- Multi-layer fusion: trọng số L4/L8/L12, learnable gates, dynamic attention.
- Local–Context fusion: gating, cross-attention, confidence weighting.
- Projection: 384→64/128/256, shared vs separate Local/Context.
- Decoder: depth, skip/refinement, multi-scale, boundary-aware refinement.
- Training strategy: LR, weight decay, module-specific LR, warmup, cosine decay, hard-example sampling.
- Synthetic anomaly: tiny defects, boundary probability, shape, texture/intensity.

## 10. Ablation rule

Không thay nhiều biến cùng lúc.

```text
B0 = Day-03 locked baseline
B1 = B0 + fusion_dim 128
B2 = B0 + Local/Context gate
B3 = B0 + multi-layer dynamic weighting
B4 = B0 + decoder refinement
```

Giữ cố định:

```text
dataset split
seed
DINO checkpoint
cache
training schedule
evaluation code
```

Metric:

```text
Primary   : AU-PRO0.05
Secondary : SegF1 / AUROC / AUPR
Analysis  : tiny defect / boundary defect / illumination
```

## 11. Week-04 entry gate

```text
[PASS] DINO frozen
[PASS] Feature cache
[PASS] CachedFeatureDataset
[PASS] Adapter gradient
[PASS] Projection gradient
[PASS] Fusion gradient
[PASS] Decoder gradient
[PASS] Overfit-16
[PASS] Checkpoint
[PASS] Resume
[PASS] Visualization
[PASS] NaN/Inf = 0
```

# Day-03 Architecture QA: PASS

Week-04 có thể bắt đầu từ baseline này và chuyển từ “code có chạy không” sang controlled ablation để kiểm tra module nào thực sự cải thiện anomaly localization, đặc biệt AU-PRO0.05.
