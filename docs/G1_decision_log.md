# G1 Decision Log — Data & Feature Pipeline

## Decision

**G1 status: PASS**

## Locked API

### Loader

Input:
- path tới ảnh MVTec AD 2.

Output:
- `image`: `float32 [3,H,W]`, native resolution, range `[0,1]`;
- `image_norm`: DINO-normalized `[3,H,W]`;
- `mask`: `uint8 [H,W]` hoặc `None`;
- metadata: `category, split, H, W, path, mask_path`.

### Coordinate convention

- `(x0,y0,x1,y1)`;
- `x1,y1` exclusive;
- local tile = `512×512`;
- overlap = `128`;
- stride = `384`;
- context = `768×768` cùng tâm local;
- context resize `768→512` trước backbone;
- border context được padding.

### Stitching

- anomaly/feature map: Hann weighted average;
- mask: binary OR/max;
- mask resize: nearest-neighbor only.

### Feature extraction

- backbone: `dinov3_vits16`;
- frozen: `requires_grad=False` cho toàn bộ backbone;
- selected block IDs: `[-4, -2, -1]`;
- feature extraction qua forward hook;
- downstream code không được phụ thuộc trực tiếp vào class token nếu chưa xác nhận token layout.

### Evaluator

- SegF1 threshold là explicit argument;
- threshold selection chỉ dùng validation;
- AU-PRO tích phân đến FPR `0.05`;
- báo cáo per-category và macro;
- reject NaN/Inf và shape mismatch.

## Gate criteria

- coverage = 100%;
- Hann reconstruction max error < `1e-5`;
- reconstructed mask IoU = `1.0`;
- loader/metric unit tests PASS;
- frozen DINOv3 feature finite;
- end-to-end QA finite và đúng image shape.

## Deferred to Week 03+

- adapter training;
- anomaly head chính thức;
- threshold calibration chính thức;
- benchmark SuperADD/PatchCore/DINOv3 control;
- AU-PRO/SegF1 khoa học trên benchmark.
