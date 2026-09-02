# Week 02 QA Log — MS-ILA

Generated: 2026-09-02T12:29:35

## Protocol

- DATA_ROOT: `/content/msila/src/data`
- Local tile: `512×512`
- Local overlap: `128`
- Local stride: `384`
- Context crop: `768×768`
- Context backbone input: `512×512`
- Coordinate convention: `(x0,y0,x1,y1)`, x1/y1 exclusive
- DINOv3 normalization mean: `(0.485, 0.456, 0.406)`
- DINOv3 normalization std: `(0.229, 0.224, 0.225)`
- DINOv3 model: `facebookresearch/dinov3:dinov3_vits16`
- Frozen backbone: yes

## Gate status

| Item | Status |
|---|---|
| High-resolution loader | PASS |
| 512 local / 768 context tiling | PASS |
| Hann stitching | PASS |
| Mask crop/stitch | PASS |
| SegF1 + AU-PRO evaluator | PASS |
| Frozen DINOv3 smoke | PASS |
| End-to-end visual QA | PASS |
| **G1** | **PASS** |

## Evidence files

- `outputs/week02/coverage_audit.csv`
- `outputs/week02/tile_coordinates.csv`
- `outputs/week02/hann_stitch_visual_qa.png`
- `outputs/week02/dinov3_smoke_feature_shapes.csv`
- `outputs/week02/end_to_end_qa.csv`
- `outputs/week02/qa_*.png`

## Notes

1. Feature-norm map trong end-to-end QA chỉ là projection giả lập để kiểm tra plumbing.
2. Chưa train adapter.
3. Không dùng QA-only SegF1/AU-PRO làm benchmark khoa học.
4. Threshold của benchmark sau này phải được khóa/calibrate trên validation, không tune trên private test.
