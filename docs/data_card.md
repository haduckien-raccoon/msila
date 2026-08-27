# MVTec AD 2 — Data Card v1

Dataset: MVTec Anomaly Detection 2
Abbreviation: MVTec AD 2
Task: Industrial Anomaly Detection and Segmentation
Protocol: Single-class anomaly segmentation
Dataset version: MVTec AD 2
Date downloaded: 27/08/2026
Data root: /content/data/mvtec_ad2/
Data card version: v1.0

## Purpose

MVTec AD 2 được sử dụng để đánh giá bài toán phát hiện và
phân đoạn bất thường công nghiệp trong điều kiện ảnh độ phân
giải cao và thay đổi chiếu sáng.

Trong nghiên cứu MS-ILA, dataset được sử dụng để đánh giá:

1. khả năng định vị dị vật nhỏ;
2. khả năng khai thác đặc trưng đa tỷ lệ;
3. độ bền trước thay đổi illumination;
4. hiệu quả của adapter nhẹ trên DINOv3.

## Categories

1. Can
2. Fabric
3. Fruit Jelly
4. Rice
5. Sheet Metal
6. Vial
7. Wall Plugs
8. Walnuts

| ID | Category    | Channel   | Ghi chú   |
| -: | ----------- | --------- | --------- |
|  1 | Can         | RGB       |           |
|  2 | Fabric      | RGB       |           |
|  3 | Fruit Jelly | RGB       |           |
|  4 | Rice        | RGB       |           |
|  5 | Sheet Metal | Grayscale | 1 channel |
|  6 | Vial        | Grayscale | 1 channel |
|  7 | Wall Plugs  | Grayscale | 1 channel |
|  8 | Walnuts     | RGB       |           |

| Split        | Số ảnh | Label anomaly | Pixel mask | Vai trò                 |
| ------------ | -----: | ------------- | ---------- | ----------------------- |
| TRAIN        |  2,528 | Normal only   | —          | Training                |
| VALIDATION   |    302 | Normal only   | —          | Calibration             |
| TESTpub      |  1,084 | Có abnormal   | Public GT  | Public evaluation       |
| TESTpriv     |  2,045 | Có abnormal   | Hidden     | Official server         |
| TESTpriv,mix |  2,045 | Có abnormal   | Hidden     | Illumination robustness |

## Research split

Official TRAIN được chia deterministic thành:

- TRAIN-core: 80%
- DEV-synthetic: 20%

Splitting method:
SHA-256 hash of relative image path.

Random seed:
Không sử dụng random split trực tiếp.

Split version:
split_v1

Split manifest:
configs/splits/split_v1.csv

Split checksum:
configs/splits/split_v1.sha256

## Image properties

Resolution policy:
Images have heterogeneous spatial resolutions and aspect ratios.

Preprocessing policy:
Original aspect ratio is preserved.

Image interpolation:
bicubic

Mask interpolation:
nearest-neighbor

## Annotation

TRAIN:
- normal images only
- no real anomaly used for model training

VALIDATION:
- normal images only

TESTpub:
- anomaly labels available
- pixel-level ground-truth masks available

TESTpriv:
- ground truth hidden

TESTpriv,mix:
- ground truth hidden
- evaluated using official benchmark server

## License

License:
CC BY-NC-SA 4.0

Usage:
Research / non-commercial use according to dataset license.

Redistribution:
Dataset files are not included in the project repository.

Source:
Official MVTec AD 2 distribution.

## Dataset integrity

Checksum algorithm:
SHA-256

Manifest:
metadata/dataset_sha256.csv

Fields:
- relative_path
- file_size
- sha256
- split
- category

Expected image count:
8,004

Integrity status:
PASS / FAIL

## Processing policy

Original image:
Preserve original aspect ratio.

Training/inference:
Overlapping tiled processing.

Default local tile:
512 × 512

Default context crop:
768 × 768 → resize to 512 × 512

Overlap:
128 px

RGB images:
loaded as 3-channel RGB.

Grayscale images:
loaded as 1-channel;
replicated to 3 channels only before DINOv3 input.

Image interpolation:
bicubic.

Mask interpolation:
nearest-neighbor.

Final prediction:
stitched back to original image resolution.

## Experimental data roles

TRAIN-core:
model optimization

DEV-synthetic:
architecture and hyperparameter selection

VALIDATION:
normal-only calibration

TESTpub:
final public evaluation

TESTpriv:
official private evaluation

TESTpriv,mix:
official illumination-shift evaluation

All experiment runs must record:
- dataset version
- split version
- split hash
- config hash
- seed