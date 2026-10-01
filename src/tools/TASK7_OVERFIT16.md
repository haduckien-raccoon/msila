# MS-ILA — Day 3 / Task 7: Overfit-16 Dataset

## 1. Phạm vi

Task này **chỉ xây dựng tập Overfit-16 cố định** để kiểm tra kiến trúc ở bước sau có khả năng học thuộc một bài toán rất nhỏ hay không.

Không thuộc Task 7:

- Loss.
- Optimizer.
- Trainer.
- Visualization khi training.
- Checkpoint / Resume.
- Integration training QA.
- Day-03 report.

Do đó Task 7 **không train model** và **không tính AU-PRO0.05/SegF1 để báo cáo nghiên cứu**.

---

## 2. Mục tiêu khoa học đúng của Overfit-16

Overfit-16 là một **controlled debugging dataset**.

Câu hỏi duy nhất nó chuẩn bị để kiểm tra ở Task 10 là:

> Với 16 mẫu cố định và supervision chính xác, architecture hiện tại có đủ khả năng tối ưu để học thuộc tập nhỏ hay không?

Nếu model không học nổi 16 mẫu này, cần nghi ngờ computational graph, loss, optimizer, alignment hoặc capacity trước khi chuyển sang full training.

Nếu model học được 16 mẫu này, **không được kết luận model generalize tốt**.

---

## 3. Thành phần tập dữ liệu đã khóa

```text
Overfit-16
├── 8 normal samples
│   ├── RGB = ảnh normal gốc sau decode RGB
│   └── mask = toàn 0
│
└── 8 synthetic-anomaly samples
    ├── RGB = Task-6 SyntheticAnomalyGenerator(normal RGB)
    └── mask = binary mask chính xác từ Task 6
```

Tổng:

```text
16 samples
8 normal
8 synthetic anomaly
```

Mỗi sample dùng **một source image khác nhau**. Script yêu cầu ít nhất 16 ảnh normal khác nhau trong `normal_root`.

---

## 4. Vị trí đúng trong pipeline

Synthetic anomaly phải xuất hiện **trước DINOv3 / feature cache**:

```text
NORMAL SOURCE IMAGE
        ↓
Task 7 chọn source image cố định
        ↓
Task 6 SyntheticAnomalyGenerator
        ↓
Overfit-16 RGB + exact binary mask
        ↓
[sang task khác sau này]
Local / Context
        ↓
DINO preprocessing
        ↓
Frozen DINOv3
        ↓
Feature cache / Adapter / Alignment / Projection / Fusion / Decoder
```

Không được tạo mask anomaly sau khi đã cache feature từ ảnh normal, vì khi đó input feature không chứa anomaly nhưng target lại chứa anomaly — supervision sai về mặt logic.

---

## 5. File được thêm

```text
src/tools/build_overfit16.py
tests/test_build_overfit16.py
TASK7_OVERFIT16.md
```

Task 7 sử dụng lại prerequisite từ Task 6:

```text
src/data/synthetic_anomaly.py
```

Không sửa các module model của Day 1/Day 2.

---

## 6. Cấu trúc output

Ví dụ:

```text
outputs/day03/overfit16/
├── manifest.json
├── images/
│   ├── ov16_00_normal.png
│   ├── ...
│   ├── ov16_07_normal.png
│   ├── ov16_08_anomaly.png
│   ├── ...
│   └── ov16_15_anomaly.png
└── masks/
    ├── ov16_00_normal.png
    ├── ...
    └── ov16_15_anomaly.png
```

Ảnh output được chuẩn hóa thành PNG RGB 8-bit; mask là PNG grayscale chỉ có `0` và `255`.

---

## 7. Reproducibility contract

Với cùng:

1. normal source pool;
2. nội dung các source file;
3. `base_seed`;
4. `selection_seed`;
5. Task-6 generator + config;

thì phải sinh lại cùng:

- 16 source images được chọn;
- normal/anomaly assignment;
- synthetic anomaly;
- binary masks;
- metadata;
- pixel hashes.

Mặc định:

```text
base_seed = 2026
selection_seed = base_seed
sample_seed = base_seed + sample_index
```

Không lưu timestamp vào manifest vì timestamp không liên quan đến khoa học và sẽ làm manifest khác nhau giữa các lần build.

---

## 8. `manifest.json`

Mỗi sample lưu tối thiểu:

```json
{
  "index": 8,
  "sample_id": "ov16_08_anomaly",
  "split": "overfit16",
  "purpose": "architecture_qa_only",
  "is_anomaly": true,
  "label": 1,
  "seed": 2034,
  "source_relpath": "normal_017.png",
  "source_sha256": "...",
  "image_path": "images/ov16_08_anomaly.png",
  "mask_path": "masks/ov16_08_anomaly.png",
  "image_hw": [512, 512],
  "image_pixel_sha256": "...",
  "mask_pixel_sha256": "...",
  "synthetic": {
    "anomaly_type": "...",
    "shape_type": "...",
    "area_ratio": 0.0,
    "bbox_xyxy": []
  }
}
```

Dataset-level manifest cũng khóa:

```text
purpose = architecture_qa_only
benchmark_use_allowed = false
realism_claim_allowed = false
```

Điều này tránh sử dụng nhầm Overfit-16 như benchmark khoa học.

---

## 9. Cách chạy

Đặt file Task-6 đúng vị trí:

```text
src/data/synthetic_anomaly.py
```

Sau đó từ root repo:

```bash
python -m src.tools.build_overfit16 \
  --normal-root /PATH/TO/NORMAL_ONLY_IMAGES \
  --output-root outputs/day03/overfit16 \
  --base-seed 2026 \
  --category fabric
```

`normal-root` phải là tập **normal-only**. Script không tự suy đoán nhãn normal từ filename.

Nếu muốn build lại có chủ đích:

```bash
python -m src.tools.build_overfit16 \
  --normal-root /PATH/TO/NORMAL_ONLY_IMAGES \
  --output-root outputs/day03/overfit16 \
  --base-seed 2026 \
  --category fabric \
  --overwrite
```

### Verify dataset đã build

```bash
python -m src.tools.build_overfit16 \
  --normal-root /PATH/TO/NORMAL_ONLY_IMAGES \
  --output-root outputs/day03/overfit16 \
  --verify-only
```

Expected:

```json
{
  "status": "PASS",
  "schema_name": "msila_overfit16",
  "schema_version": 1,
  "n_samples": 16,
  "n_normal": 8,
  "n_anomaly": 8
}
```

---

## 10. Unit tests

Chạy:

```bash
pytest -q tests/test_build_overfit16.py
```

Tests kiểm tra:

1. đúng 16 mẫu;
2. đúng `8 normal + 8 anomaly`;
3. 16 source image khác nhau;
4. 16 sample ID khác nhau;
5. cùng source pool + seed → cùng manifest và pixel output;
6. normal sample: RGB giữ nguyên, mask toàn 0;
7. anomaly sample: mask non-empty;
8. ngoài anomaly mask, RGB phải giữ nguyên chính xác;
9. trong anomaly mask phải có thay đổi RGB sau khi lưu PNG;
10. mask lưu trên đĩa chỉ có `{0,255}`;
11. dataset có dưới 16 source images → fail;
12. output đã tồn tại → fail nếu không truyền `--overwrite`;
13. `verify_overfit16()` kiểm lại pixel hashes và composition.

---

## 11. PASS gate của Task 7

Task 7 được xem là **PASS** khi tất cả điều kiện sau đúng:

```text
[PASS] Exactly 16 samples
[PASS] Exactly 8 normal
[PASS] Exactly 8 synthetic anomaly
[PASS] 16 distinct source images
[PASS] Fixed seeds reproduce the exact dataset
[PASS] Normal images are unchanged
[PASS] Normal masks are all zero
[PASS] Synthetic masks are binary and non-empty
[PASS] Image/mask H,W match
[PASS] Pixels outside anomaly masks remain unchanged
[PASS] Pixels inside anomaly masks contain a persisted RGB change
[PASS] manifest.json records provenance + hashes + Task-6 metadata
[PASS] verify_overfit16() returns PASS
```

Không yêu cầu ở Task 7:

```text
loss decreases
backward works
optimizer updates parameters
Dice/IoU improves
checkpoint/resume
AU-PRO0.05
```

Các mục trên thuộc các task sau.

---

## 12. Phân tích khoa học được phép từ Task 7

Có thể mô tả:

> Overfit-16 là một tập kiểm thử kiến trúc có kiểm soát gồm 8 ảnh normal và 8 ảnh được gắn synthetic anomaly với pixel-level ground truth chính xác. Source selection và anomaly generation được khóa bằng seed nhằm bảo đảm reproducibility.

Có thể thống kê từ manifest:

- anomaly type distribution;
- shape distribution;
- anomaly area ratio;
- normalized centroid/location;
- boundary fraction.

Không được kết luận:

> Synthetic anomalies are realistic industrial defects.

Không được dùng kết quả trên Overfit-16 để tuyên bố:

> MS-ILA generalizes tốt / vượt baseline / đạt AU-PRO cao.

Đó không phải mục đích của tập này.

---

## 13. Output bắt buộc sau Task 7

Sau khi build xong, chỉ cần lưu:

```text
outputs/day03/overfit16/manifest.json
outputs/day03/overfit16/images/*.png
outputs/day03/overfit16/masks/*.png
```

Sau đó mới chuyển sang **Task 8 — Loss**. Task 8 không được triển khai trong gói này.
