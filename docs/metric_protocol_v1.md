# MS-ILA — Evaluation Metric Protocol v1

**Version:** 1.0  
**Dataset:** MVTec AD 2

---

## 1. Primary Scientific Metric

### AU-PRO_0.05

AU-PRO_0.05 là metric chính cho đánh giá pixel-level anomaly
localization trong nghiên cứu.

Metric được tính từ anomaly map liên tục, trước khi áp dụng threshold
nhị phân cố định.

Kết quả được báo cáo:

- theo từng category;
- macro average trên tám categories.

---

## 2. Segmentation F1

SegF1 được sử dụng để đánh giá anomaly mask sau thresholding.

SegF1 là metric phụ thuộc threshold và được báo cáo tách biệt với
AU-PRO_0.05.

Không sử dụng SegF1 để thay thế trực tiếp AU-PRO_0.05 trong phép
so sánh benchmark MVTec AD 2.

---

## 3. Boundary-F1

Boundary-F1 được sử dụng chủ yếu cho RQ4 nhằm đánh giá khả năng
định vị biên của dị vật.

Metric này có vai trò secondary/diagnostic.

---

## 4. Illumination Robustness

Định nghĩa illumination performance gap:

Gap_illum = Score_private - Score_private_mixed

Báo cáo đồng thời:

- Score_private;
- Score_private_mixed;
- Gap_illum.

Việc giảm gap chỉ được diễn giải cùng với absolute performance trên
hai tập.

---

## 5. Computational Metrics

Mỗi mô hình phải báo cáo:

- number of trainable parameters;
- inference latency;
- images per second;
- peak GPU memory;
- input resolution/tile configuration.

Latency phải được đo sau warm-up trên cùng môi trường benchmark.

---

## 6. Reporting Format

Bảng chính:

| Method | Can | Fabric | Fruit Jelly | Rice | Sheet Metal | Vial | Wall Plugs | Walnuts | Macro |
|---|---|---|---|---|---|---|---|---|---|

Các run nhiều seed:

mean ± standard deviation.

Metric phải được ghi kèm protocol và dataset split.