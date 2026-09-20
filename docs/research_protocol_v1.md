# MS-ILA — Research Protocol v1

**Protocol Version:** 1.0  
**Date:** 06/09/2026  
**Dataset:** MVTec AD 2  
**Primary Task:** Pixel-level industrial anomaly segmentation

---

## 1. Research Objective

Mục tiêu của nghiên cứu là xác định liệu việc thích ứng có học trên
biểu diễn DINOv3 có thể cải thiện khả năng phân đoạn dị vật công nghiệp,
đặc biệt đối với dị vật kích thước nhỏ và thay đổi chiếu sáng.

Kiến trúc nghiên cứu tập trung vào ba cơ chế chính:

1. residual feature adaptation;
2. local/context multi-scale representation;
3. illumination-consistency learning.

Synthetic anomaly generation được sử dụng để cung cấp supervision
pixel-level mà không sử dụng dị vật thật trong tập huấn luyện.

---

## 2. Research Questions

### RQ1 — Residual Feature Adaptation

**Question**

Liệu residual adapter có cải thiện khả năng phân đoạn dị vật so với
việc sử dụng trực tiếp frozen DINOv3 features hay không?

**Primary comparison**

Frozen DINOv3 + decoder

versus

Frozen DINOv3 + residual adapters + decoder

**Primary metric**

Macro AU-PRO_0.05.

**Practical criterion**

Cải thiện ≥ 1.0 percentage point macro AU-PRO_0.05 và tốt hơn trên
ít nhất 6/8 categories.

---

### RQ2 — Multi-Scale Representation

**Question**

Việc kết hợp thông tin ở nhiều tầng và hai trường nhìn local/context
có cải thiện khả năng định vị dị vật kích thước nhỏ hay không?

**Ablation sequence**

C1: deep feature + local view

C2: multi-layer feature + local view

C3: multi-layer + local/context views

C4: multi-layer + local/context + learnable fusion

**Primary analysis**

- macro AU-PRO_0.05;
- performance trên nhóm tiny defects;
- performance theo category.

---

### RQ3 — Illumination Consistency

**Question**

Illumination-consistency learning có làm giảm feature drift do thay đổi
chiếu sáng và cải thiện anomaly segmentation dưới illumination shift
hay không?

**Primary comparison**

Illumination augmentation only

versus

Illumination augmentation + illumination-consistency loss

**Evaluation**

So sánh hiệu năng trên:

- TESTpriv;
- TESTpriv,mix;
- khoảng cách hiệu năng giữa hai tập.

---

### RQ4 — Synthetic Anomaly Strategy

**Question**

Tiny/boundary-aware synthetic anomaly generation có tạo supervision
tốt hơn random synthetic masks hay không?

**Primary comparison**

Random synthetic mask

versus

tiny/boundary-aware synthetic mask

**Metrics**

- SegF1;
- Boundary-F1;
- AU-PRO_0.05.

---

## 3. Dataset Protocol

Sử dụng trực tiếp các split chính thức của MVTec AD 2:

TRAIN -> model optimization.

VALIDATION -> model validation, model selection and calibration.

TESTpub -> public evaluation.

TESTpriv -> official private evaluation.

TESTpriv,mix -> illumination robustness evaluation.

Không tạo thêm random train/validation split.

---

## 4. Random Seeds

Primary development seed:

42

Replication seeds:

17
2026

Mọi experiment phải ghi seed trong configuration file và result manifest.

---

## 5. Experimental Unit

Kết quả được tổng hợp ở mức category và toàn bộ dataset.

Mỗi category được xem là một đơn vị benchmark riêng trong phép tổng hợp
macro.

Các kết quả nhiều seed phải báo cáo mean và standard deviation.

---

## 6. Experimental Traceability

Mỗi run phải lưu:

- experiment name;
- timestamp;
- git commit;
- dataset manifest version;
- configuration;
- random seed;
- trainable parameter count;
- checkpoint;
- validation metrics;
- runtime;
- peak VRAM.

---

## 7. Protocol Versioning

Mọi thay đổi đối với:

- dataset role;
- metric;
- architecture comparison;
- threshold calibration;
- random seed;
- evaluation procedure

phải tạo protocol version mới.

Current protocol:

v1.0