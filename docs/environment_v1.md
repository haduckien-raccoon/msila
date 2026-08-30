# MS-ILA — Environment Specification v1

**Environment Version:** 1.0  
**Python:** 3.12

---

## 1. Software Environment

Python:
3.12.x

Operating environment:
Google Colab Pro / local development

Deep learning framework:
PyTorch

CUDA:
<record automatically>

cuDNN:
<record automatically>

DINOv3 implementation:
<record package/repository and commit>

---

## 2. Dependency Management

Các package của project được khai báo tại:

requirements.txt

hoặc:

pyproject.toml

Phiên bản package phải được cố định trước các experiment chính.

---

## 3. Hardware Logging

Mỗi experiment ghi:

- GPU model;
- GPU memory;
- CUDA version;
- CPU;
- system RAM;
- precision mode.

Không giả định Colab luôn cung cấp cùng GPU.

---

## 4. Numerical Precision

Precision được ghi trực tiếp trong config của từng experiment.

Ví dụ:

precision: bf16

hoặc

precision: fp16

---

## 5. Reproducibility

Các nguồn random được seed gồm:

- Python;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA.

Primary seed: 42.

Replication seeds: 17, 2026.

---

## 6. Environment Snapshot

Mỗi run lưu environment snapshot tại:

runs/<run_id>/environment.json