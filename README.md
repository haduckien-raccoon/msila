# MS-ILA

**Multi-Scale Illumination-Invariant Lightweight Adapters for Industrial Anomaly Segmentation on MVTec AD 2**

## Environment

- Python: 3.12
- Framework: PyTorch
- Dataset: MVTec AD 2
- Backbone: DINOv3
- Task: Industrial Anomaly Detection and Segmentation

## Project Structure

```text
msila/
├── benchmark/
├── configs/
├── metadata/
├── docs/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── runs/
├── outputs/
└── README.md
````

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python --version
# Python 3.12.x

pip install --upgrade pip
pip install -r requirements.txt
```

## Research Pipeline

```text
MVTec AD 2
    ↓
High-resolution tiling
    ↓
DINOv3 frozen backbone
    ↓
Multi-scale residual adapters
    ↓
Local + Context feature fusion
    ↓
Lightweight segmentation decoder
    ↓
Hann stitching
    ↓
Anomaly map
```

## Main Evaluation

* AU-PRO@0.05
* SegF1
* Boundary-F1
* Illumination robustness
* Latency
* Peak VRAM
* Trainable parameters

## Reproducibility

Main random seeds:

```text
17
42
2026
```

Each experiment should record:

* dataset version
* split version
* config
* random seed
* environment
* checkpoint
* evaluation results

## Status

Current phase: **Week 1 — Dataset audit and research protocol setup**
