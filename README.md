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

## Day 05 — TV1/TV2 integration (DINOv3 ViT-S/16)

See [the Day 05 Colab runbook](docs/DAY05_PIPELINE_RUNBOOK.md) for audit, preflight, sequential R0/R1/R2 training, native tiled inference, evaluation and resume. The runner preserves the full Day 04 notebook protocol; missing real assets or diagnostic locks stop their dependent gates.

## Full-scale training

See [the full-scale Colab runbook](docs/FULL_SCALE_TRAINING.md) for all five DINOv3
backbones, configurable adapter r/d, source-disjoint tiny/mixed DEV, native
evaluation and checkpoint resume. `configs/full_scale_grid.yaml` declares the
complete 3,240-job grid with 150 epochs per job. `--train-all` executes every
configuration and selects models using DEV; TEST_PUBLIC runs only after the DEV
selection lock. OOM preserves the configuration and stops with a recorded error.
