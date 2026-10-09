# Full-scale MS-ILA training

The full-scale entry point extends the existing Day-05 cache builder, trainer,
Mean Fusion, adapters, projection, decoder and native Hann inference. Historical
Day-04/05 artifacts remain readable. No historical checkpoint or result is edited.

The shipped grid declares **3,240 training jobs**: five backbones, nine adapter
pairs, R0/R1/R2, seeds 17/42/2026 and all eight MVTec AD 2 categories. Each job
runs **150 epochs with batch size 64**, Local 512, Context 768 resized to network
input 512, overlap 128. `--train-all` executes this complete declared grid.
It does not select a smaller grid when memory or session time is insufficient.
GPU OOM stops the run, records the failed configuration and preserves all settings.

These changes have software tests, including actual CPU optimization and exact
resume on synthetic feature fixtures. They do not establish real DINOv3 accuracy,
GPU memory requirements, throughput or a winning backbone. Real MVTec data and
the five official pretrained checkpoints were unavailable in the development
workspace. Historical DEV GT statistics also remain unmeasured; see
`reports/full_scale_initial_audit.json`.

## Data and scientific protocol

Image/GT pairing uses category, split, defect type and nested image identity.
Supported explicit GT layouts include `category/test_public/ground_truth/bad/`
and `category/ground_truth/test_public/bad/`. Unscoped GT is never guessed across
splits. `good` supplies a native zero mask. Missing/ambiguous abnormal GT raises
an error for pixel evaluation. Images retain native H×W; mask resizing uses nearest
neighbor when required.

Before evaluating any new model, audit available old DEV GT:

```bash
python -m scripts.audit_synthetic_gt \
  --inputs OLD_DEV_RECORDS.json --mask-root OLD_MASK_ROOT \
  --output OLD_DEV_GT_AUDIT.json
python -m scripts.audit_dataset \
  --data-root /content/drive/MyDrive/msila/data/MVTec_AD_2 \
  --output /content/drive/MyDrive/msila/dataset_audit.json
```

`configs/full_scale_synthetic.yaml` locks a **synthetic proxy**, without using
real TEST defects to calibrate it. Foreground area bins, in native pixels, are
4–32, 33–128 and 129–2048. The first two define DEV tiny; DEV mixed covers all
three. The rationale relates the first two bins to a 16×16 Local DINO patch and
uses the third as a larger-size control. These bins do not claim to represent
the real TEST distribution or a universal definition of tiny.

Only official TRAIN normal images are used. A deterministic hash with fixed
data seed 42 assigns approximately 80% of sources to TRAIN core, 10% to DEV tiny
and 10% to DEV mixed. The three sets are source-disjoint; an empty partition
raises an error. All source images are retained. TRAIN supplies an unchanged
normal and one stratified defect per source. Each DEV source supplies a normal
and the complete declared `type × size bin × placement` set: pinhole, thin
scratch, texture, contamination; interior and image boundary.

Defects are generated on the native image before tiling. All Local tiles are
cached, including negative tiles. Pixels outside GT are bitwise unchanged.
`assets/dataset/gt_statistics.json` records native component area, width, height,
ratio, centroid, edge distance, Local tile geometry and Context mask geometry
before/after nearest resize. Image-edge defects and defect-contour localization
are distinct diagnostics. Empty metric groups return `null` with a reason.

Checkpoint selection maximizes the predeclared
`0.5*DEV_tiny_AU-PRO005 + 0.5*DEV_mixed_AU-PRO005` **on cached native Local tiles**,
keeping the earliest epoch on a tie. Every epoch runs; there is no early stopping.
Each best checkpoint is then evaluated on **full native DEV images with Hann
stitching**. Families are ranked per category by the mean of the native DEV score
over all declared seeds; ties use the declared lexicographic rule. TEST_PUBLIC is
reachable only after all jobs in the execution scope have produced native DEV
results and a valid `selection_lock.json`. TEST never selects checkpoints or
hyperparameters. Single mode compares only its explicitly executed configuration;
it cannot establish a winner over the unexecuted YAML grid.

AU-PRO uses the existing exact empirical evaluator. The full-scale backend
stores sorted background chunks on disk and counts them at foreground score
events, preserving tied-score handling and integration. Tests compare it with
the reference evaluator on continuous and tied scores. It avoids holding every
native background pixel in RAM, while retaining every image and region.

## Backbone and adapter contracts

The registry is checked against the loaded model's embedding dimension, depth
and patch size. Official architecture source:
[DINOv3 hub backbones](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py).

| Backbone | C | Depth | Default physical blocks |
| --- | ---: | ---: | --- |
| `dinov3_vits16` | 384 | 12 | 4, 8, 12 |
| `dinov3_vits16plus` | 384 | 12 | 4, 8, 12 |
| `dinov3_vitb16` | 768 | 12 | 4, 8, 12 |
| `dinov3_vitl16` | 1024 | 24 | 8, 16, 24 |
| `dinov3_vith16plus` | 1280 | 32 | 11, 21, 32 |

Three explicit block overrides are supported and validated. Existing cache keys
`b4/b8/b12` remain shallow/middle/deep **slots**; the signed `cache_slot_blocks`
maps them to actual physical blocks. For L, `local_b12` holds physical block 24.
This preserves existing selector, alignment and projection contracts.

The 3×3 adapter search scales Day-04 widths with C: r/C = 1/12, 1/6, 1/3;
d/C = 1/3, 2/3, 1. Widths round up to a multiple of 8. These are candidates, not
optimal widths. A backbone entry may override the adapter section with explicit
`pairs: [[r1, d1], [r2, d2]]`. Adapter internal d remains independent of fusion
dimension 64. Backbone parameters stay frozen.

Cache identity includes backbone ID, checkpoint SHA256, actual architecture,
physical blocks, slot mapping, normalization, preprocessing, source hashes,
synthetic protocol, producer code and DINOv3 source files/revision. Full-scale
shards and training checkpoints have checksums. A different backbone, weights,
blocks, adapter, protocol, code or cached tensor identity causes a mismatch error.
New DEV caches are never relabeled legacy caches.
Local weights are read directly from the specified checkpoint after constructing
the official architecture, avoiding Torch Hub's basename cache aliasing.

Full-scale mode uses a fixed-geometry bilinear gather/index-select backend for
R2 alignment and decoder resize. It preserves the bilinear kernel, half-pixel
coordinates, zero/border padding and trainable parameter layout. Forward values
and input gradients are tested against the original CPU operations. This enables
strict deterministic CUDA backward without relaxing the declared protocol:
[PyTorch 2.6 deterministic algorithms](https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html)
documents that CUDA bilinear interpolation and grid sampling backward raise in
strict mode, while gather/index-select support deterministic input gradients.
CUDA execution remains an external integration gate until an actual GPU is
available. Historical mode retains its original sampling backend.

## Colab Pro

Mount Drive and make the **updated repository** available at `/content/msila`.
If cloning the public repository, apply the supplied local patch first; the
uncommitted changes in this workspace are not automatically on GitHub:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
git clone https://github.com/haduckien-raccoon/msila /content/msila
cd /content/msila
git checkout 6ad23e66f10af08a70e6f5731de336db7de159c8
git apply /content/drive/MyDrive/msila/full_scale_changes.patch
pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3 /content/dinov3
```

Record the DINOv3 commit (`git -C /content/dinov3 rev-parse HEAD`) in Drive and
check out that same revision after reconnecting. The cache also records all
DINOv3 Python source hashes. Use the same patched project code and Python
package versions when resuming.

Place the official checkpoints at the exact paths in
`configs/full_scale_grid.yaml`, or edit those paths **before starting**. Preserve
official checkpoint basenames: the upstream ViT-L loader inspects the hash in
the filename. The dataset root must directly contain the eight category folders.
Use a GPU that supports the declared bfloat16 protocol; an unsupported GPU raises
an error. Select Colab's high-RAM runtime and provide persistent storage for the
complete caches, native images, maps and checkpoints. The code measures resource
use; no successful Colab memory fit is claimed without running the real assets.

From the updated repository root, run the entire grid:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.run_day05_pipeline \
  --train-all --grid-config configs/full_scale_grid.yaml \
  --data-root /content/drive/MyDrive/msila/data/MVTec_AD_2 \
  --dinov3-repo /content/dinov3 \
  --output-root /content/drive/MyDrive/msila/full_scale_v2 \
  --device cuda:0 --resume
```

`--train-all` defaults to `--stage all`. It builds all backbone caches, executes
every training job, evaluates every best checkpoint on native DEV and writes
the ranking/selection lock. Re-run this **same command and same configuration**
after a session interruption. Valid complete jobs are verified and skipped;
partial jobs restore model, optimizer, RNG, sampler order and progress. Committed
update prefixes are not replayed. Updates after the last checkpoint may be
replayed; checkpoint interval is explicitly 100 steps. CSV logs are rolled back
to the committed step before resuming. Pending native maps also resume with
checksum/provenance validation. No invalid checkpoint is silently overwritten.

After DEV selection, evaluate only the selected family for each category and
each of its declared seeds:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.run_day05_pipeline \
  --train-all --grid-config configs/full_scale_grid.yaml --stage test-public \
  --data-root /content/drive/MyDrive/msila/data/MVTec_AD_2 \
  --dinov3-repo /content/dinov3 \
  --output-root /content/drive/MyDrive/msila/full_scale_v2 \
  --device cuda:0 --resume
```

For an explicitly chosen configuration, use a separate output root:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.run_day05_pipeline \
  --grid-config configs/full_scale_grid.yaml \
  --backbone dinov3_vitl16 --adapter-r 88 --adapter-d 1024 \
  --representation R2 --seed 42 --category fabric \
  --dino-checkpoint /content/drive/MyDrive/msila/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --data-root /content/drive/MyDrive/msila/data/MVTec_AD_2 \
  --dinov3-repo /content/dinov3 \
  --output-root /content/drive/MyDrive/msila/single_L_r88_d1024_R2_seed42 \
  --device cuda:0 --resume
```

To build a backbone cache independently (replace backbone/weights together):

```bash
python -m scripts.build_day05_cache \
  --backbone dinov3_vith16plus \
  --dino-checkpoint /content/drive/MyDrive/msila/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --dinov3-repo /content/dinov3 \
  --data-root /content/drive/MyDrive/msila/data/MVTec_AD_2 \
  --output-root /content/drive/MyDrive/msila/Hplus_cache_v2 \
  --categories can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts \
  --synthetic-protocol configs/full_scale_synthetic.yaml --seed 42 --device cuda:0
```

The same single/grid runner supports `--stage inference` to resume native DEV
evaluation of completed checkpoints. `--stage audit`/`--dry-run` checks paths,
source partitions and declared configurations without claiming VRAM validation.
`--stage preflight` executes real model forward/backward/optimizer checks after
building the full caches. These stages do not replace full training.

## Artifacts and verification

`grid_manifest.json` records every declared job, completion, interruption/OOM and
native DEV score. Runs live under
`runs/BACKBONE/adapter_rR_dD/seed_SEED/CATEGORY/REPRESENTATION/` and contain resolved
config, model/optimizer/RNG checkpoints, checksums, per-update and per-epoch CSV
logs, trainable parameter counts, preflight evidence and native DEV maps/metrics.
Runtime fields distinguish preparation, training, shared cache extraction and
native evaluation. VRAM is measured per stage and a combined peak is recorded.
`ranking.csv`, `ranking.json` and `selection_lock.json` appear only after complete
DEV evaluation. Package/GPU environment snapshots are stored under `environments/`.

Run software verification:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

Real-data/GPU integration gates are skipped unless their external assets are
provided. CPU fixture training, mock backbone tests and reference metric tests
are software evidence only. `reports/full_scale_validation.json` records the
local test counts, versions, limitations and precise changed-file inventory.
`reports/full_scale_synthetic_fixture_coverage.json` audits all 40 declared DEV
type/size/placement strata on a constant native image fixture, including signal,
component geometry, Local/Context transforms and Hann reconstruction. Actual
dataset coverage is recorded separately in `assets/dataset/gt_statistics.json`.
The fixture report does not measure real defect distributions or model accuracy.
Bitwise resume has been verified on CPU fixtures. CUDA reproducibility and
cross-session behavior still require GPU verification; identical results across
different GPU models or software versions are not claimed.
