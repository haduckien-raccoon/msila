# G1 — E1 solo runbook

G1 implements frozen DINOv3 → one deepest dense feature → `BasicDecoder` →
BCEWithLogits + positive-mask Dice. `E1` constructs only an extractor and a
decoder. Adapter r/d, fusion, projection, context and illumination consistency
are outside this run. Existing Adapter modules and grids remain unchanged.

## Audit and changes

| Existing module | Audit | G1 action |
| --- | --- | --- |
| `backbone_registry.py` | Official names, C/depth/patch size and C-based Adapter ratios already present | Reuse unchanged; `dinov3_vitsh16` is rejected |
| `dinov3_extractor.py` | Frozen/eval behavior and exact local strict state loading already present; requires three blocks | Add `feature_mode=deepest`, request only final index, contextual checkpoint mismatch error |
| `basic_decoder.py` | Accepts C dynamically and upsamples raw logits | Reuse unchanged with 64 hidden channels |
| `msila.py` | Existing baselines construct Adapters/fusion | Add small `E1` class with two modules only |
| `loader.py` | Native RGB, official normalization and split scanning present | Add good-only native pairs and local tile datasets; retain coordinates |
| `synthetic_anomaly.py` | Native tiny generator already meets exact support/bin contracts | Reuse unchanged; legacy area-ratio generator is bypassed |
| `tiling.py` | Coordinate coverage and positive Hann boundary weights already present | Fix constant mask padding incorrectly falling back to replicate on small images |
| `anomaly_loss.py` | BCEWithLogits and Dice; no mask resize | Reuse unchanged |
| `aupro.py`, evaluator | Existing AU-PRO@0.05, native geometry QA and DEV accumulator | Reuse native QA and exact disk-backed metric accumulator |
| Training/utilities | Cache/Adapter orchestration cannot directly run online E1; generic train step, optimizer, checkpoint, resume and curves work | `g1_e1.py` orchestrates these existing components; no second train-step implementation |

Official architecture/checkpoint names are taken from
[DINOv3 hub backbones](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py).
RGB float [0,1] uses the
[official LVD1689M mean/std](https://github.com/facebookresearch/dinov3#image-transforms),
then native 512×512 crops with overlap 128, without global image resizing.

## Environment and assets

Run commands at the repo root. Install the existing requirements in the local
virtual environment. On GPU machines retain an appropriate CUDA PyTorch build.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.train.g1_e1 --help
```

Supply an official DINOv3 checkout containing `hubconf.py`, the official local
pretrained checkpoint, and MVTec AD2 with `<root>/rice/TRAIN/good` and
`<root>/rice/VALIDATION/good` (lowercase directories also work). Get official
weights through [Meta's download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
The CLI never substitutes randomly initialized weights or TEST imagery.

The defaults in `configs/g1_e1.yaml` expect `data/mvtec_ad2`,
`third_party/dinov3`, and official files under `models/`. Change those YAML paths
or use `--data-root`, `--repo-dir`, `--weights`. Relative paths resolve from the
repository root. The selected checkpoint is loaded from that exact local file
with `strict=True`. Its SHA256 and TRAIN/DEV source hashes are saved with the
checkpoint; incompatible keys/shapes fail before training. Shape compatibility
does not independently certify who produced an arbitrary user-supplied file:
use the official downloaded asset.

## D1/D2 independent checks

```bash
.venv/bin/python -m pytest -q tests/test_g1_e1.py tests/test_dinov3_extractor.py tests/test_tiling.py tests/test_anomaly_loss.py
```

On this workstation ROS exports pytest plugins that require unrelated ROS
packages. Prefix test commands with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to isolate
project tests. `MPLCONFIGDIR=/tmp/msila-g1-mpl` also gives matplotlib a writable
cache under the workspace sandbox.

Unit fixtures are explicitly API fixtures, not pretrained DINOv3. They check
384/768/1280 channels, patch16 32×32 feature grids for 512px tiles, deepest block
selection, strict state mismatch, frozen eval behavior, decoder gradients and
updates, exact native mask support, reproducibility, zero normal masks, and
asymmetric edge/corner Hann reconstruction. The real integration test is skipped
unless `MVTEC_AD2_ROOT`, `DINOV3_REPO`, and `DINOV3_WEIGHTS` are supplied.

## Smoke, Overfit-16 and training

All modes use `Overfit16Trainer.train_step` with the same E1 model and loss.
Smoke limits work; it still requires real data and pretrained weights.

```bash
.venv/bin/python -m src.train.g1_e1 --config configs/g1_e1.yaml --category rice --smoke --device cuda --output-root outputs/G1/smoke/E1
.venv/bin/python -m src.train.g1_e1 --config configs/g1_e1.yaml --category rice --overfit16 --device cuda --output-root outputs/G1/overfit16/E1
.venv/bin/python -m src.train.g1_e1 --config configs/g1_e1.yaml --category rice --device cuda
```

Overfit-16 uses eight distinct TRAIN/good sources, with one fixed normal tile and
one fixed positive synthetic tile per source. Its seeds, masks and tile choices
stay fixed across epochs. It reports the same-set initial/final loss and whether
loss decreased. This is an architecture learning check, not a research result.

Regular training uses each selected TRAIN/good source, one normal variant plus
`train_variants_per_image` synthetic variants, and all local tiles. TRAIN
perturbations change deterministically by epoch; fixed DEV perturbations never
backpropagate. TRAIN/DEV pools must have distinct image content, checked by
SHA256 before source limits are applied. Training and synthetic DEV never load
TEST images or real TEST masks. Optional final TEST_PUBLIC evaluation runs after
the checkpoint has been selected on synthetic DEV. No fallback split is created
when VALIDATION/good is absent.

Adjust `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `seed`, and
`dev_seed` in YAML. `max_steps` caps cumulative updates, including resumed steps.
`max_minutes` limits training wall time checked between steps. Each stopping
point still evaluates DEV and writes artifacts; that evaluation can extend
beyond the training time cap. Set source limits, DEV variant count and
`evaluation.tile_batch_size` to control evaluation work and memory. GPU OOM fails
clearly; it does not silently alter the experiment.

Training logs loss, BCE, Dice, learning rate, step time and checkpoints. Full
backbone state fingerprints before/after the run confirm unchanged weights;
each step checks frozen/eval state, absent backbone gradients and finite decoder
gradients. Optimizer groups contain only the decoder.

## Resume and evaluate

```bash
.venv/bin/python -m src.train.g1_e1 --config configs/g1_e1.yaml --category rice --device cuda --resume outputs/G1/E1/rice/last.pt
.venv/bin/python -m src.train.g1_e1 --config configs/g1_e1.yaml --category rice --device cuda --evaluate outputs/G1/E1/rice/best.pt
```

Resume in the same output directory with the same mode and scientific config.
For smoke/Overfit-16, repeat its mode flag and original `--output-root`.
Only epochs/step/time budgets may be extended; data hashes, generator settings,
backbone checkpoint SHA256, decoder topology and learning parameters stay locked.
`last.pt` includes optimizer/RNG state, history and the next batch position for
partial-epoch continuation. Existing checkpoint SHA256 sidecars are required.
The checkpoints contain the decoder, not a redundant copy of the frozen DINOv3;
inference must also have the original official backbone asset.

`--evaluate` writes `evaluation_metrics.json`, with native examples and map QA
under `evaluation/` so training artifacts continue to describe `best.pt`.
Training's `metrics.json` and final examples describe `best.pt`, chosen by the
highest synthetic DEV AU-PRO@0.05; earliest epoch wins an exact tie. An existing
run cannot be overwritten by a new training invocation; choose a new output root.

## Colab: eight categories and final TEST_PUBLIC

Open `notebooks/G1_E1_End_to_End_Colab.ipynb` in Colab. Cell 5 installs
`safetensors>=0.8` from the embedded requirements and verifies the installed
version in the training venv before proceeding. Cell 2 defaults to:

```python
TRAIN_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
RUN_TEST_PUBLIC = True
```

Use `TRAIN_CATEGORIES = ['rice']` for one category. This is a sequential set of
eight independent E1 models on one GPU: one model per category. Each training
stage launches a new Python process with a fresh E1 model and optimizer, or
resumes only that category's checkpoint. The frozen pretrained DINOv3 file can
initialize every model; trained decoder state stays specific to its category.
Each category has its own smoke,
Overfit-16, train checkpoint, QA, synthetic DEV and final public results. The
epoch/step/minute budgets apply to each category. Cell 3 owns the output layout
and `DRIVE_SYNC_EVERY_STEPS=20`; changing this to 10 changes Drive backup cadence,
while local loss logging still records every optimizer step. Local checkpoints
remain epoch/budget checkpoints. Re-running the same category list reuses
completed stages and resumes interrupted stages with `last.pt` automatically.
When moving from an earlier embedded-code snapshot, choose a new `RUN_ID`;
the code provenance is intentionally checked rather than silently overwritten.

All selected archives are copied to `/content` first. The training preparation
extracts only TRAIN/good and VALIDATION/good. After training and synthetic DEV,
cell 18 extracts TEST_PUBLIC images and masks from those local archives, then
loads each category's selected `best.pt`. No TEST_PRIVATE files are extracted.
If archives use a generic filename, set `ARCHIVE_PATTERNS=['*.tar.gz']`.

The public evaluator can also run from the repository root:

```bash
.venv/bin/python -m src.eval.g1_test_public --config configs/g1_e1.yaml --checkpoint outputs/G1/E1/rice/best.pt --output-dir outputs/G1/E1/rice/test_public --device cuda
```

The config must match the training checkpoint's scientific settings and exact
TRAIN/DEV/pretrained provenance. Public evaluation uses all public good/bad
images, strict native image/GT geometry, existing Hann inference, map QA and
disk-backed AU-PRO@0.05. Missing, ambiguous, empty abnormal masks and shape
mismatches fail explicitly. Images and GT are never resized by this evaluator.
TEST is used only for final scoring, not model selection or synthetic tuning.

Public output: `train/test_public/{metrics.json,qa_report.json,source_manifest.json,test_public.log,examples/}`.
The example limit controls files saved, not the metric population. Eight-category
summaries live at `_batch/<RUN_ID>/category_summary.csv` and `.json`, with
synthetic DEV and real public scores in separate columns. Public inference and
Colab GPU execution remain NOT RUN on the CPU-only development machine; unit
fixtures check the new evaluator's geometry, mask errors and checkpoint loading.

## Synthetic protocol

`configs/full_scale_synthetic.yaml` is the only source of generator parameters.
Its full resolved content is embedded in `resolved_config.yaml` and checkpoints.
All areas below count native foreground pixels, not bounding boxes.

| Parameter | Meaning / current value |
| --- | --- |
| `size_bins` | `sub_patch` 4–32, `small` 33–128, `mixed_control` 129–2048 pixels; ordered, positive, disjoint |
| `defect_types` | `pinhole`, `thin_scratch`, `texture`, `contamination` |
| `placements` | `interior` and `image_boundary`; anchor actual foreground to the edge |
| `scratch_width_px` | Inclusive native width range 1–4; actual support has exact chosen area |
| `contrast_range` | Uniform intensity perturbation magnitude 0.04–0.12 on [0,1] RGB |
| `min_mean_abs_change` | Minimum mean foreground RGB change 0.015; weak saturated pixels move toward the intensity interior; no background change |
| `boundary_band_px` | 2 native pixels; boundary-region grouping and interior placement margin |
| `train_core_bins` | All three bins, cycled with types and placements across sources/epochs |
| `dev_tiny_bins` | `sub_patch`, `small`; subset includes normal-only DEV images |
| `dev_mixed_bins` | All three bins; defines full synthetic DEV population |

`prediction_threshold=0.5` and `contour_tolerance_px=2` are existing optional
contour diagnostics; AU-PRO uses continuous scores rather than this threshold.
The generator rejects defects that cannot fit an image at the declared size;
it does not shrink masks or substitute the legacy `min_area_ratio=0.002` rule.
Do not tune this proxy using TEST defects or TEST masks.

The CPU illustration gallery at `reports/g1_synthetic_examples/` uses a clearly
labelled procedural good-image fixture. It illustrates sub-patch, small,
thin-scratch and low-contrast examples, with native and zoomed
original/synthetic/exact-mask/GT-overlay views. These are generator QA, not MVTec
results, and contain no predicted benchmark scores.

## Evaluation and outputs

Sigmoid converts tile logits to probabilities; high score means anomaly. No
per-image min/max scaling is applied. Full 512px scores use existing positive
Hann weights and original tile coordinates, then crop to exact native H×W.
Padding never enlarges the native prediction or mask. Each map passes existing
finite/probability/binary/native-coordinate QA before the existing exact
AU-PRO@0.05 implementation runs. No alternative AU-PRO formula is implemented.

`synthetic_dev_aupro_0_05` and `dev_mixed` use the same full population defined by
`dev_mixed_bins`; `dev_tiny` keeps tiny-bin anomalies and all normal-only images.
Grouping preserves original normal pixels in the FPR denominator. A group with
no valid regions or normal pixels reports null/undefined. Normal score P99 is
the exact 99th percentile of all pixels in normal-only native DEV images.
These metrics measure synthetic DEV only, never real anomaly performance.

Successful training writes `outputs/G1/E1/<category>/`:

- `best.pt`, `last.pt` and SHA256 sidecars;
- `resolved_config.yaml`, `metrics.json`, `train.log`, `qa_report.json`;
- `loss_curve.png`, `bce.png`, `dice_loss.png`;
- `examples/<sample_id>/original.png`, `synthetic.png`, `mask.png`,
  `predicted_map.npy`, `predicted_map.png`, `overlay.png`, `comparison.png`,
  `metadata.json`.

`predicted_map.npy` preserves native float probabilities; PNGs use a fixed
[0,1] color scale for inspection. If assets are absent, the CLI exits nonzero
and records **NOT RUN** in `metrics.json` with a reason, configuration and log.
It does not create a fake checkpoint, loss curve or DEV score.

## Change backbone

Change only `backbone.name` to one of these names after installing its mapped
checkpoint. Decoder C and final block are inferred from the loaded architecture
and checked against the existing registry.

| Name | C | Depth / selected block | Patch size | 512px dense shape |
| --- | --- | --- | --- | --- |
| `dinov3_vits16` | 384 | 12 | 16 | 384×32×32 |
| `dinov3_vits16plus` | 384 | 12 | 16 | 384×32×32 |
| `dinov3_vitb16` | 768 | 12 | 16 | 768×32×32 |
| `dinov3_vith16plus` | 1280 | 32 | 16 | 1280×32×32 |

G1 requires a real ViT-S/16 run when assets are available. Other backbones have
contract tests but receive no pretrained PASS claim without their actual weights.
`dinov3_vitsh16` is unsupported. G2 can keep using the existing C-based Adapter
ratio grid, without any r/d dependency in E1.

## Validation on 2026-10-10

Two focused suites passed: 55 tests for G1 and reused training/extractor/loss/
checkpoint utilities, and 74 regression tests for loader, masks, evaluator,
AU-PRO and existing model contracts. Four real-backbone integration tests were
skipped for missing assets. No full-scale grid job was executed.

| Day | Work | Changed source files | Checks actually run | Status |
| --- | --- | --- | --- | --- |
| D1 | Data + frozen DINO | `loader.py`, `dinov3_extractor.py`, `g1_e1.yaml`, `test_g1_e1.py` | Four backbone API fixture shapes, deepest-only API call, strict mismatch, frozen/eval, split isolation | PASS unit; NOT RUN pretrained |
| D2 | Synthetic + decoder | `loader.py`, `msila.py`, `test_g1_e1.py`; generator gallery artifacts | 24 bin/type/placement combinations, binary support, unchanged background, normal zeros, seed repeatability, decoder gradients/updates | PASS CPU fixture |
| D3 | Training + DEV | `g1_e1.py`, `g1_e1.yaml`, `test_g1_e1.py` | Fixed 16-tile fixture loss decrease, native metric path, logging/artifacts, checkpoint, exact partial-epoch resume | PASS unit; NOT RUN MVTec training |
| D4 | Inference + acceptance | `g1_e1.py`, `tiling.py`, this runbook, `test_g1_e1.py` | Asymmetric native coordinate reconstruction, orientation, tiny-image zero padding, finite maps, CLI missing-asset failure | PASS unit; NOT RUN pretrained inference |

The workstation has PyTorch 2.6.0+cpu and no CUDA device, MVTec AD2 dataset,
official DINOv3 checkout or pretrained checkpoint. The actual CUDA smoke command
was invoked and correctly exited nonzero with **NOT RUN**. Its
`outputs/G1/E1/rice/metrics.json`, resolved configuration and log record the
missing assets; no `best.pt`, research score or training curve was fabricated.

G1 experimental acceptance remains **NOT VERIFIED** until real pretrained
ViT-S/16, real-source Overfit-16 and synthetic DEV inference/training are run.
Contract fixtures cannot establish pretrained backbone correctness or real
anomaly performance. Machine-readable evidence is in
`reports/g1_acceptance.json`, `reports/g1_pytest.xml` and
`reports/g1_regression_pytest.xml`.
