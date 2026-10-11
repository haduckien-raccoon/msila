# G2 D11 — solo Colab runbook

TV1 D11 inspected: `g2/member1@2605739571b55caab4418dc8ac3bb648243e728a`.
This checkout incorporates that commit without modifying its Model, runner or
training configuration. Use **one common committed revision** containing both
TV1 D11 and TV2 final reporting for both owners. Record its SHA; do not move a
branch halfway through a study. Merge/push these changes before running Colab.

## Before main training

Use `notebooks/G2_03_Training.ipynb` from the shared revision. It already supports
`TRAINING_STAGE='main'`, either owner, epoch/job Drive sync and resume. GPU is
whichever Colab GPU is available; report its actual name, not an assumed T4.
Copy all eight `.tar.gz` archives **from Drive into `/content` first**, then
extract locally with the notebook's existing archive preparation. Keep dataset,
pretrained DINOv3 checkpoint and exact official DINOv3 source checkout local.

Main requires the **existing** D10 `joint_selection_lock.json`, TV1 and TV2
`joint_search_180.csv`, sibling spec/protocol and their checkpoint/evaluation
trees. Copy whole evidence trees preserving relative directories. The runner
re-audits all 360 D10 cells. A code acceptance report, 45 CPU fixtures, or a D9
skeleton cannot replace a selection lock. If D10 has not run, complete D10
first using the unchanged integrated notebook/runner; D11 remains BLOCKED.

Set these Python variables once in Colab (example local paths; adjust Drive
paths to actual storage). Use the Colab Python executable with CUDA dependencies.

```python
from pathlib import Path
import subprocess, sys
REPO = Path('/content/msila')
DATA = Path('/content/mvtec_ad2')
DINO = REPO/'third_party/dinov3'
WEIGHTS = Path('/content/models/dinov3_vitb16.pth')
D10 = Path('/content/G2_D10')
LOCK = D10/'selection/joint_selection_lock.json'
TV1_D10 = D10/'tv1/joint_search_180.csv'
TV2_D10 = D10/'tv2/joint_search_180.csv'
MAIN = Path('/content/G2_D11')
DRIVE_MAIN = Path('/content/drive/MyDrive/msila/G2/D11')
COMMON = ['--config', 'configs/g2_main.yaml', '--stage', 'main',
          '--joint-lock', str(LOCK), '--tv1-manifest', str(TV1_D10),
          '--tv2-manifest', str(TV2_D10), '--data-root', str(DATA),
          '--repo-dir', str(DINO), '--weights', str(WEIGHTS),
          '--output-root', str(MAIN), '--drive-output', str(DRIVE_MAIN),
          '--device', 'cuda']
```

## Main runs — same runner, no independent training loop

```python
# Preflight: use both owners; stop if either returns BLOCKED.
for owner in ('tv1', 'tv2'):
    subprocess.run([sys.executable, 'scripts/run_g2.py', *COMMON,
                    '--owner', owner, '--preflight'], cwd=REPO, check=True)
# Technical smoke is optional and isolated. It never counts as a main run.
subprocess.run([sys.executable, 'scripts/run_g2.py', *COMMON,
                '--owner', 'tv2', '--categories', 'sheet_metal',
                '--experiments', 'E4', '--smoke'], cwd=REPO, check=True)
# Run/resume both owners sequentially in the same study. No 3,240-job grid.
for owner in ('tv1', 'tv2'):
    subprocess.run([sys.executable, 'scripts/run_g2.py', *COMMON,
                    '--owner', owner, '--resume'], cwd=REPO, check=True)
```

TV2 owns `sheet_metal,vial,wallplugs,walnuts`: **36 logical cells**, normally
**32 physical main runs** with strict E4/E5-mean equivalence. Both owners total
72 logical main cells / 64 physical runs. Add eight E0 banks for **80 final
logical evaluation cells**. E1 may reuse G1 through TV1's existing
`g1_checkpoints` setting only after its full protocol/budget audit. Do not edit
the runner to force reuse. All five E5 methods run; the selected D10 Fusion
does not remove the other four main controls.

`--max-jobs N` limits work per session while retaining all coverage rows.
Rerun `--resume` to continue: valid completed runs skip; interrupted runs restore
optimizer/RNG/cursor. Incomplete/OOM is not PASS. Keep the locked batch, seed,
training budget and synthetic protocol; do not silently lower them on T4.

## E0 — all eight categories, same DEV

```python
E0 = Path('/content/G2_E0')
from scripts.eval_g2 import run_logged
e0_code = run_logged([sys.executable, 'scripts/run_g2_e0.py',
    '--config', 'configs/g2_e0.yaml', '--categories', 'all', '--device', 'cuda',
    '--data-root', str(DATA), '--repo-dir', str(DINO), '--weights', str(WEIGHTS),
    '--output-root', str(E0), '--save-maps', '--resume'], cwd=REPO,
    output_root=E0, drive_output=DRIVE_MAIN/'E0', stage='e0_full')
assert e0_code == 0, 'E0 incomplete: inspect actual per-category status'
```

The final audit checks E0's actual TRAIN/DEV source hashes, pretrained weights,
official source hashes, seed and synthetic protocol against D10. A historical
E0 with a different split/seed is BLOCKED. Old exports without signed
`native_maps.json` require a fresh E0 output directory; do not overwrite old
metrics or manufacture hashes. E0 keeps its declared bounded distance transform
and `.5` Dice threshold; scores are **not calibrated sigmoid probabilities**.

## Final evaluation and Drive export

Find the real owner manifests under
`MAIN/main_v1/<main_protocol_sha256>/{tv1,tv2}/main_training_36.csv`.
Do not select a `prepared/` or `smoke/` manifest. Both owners must have the same
main protocol SHA, sources and code. When importing results from Drive, copy
the whole owner directory, not CSV files alone.

```python
from scripts.eval_g2 import run_logged, sync_tree
studies = [p for p in (MAIN/'main_v1').iterdir()
           if (p/'tv1/main_training_36.csv').is_file()
           and (p/'tv2/main_training_36.csv').is_file()]
assert len(studies) == 1, 'Choose one matching main protocol explicitly'
STUDY = studies[0]
FINAL = Path('/content/G2_FINAL')
DRIVE_FINAL = DRIVE_MAIN/'final'
EVAL = ['--final', '--joint-lock', str(LOCK),
        '--tv1-manifest', str(TV1_D10), '--tv2-manifest', str(TV2_D10),
        '--main-tv1-manifest', str(STUDY/'tv1/main_training_36.csv'),
        '--main-tv2-manifest', str(STUDY/'tv2/main_training_36.csv'),
        '--e0-root', str(E0), '--experiments', 'E1','E2','E3','E4','E5',
        '--categories', 'all', '--data-root', str(DATA), '--repo-dir', str(DINO),
        '--weights', str(WEIGHTS), '--output-root', str(FINAL), '--device', 'cuda']
code = run_logged([sys.executable, 'scripts/eval_g2.py', *EVAL, '--resume', '--benchmark'],
    cwd=REPO, output_root=FINAL, drive_output=DRIVE_FINAL, stage='final')
sync_tree(FINAL, DRIVE_FINAL)
print('Final exit:', code)  # 2 = BLOCKED/incomplete; never treat it as PASS.
```

Without `--benchmark`, the final audit reuses signed native DEV exports and
computes metrics on CPU; **no second DINOv3 inference pass is needed**.
`--resume` skips exact metric recomputation only after evidence/code/hash audit.
`--report-only --resume` audits and summarizes without GPU inference. Restrict
`--categories`/`--experiments` to repair specific evaluations; all other cells
stay NOT RUN in the complete 80-cell table.

Outputs include `metrics_g2.csv` (80 cells), `metrics_g2_macro.csv`,
`fusion_comparison.csv` (40 cells), `ablation_g2.csv` (8 contrasts × 9 rows,
including the zero Average−Average control), coverage/summary, signed
per-category metrics, conditional macro figure, examples, log, code hashes,
Git SHA and the read-only selected D10 Fusion. Macro is null unless 8/8 pass.
Never copy fixture diagnostics into these study directories.
`params` counts trainable head parameters (E0 = 0); the common frozen DINOv3
is excluded. `G2 EXP PASS` additionally requires actual CUDA benchmark evidence
for all logical cells. Otherwise metric coverage and benchmark coverage are
reported separately, and final acceptance remains BLOCKED.

## GPU throughput and measurement limits

TV1 D11 already caches frozen DINOv3 between jobs. TV2 now also caches a single
frozen extractor between compatible evaluation/benchmark jobs, loads each
checkpoint's head strictly, and releases the previous feature mode before
allocating the next. E4/E5-mean share the same native evidence and benchmark
only when exact equivalence is proved. Native scores, sigmoid, Hann, precision,
DEV coverage and AU-PRO rules remain unchanged.

The D10 evidence re-audit is CPU/file I/O: copying full trees to local Colab
storage prevents hundreds of checkpoint/hash reads through Drive's mount.
Run several jobs per session rather than restarting the 360-cell preflight for
every single job. Use the runner's epoch/job sync, not per-step Drive writes.

Remaining TV1-owned candidates for profiling: repeated gradient/finite checks
can synchronize GPU; native prediction copies tile scores to CPU before Hann;
paired CPU preparation/resize can delay GPU batches. These are **code inspection
findings**, not measured bottleneck percentages. No Model/training-loop change,
AMP/TF32 change, cache of training features, DEV truncation or approximate
AU-PRO is introduced here. Exact AU-PRO/P99 sorting on CPU is intentional.
E0's declared bank device is CPU and chunking is bounded; assess a GPU bank
separately before freezing an E0 configuration, not through a hidden override.

Latency/VRAM use real synchronized CUDA native inference and save GPU name,
native H/W, batch size, warmup, repeats, individual durations and allocated vs
reserved bytes. No local CPU timing is called a T4 measurement. GPU speedup
and Colab Run All are **NOT RUN** during code authoring.

## Interpretation

Main validation is conditional on D10's selected `(r*,d*,fusion*)`; never
publish a new lock after observing main metrics. The main report keeps selected
Fusion D10 and all-five-Fusion main controls separate. Synthetic DEV was also
used for checkpoint/D10 selection: it is not evidence of held-out real-defect
generalization. No TEST assets are used by these commands. Run an independently
approved held-out protocol only after selection is locked and without tuning.
