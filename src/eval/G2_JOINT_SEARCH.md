# D9-TV2 joint-search intake

One evaluator: `scripts/eval_g2.py` reuses the D6 DEV evaluator, metrics and
native Hann exports. E5 inference calls TV1's `predict_native_e4`, because the
paired/aligned views are identical. No training or selection runs in this harness.
TV1 integration inspected: `f0c51c883c5e9bdd3740cf5e323cb6a29460b10e`.
Run E5 intake on the integration checkout containing its factory/model;
an older checkout reports `BLOCKED` rather than substituting a model.

## D9 skeleton

```bash
python scripts/eval_g2.py --init-joint --device cpu --output-root outputs/G2/D9
```

Outputs: `joint_search_360.csv`, `joint_search_spec.json`,
`joint_search_manifest.schema.json`, `joint_search_macro.csv` (45 rows),
`joint_search_coverage.json` and `run_manifest.json`. Blank CSV metrics mean
JSON null, never zero. An existing manifest is preserved; initialization cannot
erase results. The CLI exiting successfully means skeleton creation succeeded,
not that experimental coverage passed.

The grid/seed come from TV1 config, using the existing `adapter_pairs()` registry
API. The current ViT-B16 grid is 3 r x 3 d x 5 Fusion x 8 category = 360 cells.
`budget` contains exactly `epochs`, `batch_size`, `max_steps`, `max_minutes`.
Until TV1 declares `screening_budget` in model/runner config, the budget is null
and every cell is `BLOCKED`. Main-training defaults do not fill this gap.
After the D10 protocol is declared, initialize a separate D10 output directory.

## D10 checkpoint handoff (future intake, no training here)

Populate each planned row with its real `checkpoint`, `config_hash` (the existing
`sha256_json(saved_config)`), and **training** `git_sha`. Paths in an input CSV
may be absolute or relative to that CSV. All 360 IDs must remain present and
unique, even when evaluating one category/candidate. Canonical IDs are
`mean|weighted_sum|concat|gated|attention`; labels are not silently remapped.

The saved config must declare `stage: E5`, `mode: screening`, category,
numeric Adapter r/d, Fusion method, resolved protocol/sources, expected steps,
pretrained checkpoint and DINO source checksums. Metadata must retain
`config_sha256`, `git_sha` (or `git_commit`), optimizer steps, DEV selection
evidence and training history. The standard SHA-verified checkpoint loader is
used. E5 `model_state` contains `adapters.*`, `projection.*`, `decoder.*`,
`fusion.*`; the frozen backbone is loaded from the verified pretrained file.
Keep `last.pt`, its SHA sidecar and the TV1 training `metrics.json` so the
completed budget and best/last checkpoint hashes can be audited.

```bash
python scripts/eval_g2.py --joint-manifest outputs/G2/D10/joint_search_360.csv \
  --categories rice --r 64 --d 256 --fusions mean gated --device cuda \
  --output-root outputs/G2/D10 --resume --benchmark
```

Optional relocation flags `--data-root`, `--repo-dir`, `--weights` retain D6's
strict source/content checks. `--smoke` stays separate and cannot count toward
scientific coverage. `--report-only` audits saved results without inference.
No selection lock is required or created by D9.

## Aggregation and GPU benchmark

```bash
python scripts/summarize_g2_joint.py --manifest outputs/G2/D10/joint_search_360.csv
# Display only a pair already selected externally; no ranking or new lock:
python scripts/summarize_g2_joint.py --manifest outputs/G2/D10/joint_search_360.csv \
  --selected-r 128 --selected-d 512
```

Aggregation audits checkpoints, saved config, map/mask hashes, native shape/range
QA, metric protocol and same-category DEV parity. Only eligible full CUDA
results enter CSV metrics. Each candidate needs 8/8 categories before its macro
is computed. The selected-pair display has five Fusion rows and eight category
columns with explicit statuses; partial rows have no macro.

`params` counts Adapter/projection/Decoder/Fusion parameters, excluding the
frozen backbone. `latency` is median synchronized wall-clock milliseconds of
native inference on the first fixed DEV image, including tiling, transfers and
Hann stitching. `peak_vram` is peak allocated CUDA bytes including the resident
model. GPU name, shape, tile batch size, warmup/repeat counts and raw timings
are stored in the signed per-category metrics JSON. CPU supplies no proxy
latency/VRAM and no T4 claim. Benchmark status may remain `NOT RUN` while genuine
GPU metric evaluation passes. No raw untrained AU-PRO is used for ranking.
