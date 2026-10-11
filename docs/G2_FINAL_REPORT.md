# G2 D11-TV2 — current acceptance

TV1 implementation inspected and incorporated from
`g2/member1@2605739571b55caab4418dc8ac3bb648243e728a`.
Scope: final evidence intake/reporting through `scripts/eval_g2.py`, E0 native
export provenance, targeted TV2 tests and this runbook. Model, training runner,
TV1 configuration and training notebook are unchanged relative to that commit.

**G2 experiment: BLOCKED / NOT RUN.** Local PyTorch is `2.6.0+cpu`; CUDA is
unavailable. No actual D10 selection lock, pretrained checkpoint, trained main
checkpoint or eight-category dataset is available in this checkout. No training,
selection or GPU benchmark was performed. No metric or anomaly map is fabricated.

**Code acceptance: PASS — 289 CPU tests passed, 0 failed, 1 skipped.** The
skipped real-pretrained E0 smoke requires the missing dataset, official DINOv3
checkout, pretrained weights and CUDA. Tests cover exact known AU-PRO/Dice/P99,
native H/W, corrupted evidence, incomplete macros, resume, strict main checkpoint
intake, E0 exports and unchanged TV1 D11 contracts.
Actual commands, source hashes and JUnit evidence are in
[`tv2_code_acceptance.json`](../outputs/G2/D11/tv2_code_acceptance.json) and
[`tv2_cpu_tests.xml`](../outputs/G2/D11/tv2_cpu_tests.xml).

The real CPU CLI attempts returned exit 2 as expected: TV2 preparation
`BLOCKED 0/36`; final audit `BLOCKED 0/80`, Fusion `0/40`. They produced only
status/coverage artifacts. No real checkpoint was present to evaluate.

| Required evidence | Actual valid results | Status |
|---|---:|---|
| TV2 E1–E4 + five E5 Fusion, four categories | 0/36 logical cells | BLOCKED |
| TV1 main, four categories | 0/36 logical cells found locally | NOT RUN |
| E0 baseline | 0/8 | NOT RUN |
| All final evaluations | 0/80 logical cells | BLOCKED |
| Five Fusion comparison | 0/40 | BLOCKED |
| Real CUDA latency/VRAM measurements | 0 | NOT RUN |
| Held-out TEST / real defects | 0 | NOT RUN |

TV2 categories: `sheet_metal,vial,wallplugs,walnuts`. E4/E5-Average can account
for two logical cells using one physical checkpoint only when config/protocol
and explicit equivalence agree. E3 has three Local sources; E4/E5 have six
aligned Local/Context sources. E2 has one Local source and no Fusion.

`outputs/G2/D11/final/G2_FINAL_REPORT.md` is generated from actual audited
evidence and supersedes this initial status after execution. Its macros require
eight valid categories; selected D10 Fusion remains unchanged. A missing lock
blocks all dependent runs without creating a replacement selection lock.

Use [G2_SOLO_RUNBOOK.md](G2_SOLO_RUNBOOK.md) with the integrated TV1 notebook.
Keep the full results/evidence tree on Drive, including config, SHA, seed,
native maps, benchmark conditions, metrics, figures and logs.

Limits: synthetic DEV only; DEV reused for selection/checkpointing; short
screening and one seed may rank candidates differently from main training.
GPU throughput improvement is not yet measured. CPU tests are code/fixture QA,
not scientific category coverage.
