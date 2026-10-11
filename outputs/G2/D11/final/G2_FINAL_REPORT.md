# G2 final report

Status: **BLOCKED**. Actual results: **0/80**; Fusion: **0/40**.

D10 selection (unchanged): `null`.

| Experiment | Valid categories | Macro synthetic DEV AU-PRO@0.05 |
|---|---:|---:|
| E0 | 0/8 | — |
| E1 | 0/8 | — |
| E2 | 0/8 | — |
| E3 | 0/8 | — |
| E4 | 0/8 | — |
| E5/mean | 0/8 | — |
| E5/weighted_sum | 0/8 | — |
| E5/concat | 0/8 | — |
| E5/gated | 0/8 | — |
| E5/attention | 0/8 | — |

E2 uses one Local source; E3 Average uses three Local sources; E4/E5 Average uses six aligned Local/Context sources.
E4 reuse for E5 Average requires an identical physical checkpoint/config and explicit equivalent_experiments.

GPU benchmark numbers are published only with the measured GPU, native H/W, batch size and synchronized timing conditions.

## Limits

- Synthetic DEV also used for checkpoint and D10 selection; not held-out generalization evidence.
- Short screening budget and one seed may rank candidates differently from full training.
- Macro means require eight valid categories; missing results remain null.

Held-out TEST/real defects: **NOT RUN**. No tuning on TEST.

See CSV rows and per-category attempt_status.json for missing/failed evidence. All artifacts are synthetic DEV only.
