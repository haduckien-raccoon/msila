# MS-ILA E8 — Adapter Selection Report

- **Analysis:** PASS
- **Rule:** `synthetic_e8_v1`
- **Rule SHA-256:** `258bd78cd5b5277f181a92f9c179c5e5f101afa3f4548720919feb9976775fa9`
- **Reference:** `candidate_a`
- **Split:** `dev_synthetic`
- **Selection status:** **SELECTED**
- **Selected candidate:** `candidate_b`

## Candidate summary

| Candidate | Eligible | Macro AU-PRO0.05 mean±std | Δ vs ref mean±std | Improved cats | Trainable params | Median latency mean±std (ms) | Peak VRAM mean±std (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_a | YES | 0.705500 ± 0.000707 | 0.000000 ± 0.000000 | 0 | 400000 | 4.400000 ± 0.000000 | 512.000000 ± 0.000000 |
| candidate_b | YES | 0.725500 ± 0.000707 | 0.020000 ± 0.000000 | 2 | 450000 | 4.900000 ± 0.000000 | 517.000000 ± 0.000000 |

## Per-category AU-PRO delta vs reference

| Candidate | fabric | vial |
|---|---:|---:|
| candidate_a | 0.000000 | 0.000000 |
| candidate_b | 0.020000 | 0.020000 |

## Selection trace

- `primary` — macro_aupro_mean (max, tolerance=0.001): candidate_b

## Interpretation

The report applies only the pre-locked E8 rule. It does not perform the E9 fairness audit; selection is conditional on E9.
