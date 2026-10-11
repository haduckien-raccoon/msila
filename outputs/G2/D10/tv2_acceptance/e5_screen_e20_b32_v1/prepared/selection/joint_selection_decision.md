# D10 joint selection decision

Status: **BLOCKED**; audited terminal cells 0/360; eligible complete candidates 0/45.

- tv1: PASS 0/180.
- tv2: PASS 0/180.

Rule: higher eight-category macro synthetic DEV AU-PRO@0.05, then fewer trainable params, smaller r, smaller d, then mean/weighted_sum/concat/gated/attention. Exact ties; no tolerance.

The merged OOM policy records FAIL without changing batch. It provides no terminal infeasibility exclusion, so FAIL/OOM cannot complete coverage or authorize a lock.

Scope: provisional_incomplete_search. Selection lock: NOT CREATED.

- Incomplete search: 0/360 terminal audited cells; FAIL/OOM/INFEASIBLE are not terminal under the predeclared TV1 resource policy

- Screening budget and one seed can rank candidates differently from full training.
- Synthetic DEV is reused for checkpoint and joint selection; selection bias remains.
- Multi-seed confirmation, if performed, must be reported separately without changing this rule.
- No TEST scores are used. GPU latency/VRAM require named hardware and measurement conditions; they are not tie-break variables.
