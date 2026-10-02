# E7 — Day-04 Candidate Aggregation

**File:** `scripts/eval_day04.py`  
**Task:** aggregate `candidate × category × seed` into CSV/JSON.  
**E7 does not select a winner.** Mean±std, deltas, and candidate selection belong to E8.

## 1. Scientific unit

The repository research protocol defines each MVTec AD 2 category as a
benchmark unit and requires multi-seed experiments to be reported across seeds.

For E7, the atomic row is therefore:

\[
u=(c,k,s)
\]

where:

- \(c\): candidate architecture;
- \(k\): category;
- \(s\): random seed.

If the experiment plan contains \(C\) candidates, \(K\) categories and \(S\)
seeds, the expected number of rows is

\[
N_{\text{expected}}=C\times K\times S.
\]

E7 is PASS only when:

\[
N_{\text{observed}}=N_{\text{expected}}
\]

with no duplicate or missing `(candidate, category, seed)` key.

This prevents a scientifically dangerous situation where one difficult run is
missing but a partial table is still treated as a complete comparison.

---

## 2. Why an explicit plan is required

The script does **not** infer the intended grid from whatever files happen to
exist. The expected candidates, categories, seeds, split, and directory layout
are declared first in a JSON plan.

This implements the principle:

```text
expected experiment grid
        ↓
read results
        ↓
check completeness
        ↓
write table
```

rather than:

```text
read available results
        ↓
assume that is the intended experiment
```

The latter cannot detect a missing experiment.

The plan receives a SHA-256 fingerprint and every output records that hash, so
the table can be tied back to the exact pre-declared grid.

---

## 3. Expected directory structure

Default:

```text
outputs/day04/
└── runs/
    ├── candidate_01/
    │   ├── can/
    │   │   ├── seed_42/
    │   │   │   ├── metrics.json
    │   │   │   ├── params.json
    │   │   │   └── efficiency.json
    │   │   ├── seed_17/
    │   │   └── seed_2026/
    │   └── ...
    └── candidate_02/
        └── ...
```

This is configurable through:

```json
"run_layout": "runs/{candidate}/{category}/seed_{seed}"
```

E7 assumes the project's **single-class anomaly segmentation** protocol:
one `metrics.json` corresponds to exactly one category.

---

## 4. Upstream files

### `metrics.json`

Produced by E2/E3.

E7 requires:

```text
schema_version == msila-evaluator-v1
dataset == mvtec_ad2
split == plan.split
metric_protocol_version == plan.metric_protocol_version
exactly one expected category
AU-PRO max FPR == 0.05
evaluator validation == PASS
E3 QA == PASS, if required by plan
```

The script extracts:

```text
AU-PRO_0.05
SegF1
sample counts
QA status
```

It does not recompute the metrics.

### `params.json`

Produced by E4.

Both supported E4 forms are accepted:

```text
msila.e4.params.v1
msila.e4.manual_check.v1
```

When the plan requires E4 PASS, manual/internal consistency must be PASS.

Extracted values:

```text
total parameters
trainable parameters
frozen parameters
parameter scope
```

For the same `candidate_id`, these values must remain identical across category
and seed. Candidate identity represents architecture identity; random seed or
dataset category must not change the number of model parameters.

### `efficiency.json`

Produced by the combined E5/E6 function:

```text
msila.e5_e6.efficiency.v1
```

Extracted values:

```text
latency mean
latency median
latency p95
latency std
latency repeated-run CV
peak allocated VRAM (MiB)
incremental peak allocated VRAM (MiB)
benchmark scope fingerprint
```

By default all Day-04 rows must share one efficiency-scope fingerprint. This
prevents mixing, for example, cached-head latency with full-image end-to-end
latency.

---

## 5. Plan file

Copy:

```text
configs/day04_eval_plan.example.json
```

to an experiment-specific file and edit it **before viewing the candidate
results**.

Example:

```json
{
  "schema_version": "msila.day04.plan.v1",
  "split": "dev_synthetic",
  "metric_protocol_version": "1.0",
  "candidates": [
    "candidate_01",
    "candidate_02"
  ],
  "categories": [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts"
  ],
  "seeds": [42, 17, 2026],
  "run_layout": "runs/{candidate}/{category}/seed_{seed}",
  "required_files": {
    "metrics": "metrics.json",
    "params": "params.json",
    "efficiency": "efficiency.json"
  }
}
```

The repo research protocol currently names seed `42` as development and `17`,
`2026` as replication seeds. If Day-04 uses only three pilot categories, replace
the category array with **exactly those pre-registered three categories**.
Do not let E7 choose categories after seeing results.

`split` must match the split actually used by E2/E3. E7 deliberately does not
silently reconcile protocol differences.

---

## 6. Run command

```bash
python scripts/eval_day04.py \
  --root outputs/day04 \
  --plan configs/day04_eval_plan.json \
  --output-dir outputs/day04/table
```

Successful execution prints:

```text
[E7 PASS]
rows=...
plan_sha256=...
csv=...
json=...
completeness=...
```

Failure returns exit code `2`.

---

## 7. Outputs

### `day04_category_runs.csv`

Primary long-form Day-04 table.

One row:

```text
candidate_id
category
seed
split
AU-PRO0.05
SegF1
sample counts
params
latency
VRAM
scope fingerprint
source file paths
```

Long form is intentional. It preserves every experimental unit without
premature averaging.

### `day04_category_runs.json`

Same raw rows plus:

```text
schema version
plan hash
split
aggregation policy
```

### `day04_completeness.json`

Integrity gate:

```text
expected cells
observed cells
missing cells
duplicate cells
candidate/category/seed grid
parameter consistency
efficiency scope fingerprints
upstream requirements
PASS/FAIL
```

---

## 8. What E7 deliberately does not calculate

E7 does **not** calculate:

\[
\mu,\quad \sigma,\quad \Delta\text{AU-PRO}
\]

across seeds and does not rank candidates.

Reason:

```text
E7 = data integrity + deterministic aggregation
E8 = statistical analysis + candidate selection
```

Keeping the two stages separate prevents selection logic from being silently
embedded inside data collection.

A `metrics.json` may contain a macro score, but because one E7 run is
single-category, the script checks:

\[
\text{macro}=\text{category score}
\]

and stores it only as an upstream consistency check.

---

## 9. Hard PASS

E7 PASS requires all of the following:

```text
1. Plan schema valid.
2. Every expected candidate × category × seed directory exists.
3. metrics.json exists and passes E2/E3 contracts.
4. AU-PRO cutoff is exactly 0.05.
5. params.json exists and E4 is valid.
6. efficiency.json exists and E5/E6 are valid.
7. candidate_id in E4/E5/E6 matches the planned candidate.
8. No duplicate experimental unit.
9. Parameter signature is constant within a candidate.
10. Efficiency benchmark scope is common across candidates, when required.
11. observed rows == candidates × categories × seeds.
```

One missing run causes the script to fail and **no valid Day-04 table is
claimed**.

---

## 10. Scientific interpretation

Valid conclusion from E7:

> The Day-04 raw result table is complete and traceable for every pre-declared
> candidate/category/seed experimental unit, with upstream metric and efficiency
> QA satisfied.

Not valid from E7 alone:

```text
candidate A is best
candidate A significantly improves AU-PRO
candidate A is the preferred accuracy/efficiency trade-off
```

Those are E8 analyses.

---

## 11. Project evidence

This implementation follows the repository's existing protocols:

- `docs/research_protocol_v1.md`
  - category-level experimental units;
  - seeds `42`, `17`, `2026`;
  - multiple seeds should later be reported with mean and standard deviation;
  - each run must preserve traceability including parameters, runtime and peak
    VRAM.

- `docs/metric_protocol_v1.md`
  - primary metric is AU-PRO\(_{0.05}\);
  - results are reported by category;
  - computational metrics include trainable parameters, latency and peak GPU
    memory.

E7 therefore aggregates those already-produced measurements without changing
their scientific meaning.
