#!/usr/bin/env bash
#
# MS-ILA Day-04 — smoke-gated Adapter r×d screen orchestrator
#
# This script orchestrates the REAL Day-04 path:
#
#   src/train/screen_adapter.py
#       -> config.yaml / best.pt / train_log.csv / predictions/*
#       -> scripts/eval_day04_candidate.py
#       -> metrics.json / qa_report.json / params.json / efficiency.json
#
# It deliberately separates:
#
#   MODE=smoke
#       one real candidate x one real category x one seed
#       -> train
#       -> real prediction artifacts
#       -> E2/E3/E4/E5/E6
#       -> only then can a final full screen be unlocked
#
#   MODE=full
#       requires a previously completed REAL smoke gate with matching
#       candidate/category/seed/protocol artifacts before launching the grid.
#
# Scientific contract
# -------------------
# - r = Adapter bottleneck_dim
# - d = Adapter projection_dim
# - only (r,d) vary across candidates in a controlled screen
# - b4/b8/b12 are independent Adapters with the same candidate (r,d)
# - frozen DINO features are reused from cache; no DINO recomputation here
# - candidate selection split defaults to dev_synthetic
# - SegF1 threshold must be fixed BEFORE evaluation
# - AU-PRO_0.05 / SegF1 are computed by the locked repository evaluator
# - no dummy anomaly maps
# - no silent score-map resizing
# - final full screen requires E5/E6 CUDA efficiency evidence
# - candidates run sequentially to keep GPU memory behavior reproducible
#
# Recommended sequence
# --------------------
#
# 1) Real smoke:
#
#   bash scripts/run_day04.sh \
#       --mode smoke \
#       --categories fabric,vial,wallplugs \
#       --seed 42 \
#       --device cuda:0 \
#       --seg-f1-threshold 0.5
#
# 2) Full grid only AFTER smoke PASS:
#
#   bash scripts/run_day04.sh \
#       --mode full \
#       --categories fabric,vial,wallplugs \
#       --seed 42 \
#       --device cuda:0 \
#       --seg-f1-threshold 0.5 \
#       --skip-existing
#
# IMPORTANT:
# - 0.5 above is only an example. Use the threshold locked by your protocol.
# - The current runner formats {category}, but not {seed}; therefore exactly
#   one seed is accepted per invocation.
# - If more than one category is requested, protocol.output_root MUST contain
#   literal "{category}".

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

GRID="configs/day04_adapter_grid.yaml"
PROTOCOL="configs/day04_train_protocol.yaml"
DEVICE="auto"

RUNNER="src/train/screen_adapter.py"
EVALUATOR="scripts/eval_day04_candidate.py"

MODE="smoke"
CATEGORIES_CSV=""
SEED=""
SEG_F1_THRESHOLD=""
EXPECTED_SPLIT="dev_synthetic"

# Smoke candidate is a pipeline sanity candidate only; it is NOT declared best.
SMOKE_CATEGORY=""
SMOKE_R=64
SMOKE_D=256

SKIP_EXISTING=0
OVERWRITE_EVALUATION=0
SKIP_EFFICIENCY=0
DRY_RUN=0

# E5/E6 defaults mirror src.eval.efficiency.
BENCHMARK_BATCH_SIZE=1
LATENCY_WARMUP=10
LATENCY_ITERATIONS=50
LATENCY_ROUNDS=3
LATENCY_STABILITY_CV_THRESHOLD="0.10"
VRAM_WARMUP=10
VRAM_ITERATIONS=1


usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_day04.sh \
      --mode smoke|full \
      --categories <cat1,cat2,...> \
      --seed <int> \
      --seg-f1-threshold <0..1> \
      [--device auto|cpu|cuda|cuda:0] \
      [--expected-split dev_synthetic] \
      [--grid path/to/day04_adapter_grid.yaml] \
      [--protocol path/to/day04_train_protocol.yaml] \
      [--smoke-category fabric] \
      [--smoke-r 64] \
      [--smoke-d 256] \
      [--skip-existing] \
      [--overwrite-evaluation] \
      [--skip-efficiency] \
      [--dry-run]

Required:
  --categories
      Comma-separated categories available to the screen.
      Example: fabric,vial,wallplugs

  --seed
      One integer seed shared by the invocation.

  --seg-f1-threshold
      Pre-locked probability threshold for SegF1.
      This launcher does NOT tune threshold on the evaluated split.

Modes:
  --mode smoke
      DEFAULT.
      Runs exactly one real candidate on one real category, then runs
      scripts/eval_day04_candidate.py.
      Default smoke candidate: r=64, d=256.
      Default smoke category: the first value in --categories.

  --mode full
      Runs all grid candidates for all requested categories.
      HARD GATE: refuses to start unless the smoke candidate/category from the
      same seed has already produced PASS:
        config.yaml
        best.pt
        train_log.csv
        predictions/manifest.jsonl
        metrics.json
        qa_report.json
        params.json
        efficiency.json

Optional:
  --device
      Forwarded to training/evaluation. Default: auto.
      Final full mode requires efficiency evidence, therefore a CUDA runtime is
      normally required by eval_day04_candidate.py.

  --expected-split
      Default: dev_synthetic.
      Day-04 Adapter architecture selection should use dev_synthetic, not
      test_public.

  --grid
      Default: configs/day04_adapter_grid.yaml

  --protocol
      Default: configs/day04_train_protocol.yaml

  --smoke-category
      Smoke category. Default: first category in --categories.

  --smoke-r / --smoke-d
      Smoke candidate dimensions. Defaults: 64 / 256.
      The pair must exist in the locked grid.

  --skip-existing
      Reuse an existing run ONLY when its artifacts are complete and validated.
      Incomplete training/evaluation artifacts are never silently accepted.

  --overwrite-evaluation
      If training artifacts are complete but evaluation artifacts are partial
      or intentionally need regeneration, forward --overwrite to
      eval_day04_candidate.py. Training artifacts are never overwritten.

  --skip-efficiency
      Debug only: evaluate E2-E4 without E5/E6.
      Allowed in smoke mode, but such a smoke does NOT unlock --mode full.
      Forbidden in full mode.

  --dry-run
      Validate grid/protocol/smoke gate and print commands without executing
      training/evaluation.

E5/E6 optional controls:
  --benchmark-batch-size <int>       default 1
  --latency-warmup <int>             default 10
  --latency-iterations <int>         default 50
  --latency-rounds <int>             default 3
  --latency-stability-cv <float>     default 0.10
  --vram-warmup <int>                default 10
  --vram-iterations <int>            default 1
EOF
}


die() {
    echo "[DAY04 ERROR] $*" >&2
    exit 2
}


log() {
    echo "[DAY04] $*"
}


on_error() {
    local exit_code=$?
    local line_no="${1:-unknown}"
    echo "[DAY04 FAIL] command failed at line ${line_no}, exit=${exit_code}" >&2
    exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR


require_value() {
    local option="$1"
    local remaining="$2"
    [[ "${remaining}" -ge 2 ]] || die "${option} requires a value"
}


require_positive_int_shell() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a positive integer, got: ${value}"
    (( value > 0 )) || die "${name} must be > 0, got: ${value}"
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            require_value "$1" "$#"
            MODE="$2"
            shift 2
            ;;
        --categories)
            require_value "$1" "$#"
            CATEGORIES_CSV="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "$#"
            SEED="$2"
            shift 2
            ;;
        --seg-f1-threshold)
            require_value "$1" "$#"
            SEG_F1_THRESHOLD="$2"
            shift 2
            ;;
        --expected-split)
            require_value "$1" "$#"
            EXPECTED_SPLIT="$2"
            shift 2
            ;;
        --device)
            require_value "$1" "$#"
            DEVICE="$2"
            shift 2
            ;;
        --grid)
            require_value "$1" "$#"
            GRID="$2"
            shift 2
            ;;
        --protocol)
            require_value "$1" "$#"
            PROTOCOL="$2"
            shift 2
            ;;
        --smoke-category)
            require_value "$1" "$#"
            SMOKE_CATEGORY="$2"
            shift 2
            ;;
        --smoke-r)
            require_value "$1" "$#"
            SMOKE_R="$2"
            shift 2
            ;;
        --smoke-d)
            require_value "$1" "$#"
            SMOKE_D="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --overwrite-evaluation)
            OVERWRITE_EVALUATION=1
            shift
            ;;
        --skip-efficiency)
            SKIP_EFFICIENCY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --benchmark-batch-size)
            require_value "$1" "$#"
            BENCHMARK_BATCH_SIZE="$2"
            shift 2
            ;;
        --latency-warmup)
            require_value "$1" "$#"
            LATENCY_WARMUP="$2"
            shift 2
            ;;
        --latency-iterations)
            require_value "$1" "$#"
            LATENCY_ITERATIONS="$2"
            shift 2
            ;;
        --latency-rounds)
            require_value "$1" "$#"
            LATENCY_ROUNDS="$2"
            shift 2
            ;;
        --latency-stability-cv)
            require_value "$1" "$#"
            LATENCY_STABILITY_CV_THRESHOLD="$2"
            shift 2
            ;;
        --vram-warmup)
            require_value "$1" "$#"
            VRAM_WARMUP="$2"
            shift 2
            ;;
        --vram-iterations)
            require_value "$1" "$#"
            VRAM_ITERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------

[[ "${MODE}" == "smoke" || "${MODE}" == "full" ]] \
    || die "--mode must be smoke or full, got: ${MODE}"

[[ -n "${CATEGORIES_CSV}" ]] || die "--categories is required"
[[ -n "${SEED}" ]] || die "--seed is required"
[[ -n "${SEG_F1_THRESHOLD}" ]] || die "--seg-f1-threshold is required"
[[ -n "${EXPECTED_SPLIT}" ]] || die "--expected-split must not be empty"

[[ "${SEED}" =~ ^-?[0-9]+$ ]] \
    || die "--seed must be an integer, got: ${SEED}"

require_positive_int_shell "--smoke-r" "${SMOKE_R}"
require_positive_int_shell "--smoke-d" "${SMOKE_D}"
require_positive_int_shell "--benchmark-batch-size" "${BENCHMARK_BATCH_SIZE}"
require_positive_int_shell "--latency-warmup" "${LATENCY_WARMUP}"
require_positive_int_shell "--latency-iterations" "${LATENCY_ITERATIONS}"
require_positive_int_shell "--latency-rounds" "${LATENCY_ROUNDS}"
require_positive_int_shell "--vram-warmup" "${VRAM_WARMUP}"
require_positive_int_shell "--vram-iterations" "${VRAM_ITERATIONS}"

if [[ "${MODE}" == "full" && "${SKIP_EFFICIENCY}" -eq 1 ]]; then
    die "--skip-efficiency is forbidden in --mode full; E7 requires efficiency.json"
fi

# Scientific guard: architecture selection should not accidentally move to the
# public test split. This script can evaluate another explicit split for a
# different protocol, but test_public is not accepted for Day-04 screening.
if [[ "${EXPECTED_SPLIT}" == "test_public" && "${MODE}" == "full" ]]; then
    die "Day-04 full candidate screen must not select architecture on test_public; use dev_synthetic"
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || die "Python executable not found: ${PYTHON_BIN}"

[[ -f "${RUNNER}" ]] \
    || die "missing runner: ${RUNNER}"

[[ -f "${EVALUATOR}" ]] \
    || die "missing evaluator: ${EVALUATOR}"

[[ -f "${GRID}" ]] \
    || die "missing grid: ${GRID}"

[[ -f "${PROTOCOL}" ]] \
    || die "missing protocol: ${PROTOCOL}"


# Validate numeric floating-point CLI arguments without depending on bc.
"${PYTHON_BIN}" - \
    "${SEG_F1_THRESHOLD}" \
    "${LATENCY_STABILITY_CV_THRESHOLD}" <<'PY'
import math
import sys

seg = float(sys.argv[1])
cv = float(sys.argv[2])

if not math.isfinite(seg) or not (0.0 <= seg <= 1.0):
    raise SystemExit("--seg-f1-threshold must be finite in [0,1]")
if not math.isfinite(cv) or cv <= 0.0:
    raise SystemExit("--latency-stability-cv must be finite and > 0")
PY


IFS=',' read -r -a RAW_CATEGORIES <<< "${CATEGORIES_CSV}"

CATEGORIES=()
declare -A SEEN_CATEGORY=()

for raw in "${RAW_CATEGORIES[@]}"; do
    category="${raw//[[:space:]]/}"

    [[ -n "${category}" ]] \
        || die "empty category in --categories=${CATEGORIES_CSV}"

    if [[ -n "${SEEN_CATEGORY[${category}]+x}" ]]; then
        die "duplicate category: ${category}"
    fi

    SEEN_CATEGORY["${category}"]=1
    CATEGORIES+=("${category}")
done

[[ ${#CATEGORIES[@]} -gt 0 ]] \
    || die "no valid categories supplied"

if [[ -z "${SMOKE_CATEGORY}" ]]; then
    SMOKE_CATEGORY="${CATEGORIES[0]}"
fi

[[ -n "${SEEN_CATEGORY[${SMOKE_CATEGORY}]+x}" ]] \
    || die "--smoke-category=${SMOKE_CATEGORY} must also appear in --categories"


# ---------------------------------------------------------------------------
# Python import preflight
# ---------------------------------------------------------------------------

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" - <<'PY'
import importlib

required = (
    "torch",
    "numpy",
    "yaml",
    "src.train.screen_adapter",
    "src.train.day04_project_hooks",
    "src.eval.evaluator",
    "src.eval.efficiency",
)

for name in required:
    importlib.import_module(name)

print("[DAY04] Python imports: PASS")
PY


# ---------------------------------------------------------------------------
# Grid preflight
# ---------------------------------------------------------------------------

mapfile -t CANDIDATE_ROWS < <(
    "${PYTHON_BIN}" - "${GRID}" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))

if not isinstance(payload, dict):
    raise SystemExit("grid root must be a mapping")

candidates = payload.get("candidates")
if not isinstance(candidates, list) or not candidates:
    raise SystemExit("grid.candidates must be a non-empty list")

seen_names = set()
seen_pairs = set()
rows = []

for i, raw in enumerate(candidates):
    if not isinstance(raw, dict):
        raise SystemExit(f"grid.candidates[{i}] must be a mapping")

    if "bottleneck_dim" not in raw or "projection_dim" not in raw:
        raise SystemExit(
            f"grid.candidates[{i}] must define bottleneck_dim and projection_dim"
        )

    r = raw["bottleneck_dim"]
    d = raw["projection_dim"]

    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise SystemExit(f"candidate[{i}].bottleneck_dim must be positive int")
    if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
        raise SystemExit(f"candidate[{i}].projection_dim must be positive int")

    expected_name = f"adapter_r{r}_d{d}"
    run_name = raw.get("run_name", expected_name)

    if run_name != expected_name:
        raise SystemExit(
            f"candidate[{i}] run_name={run_name!r}; expected {expected_name!r}"
        )

    pair = (r, d)
    if pair in seen_pairs:
        raise SystemExit(f"duplicate candidate pair: r={r}, d={d}")
    if run_name in seen_names:
        raise SystemExit(f"duplicate candidate run_name: {run_name}")

    seen_pairs.add(pair)
    seen_names.add(run_name)
    rows.append((r, d, run_name))

expected_n = payload.get("expected_num_candidates")
if expected_n is not None:
    if isinstance(expected_n, bool) or not isinstance(expected_n, int):
        raise SystemExit("expected_num_candidates must be int")
    if expected_n != len(rows):
        raise SystemExit(
            f"expected_num_candidates={expected_n}, actual={len(rows)}"
        )

for r, d, run_name in rows:
    print(f"{r}\t{d}\t{run_name}")
PY
)

[[ ${#CANDIDATE_ROWS[@]} -gt 0 ]] \
    || die "grid preflight produced zero candidates"

SMOKE_RUN_NAME="adapter_r${SMOKE_R}_d${SMOKE_D}"
SMOKE_IN_GRID=0

for row in "${CANDIDATE_ROWS[@]}"; do
    IFS=$'\t' read -r r d run_name <<< "${row}"
    if [[ "${r}" == "${SMOKE_R}" && "${d}" == "${SMOKE_D}" ]]; then
        [[ "${run_name}" == "${SMOKE_RUN_NAME}" ]] \
            || die "smoke candidate name mismatch: ${run_name} != ${SMOKE_RUN_NAME}"
        SMOKE_IN_GRID=1
    fi
done

[[ "${SMOKE_IN_GRID}" -eq 1 ]] \
    || die "smoke candidate ${SMOKE_RUN_NAME} does not exist in locked grid"

log "Grid candidates: ${#CANDIDATE_ROWS[@]}"
log "Smoke candidate: ${SMOKE_RUN_NAME}"


# ---------------------------------------------------------------------------
# Protocol preflight
# ---------------------------------------------------------------------------
#
# Important: use the REAL screen_adapter.resolve_protocol() so placeholder/null
# scientific settings are rejected before any GPU work starts.
# ---------------------------------------------------------------------------

if [[ ${#CATEGORIES[@]} -gt 1 ]]; then
    "${PYTHON_BIN}" - "${PROTOCOL}" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))

if not isinstance(payload, dict):
    raise SystemExit("protocol root must be a mapping")

output_root = payload.get("output_root")
if not isinstance(output_root, str) or not output_root.strip():
    raise SystemExit("protocol.output_root must be a non-empty string")

if "{category}" not in output_root:
    raise SystemExit(
        "Multiple categories requested, but protocol.output_root does not "
        "contain literal '{category}'. Example: outputs/day04/screens/{category}"
    )
PY
fi


declare -A CATEGORY_OUTPUT_ROOT=()

for category in "${CATEGORIES[@]}"; do
    resolved_output_root="$(
        "${PYTHON_BIN}" - "${PROTOCOL}" "${category}" <<'PY'
from pathlib import Path
import sys

from src.train.screen_adapter import load_yaml, resolve_protocol

protocol_path = Path(sys.argv[1])
category = sys.argv[2]

raw = load_yaml(protocol_path)
resolved = resolve_protocol(raw, category=category)

# Real Day-04 screen requires actual cache/index paths.
data = resolved["data"]
for key in ("cache_dir", "train_records", "val_records"):
    value = data.get(key)
    if value is None or not str(value).strip():
        raise SystemExit(
            f"protocol.data.{key} is not configured for category={category!r}"
        )

print(str(resolved["output_root"]))
PY
    )"

    [[ -n "${resolved_output_root}" ]] \
        || die "resolved empty output_root for category=${category}"

    CATEGORY_OUTPUT_ROOT["${category}"]="${resolved_output_root}"

    log "Protocol PASS: category=${category}"
    log "Resolved output_root: ${resolved_output_root}"
done


# ---------------------------------------------------------------------------
# Artifact state helpers
# ---------------------------------------------------------------------------

training_is_complete() {
    local run_dir="$1"

    [[ -s "${run_dir}/config.yaml" ]] \
        && [[ -s "${run_dir}/best.pt" ]] \
        && [[ -s "${run_dir}/train_log.csv" ]] \
        && [[ -s "${run_dir}/predictions/manifest.jsonl" ]]
}


evaluation_is_complete() {
    local run_dir="$1"

    [[ -s "${run_dir}/metrics.json" ]] \
        && [[ -s "${run_dir}/qa_report.json" ]] \
        && [[ -s "${run_dir}/params.json" ]] \
        && [[ -s "${run_dir}/efficiency.json" ]] \
        && [[ -s "${run_dir}/evaluation_manifest.jsonl" ]]
}


evaluation_debug_is_complete() {
    local run_dir="$1"

    [[ -s "${run_dir}/metrics.json" ]] \
        && [[ -s "${run_dir}/qa_report.json" ]] \
        && [[ -s "${run_dir}/params.json" ]] \
        && [[ -s "${run_dir}/evaluation_manifest.jsonl" ]]
}


run_is_nonempty() {
    local run_dir="$1"

    [[ -d "${run_dir}" ]] \
        && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}


validate_real_evaluation() {
    local run_dir="$1"
    local expected_candidate="$2"
    local expected_category="$3"
    local expected_seed="$4"
    local expected_split="$5"
    local require_efficiency="$6"

    "${PYTHON_BIN}" - \
        "${run_dir}" \
        "${expected_candidate}" \
        "${expected_category}" \
        "${expected_seed}" \
        "${expected_split}" \
        "${require_efficiency}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidate = sys.argv[2]
category = sys.argv[3]
seed = int(sys.argv[4])
split = sys.argv[5]
require_eff = bool(int(sys.argv[6]))

def load_json(name):
    path = run_dir / name
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value

# config.yaml is read with yaml only here because it is the run provenance root.
import yaml
config_path = run_dir / "config.yaml"
if not config_path.is_file():
    raise SystemExit(f"missing {config_path}")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("config.yaml root must be mapping")

c = config.get("candidate", {})
if c.get("run_name") != candidate:
    raise SystemExit(
        f"candidate mismatch in config: {c.get('run_name')!r} != {candidate!r}"
    )
if str(config.get("category")) != category:
    raise SystemExit(
        f"category mismatch in config: {config.get('category')!r} != {category!r}"
    )
if int(config.get("seed")) != seed:
    raise SystemExit(
        f"seed mismatch in config: {config.get('seed')!r} != {seed!r}"
    )

metrics = load_json("metrics.json")
if metrics.get("schema_version") != "msila-evaluator-v1":
    raise SystemExit("metrics.json schema_version mismatch")
if metrics.get("validation", {}).get("status") != "PASS":
    raise SystemExit("metrics.json evaluator validation is not PASS")
if metrics.get("split") != split:
    raise SystemExit(
        f"metrics split mismatch: {metrics.get('split')!r} != {split!r}"
    )
if metrics.get("categories") != [category]:
    raise SystemExit(
        f"metrics categories mismatch: {metrics.get('categories')!r}"
    )
qa_summary = metrics.get("validation", {}).get("anomaly_map_qa", {})
if qa_summary.get("status") != "PASS":
    raise SystemExit("metrics.json E3 anomaly_map_qa is not PASS")
if float(qa_summary.get("valid_fraction", -1)) != 1.0:
    raise SystemExit("metrics.json E3 valid_fraction != 1.0")

qa = load_json("qa_report.json")
if qa.get("summary", {}).get("status") != "PASS":
    raise SystemExit("qa_report.json is not PASS")
if float(qa.get("summary", {}).get("valid_fraction", -1)) != 1.0:
    raise SystemExit("qa_report.json valid_fraction != 1.0")

params = load_json("params.json")
if params.get("schema_version") != "msila.e4.manual_check.v1":
    raise SystemExit(
        f"params.json must be E4 manual-check report; got "
        f"{params.get('schema_version')!r}"
    )
if params.get("candidate_id") != candidate:
    raise SystemExit("params.json candidate_id mismatch")
if params.get("status") != "PASS":
    raise SystemExit("params.json E4 status is not PASS")

manual = params.get("manual_check", {})
for key in ("total_match", "trainable_match", "component_match"):
    if manual.get(key) is not True:
        raise SystemExit(f"params.json manual_check.{key} is not true")
if int(manual.get("unexpected_parameters", -1)) != 0:
    raise SystemExit("params.json reports unexpected parameters")

manifest = run_dir / "evaluation_manifest.jsonl"
if not manifest.is_file() or manifest.stat().st_size <= 0:
    raise SystemExit("evaluation_manifest.jsonl missing/empty")

if require_eff:
    eff = load_json("efficiency.json")
    if eff.get("schema_version") != "msila.e5_e6.efficiency.v1":
        raise SystemExit("efficiency.json schema_version mismatch")
    if eff.get("candidate_id") != candidate:
        raise SystemExit("efficiency.json candidate_id mismatch")
    if eff.get("status") != "PASS":
        raise SystemExit("efficiency.json combined status is not PASS")
    if eff.get("latency", {}).get("status") != "PASS":
        raise SystemExit("efficiency.json latency status is not PASS")
    if eff.get("peak_vram", {}).get("status") != "PASS":
        raise SystemExit("efficiency.json peak_vram status is not PASS")

print(
    f"[DAY04] Evaluation artifacts PASS: "
    f"candidate={candidate} category={category} seed={seed} split={split}"
)
PY
}


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

print_command() {
    printf '[DAY04] command:'
    printf ' %q' "$@"
    printf '\n'
}


run_training() {
    local category="$1"
    local r="$2"
    local d="$3"
    local run_name="$4"
    local run_dir="$5"

    if run_is_nonempty "${run_dir}"; then
        if training_is_complete "${run_dir}"; then
            if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
                log "Reuse complete training artifacts: ${run_dir}"
                return 0
            fi

            die \
                "complete training run already exists: ${run_dir}. " \
                "Use --skip-existing only when intentionally resuming."
        fi

        die \
            "existing run is incomplete: ${run_dir}. " \
            "Refusing to overwrite or silently resume partial training state."
    fi

    local cmd=(
        "${PYTHON_BIN}"
        "${RUNNER}"
        --r "${r}"
        --d "${d}"
        --category "${category}"
        --seed "${SEED}"
        --grid "${GRID}"
        --protocol "${PROTOCOL}"
        --device "${DEVICE}"
    )

    print_command "${cmd[@]}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 0
    fi

    "${cmd[@]}"

    training_is_complete "${run_dir}" \
        || die \
            "runner returned success but required training artifacts are missing: ${run_dir}"

    log "TRAIN PASS candidate=${run_name} category=${category}"
}


run_evaluation() {
    local category="$1"
    local run_name="$2"
    local run_dir="$3"
    local require_efficiency="$4"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        local preview=(
            "${PYTHON_BIN}"
            "${EVALUATOR}"
            --run-dir "${run_dir}"
            --expected-split "${EXPECTED_SPLIT}"
            --seg-f1-threshold "${SEG_F1_THRESHOLD}"
            --device "${DEVICE}"
            --benchmark-batch-size "${BENCHMARK_BATCH_SIZE}"
            --latency-warmup "${LATENCY_WARMUP}"
            --latency-iterations "${LATENCY_ITERATIONS}"
            --latency-rounds "${LATENCY_ROUNDS}"
            --latency-stability-cv-threshold "${LATENCY_STABILITY_CV_THRESHOLD}"
            --vram-warmup "${VRAM_WARMUP}"
            --vram-iterations "${VRAM_ITERATIONS}"
        )

        if [[ "${require_efficiency}" -eq 0 ]]; then
            preview+=(--skip-efficiency)
        fi
        if [[ "${OVERWRITE_EVALUATION}" -eq 1 ]]; then
            preview+=(--overwrite)
        fi

        print_command "${preview[@]}"
        return 0
    fi

    training_is_complete "${run_dir}" \
        || die "cannot evaluate incomplete training run: ${run_dir}"

    if [[ "${require_efficiency}" -eq 1 ]]; then
        if evaluation_is_complete "${run_dir}"; then
            if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
                validate_real_evaluation \
                    "${run_dir}" "${run_name}" "${category}" "${SEED}" \
                    "${EXPECTED_SPLIT}" 1
                log "Reuse complete REAL evaluation: ${run_dir}"
                return 0
            fi

            die \
                "complete evaluation already exists: ${run_dir}. " \
                "Use --skip-existing to validate/reuse it."
        fi
    else
        if evaluation_debug_is_complete "${run_dir}" \
            && [[ "${SKIP_EXISTING}" -eq 1 ]]; then
            validate_real_evaluation \
                "${run_dir}" "${run_name}" "${category}" "${SEED}" \
                "${EXPECTED_SPLIT}" 0
            log "Reuse complete debug E2-E4 evaluation: ${run_dir}"
            return 0
        fi
    fi

    local cmd=(
        "${PYTHON_BIN}"
        "${EVALUATOR}"
        --run-dir "${run_dir}"
        --expected-split "${EXPECTED_SPLIT}"
        --seg-f1-threshold "${SEG_F1_THRESHOLD}"
        --device "${DEVICE}"
        --benchmark-batch-size "${BENCHMARK_BATCH_SIZE}"
        --latency-warmup "${LATENCY_WARMUP}"
        --latency-iterations "${LATENCY_ITERATIONS}"
        --latency-rounds "${LATENCY_ROUNDS}"
        --latency-stability-cv-threshold "${LATENCY_STABILITY_CV_THRESHOLD}"
        --vram-warmup "${VRAM_WARMUP}"
        --vram-iterations "${VRAM_ITERATIONS}"
    )

    if [[ "${require_efficiency}" -eq 0 ]]; then
        cmd+=(--skip-efficiency)
    fi
    if [[ "${OVERWRITE_EVALUATION}" -eq 1 ]]; then
        cmd+=(--overwrite)
    fi

    print_command "${cmd[@]}"
    "${cmd[@]}"

    validate_real_evaluation \
        "${run_dir}" "${run_name}" "${category}" "${SEED}" \
        "${EXPECTED_SPLIT}" "${require_efficiency}"

    log "EVAL PASS candidate=${run_name} category=${category}"
}


run_one_candidate() {
    local category="$1"
    local r="$2"
    local d="$3"
    local run_name="$4"
    local require_efficiency="$5"

    local output_root="${CATEGORY_OUTPUT_ROOT[${category}]}"
    local run_dir="${output_root}/${run_name}"

    log "------------------------------------------------------------"
    log "category=${category} candidate=${run_name}"
    log "r=${r} d=${d} seed=${SEED}"
    log "run_dir=${run_dir}"

    # Existing training + partial evaluation is a valid resume point only when
    # --skip-existing is supplied: training is reused and evaluation continues.
    if training_is_complete "${run_dir}"; then
        if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
            log "Training already complete; continue/validate evaluation."
        else
            die \
                "training artifacts already exist: ${run_dir}. " \
                "Use --skip-existing to reuse them."
        fi
    else
        run_training \
            "${category}" "${r}" "${d}" "${run_name}" "${run_dir}"
    fi

    run_evaluation \
        "${category}" "${run_name}" "${run_dir}" "${require_efficiency}"
}


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

SMOKE_OUTPUT_ROOT="${CATEGORY_OUTPUT_ROOT[${SMOKE_CATEGORY}]}"
SMOKE_RUN_DIR="${SMOKE_OUTPUT_ROOT}/${SMOKE_RUN_NAME}"


validate_smoke_gate_for_full() {
    training_is_complete "${SMOKE_RUN_DIR}" \
        || die \
            "FULL SCREEN BLOCKED: real smoke training has not completed: " \
            "${SMOKE_RUN_DIR}"

    evaluation_is_complete "${SMOKE_RUN_DIR}" \
        || die \
            "FULL SCREEN BLOCKED: smoke lacks complete E2-E6 artifacts. " \
            "Run --mode smoke on CUDA first: ${SMOKE_RUN_DIR}"

    validate_real_evaluation \
        "${SMOKE_RUN_DIR}" \
        "${SMOKE_RUN_NAME}" \
        "${SMOKE_CATEGORY}" \
        "${SEED}" \
        "${EXPECTED_SPLIT}" \
        1

    log "REAL SMOKE GATE: PASS"
}


# ---------------------------------------------------------------------------
# Summary before execution
# ---------------------------------------------------------------------------

log "============================================================"
log "MS-ILA Day-04 smoke-gated r×d experiment"
log "============================================================"
log "Project root       : ${PROJECT_ROOT}"
log "Mode               : ${MODE}"
log "Grid               : ${GRID}"
log "Protocol           : ${PROTOCOL}"
log "Runner             : ${RUNNER}"
log "Evaluator          : ${EVALUATOR}"
log "Device             : ${DEVICE}"
log "Seed               : ${SEED}"
log "Expected split     : ${EXPECTED_SPLIT}"
log "SegF1 threshold    : ${SEG_F1_THRESHOLD}"
log "Categories         : ${CATEGORIES[*]}"
log "Grid candidates    : ${#CANDIDATE_ROWS[@]}"
log "Smoke category     : ${SMOKE_CATEGORY}"
log "Smoke candidate    : ${SMOKE_RUN_NAME}"
log "Skip existing      : ${SKIP_EXISTING}"
log "Overwrite eval     : ${OVERWRITE_EVALUATION}"
log "Skip efficiency    : ${SKIP_EFFICIENCY}"
log "Dry run            : ${DRY_RUN}"
log "============================================================"


# ---------------------------------------------------------------------------
# MODE=smoke
# ---------------------------------------------------------------------------

if [[ "${MODE}" == "smoke" ]]; then
    REQUIRE_EFFICIENCY=1
    if [[ "${SKIP_EFFICIENCY}" -eq 1 ]]; then
        REQUIRE_EFFICIENCY=0
    fi

    run_one_candidate \
        "${SMOKE_CATEGORY}" \
        "${SMOKE_R}" \
        "${SMOKE_D}" \
        "${SMOKE_RUN_NAME}" \
        "${REQUIRE_EFFICIENCY}"

    log "============================================================"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "SMOKE DRY-RUN PASS"
        log "No training/evaluation was executed."
    elif [[ "${REQUIRE_EFFICIENCY}" -eq 1 ]]; then
        log "REAL DAY-04 SMOKE PASS"
        log "Full-screen gate is now satisfied for this category/seed/split."
        log "Next: rerun with --mode full --skip-existing."
    else
        log "DEBUG SMOKE E2-E4 PASS"
        log "E5/E6 were skipped, so this does NOT unlock --mode full."
    fi

    log "============================================================"
    exit 0
fi


# ---------------------------------------------------------------------------
# MODE=full — HARD smoke gate BEFORE the first full-grid candidate
# ---------------------------------------------------------------------------

validate_smoke_gate_for_full

TOTAL=0
RAN=0
REUSED=0

for category in "${CATEGORIES[@]}"; do
    output_root="${CATEGORY_OUTPUT_ROOT[${category}]}"

    for row in "${CANDIDATE_ROWS[@]}"; do
        IFS=$'\t' read -r r d run_name <<< "${row}"

        TOTAL=$((TOTAL + 1))
        run_dir="${output_root}/${run_name}"

        log "------------------------------------------------------------"
        log "[$TOTAL] category=${category} candidate=${run_name}"
        log "r=${r} d=${d} seed=${SEED}"
        log "run_dir=${run_dir}"

        # The smoke candidate itself is a valid full-grid cell if it has the
        # exact same category/seed/split and complete REAL E2-E6 evidence.
        if [[ "${category}" == "${SMOKE_CATEGORY}" \
              && "${run_name}" == "${SMOKE_RUN_NAME}" ]]; then
            validate_real_evaluation \
                "${run_dir}" "${run_name}" "${category}" "${SEED}" \
                "${EXPECTED_SPLIT}" 1
            log "REUSE validated smoke cell in full grid."
            REUSED=$((REUSED + 1))
            continue
        fi

        before_complete=0
        if training_is_complete "${run_dir}" && evaluation_is_complete "${run_dir}"; then
            before_complete=1
        fi

        run_one_candidate \
            "${category}" \
            "${r}" \
            "${d}" \
            "${run_name}" \
            1

        if [[ "${DRY_RUN}" -eq 0 ]]; then
            if [[ "${before_complete}" -eq 1 ]]; then
                REUSED=$((REUSED + 1))
            else
                RAN=$((RAN + 1))
            fi
        fi
    done
done


# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------

log "============================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "FULL-SCREEN DRY-RUN PASS"
    log "Smoke gate + commands validated; no training/evaluation was executed."
else
    log "DAY-04 FULL SCREEN COMPLETE"
    log "New/continued cells : ${RAN}"
    log "Reused cells        : ${REUSED}"
    log "Expected cells      : ${TOTAL}"
    log "Every accepted cell has real E2-E6 artifacts."
    log "Next step           : scripts/eval_day04.py (E7 aggregation)"
fi

log "============================================================"
