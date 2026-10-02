#!/usr/bin/env bash
#
# MS-ILA Day-04 — controlled full r×d Adapter screen orchestrator
#
# This script does NOT implement training logic. It only orchestrates the
# existing runner:
#
#   src/train/screen_adapter.py
#
# Scientific contract
# -------------------
# - r = bottleneck_dim
# - d = projection_dim
# - only (r,d) vary across candidates
# - category protocol is resolved once and then locked by screen_adapter.py
# - one seed is used for the whole screen
# - candidates run sequentially
# - existing incomplete runs are never overwritten
#
# Example:
#
#   bash scripts/run_day04.sh \
#       --categories fabric,vial,wallplugs \
#       --seed 42 \
#       --device cuda
#
# Resume only already-complete runs:
#
#   bash scripts/run_day04.sh \
#       --categories fabric,vial,wallplugs \
#       --seed 42 \
#       --device cuda \
#       --skip-existing
#
# IMPORTANT:
# If more than one category is requested, protocol.output_root MUST contain
# literal "{category}", for example:
#
#   output_root: /content/drive/MyDrive/.../msila_day04/screens/{category}
#
# The current Python runner formats {category}, but does not format {seed}.
# Therefore this shell orchestrator intentionally accepts exactly one seed.

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

CATEGORIES_CSV=""
SEED=""

SKIP_EXISTING=0
DRY_RUN=0

RUNNER="src/train/screen_adapter.py"


usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_day04.sh \
      --categories <cat1,cat2,...> \
      --seed <int> \
      [--device auto|cpu|cuda|cuda:0] \
      [--grid path/to/grid.yaml] \
      [--protocol path/to/protocol.yaml] \
      [--skip-existing] \
      [--dry-run]

Required:
  --categories
      Comma-separated categories to screen.
      Example: fabric,vial,wallplugs

  --seed
      One integer seed shared by every candidate/category run.

Optional:
  --device
      Device forwarded to screen_adapter.py. Default: auto

  --grid
      Locked Day-04 r×d grid.
      Default: configs/day04_adapter_grid.yaml

  --protocol
      Fixed Day-04 training protocol.
      Default: configs/day04_train_protocol.yaml

  --skip-existing
      Skip an existing candidate ONLY if the run is complete:
        config.yaml
        best.pt
        train_log.csv
        predictions/manifest.jsonl
      An incomplete existing run is always treated as an error.

  --dry-run
      Run preflight checks and print commands without training.

Notes:
  - This script runs candidates sequentially.
  - It never changes LR, loss, split, Fusion, Decoder, augmentation, etc.
  - It does not support multiple seeds in one invocation because the current
    screen runner does not include seed in run_dir/output_root formatting.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --categories)
            [[ $# -ge 2 ]] || die "--categories requires a value"
            CATEGORIES_CSV="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || die "--seed requires a value"
            SEED="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || die "--device requires a value"
            DEVICE="$2"
            shift 2
            ;;
        --grid)
            [[ $# -ge 2 ]] || die "--grid requires a value"
            GRID="$2"
            shift 2
            ;;
        --protocol)
            [[ $# -ge 2 ]] || die "--protocol requires a value"
            PROTOCOL="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

[[ -n "${CATEGORIES_CSV}" ]] || die "--categories is required"
[[ -n "${SEED}" ]] || die "--seed is required"

[[ "${SEED}" =~ ^-?[0-9]+$ ]] || die "--seed must be an integer, got: ${SEED}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || die "Python executable not found: ${PYTHON_BIN}"

[[ -f "${RUNNER}" ]] \
    || die "missing runner: ${RUNNER}"

[[ -f "${GRID}" ]] \
    || die "missing grid: ${GRID}"

[[ -f "${PROTOCOL}" ]] \
    || die "missing protocol: ${PROTOCOL}"


IFS=',' read -r -a RAW_CATEGORIES <<< "${CATEGORIES_CSV}"

CATEGORIES=()
declare -A SEEN_CATEGORY=()

for raw in "${RAW_CATEGORIES[@]}"; do
    # Category tokens are identifiers; whitespace is never meaningful here.
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
)

for name in required:
    importlib.import_module(name)

print("[DAY04] Python imports: PASS")
PY


# ---------------------------------------------------------------------------
# Grid preflight
#
# Read candidates from the locked grid instead of duplicating the 9 pairs in
# this shell script. This prevents candidate drift between YAML and launcher.
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

log "Grid candidates: ${#CANDIDATE_ROWS[@]}"


# ---------------------------------------------------------------------------
# Protocol preflight
#
# Use screen_adapter.resolve_protocol() itself. This catches:
#   - REPLACE_ME / null scientific settings
#   - invalid epochs / batch size
#   - non-frozen backbone
#   - invalid checkpoint mode
#
# For multiple categories, output_root must contain {category}; otherwise
# candidate directories from category 2 would collide with category 1.
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
        "contain literal '{category}'. This would cause run-directory "
        "collisions. Example: outputs/day04/{category}"
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
# Existing-run safety
# ---------------------------------------------------------------------------

run_is_complete() {
    local run_dir="$1"

    [[ -s "${run_dir}/config.yaml" ]] \
        && [[ -s "${run_dir}/best.pt" ]] \
        && [[ -s "${run_dir}/train_log.csv" ]] \
        && [[ -s "${run_dir}/predictions/manifest.jsonl" ]]
}


run_is_nonempty() {
    local run_dir="$1"

    [[ -d "${run_dir}" ]] \
        && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}


# ---------------------------------------------------------------------------
# Summary before execution
# ---------------------------------------------------------------------------

log "============================================================"
log "MS-ILA Day-04 controlled r×d screen"
log "============================================================"
log "Project root : ${PROJECT_ROOT}"
log "Grid         : ${GRID}"
log "Protocol     : ${PROTOCOL}"
log "Runner       : ${RUNNER}"
log "Device       : ${DEVICE}"
log "Seed         : ${SEED}"
log "Categories   : ${CATEGORIES[*]}"
log "Candidates   : ${#CANDIDATE_ROWS[@]}"
log "Total runs   : $((${#CATEGORIES[@]} * ${#CANDIDATE_ROWS[@]}))"
log "Skip existing: ${SKIP_EXISTING}"
log "Dry run      : ${DRY_RUN}"
log "============================================================"


# ---------------------------------------------------------------------------
# Full sequential screen
# ---------------------------------------------------------------------------

TOTAL=0
RAN=0
SKIPPED=0

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

        if run_is_nonempty "${run_dir}"; then
            if run_is_complete "${run_dir}"; then
                if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
                    log "SKIP complete existing run: ${run_dir}"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                die \
                    "complete run already exists: ${run_dir}. " \
                    "Use --skip-existing only when intentionally resuming."
            fi

            die \
                "existing run is incomplete: ${run_dir}. " \
                "Refusing to overwrite or silently resume partial state."
        fi

        cmd=(
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

        printf '[DAY04] command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'

        if [[ "${DRY_RUN}" -eq 1 ]]; then
            continue
        fi

        "${cmd[@]}"
        RAN=$((RAN + 1))

        # Postcondition: one successful runner invocation must leave the exact
        # artifact set required for downstream Day-04 QA.
        if ! run_is_complete "${run_dir}"; then
            die \
                "runner returned success but required artifacts are missing: " \
                "${run_dir}"
        fi

        log "PASS candidate=${run_name} category=${category}"
    done
done


# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------

log "============================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN PASS"
    log "Commands validated/printed; no training was executed."
else
    log "DAY-04 SCREEN COMPLETE"
    log "Executed : ${RAN}"
    log "Skipped  : ${SKIPPED}"
    log "Expected : ${TOTAL}"
fi

log "============================================================"
