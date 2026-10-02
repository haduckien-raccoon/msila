#!/usr/bin/env python3
"""
MS-ILA E7 — Day-04 candidate aggregation.

Purpose
-------
Aggregate the *already evaluated* experimental grid

    candidate x category x seed

into deterministic CSV/JSON tables.

This script does NOT:
    - recompute AU-PRO / SegF1 (E1-E3 own metric evaluation),
    - benchmark params / latency / VRAM (E4-E6 own efficiency),
    - average/rank/select candidates (E8 owns statistical selection),
    - compare architecture configs (E9 owns fairness audit).

Hard E7 principle
-----------------
The expected experiment grid is declared *before reading results* in a plan
JSON.  A Day-04 table is valid only when every expected
(candidate, category, seed) cell exists exactly once and all upstream reports
pass their locked contracts.

Default per-run layout
----------------------
Relative to --root:

    runs/{candidate}/{category}/seed_{seed}/
        metrics.json
        params.json
        efficiency.json

The layout is configurable in the plan using ``run_layout``.

Outputs
-------
    day04_category_runs.csv
    day04_category_runs.json
    day04_completeness.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLAN_SCHEMA_VERSION = "msila.day04.plan.v1"
OUTPUT_SCHEMA_VERSION = "msila.day04.aggregate.v1"
LOCKED_AUPRO_MAX_FPR = 0.05

MVTEC_AD2_CATEGORIES: tuple[str, ...] = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)


class Day04AggregationError(RuntimeError):
    """Raised when the declared Day-04 grid or an upstream report is invalid."""


@dataclass(frozen=True)
class Plan:
    split: str
    metric_protocol_version: str
    candidates: tuple[str, ...]
    categories: tuple[str, ...]
    seeds: tuple[int, ...]
    run_layout: str
    metrics_filename: str
    params_filename: str
    efficiency_filename: str
    require_e3_qa: bool
    require_parameter_pass: bool
    require_efficiency_pass: bool
    require_single_efficiency_scope: bool
    raw: Mapping[str, Any]
    sha256: str


CSV_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "category",
    "seed",
    "split",
    "aupro_0.05",
    "seg_f1",
    "n_samples",
    "anomalous_samples",
    "normal_samples",
    "qa_status",
    "eval_macro_aupro_0.05",
    "eval_macro_seg_f1",
    "total_parameters",
    "trainable_parameters",
    "frozen_parameters",
    "parameter_scope",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_std_ms",
    "latency_stability_cv",
    "peak_vram_mib",
    "incremental_peak_vram_mib",
    "benchmark_scope_fingerprint",
    "metrics_path",
    "params_path",
    "efficiency_path",
)


def _nonempty_string(value: Any, *, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise Day04AggregationError(f"{field} must be a non-empty string")
    return value


def _positive_or_zero_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise Day04AggregationError(f"{field} must be an integer")
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise Day04AggregationError(f"{field} must be an integer") from exc
    if ivalue < 0 or ivalue != value:
        raise Day04AggregationError(f"{field} must be an integer >= 0")
    return ivalue


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise Day04AggregationError(f"{field} must be numeric")
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise Day04AggregationError(f"{field} must be numeric") from exc
    if not math.isfinite(fvalue):
        raise Day04AggregationError(f"{field} must be finite, got {value!r}")
    return fvalue


def _unit_interval(value: Any, *, field: str, tol: float = 1e-9) -> float:
    fvalue = _finite_float(value, field=field)
    if fvalue < -tol or fvalue > 1.0 + tol:
        raise Day04AggregationError(
            f"{field} must be in [0,1], got {fvalue}"
        )
    return min(1.0, max(0.0, fvalue))


def _unique_nonempty_strings(values: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Day04AggregationError(f"{field} must be a non-empty list")
    out = tuple(_nonempty_string(v, field=field) for v in values)
    if not out:
        raise Day04AggregationError(f"{field} must not be empty")
    if len(set(out)) != len(out):
        raise Day04AggregationError(f"{field} contains duplicates: {out}")
    return out


def _unique_seeds(values: Any) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Day04AggregationError("seeds must be a non-empty list")
    out: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise Day04AggregationError("seed values must be integers")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise Day04AggregationError(
                f"invalid seed {value!r}"
            ) from exc
        if seed != value:
            raise Day04AggregationError(f"seed must be integer, got {value!r}")
        out.append(seed)
    if not out:
        raise Day04AggregationError("seeds must not be empty")
    if len(set(out)) != len(out):
        raise Day04AggregationError(f"seeds contains duplicates: {out}")
    return tuple(out)


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_run_layout(layout: str) -> str:
    layout = _nonempty_string(layout, field="run_layout")
    formatter = string.Formatter()
    fields = {
        name
        for _, name, _, _ in formatter.parse(layout)
        if name is not None
    }
    required = {"candidate", "category", "seed"}
    missing = sorted(required.difference(fields))
    if missing:
        raise Day04AggregationError(
            "run_layout must contain placeholders "
            "{candidate}, {category}, {seed}; missing "
            + ", ".join(missing)
        )
    unknown = sorted(fields.difference(required))
    if unknown:
        raise Day04AggregationError(
            f"run_layout contains unsupported placeholders: {unknown}"
        )
    return layout


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, Mapping):
        raise Day04AggregationError("plan root must be a JSON object")

    if raw.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise Day04AggregationError(
            f"plan.schema_version must be {PLAN_SCHEMA_VERSION!r}"
        )

    candidates = _unique_nonempty_strings(
        raw.get("candidates"), field="candidates"
    )
    categories = _unique_nonempty_strings(
        raw.get("categories"), field="categories"
    )
    unknown_categories = sorted(set(categories).difference(MVTEC_AD2_CATEGORIES))
    if unknown_categories:
        raise Day04AggregationError(
            f"Unknown MVTec AD 2 categories in plan: {unknown_categories}"
        )

    seeds = _unique_seeds(raw.get("seeds"))
    split = _nonempty_string(raw.get("split"), field="split")
    metric_protocol_version = _nonempty_string(
        raw.get("metric_protocol_version"),
        field="metric_protocol_version",
    )
    run_layout = _validate_run_layout(
        raw.get(
            "run_layout",
            "runs/{candidate}/{category}/seed_{seed}",
        )
    )

    required_files = raw.get("required_files", {})
    if not isinstance(required_files, Mapping):
        raise Day04AggregationError("required_files must be a JSON object")

    metrics_filename = _nonempty_string(
        required_files.get("metrics", "metrics.json"),
        field="required_files.metrics",
    )
    params_filename = _nonempty_string(
        required_files.get("params", "params.json"),
        field="required_files.params",
    )
    efficiency_filename = _nonempty_string(
        required_files.get("efficiency", "efficiency.json"),
        field="required_files.efficiency",
    )

    for name, filename in (
        ("metrics", metrics_filename),
        ("params", params_filename),
        ("efficiency", efficiency_filename),
    ):
        p = Path(filename)
        if p.is_absolute() or ".." in p.parts:
            raise Day04AggregationError(
                f"required_files.{name} must be a safe relative filename/path"
            )

    def flag(name: str, default: bool) -> bool:
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise Day04AggregationError(f"{name} must be boolean")
        return value

    return Plan(
        split=split,
        metric_protocol_version=metric_protocol_version,
        candidates=candidates,
        categories=categories,
        seeds=seeds,
        run_layout=run_layout,
        metrics_filename=metrics_filename,
        params_filename=params_filename,
        efficiency_filename=efficiency_filename,
        require_e3_qa=flag("require_e3_qa", True),
        require_parameter_pass=flag("require_parameter_pass", True),
        require_efficiency_pass=flag("require_efficiency_pass", True),
        require_single_efficiency_scope=flag(
            "require_single_efficiency_scope", True
        ),
        raw=dict(raw),
        sha256=_canonical_json_sha256(raw),
    )


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise Day04AggregationError(f"Missing {label}: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise Day04AggregationError(
            f"Invalid JSON in {label}: {path}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise Day04AggregationError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_mapping(
    obj: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    value = obj.get(key)
    if not isinstance(value, Mapping):
        raise Day04AggregationError(f"{label}.{key} must be an object")
    return value


def _validate_metrics(
    payload: Mapping[str, Any],
    *,
    plan: Plan,
    category: str,
    path: Path,
) -> dict[str, Any]:
    label = f"metrics[{path}]"

    if payload.get("schema_version") != "msila-evaluator-v1":
        raise Day04AggregationError(
            f"{label}: unexpected schema_version={payload.get('schema_version')!r}"
        )
    if str(payload.get("metric_protocol_version")) != plan.metric_protocol_version:
        raise Day04AggregationError(
            f"{label}: metric_protocol_version mismatch"
        )
    if str(payload.get("dataset")) != "mvtec_ad2":
        raise Day04AggregationError(f"{label}: dataset must be 'mvtec_ad2'")
    if str(payload.get("split")) != plan.split:
        raise Day04AggregationError(
            f"{label}: split={payload.get('split')!r}, expected {plan.split!r}"
        )

    categories = payload.get("categories")
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise Day04AggregationError(f"{label}.categories must be a list")

    metric_categories = tuple(str(x) for x in categories)
    if metric_categories != (category,):
        raise Day04AggregationError(
            f"{label}: E7 expects one single-class category per run; "
            f"expected [{category!r}], got {list(metric_categories)!r}"
        )

    n_categories = _positive_or_zero_int(
        payload.get("n_categories"), field=f"{label}.n_categories"
    )
    if n_categories != 1:
        raise Day04AggregationError(
            f"{label}: n_categories must be 1 for single-class Day-04 runs"
        )

    settings = _require_mapping(payload, "settings", label=label)
    max_fpr = _finite_float(
        settings.get("aupro_max_fpr"),
        field=f"{label}.settings.aupro_max_fpr",
    )
    if not math.isclose(max_fpr, LOCKED_AUPRO_MAX_FPR, rel_tol=0.0, abs_tol=1e-12):
        raise Day04AggregationError(
            f"{label}: AU-PRO max FPR must be {LOCKED_AUPRO_MAX_FPR}, got {max_fpr}"
        )

    validation = _require_mapping(payload, "validation", label=label)
    if validation.get("status") != "PASS":
        raise Day04AggregationError(f"{label}: evaluator validation is not PASS")

    qa_status = "NOT_REQUIRED"
    if plan.require_e3_qa:
        qa = validation.get("anomaly_map_qa")
        if not isinstance(qa, Mapping):
            raise Day04AggregationError(
                f"{label}: E3 anomaly_map_qa summary is required but missing"
            )
        if qa.get("status") != "PASS":
            raise Day04AggregationError(
                f"{label}: anomaly_map_qa status is not PASS"
            )
        valid_fraction = _finite_float(
            qa.get("valid_fraction"),
            field=f"{label}.validation.anomaly_map_qa.valid_fraction",
        )
        if not math.isclose(valid_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise Day04AggregationError(
                f"{label}: anomaly_map_qa valid_fraction must be 1.0"
            )
        qa_status = "PASS"

    metrics = _require_mapping(payload, "metrics", label=label)
    au = _require_mapping(metrics, "aupro_0.05", label=f"{label}.metrics")
    sf = _require_mapping(metrics, "seg_f1", label=f"{label}.metrics")

    au_per = _require_mapping(au, "per_category", label=f"{label}.metrics.aupro_0.05")
    sf_per = _require_mapping(sf, "per_category", label=f"{label}.metrics.seg_f1")

    if set(au_per.keys()) != {category}:
        raise Day04AggregationError(
            f"{label}: AU-PRO per_category keys must be exactly {{{category!r}}}"
        )
    if set(sf_per.keys()) != {category}:
        raise Day04AggregationError(
            f"{label}: SegF1 per_category keys must be exactly {{{category!r}}}"
        )

    aupro = _unit_interval(
        au_per[category],
        field=f"{label}.metrics.aupro_0.05.per_category.{category}",
    )
    au_macro = _unit_interval(
        au.get("macro"),
        field=f"{label}.metrics.aupro_0.05.macro",
    )

    sf_cat = sf_per[category]
    if not isinstance(sf_cat, Mapping):
        raise Day04AggregationError(
            f"{label}.metrics.seg_f1.per_category.{category} must be an object"
        )
    segf1 = _unit_interval(
        sf_cat.get("f1"),
        field=f"{label}.metrics.seg_f1.per_category.{category}.f1",
    )
    sf_macro = _unit_interval(
        sf.get("macro"),
        field=f"{label}.metrics.seg_f1.macro",
    )

    # With one category per run, macro must equal that category exactly.
    if not math.isclose(aupro, au_macro, rel_tol=0.0, abs_tol=1e-12):
        raise Day04AggregationError(
            f"{label}: single-category AU-PRO macro != category value"
        )
    if not math.isclose(segf1, sf_macro, rel_tol=0.0, abs_tol=1e-12):
        raise Day04AggregationError(
            f"{label}: single-category SegF1 macro != category value"
        )

    counts = _require_mapping(payload, "counts", label=label)
    counts_per = _require_mapping(
        counts, "per_category", label=f"{label}.counts"
    )
    if set(counts_per.keys()) != {category}:
        raise Day04AggregationError(
            f"{label}: counts.per_category keys must be exactly {{{category!r}}}"
        )
    cat_counts = counts_per[category]
    if not isinstance(cat_counts, Mapping):
        raise Day04AggregationError(
            f"{label}.counts.per_category.{category} must be an object"
        )

    n_samples = _positive_or_zero_int(
        cat_counts.get("samples"),
        field=f"{label}.counts.{category}.samples",
    )
    anomalous_samples = _positive_or_zero_int(
        cat_counts.get("anomalous_samples"),
        field=f"{label}.counts.{category}.anomalous_samples",
    )
    normal_samples = _positive_or_zero_int(
        cat_counts.get("normal_samples"),
        field=f"{label}.counts.{category}.normal_samples",
    )

    if anomalous_samples + normal_samples != n_samples:
        raise Day04AggregationError(
            f"{label}: anomalous_samples + normal_samples != samples"
        )
    if _positive_or_zero_int(
        payload.get("n_samples"),
        field=f"{label}.n_samples",
    ) != n_samples:
        raise Day04AggregationError(
            f"{label}: top-level n_samples disagrees with category count"
        )

    return {
        "aupro_0.05": aupro,
        "seg_f1": segf1,
        "n_samples": n_samples,
        "anomalous_samples": anomalous_samples,
        "normal_samples": normal_samples,
        "qa_status": qa_status,
        "eval_macro_aupro_0.05": au_macro,
        "eval_macro_seg_f1": sf_macro,
    }


def _normalize_params(
    payload: Mapping[str, Any],
    *,
    candidate: str,
    path: Path,
    require_pass: bool,
) -> dict[str, Any]:
    label = f"params[{path}]"
    schema = payload.get("schema_version")

    if schema == "msila.e4.manual_check.v1":
        if require_pass and payload.get("status") != "PASS":
            raise Day04AggregationError(f"{label}: manual-check status is not PASS")
        if str(payload.get("candidate_id")) != candidate:
            raise Day04AggregationError(
                f"{label}: candidate_id mismatch; "
                f"got {payload.get('candidate_id')!r}, expected {candidate!r}"
            )
        report = payload.get("report")
        if not isinstance(report, Mapping):
            raise Day04AggregationError(f"{label}.report must be an object")
        manual_check = payload.get("manual_check")
        if require_pass:
            if not isinstance(manual_check, Mapping):
                raise Day04AggregationError(
                    f"{label}: manual_check required but missing"
                )
            required_true = (
                "total_match",
                "trainable_match",
                "component_match",
            )
            if any(manual_check.get(k) is not True for k in required_true):
                raise Day04AggregationError(
                    f"{label}: E4 manual count did not fully PASS"
                )
            if int(manual_check.get("unexpected_parameters", -1)) != 0:
                raise Day04AggregationError(
                    f"{label}: unexpected parameters detected"
                )
    elif schema == "msila.e4.params.v1":
        report = payload
        if require_pass:
            internal = report.get("internal_consistency")
            if not isinstance(internal, Mapping) or internal.get("status") != "PASS":
                raise Day04AggregationError(
                    f"{label}: internal parameter consistency is not PASS"
                )
        if str(report.get("candidate_id")) != candidate:
            raise Day04AggregationError(
                f"{label}: candidate_id mismatch; "
                f"got {report.get('candidate_id')!r}, expected {candidate!r}"
            )
    else:
        raise Day04AggregationError(
            f"{label}: unsupported schema_version={schema!r}"
        )

    totals = report.get("totals")
    if not isinstance(totals, Mapping):
        raise Day04AggregationError(f"{label}: totals must be an object")

    total = _positive_or_zero_int(
        totals.get("total_parameters"),
        field=f"{label}.total_parameters",
    )
    trainable = _positive_or_zero_int(
        totals.get("trainable_parameters"),
        field=f"{label}.trainable_parameters",
    )
    frozen = _positive_or_zero_int(
        totals.get("frozen_parameters"),
        field=f"{label}.frozen_parameters",
    )

    if total != trainable + frozen:
        raise Day04AggregationError(
            f"{label}: total_parameters != trainable + frozen"
        )

    scope = _nonempty_string(report.get("scope"), field=f"{label}.scope")

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "parameter_scope": scope,
    }


def _validate_efficiency(
    payload: Mapping[str, Any],
    *,
    candidate: str,
    path: Path,
    require_pass: bool,
) -> dict[str, Any]:
    label = f"efficiency[{path}]"

    if payload.get("schema_version") != "msila.e5_e6.efficiency.v1":
        raise Day04AggregationError(
            f"{label}: unsupported schema_version={payload.get('schema_version')!r}"
        )
    if str(payload.get("candidate_id")) != candidate:
        raise Day04AggregationError(
            f"{label}: candidate_id mismatch; "
            f"got {payload.get('candidate_id')!r}, expected {candidate!r}"
        )
    if require_pass and payload.get("status") != "PASS":
        raise Day04AggregationError(f"{label}: combined E5/E6 status is not PASS")

    fingerprint = _nonempty_string(
        payload.get("scope_fingerprint_sha256"),
        field=f"{label}.scope_fingerprint_sha256",
    )
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint.lower()):
        raise Day04AggregationError(
            f"{label}: scope fingerprint is not a SHA-256 hex digest"
        )

    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise Day04AggregationError(f"{label}.scope must be an object")
    if str(scope.get("fingerprint_sha256")) != fingerprint:
        raise Day04AggregationError(
            f"{label}: top-level and scope fingerprints disagree"
        )

    latency = payload.get("latency")
    if not isinstance(latency, Mapping):
        raise Day04AggregationError(f"{label}.latency must be an object")
    if require_pass and latency.get("status") != "PASS":
        raise Day04AggregationError(f"{label}: latency status is not PASS")

    latency_ms = latency.get("latency_ms")
    stability = latency.get("stability")
    if not isinstance(latency_ms, Mapping) or not isinstance(stability, Mapping):
        raise Day04AggregationError(
            f"{label}: latency_ms/stability objects are required"
        )
    if require_pass and stability.get("status") != "PASS":
        raise Day04AggregationError(
            f"{label}: repeated-run latency stability is not PASS"
        )

    mean_ms = _finite_float(
        latency_ms.get("mean"),
        field=f"{label}.latency.mean",
    )
    median_ms = _finite_float(
        latency_ms.get("median"),
        field=f"{label}.latency.median",
    )
    p95_ms = _finite_float(
        latency_ms.get("p95"),
        field=f"{label}.latency.p95",
    )
    std_ms = _finite_float(
        latency_ms.get("std"),
        field=f"{label}.latency.std",
    )
    stability_cv = _finite_float(
        stability.get("round_median_cv"),
        field=f"{label}.latency.round_median_cv",
    )

    if min(mean_ms, median_ms, p95_ms, std_ms, stability_cv) < 0.0:
        raise Day04AggregationError(
            f"{label}: latency/stability values must be >= 0"
        )
    if p95_ms + 1e-12 < median_ms:
        raise Day04AggregationError(
            f"{label}: p95 latency cannot be smaller than median latency"
        )

    peak = payload.get("peak_vram")
    if not isinstance(peak, Mapping):
        raise Day04AggregationError(f"{label}.peak_vram must be an object")
    if require_pass and peak.get("status") != "PASS":
        raise Day04AggregationError(f"{label}: peak VRAM status is not PASS")

    memory = peak.get("memory")
    if not isinstance(memory, Mapping):
        raise Day04AggregationError(f"{label}.peak_vram.memory must be an object")

    peak_alloc = memory.get("peak_allocated")
    incr_alloc = memory.get("incremental_peak_allocated")
    if not isinstance(peak_alloc, Mapping) or not isinstance(incr_alloc, Mapping):
        raise Day04AggregationError(
            f"{label}: peak/incremental allocated memory objects are required"
        )

    peak_mib = _finite_float(
        peak_alloc.get("MiB"),
        field=f"{label}.peak_allocated.MiB",
    )
    incr_mib = _finite_float(
        incr_alloc.get("MiB"),
        field=f"{label}.incremental_peak_allocated.MiB",
    )
    if peak_mib < 0.0 or incr_mib < 0.0:
        raise Day04AggregationError(
            f"{label}: VRAM measurements must be >= 0"
        )
    if incr_mib > peak_mib + 1e-9:
        raise Day04AggregationError(
            f"{label}: incremental peak cannot exceed absolute peak"
        )

    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p95_ms": p95_ms,
        "latency_std_ms": std_ms,
        "latency_stability_cv": stability_cv,
        "peak_vram_mib": peak_mib,
        "incremental_peak_vram_mib": incr_mib,
        "benchmark_scope_fingerprint": fingerprint,
    }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _expected_grid(plan: Plan) -> list[tuple[str, str, int]]:
    return [
        (candidate, category, seed)
        for candidate in plan.candidates
        for category in plan.categories
        for seed in plan.seeds
    ]


def _run_dir(root: Path, plan: Plan, candidate: str, category: str, seed: int) -> Path:
    relative = plan.run_layout.format(
        candidate=candidate,
        category=category,
        seed=seed,
    )
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise Day04AggregationError(
            f"run_layout produced unsafe path: {relative!r}"
        )
    return root / p


def collect_day04(
    *,
    root: str | Path,
    plan: Plan,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    expected = _expected_grid(plan)
    expected_set = set(expected)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    missing: list[dict[str, Any]] = []

    for candidate, category, seed in expected:
        key = (candidate, category, seed)
        if key in seen:
            raise Day04AggregationError(f"duplicate expected grid key {key}")
        seen.add(key)

        run_dir = _run_dir(root, plan, candidate, category, seed)
        metrics_path = run_dir / plan.metrics_filename
        params_path = run_dir / plan.params_filename
        efficiency_path = run_dir / plan.efficiency_filename

        missing_files = [
            str(p)
            for p in (metrics_path, params_path, efficiency_path)
            if not p.is_file()
        ]
        if missing_files:
            missing.append(
                {
                    "candidate_id": candidate,
                    "category": category,
                    "seed": seed,
                    "run_dir": _relative_or_absolute(run_dir, root),
                    "missing_files": [
                        _relative_or_absolute(Path(p), root)
                        for p in missing_files
                    ],
                }
            )
            continue

        metrics_payload = _load_json(metrics_path, label="metrics")
        params_payload = _load_json(params_path, label="params")
        efficiency_payload = _load_json(efficiency_path, label="efficiency")

        metrics = _validate_metrics(
            metrics_payload,
            plan=plan,
            category=category,
            path=metrics_path,
        )
        params = _normalize_params(
            params_payload,
            candidate=candidate,
            path=params_path,
            require_pass=plan.require_parameter_pass,
        )
        efficiency = _validate_efficiency(
            efficiency_payload,
            candidate=candidate,
            path=efficiency_path,
            require_pass=plan.require_efficiency_pass,
        )

        row = {
            "candidate_id": candidate,
            "category": category,
            "seed": seed,
            "split": plan.split,
            **metrics,
            **params,
            **efficiency,
            "metrics_path": _relative_or_absolute(metrics_path, root),
            "params_path": _relative_or_absolute(params_path, root),
            "efficiency_path": _relative_or_absolute(efficiency_path, root),
        }
        rows.append(row)

    if missing:
        preview = "\n".join(
            f"  - {m['candidate_id']} / {m['category']} / seed={m['seed']}: "
            + ", ".join(m["missing_files"])
            for m in missing[:20]
        )
        suffix = "" if len(missing) <= 20 else f"\n  ... +{len(missing)-20} more"
        raise Day04AggregationError(
            f"Day-04 grid is incomplete: {len(missing)} missing run(s).\n"
            f"{preview}{suffix}"
        )

    row_keys = {
        (str(r["candidate_id"]), str(r["category"]), int(r["seed"]))
        for r in rows
    }
    if row_keys != expected_set:
        missing_keys = sorted(expected_set.difference(row_keys))
        unexpected_keys = sorted(row_keys.difference(expected_set))
        raise Day04AggregationError(
            "Grid-key mismatch after loading. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    # Candidate identity means architecture identity; therefore parameter counts
    # must not change with category or random seed.
    parameter_signatures: dict[str, set[tuple[Any, ...]]] = {}
    for row in rows:
        parameter_signatures.setdefault(row["candidate_id"], set()).add(
            (
                row["total_parameters"],
                row["trainable_parameters"],
                row["frozen_parameters"],
                row["parameter_scope"],
            )
        )
    inconsistent_candidates = {
        c: sorted(list(sigs), key=str)
        for c, sigs in parameter_signatures.items()
        if len(sigs) != 1
    }
    if inconsistent_candidates:
        raise Day04AggregationError(
            "Parameter report changed within the same candidate across "
            f"category/seed runs: {inconsistent_candidates}"
        )

    fingerprints = sorted(
        {str(r["benchmark_scope_fingerprint"]) for r in rows}
    )
    if plan.require_single_efficiency_scope and len(fingerprints) != 1:
        raise Day04AggregationError(
            "Efficiency benchmark scope differs across Day-04 runs. "
            f"Observed {len(fingerprints)} fingerprints: {fingerprints}"
        )

    completeness = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "PASS",
        "plan_sha256": plan.sha256,
        "split": plan.split,
        "expected_cells": len(expected),
        "observed_cells": len(rows),
        "missing_cells": 0,
        "duplicate_cells": 0,
        "candidates": list(plan.candidates),
        "categories": list(plan.categories),
        "seeds": list(plan.seeds),
        "expected_formula": (
            f"{len(plan.candidates)} candidates x "
            f"{len(plan.categories)} categories x "
            f"{len(plan.seeds)} seeds = {len(expected)} runs"
        ),
        "parameter_signature_consistent_within_candidate": True,
        "efficiency_scope_fingerprints": fingerprints,
        "single_efficiency_scope": len(fingerprints) == 1,
        "upstream_requirements": {
            "evaluator_validation_pass": True,
            "e3_qa_required": plan.require_e3_qa,
            "e4_parameter_pass_required": plan.require_parameter_pass,
            "e5_e6_efficiency_pass_required": plan.require_efficiency_pass,
            "aupro_max_fpr": LOCKED_AUPRO_MAX_FPR,
        },
    }

    return rows, completeness


def _atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_csv_dump(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(CSV_FIELDS),
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})
    os.replace(tmp, path)


def write_outputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    completeness: Mapping[str, Any],
    plan: Plan,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "day04_category_runs.csv"
    json_path = output_dir / "day04_category_runs.json"
    completeness_path = output_dir / "day04_completeness.json"

    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "plan_sha256": plan.sha256,
        "split": plan.split,
        "row_granularity": "one row per candidate x category x seed",
        "aggregation_policy": (
            "raw upstream values only; no across-seed mean/std and no ranking"
        ),
        "rows": list(rows),
    }

    _atomic_csv_dump(rows, csv_path)
    _atomic_json_dump(payload, json_path)
    _atomic_json_dump(dict(completeness), completeness_path)

    return {
        "csv": csv_path,
        "json": json_path,
        "completeness": completeness_path,
    }


def aggregate_day04(
    *,
    root: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    rows, completeness = collect_day04(root=root, plan=plan)
    paths = write_outputs(
        rows=rows,
        completeness=completeness,
        plan=plan,
        output_dir=output_dir,
    )
    return {
        "status": "PASS",
        "n_rows": len(rows),
        "plan_sha256": plan.sha256,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a complete MS-ILA Day-04 "
            "candidate x category x seed grid."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root containing run directories declared by plan.run_layout.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Pre-registered Day-04 aggregation plan JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for deterministic Day-04 CSV/JSON outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        result = aggregate_day04(
            root=args.root,
            plan_path=args.plan,
            output_dir=args.output_dir,
        )
    except (Day04AggregationError, FileNotFoundError) as exc:
        print(f"[E7 FAIL] {exc}", file=sys.stderr)
        return 2

    print("[E7 PASS]")
    print(f"rows={result['n_rows']}")
    print(f"plan_sha256={result['plan_sha256']}")
    for name, path in result["paths"].items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
