#!/usr/bin/env python3
"""
MS-ILA E8 — Adapter candidate analysis and pre-locked selection.

Input
-----
E7 outputs:
    day04_category_runs.json
    day04_completeness.json

Pre-locked rule:
    a JSON file created/committed before inspecting candidate results.

Output
------
    selection_report.json
    candidate_summary.csv
    selection_report.md

Scientific aggregation
----------------------
For candidate c, category k, seed s:

    A[c,k,s] = AU-PRO_0.05

First preserve the experimental units:
    macro_seed[c,s] = mean_k A[c,k,s]

Then summarize replication seeds:
    mean_c = mean_s macro_seed[c,s]
    std_c  = sample_std_s macro_seed[c,s]   (ddof=1)

Per-category delta against reference r is paired by seed:

    delta[c,k,s] = A[c,k,s] - A[r,k,s]

Using paired seed deltas avoids discarding the natural seed matching.

E8 intentionally does NOT implement:
    - config fairness audit (E9),
    - evaluator/efficiency regression tests (E10).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


E7_SCHEMA = "msila.day04.aggregate.v1"
E8_RULE_SCHEMA = "msila.e8.selection_rule.v1"
E8_REPORT_SCHEMA = "msila.e8.selection_report.v1"

PRIMARY_METRIC = "macro_aupro_mean"

_ALLOWED_TIE_METRICS = {
    "macro_aupro_mean",
    "macro_segf1_mean",
    "trainable_parameters",
    "total_parameters",
    "latency_median_ms_mean",
    "latency_p95_ms_mean",
    "peak_vram_mib_mean",
}

_MVTEC_AD2_CATEGORIES = {
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
}


class SelectionAnalysisError(RuntimeError):
    """Raised when E8 input, aggregation, or pre-locked rule is invalid."""


@dataclass(frozen=True)
class TieBreaker:
    metric: str
    direction: str
    tolerance: float


@dataclass(frozen=True)
class SelectionRule:
    rule_id: str
    locked_before_results: bool
    locked_at_utc: str
    expected_split: str
    expected_e7_plan_sha256: str | None
    reference_candidate: str
    min_seeds: int
    primary_tolerance: float
    category_improvement_epsilon: float
    min_macro_aupro_delta_vs_reference: float | None
    min_categories_improved_vs_reference: int | None
    max_trainable_parameters: int | None
    max_trainable_parameters_ratio_vs_reference: float | None
    max_latency_median_ms: float | None
    max_latency_median_ratio_vs_reference: float | None
    max_peak_vram_mib: float | None
    max_peak_vram_ratio_vs_reference: float | None
    tie_breakers: tuple[TieBreaker, ...]
    raw: Mapping[str, Any]
    sha256: str


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty(value: Any, *, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise SelectionAnalysisError(f"{field} must be a non-empty string")
    return value


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise SelectionAnalysisError(f"{field} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionAnalysisError(f"{field} must be numeric") from exc
    if not math.isfinite(value):
        raise SelectionAnalysisError(f"{field} must be finite")
    return value


def _optional_nonnegative_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    value = _finite_float(value, field=field)
    if value < 0:
        raise SelectionAnalysisError(f"{field} must be >= 0 or null")
    return value


def _optional_positive_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    value = _finite_float(value, field=field)
    if value <= 0:
        raise SelectionAnalysisError(f"{field} must be > 0 or null")
    return value


def _optional_nonnegative_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SelectionAnalysisError(f"{field} must be an integer or null")
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise SelectionAnalysisError(f"{field} must be an integer or null") from exc
    if ivalue != value or ivalue < 0:
        raise SelectionAnalysisError(f"{field} must be an integer >= 0 or null")
    return ivalue


def _parse_iso_utc(value: Any) -> str:
    raw = _nonempty(value, field="locked_at_utc")
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SelectionAnalysisError(
            "locked_at_utc must be ISO-8601, e.g. 2026-10-02T05:00:00Z"
        ) from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise SelectionAnalysisError("locked_at_utc must include a timezone")
    if dt.utcoffset().total_seconds() != 0:
        raise SelectionAnalysisError("locked_at_utc must be expressed in UTC")
    return raw


def _load_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SelectionAnalysisError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SelectionAnalysisError(f"{label} root must be a JSON object")
    return payload


def load_selection_rule(path: str | Path) -> SelectionRule:
    raw = _load_json(path, label="selection rule")
    if raw.get("schema_version") != E8_RULE_SCHEMA:
        raise SelectionAnalysisError(
            f"rule.schema_version must be {E8_RULE_SCHEMA!r}"
        )
    if raw.get("locked_before_results") is not True:
        raise SelectionAnalysisError(
            "E8 requires rule.locked_before_results=true. "
            "A post-hoc rule is not accepted as a locked selection protocol."
        )

    rule_id = _nonempty(raw.get("rule_id"), field="rule_id")
    locked_at_utc = _parse_iso_utc(raw.get("locked_at_utc"))
    expected_split = _nonempty(raw.get("expected_split"), field="expected_split")
    reference_candidate = _nonempty(
        raw.get("reference_candidate"),
        field="reference_candidate",
    )

    expected_hash = raw.get("expected_e7_plan_sha256")
    if expected_hash is not None:
        expected_hash = str(expected_hash).lower()
        if (
            len(expected_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_hash)
        ):
            raise SelectionAnalysisError(
                "expected_e7_plan_sha256 must be a 64-character SHA-256 hex "
                "digest or null"
            )

    min_seeds = raw.get("min_seeds", 2)
    if isinstance(min_seeds, bool) or not isinstance(min_seeds, int) or min_seeds < 2:
        raise SelectionAnalysisError("min_seeds must be an integer >= 2")

    selection = raw.get("selection")
    if not isinstance(selection, Mapping):
        raise SelectionAnalysisError("selection must be a JSON object")
    if selection.get("primary_metric") != PRIMARY_METRIC:
        raise SelectionAnalysisError(
            f"E8 v1 locks the primary metric to {PRIMARY_METRIC!r} "
            "(macro AU-PRO_0.05 mean across seeds)"
        )
    if selection.get("direction") != "max":
        raise SelectionAnalysisError("primary selection direction must be 'max'")

    primary_tolerance = _finite_float(
        selection.get("primary_tolerance", 0.0),
        field="selection.primary_tolerance",
    )
    if primary_tolerance < 0:
        raise SelectionAnalysisError("selection.primary_tolerance must be >= 0")

    raw_tie = selection.get("tie_breakers", [])
    if not isinstance(raw_tie, Sequence) or isinstance(raw_tie, (str, bytes)):
        raise SelectionAnalysisError("selection.tie_breakers must be a list")

    tie_breakers: list[TieBreaker] = []
    seen_metrics: set[str] = set()
    for i, item in enumerate(raw_tie):
        if not isinstance(item, Mapping):
            raise SelectionAnalysisError(
                f"selection.tie_breakers[{i}] must be an object"
            )
        metric = _nonempty(item.get("metric"), field=f"tie_breakers[{i}].metric")
        direction = _nonempty(
            item.get("direction"),
            field=f"tie_breakers[{i}].direction",
        )
        tolerance = _finite_float(
            item.get("tolerance", 0.0),
            field=f"tie_breakers[{i}].tolerance",
        )
        if metric not in _ALLOWED_TIE_METRICS:
            raise SelectionAnalysisError(
                f"unsupported tie-break metric {metric!r}; "
                f"allowed={sorted(_ALLOWED_TIE_METRICS)}"
            )
        if metric == PRIMARY_METRIC:
            raise SelectionAnalysisError(
                "primary metric must not be repeated as a tie-breaker"
            )
        if metric in seen_metrics:
            raise SelectionAnalysisError(f"duplicate tie-break metric {metric!r}")
        if direction not in {"min", "max"}:
            raise SelectionAnalysisError(
                f"tie-break direction must be min/max, got {direction!r}"
            )
        if tolerance < 0:
            raise SelectionAnalysisError("tie-break tolerance must be >= 0")
        tie_breakers.append(TieBreaker(metric, direction, tolerance))
        seen_metrics.add(metric)

    eligibility = raw.get("eligibility", {})
    if not isinstance(eligibility, Mapping):
        raise SelectionAnalysisError("eligibility must be a JSON object")

    category_eps = _finite_float(
        eligibility.get("category_improvement_epsilon", 0.0),
        field="eligibility.category_improvement_epsilon",
    )
    if category_eps < 0:
        raise SelectionAnalysisError(
            "eligibility.category_improvement_epsilon must be >= 0"
        )

    min_categories = _optional_nonnegative_int(
        eligibility.get("min_categories_improved_vs_reference"),
        field="eligibility.min_categories_improved_vs_reference",
    )

    return SelectionRule(
        rule_id=rule_id,
        locked_before_results=True,
        locked_at_utc=locked_at_utc,
        expected_split=expected_split,
        expected_e7_plan_sha256=expected_hash,
        reference_candidate=reference_candidate,
        min_seeds=min_seeds,
        primary_tolerance=primary_tolerance,
        category_improvement_epsilon=category_eps,
        min_macro_aupro_delta_vs_reference=_optional_nonnegative_float(
            eligibility.get("min_macro_aupro_delta_vs_reference"),
            field="eligibility.min_macro_aupro_delta_vs_reference",
        ),
        min_categories_improved_vs_reference=min_categories,
        max_trainable_parameters=_optional_nonnegative_int(
            eligibility.get("max_trainable_parameters"),
            field="eligibility.max_trainable_parameters",
        ),
        max_trainable_parameters_ratio_vs_reference=_optional_positive_float(
            eligibility.get("max_trainable_parameters_ratio_vs_reference"),
            field="eligibility.max_trainable_parameters_ratio_vs_reference",
        ),
        max_latency_median_ms=_optional_nonnegative_float(
            eligibility.get("max_latency_median_ms"),
            field="eligibility.max_latency_median_ms",
        ),
        max_latency_median_ratio_vs_reference=_optional_positive_float(
            eligibility.get("max_latency_median_ratio_vs_reference"),
            field="eligibility.max_latency_median_ratio_vs_reference",
        ),
        max_peak_vram_mib=_optional_nonnegative_float(
            eligibility.get("max_peak_vram_mib"),
            field="eligibility.max_peak_vram_mib",
        ),
        max_peak_vram_ratio_vs_reference=_optional_positive_float(
            eligibility.get("max_peak_vram_ratio_vs_reference"),
            field="eligibility.max_peak_vram_ratio_vs_reference",
        ),
        tie_breakers=tuple(tie_breakers),
        raw=dict(raw),
        sha256=_canonical_sha256(raw),
    )


def _sample_std(values: Sequence[float]) -> float | None:
    """Sample standard deviation (N-1 denominator); undefined for N < 2."""
    if len(values) < 2:
        return None
    return float(statistics.stdev(float(v) for v in values))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SelectionAnalysisError("cannot summarize an empty sequence")
    return float(statistics.fmean(float(v) for v in values))


def _mean_std(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": int(len(values)),
        "mean": _mean(values),
        "std": _sample_std(values),
    }


def _validate_e7(
    runs_payload: Mapping[str, Any],
    completeness: Mapping[str, Any],
    rule: SelectionRule,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[int, ...], str]:
    if runs_payload.get("schema_version") != E7_SCHEMA:
        raise SelectionAnalysisError(
            f"E7 runs schema must be {E7_SCHEMA!r}"
        )
    if completeness.get("schema_version") != E7_SCHEMA:
        raise SelectionAnalysisError(
            f"E7 completeness schema must be {E7_SCHEMA!r}"
        )
    if completeness.get("status") != "PASS":
        raise SelectionAnalysisError("E7 completeness status must be PASS")

    plan_hash_runs = str(runs_payload.get("plan_sha256", ""))
    plan_hash_comp = str(completeness.get("plan_sha256", ""))
    if plan_hash_runs != plan_hash_comp:
        raise SelectionAnalysisError(
            "E7 runs/completeness plan_sha256 mismatch"
        )
    if (
        rule.expected_e7_plan_sha256 is not None
        and plan_hash_runs != rule.expected_e7_plan_sha256
    ):
        raise SelectionAnalysisError(
            "E8 rule is bound to a different E7 plan SHA-256"
        )

    split_runs = str(runs_payload.get("split", ""))
    split_comp = str(completeness.get("split", ""))
    if split_runs != split_comp:
        raise SelectionAnalysisError("E7 runs/completeness split mismatch")
    if split_runs != rule.expected_split:
        raise SelectionAnalysisError(
            f"E7 split={split_runs!r} does not match locked E8 "
            f"expected_split={rule.expected_split!r}"
        )

    rows = runs_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SelectionAnalysisError("E7 rows must be a non-empty list")

    candidates_raw = completeness.get("candidates")
    categories_raw = completeness.get("categories")
    seeds_raw = completeness.get("seeds")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise SelectionAnalysisError("E7 completeness.candidates invalid")
    if not isinstance(categories_raw, list) or not categories_raw:
        raise SelectionAnalysisError("E7 completeness.categories invalid")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise SelectionAnalysisError("E7 completeness.seeds invalid")

    candidates = tuple(str(x) for x in candidates_raw)
    categories = tuple(str(x) for x in categories_raw)
    seeds = tuple(int(x) for x in seeds_raw)

    if len(set(candidates)) != len(candidates):
        raise SelectionAnalysisError("duplicate candidates in E7 completeness")
    if len(set(categories)) != len(categories):
        raise SelectionAnalysisError("duplicate categories in E7 completeness")
    if len(set(seeds)) != len(seeds):
        raise SelectionAnalysisError("duplicate seeds in E7 completeness")
    if any(c not in _MVTEC_AD2_CATEGORIES for c in categories):
        raise SelectionAnalysisError("unknown MVTec AD 2 category in E7 input")
    if len(seeds) < rule.min_seeds:
        raise SelectionAnalysisError(
            f"E8 rule requires at least {rule.min_seeds} seeds; got {len(seeds)}"
        )
    if rule.reference_candidate not in candidates:
        raise SelectionAnalysisError(
            f"reference_candidate={rule.reference_candidate!r} "
            "is not present in the E7 grid"
        )

    expected_cells = len(candidates) * len(categories) * len(seeds)
    if int(completeness.get("expected_cells", -1)) != expected_cells:
        raise SelectionAnalysisError(
            "E7 completeness.expected_cells does not match declared grid"
        )
    if int(completeness.get("observed_cells", -1)) != expected_cells:
        raise SelectionAnalysisError(
            "E7 observed_cells is not complete"
        )
    if int(completeness.get("missing_cells", -1)) != 0:
        raise SelectionAnalysisError("E7 still reports missing cells")
    if int(completeness.get("duplicate_cells", -1)) != 0:
        raise SelectionAnalysisError("E7 reports duplicate cells")

    expected_keys = {
        (candidate, category, seed)
        for candidate in candidates
        for category in categories
        for seed in seeds
    }
    row_keys: set[tuple[str, str, int]] = set()
    normalized_rows: list[dict[str, Any]] = []

    numeric_nonnegative = (
        "total_parameters",
        "trainable_parameters",
        "frozen_parameters",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_std_ms",
        "latency_stability_cv",
        "peak_vram_mib",
        "incremental_peak_vram_mib",
    )

    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SelectionAnalysisError(f"E7 rows[{i}] must be an object")
        candidate = _nonempty(row.get("candidate_id"), field=f"rows[{i}].candidate_id")
        category = _nonempty(row.get("category"), field=f"rows[{i}].category")
        if isinstance(row.get("seed"), bool):
            raise SelectionAnalysisError(f"rows[{i}].seed must be int")
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError) as exc:
            raise SelectionAnalysisError(f"rows[{i}].seed must be int") from exc

        key = (candidate, category, seed)
        if key in row_keys:
            raise SelectionAnalysisError(f"duplicate E7 row key {key}")
        row_keys.add(key)

        if str(row.get("split")) != rule.expected_split:
            raise SelectionAnalysisError(f"rows[{i}].split mismatch")

        aupro = _finite_float(row.get("aupro_0.05"), field=f"rows[{i}].aupro_0.05")
        segf1 = _finite_float(row.get("seg_f1"), field=f"rows[{i}].seg_f1")
        if not 0.0 <= aupro <= 1.0:
            raise SelectionAnalysisError(f"rows[{i}].aupro_0.05 outside [0,1]")
        if not 0.0 <= segf1 <= 1.0:
            raise SelectionAnalysisError(f"rows[{i}].seg_f1 outside [0,1]")
        if row.get("qa_status") != "PASS":
            raise SelectionAnalysisError(f"rows[{i}].qa_status is not PASS")

        normalized = dict(row)
        normalized["candidate_id"] = candidate
        normalized["category"] = category
        normalized["seed"] = seed
        normalized["aupro_0.05"] = aupro
        normalized["seg_f1"] = segf1

        for field in numeric_nonnegative:
            value = _finite_float(row.get(field), field=f"rows[{i}].{field}")
            if value < 0:
                raise SelectionAnalysisError(f"rows[{i}].{field} must be >= 0")
            normalized[field] = value

        normalized_rows.append(normalized)

    if row_keys != expected_keys:
        raise SelectionAnalysisError(
            "E7 row keys do not exactly match candidate x category x seed grid"
        )

    fingerprints = {
        str(r.get("benchmark_scope_fingerprint"))
        for r in normalized_rows
    }
    if "" in fingerprints or "None" in fingerprints or len(fingerprints) != 1:
        raise SelectionAnalysisError(
            "E8 requires one common E5/E6 benchmark scope fingerprint"
        )

    return normalized_rows, candidates, categories, seeds, plan_hash_runs


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    return {
        (str(r["candidate_id"]), str(r["category"]), int(r["seed"])): r
        for r in rows
    }


def _parameter_signature(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
) -> tuple[int, int, int, str]:
    subset = [r for r in rows if r["candidate_id"] == candidate]
    signatures = {
        (
            int(r["total_parameters"]),
            int(r["trainable_parameters"]),
            int(r["frozen_parameters"]),
            str(r["parameter_scope"]),
        )
        for r in subset
    }
    if len(signatures) != 1:
        raise SelectionAnalysisError(
            f"parameter signature changes within candidate {candidate!r}"
        )
    return next(iter(signatures))


def _candidate_summary(
    *,
    candidate: str,
    reference: str,
    categories: Sequence[str],
    seeds: Sequence[int],
    index: Mapping[tuple[str, str, int], Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_seed_aupro: dict[int, float] = {}
    per_seed_segf1: dict[int, float] = {}

    for seed in seeds:
        per_seed_aupro[seed] = _mean([
            float(index[(candidate, cat, seed)]["aupro_0.05"])
            for cat in categories
        ])
        per_seed_segf1[seed] = _mean([
            float(index[(candidate, cat, seed)]["seg_f1"])
            for cat in categories
        ])

    per_category: dict[str, Any] = {}
    categories_improved = 0
    for category in categories:
        values = [
            float(index[(candidate, category, seed)]["aupro_0.05"])
            for seed in seeds
        ]
        delta_values = [
            float(index[(candidate, category, seed)]["aupro_0.05"])
            - float(index[(reference, category, seed)]["aupro_0.05"])
            for seed in seeds
        ]
        delta_summary = _mean_std(delta_values)
        per_category[category] = {
            "aupro_0.05": _mean_std(values),
            "delta_vs_reference": delta_summary,
        }

    candidate_rows = [r for r in all_rows if r["candidate_id"] == candidate]
    total, trainable, frozen, parameter_scope = _parameter_signature(
        all_rows, candidate
    )

    efficiency = {
        "n_observations": len(candidate_rows),
        "latency_median_ms": _mean_std([
            float(r["latency_median_ms"]) for r in candidate_rows
        ]),
        "latency_p95_ms": _mean_std([
            float(r["latency_p95_ms"]) for r in candidate_rows
        ]),
        "peak_vram_mib": _mean_std([
            float(r["peak_vram_mib"]) for r in candidate_rows
        ]),
        "incremental_peak_vram_mib": _mean_std([
            float(r["incremental_peak_vram_mib"]) for r in candidate_rows
        ]),
        "benchmark_scope_fingerprint": str(
            candidate_rows[0]["benchmark_scope_fingerprint"]
        ),
    }

    macro_delta_by_seed = [
        per_seed_aupro[seed]
        - _mean([
            float(index[(reference, cat, seed)]["aupro_0.05"])
            for cat in categories
        ])
        for seed in seeds
    ]

    return {
        "candidate_id": candidate,
        "n_categories": len(categories),
        "n_seeds": len(seeds),
        "seed_macro_aupro_0.05": {
            str(seed): per_seed_aupro[seed] for seed in seeds
        },
        "macro_aupro_0.05": _mean_std(list(per_seed_aupro.values())),
        "macro_segf1": _mean_std(list(per_seed_segf1.values())),
        "macro_aupro_delta_vs_reference": _mean_std(macro_delta_by_seed),
        "parameters": {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "scope": parameter_scope,
        },
        "efficiency": efficiency,
        "per_category": per_category,
    }


def _selection_scalar(summary: Mapping[str, Any], metric: str) -> float:
    if metric == "macro_aupro_mean":
        return float(summary["macro_aupro_0.05"]["mean"])
    if metric == "macro_segf1_mean":
        return float(summary["macro_segf1"]["mean"])
    if metric == "trainable_parameters":
        return float(summary["parameters"]["trainable_parameters"])
    if metric == "total_parameters":
        return float(summary["parameters"]["total_parameters"])
    if metric == "latency_median_ms_mean":
        return float(summary["efficiency"]["latency_median_ms"]["mean"])
    if metric == "latency_p95_ms_mean":
        return float(summary["efficiency"]["latency_p95_ms"]["mean"])
    if metric == "peak_vram_mib_mean":
        return float(summary["efficiency"]["peak_vram_mib"]["mean"])
    raise SelectionAnalysisError(f"unsupported selection metric {metric!r}")


def _apply_eligibility(
    summaries: dict[str, dict[str, Any]],
    *,
    rule: SelectionRule,
    categories: Sequence[str],
) -> None:
    reference = summaries[rule.reference_candidate]
    ref_trainable = float(reference["parameters"]["trainable_parameters"])
    ref_latency = float(reference["efficiency"]["latency_median_ms"]["mean"])
    ref_vram = float(reference["efficiency"]["peak_vram_mib"]["mean"])

    for candidate, summary in summaries.items():
        reasons: list[str] = []
        mean_delta = float(
            summary["macro_aupro_delta_vs_reference"]["mean"]
        )

        improved = 0
        for category in categories:
            delta = float(
                summary["per_category"][category]["delta_vs_reference"]["mean"]
            )
            if delta > rule.category_improvement_epsilon:
                improved += 1

        if (
            rule.min_macro_aupro_delta_vs_reference is not None
            and mean_delta + 1e-15 < rule.min_macro_aupro_delta_vs_reference
        ):
            reasons.append(
                "macro AU-PRO delta below locked minimum "
                f"({mean_delta:.8f} < "
                f"{rule.min_macro_aupro_delta_vs_reference:.8f})"
            )

        if (
            rule.min_categories_improved_vs_reference is not None
            and improved < rule.min_categories_improved_vs_reference
        ):
            reasons.append(
                "improved categories below locked minimum "
                f"({improved} < {rule.min_categories_improved_vs_reference})"
            )

        trainable = float(summary["parameters"]["trainable_parameters"])
        latency = float(summary["efficiency"]["latency_median_ms"]["mean"])
        vram = float(summary["efficiency"]["peak_vram_mib"]["mean"])

        if (
            rule.max_trainable_parameters is not None
            and trainable > rule.max_trainable_parameters
        ):
            reasons.append(
                f"trainable parameters {int(trainable)} exceed locked maximum "
                f"{rule.max_trainable_parameters}"
            )
        if (
            rule.max_trainable_parameters_ratio_vs_reference is not None
            and (
                (math.inf if ref_trainable == 0 and trainable > 0 else
                 (1.0 if ref_trainable == 0 else trainable / ref_trainable))
                > rule.max_trainable_parameters_ratio_vs_reference
            )
        ):
            ratio = (
                math.inf if ref_trainable == 0 and trainable > 0
                else (1.0 if ref_trainable == 0 else trainable / ref_trainable)
            )
            reasons.append(
                f"trainable parameter ratio {ratio:.6g} exceeds locked maximum "
                f"{rule.max_trainable_parameters_ratio_vs_reference:.6g}"
            )
        if (
            rule.max_latency_median_ms is not None
            and latency > rule.max_latency_median_ms
        ):
            reasons.append(
                f"median latency {latency:.6g} ms exceeds locked maximum "
                f"{rule.max_latency_median_ms:.6g} ms"
            )
        if (
            rule.max_latency_median_ratio_vs_reference is not None
            and (
                (math.inf if ref_latency == 0 and latency > 0 else
                 (1.0 if ref_latency == 0 else latency / ref_latency))
                > rule.max_latency_median_ratio_vs_reference
            )
        ):
            ratio = (
                math.inf if ref_latency == 0 and latency > 0
                else (1.0 if ref_latency == 0 else latency / ref_latency)
            )
            reasons.append(
                f"median latency ratio {ratio:.6g} exceeds locked maximum "
                f"{rule.max_latency_median_ratio_vs_reference:.6g}"
            )
        if (
            rule.max_peak_vram_mib is not None
            and vram > rule.max_peak_vram_mib
        ):
            reasons.append(
                f"peak VRAM {vram:.6g} MiB exceeds locked maximum "
                f"{rule.max_peak_vram_mib:.6g} MiB"
            )
        if (
            rule.max_peak_vram_ratio_vs_reference is not None
            and (
                (math.inf if ref_vram == 0 and vram > 0 else
                 (1.0 if ref_vram == 0 else vram / ref_vram))
                > rule.max_peak_vram_ratio_vs_reference
            )
        ):
            ratio = (
                math.inf if ref_vram == 0 and vram > 0
                else (1.0 if ref_vram == 0 else vram / ref_vram)
            )
            reasons.append(
                f"peak VRAM ratio {ratio:.6g} exceeds locked maximum "
                f"{rule.max_peak_vram_ratio_vs_reference:.6g}"
            )

        summary["categories_improved_vs_reference"] = improved
        summary["eligibility"] = {
            "eligible": len(reasons) == 0,
            "reasons": reasons,
        }


def _shortlist_by_metric(
    candidates: Sequence[str],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    direction: str,
    tolerance: float,
) -> tuple[list[str], float]:
    values = {c: _selection_scalar(summaries[c], metric) for c in candidates}
    if direction == "max":
        best = max(values.values())
        kept = [c for c in candidates if best - values[c] <= tolerance]
    elif direction == "min":
        best = min(values.values())
        kept = [c for c in candidates if values[c] - best <= tolerance]
    else:
        raise SelectionAnalysisError(f"invalid direction {direction!r}")
    return sorted(kept), float(best)


def _select(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    rule: SelectionRule,
) -> dict[str, Any]:
    eligible = sorted(
        candidate
        for candidate, summary in summaries.items()
        if summary["eligibility"]["eligible"]
    )

    trace: list[dict[str, Any]] = []

    if not eligible:
        return {
            "status": "NO_ELIGIBLE_CANDIDATE",
            "selected_candidate": None,
            "eligible_candidates": [],
            "finalists": [],
            "trace": trace,
        }

    finalists, best_primary = _shortlist_by_metric(
        eligible,
        summaries,
        metric=PRIMARY_METRIC,
        direction="max",
        tolerance=rule.primary_tolerance,
    )
    trace.append({
        "stage": "primary",
        "metric": PRIMARY_METRIC,
        "direction": "max",
        "tolerance": rule.primary_tolerance,
        "best_value": best_primary,
        "remaining": list(finalists),
    })

    for breaker in rule.tie_breakers:
        if len(finalists) <= 1:
            break
        finalists, best = _shortlist_by_metric(
            finalists,
            summaries,
            metric=breaker.metric,
            direction=breaker.direction,
            tolerance=breaker.tolerance,
        )
        trace.append({
            "stage": "tie_break",
            "metric": breaker.metric,
            "direction": breaker.direction,
            "tolerance": breaker.tolerance,
            "best_value": best,
            "remaining": list(finalists),
        })

    if len(finalists) == 1:
        return {
            "status": "SELECTED",
            "selected_candidate": finalists[0],
            "eligible_candidates": eligible,
            "finalists": finalists,
            "trace": trace,
        }

    return {
        "status": "AMBIGUOUS_TIE",
        "selected_candidate": None,
        "eligible_candidates": eligible,
        "finalists": finalists,
        "trace": trace,
    }


def analyze(
    *,
    runs_payload: Mapping[str, Any],
    completeness_payload: Mapping[str, Any],
    rule: SelectionRule,
) -> dict[str, Any]:
    rows, candidates, categories, seeds, plan_hash = _validate_e7(
        runs_payload,
        completeness_payload,
        rule,
    )
    index = _index_rows(rows)

    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        summaries[candidate] = _candidate_summary(
            candidate=candidate,
            reference=rule.reference_candidate,
            categories=categories,
            seeds=seeds,
            index=index,
            all_rows=rows,
        )

    _apply_eligibility(
        summaries,
        rule=rule,
        categories=categories,
    )
    selection = _select(summaries, rule=rule)

    return {
        "schema_version": E8_REPORT_SCHEMA,
        "analysis_status": "PASS",
        "e7_plan_sha256": plan_hash,
        "selection_rule": {
            "rule_id": rule.rule_id,
            "locked_before_results": rule.locked_before_results,
            "locked_at_utc": rule.locked_at_utc,
            "sha256": rule.sha256,
            "expected_split": rule.expected_split,
            "reference_candidate": rule.reference_candidate,
            "primary_metric": PRIMARY_METRIC,
            "primary_tolerance": rule.primary_tolerance,
            "tie_breakers": [
                {
                    "metric": b.metric,
                    "direction": b.direction,
                    "tolerance": b.tolerance,
                }
                for b in rule.tie_breakers
            ],
        },
        "grid": {
            "candidates": list(candidates),
            "categories": list(categories),
            "seeds": list(seeds),
            "n_candidates": len(candidates),
            "n_categories": len(categories),
            "n_seeds": len(seeds),
        },
        "aggregation": {
            "macro_definition": (
                "unweighted category mean within each seed, then mean across seeds"
            ),
            "std_definition": "sample standard deviation across seeds (ddof=1)",
            "delta_definition": (
                "paired by seed: candidate score - reference score"
            ),
            "efficiency_summary": (
                "mean/sample-std over E7 efficiency observations for each candidate"
            ),
        },
        "candidate_summaries": summaries,
        "selection": selection,
        "limitations": [
            (
                "locked_before_results=true plus rule hash is an audit assertion; "
                "it does not cryptographically prove the rule existed before "
                "results unless the rule was externally timestamped/committed."
            ),
            (
                "E8 does not perform the E9 configuration fairness audit. "
                "Any selected candidate remains conditional on E9."
            ),
        ],
    }


def _atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_candidate_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = [
        "candidate_id",
        "eligible",
        "macro_aupro_mean",
        "macro_aupro_std",
        "macro_aupro_delta_vs_reference_mean",
        "macro_aupro_delta_vs_reference_std",
        "categories_improved_vs_reference",
        "macro_segf1_mean",
        "macro_segf1_std",
        "total_parameters",
        "trainable_parameters",
        "latency_median_ms_mean",
        "latency_median_ms_std",
        "latency_p95_ms_mean",
        "latency_p95_ms_std",
        "peak_vram_mib_mean",
        "peak_vram_mib_std",
        "eligibility_reasons",
    ]

    rows = []
    for candidate, summary in report["candidate_summaries"].items():
        rows.append({
            "candidate_id": candidate,
            "eligible": summary["eligibility"]["eligible"],
            "macro_aupro_mean": summary["macro_aupro_0.05"]["mean"],
            "macro_aupro_std": summary["macro_aupro_0.05"]["std"],
            "macro_aupro_delta_vs_reference_mean": (
                summary["macro_aupro_delta_vs_reference"]["mean"]
            ),
            "macro_aupro_delta_vs_reference_std": (
                summary["macro_aupro_delta_vs_reference"]["std"]
            ),
            "categories_improved_vs_reference": (
                summary["categories_improved_vs_reference"]
            ),
            "macro_segf1_mean": summary["macro_segf1"]["mean"],
            "macro_segf1_std": summary["macro_segf1"]["std"],
            "total_parameters": summary["parameters"]["total_parameters"],
            "trainable_parameters": summary["parameters"]["trainable_parameters"],
            "latency_median_ms_mean": (
                summary["efficiency"]["latency_median_ms"]["mean"]
            ),
            "latency_median_ms_std": (
                summary["efficiency"]["latency_median_ms"]["std"]
            ),
            "latency_p95_ms_mean": (
                summary["efficiency"]["latency_p95_ms"]["mean"]
            ),
            "latency_p95_ms_std": (
                summary["efficiency"]["latency_p95_ms"]["std"]
            ),
            "peak_vram_mib_mean": (
                summary["efficiency"]["peak_vram_mib"]["mean"]
            ),
            "peak_vram_mib_std": (
                summary["efficiency"]["peak_vram_mib"]["std"]
            ),
            "eligibility_reasons": " | ".join(
                summary["eligibility"]["reasons"]
            ),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summaries = report["candidate_summaries"]
    reference = report["selection_rule"]["reference_candidate"]
    selection = report["selection"]

    lines = [
        "# MS-ILA E8 — Adapter Selection Report",
        "",
        f"- **Analysis:** {report['analysis_status']}",
        f"- **Rule:** `{report['selection_rule']['rule_id']}`",
        f"- **Rule SHA-256:** `{report['selection_rule']['sha256']}`",
        f"- **Reference:** `{reference}`",
        f"- **Split:** `{report['selection_rule']['expected_split']}`",
        f"- **Selection status:** **{selection['status']}**",
        f"- **Selected candidate:** "
        + (
            f"`{selection['selected_candidate']}`"
            if selection["selected_candidate"] is not None
            else "None"
        ),
        "",
        "## Candidate summary",
        "",
        "| Candidate | Eligible | Macro AU-PRO0.05 mean±std | Δ vs ref mean±std | Improved cats | Trainable params | Median latency mean±std (ms) | Peak VRAM mean±std (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for candidate, s in summaries.items():
        au = s["macro_aupro_0.05"]
        delta = s["macro_aupro_delta_vs_reference"]
        lat = s["efficiency"]["latency_median_ms"]
        vram = s["efficiency"]["peak_vram_mib"]

        def pm(stat):
            std = "NA" if stat["std"] is None else f"{stat['std']:.6f}"
            return f"{stat['mean']:.6f} ± {std}"

        lines.append(
            "| "
            + " | ".join([
                candidate,
                "YES" if s["eligibility"]["eligible"] else "NO",
                pm(au),
                pm(delta),
                str(s["categories_improved_vs_reference"]),
                str(s["parameters"]["trainable_parameters"]),
                pm(lat),
                pm(vram),
            ])
            + " |"
        )

    lines += ["", "## Per-category AU-PRO delta vs reference", ""]
    categories = report["grid"]["categories"]
    header = "| Candidate | " + " | ".join(categories) + " |"
    sep = "|---|" + "|".join(["---:"] * len(categories)) + "|"
    lines += [header, sep]
    for candidate, s in summaries.items():
        vals = [
            f"{s['per_category'][cat]['delta_vs_reference']['mean']:.6f}"
            for cat in categories
        ]
        lines.append("| " + candidate + " | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Selection trace",
        "",
    ]
    if selection["trace"]:
        for step in selection["trace"]:
            lines.append(
                f"- `{step['stage']}` — {step['metric']} "
                f"({step['direction']}, tolerance={step['tolerance']}): "
                f"{', '.join(step['remaining'])}"
            )
    else:
        lines.append("- No eligible candidate reached the selection stage.")

    lines += [
        "",
        "## Interpretation",
        "",
        "The report applies only the pre-locked E8 rule. "
        "It does not perform the E9 fairness audit; selection is conditional on E9.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def run_analysis(
    *,
    runs_path: str | Path,
    completeness_path: str | Path,
    rule_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    runs = _load_json(runs_path, label="E7 runs")
    completeness = _load_json(completeness_path, label="E7 completeness")
    rule = load_selection_rule(rule_path)

    report = analyze(
        runs_payload=runs,
        completeness_payload=completeness,
        rule=rule,
    )

    output_dir = Path(output_dir)
    json_path = output_dir / "selection_report.json"
    csv_path = output_dir / "candidate_summary.csv"
    md_path = output_dir / "selection_report.md"

    _atomic_json_dump(report, json_path)
    _write_candidate_csv(report, csv_path)
    _write_markdown(report, md_path)

    return {
        "analysis_status": report["analysis_status"],
        "selection_status": report["selection"]["status"],
        "selected_candidate": report["selection"]["selected_candidate"],
        "rule_sha256": report["selection_rule"]["sha256"],
        "paths": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        },
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MS-ILA E8: summarize Day-04 results and apply a pre-locked adapter selection rule."
    )
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--completeness", type=Path, required=True)
    p.add_argument("--rule", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_analysis(
            runs_path=args.runs,
            completeness_path=args.completeness,
            rule_path=args.rule,
            output_dir=args.output_dir,
        )
    except (SelectionAnalysisError, FileNotFoundError) as exc:
        print(f"[E8 FAIL] {exc}", file=sys.stderr)
        return 2

    print("[E8 PASS]")
    print(f"selection_status={result['selection_status']}")
    print(f"selected_candidate={result['selected_candidate']}")
    print(f"rule_sha256={result['rule_sha256']}")
    for name, path in result["paths"].items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
