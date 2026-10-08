#!/usr/bin/env python3
"""Create a metric-based Day-5 representation lock after validated aggregation."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from .aggregate_week06_multiscale import (
        CANDIDATES, FIELDS, aggregate, numeric, read_csv, read_json, require, sha256,
    )
except ImportError:
    from aggregate_week06_multiscale import (
        CANDIDATES, FIELDS, aggregate, numeric, read_csv, read_json, require, sha256,
    )

REPRESENTATIONS = {
    "R0": "deep_only",
    "R1": "multi_layer_local",
    "R2": "multi_layer_local_context",
}
# Numerical tie tolerance, not a statistical/practical effect-size threshold.
TIE_TOLERANCE = 1e-12


def check_lock_output(path):
    path = Path(path)
    require(path.suffix.lower() in {".yaml", ".yml"}, "Lock output must end in .yaml/.yml")
    require(not path.exists(), f"Lock already exists: {path}; choose a new path")


def build_lock(result):
    """Use an aggregate() result; refuse test-set selection or incomplete evidence."""
    audit = result["provenance"]
    require(audit["schema"] == "msila.week06.aggregate.v1", "Unsupported aggregation schema")
    require(audit["split"] == "dev_synthetic",
            "Representation selection requires dev_synthetic metrics, not test-set metrics")
    require(len(result["summary"]) == 3
            and {r["candidate"] for r in result["summary"]} == set(CANDIDATES),
            "Lock requires all three candidates exactly once")
    rows = {r["candidate"]: r for r in result["summary"]}
    for c, row in rows.items():
        require(set(row) == set(FIELDS), f"{c}: missing/unexpected evidence fields")
        numeric(row["AU-PRO0.05"], "primary", score=True)
        numeric(row["params"], "params", integer=True)
        numeric(row["runtime_ms_per_batch"], "runtime")
        numeric(row["VRAM_MiB"], "VRAM")
        for field, support in (("tiny_AU-PRO0.05", "tiny_categories"),
                               ("boundary_AU-PRO0.05", "boundary_categories")):
            require((row[field] is None) == (not audit[support]), f"{c}: missing diagnostic evidence")
            if row[field] is not None:
                numeric(row[field], field, score=True)
    seeds = {r["seed"] for r in result["per_category"]}
    require(len(seeds) == 1, "Lock requires one common seed")
    require(len(result["per_category"]) == 3 * len(audit["categories"]), "Incomplete category evidence")
    best = max(rows[c]["AU-PRO0.05"] for c in CANDIDATES)
    selected = next(c for c in CANDIDATES if best - rows[c]["AU-PRO0.05"] <= TIE_TOLERANCE)

    def delta(previous, following, field):
        a, b = rows[previous][field], rows[following][field]
        return None if a is None or b is None else b - a

    def comparison(previous, following):
        primary_delta = delta(previous, following, "AU-PRO0.05")
        direction = "improved" if primary_delta > TIE_TOLERANCE else (
            "decreased" if primary_delta < -TIE_TOLERANCE else "tied"
        )
        return dict(
            delta_au_pro_0_05=primary_delta,
            primary_direction=direction,
            observed_localization_improvement=direction == "improved",
            delta_tiny=delta(previous, following, "tiny_AU-PRO0.05"),
            delta_boundary=delta(previous, following, "boundary_AU-PRO0.05"),
            delta_runtime_ms_per_batch=delta(previous, following, "runtime_ms_per_batch"),
            delta_vram_MiB=delta(previous, following, "VRAM_MiB"),
            delta_params=delta(previous, following, "params"),
        )

    comparisons = {
        "R0_to_R1": comparison("R0", "R1"),
        "R1_to_R2": comparison("R1", "R2"),
    }
    chosen = rows[selected]
    scope = audit["efficiency_scopes"][0]
    return dict(
        schema="msila.representation_lock.v1",
        selected=selected,
        representation=REPRESENTATIONS[selected],
        selection_rule=dict(
            primary="au_pro_0.05", direction="maximize",
            numerical_tie_tolerance=TIE_TOLERANCE,
            tie_order=list(CANDIDATES),
            diagnostics_role="reported evidence; not additional selection gates",
            cost_role="reported tradeoffs; not additional selection gates",
            visualization_used=False,
        ),
        evidence=dict(
            primary="au_pro_0.05", primary_value=chosen["AU-PRO0.05"],
            tiny=chosen["tiny_AU-PRO0.05"], boundary=chosen["boundary_AU-PRO0.05"],
            boundary_f1=audit["boundary_f1_macro"][selected], params=chosen["params"],
            runtime=chosen["runtime_ms_per_batch"], vram=chosen["VRAM_MiB"],
            metric_definitions=dict(tiny="tiny_au_pro_0.05", boundary="boundary_au_pro_0.05"),
            units=dict(primary="fraction [0,1]", tiny="fraction [0,1]", boundary="fraction [0,1]",
                       runtime="ms_per_batch", vram="MiB", params="trainable_parameter_elements"),
        ),
        candidates={c: dict(
            representation=REPRESENTATIONS[c], au_pro_0_05=rows[c]["AU-PRO0.05"],
            tiny=rows[c]["tiny_AU-PRO0.05"], boundary=rows[c]["boundary_AU-PRO0.05"],
            boundary_f1=audit["boundary_f1_macro"][c], params=rows[c]["params"],
            runtime_ms_per_batch=rows[c]["runtime_ms_per_batch"], vram_MiB=rows[c]["VRAM_MiB"],
        ) for c in CANDIDATES},
        comparisons=comparisons,
        scientific_assessment=dict(
            multi_layer_improves_deep_only=comparisons["R0_to_R1"]["observed_localization_improvement"],
            context_improves_multi_layer=comparisons["R1_to_R2"]["observed_localization_improvement"],
            both_steps_improve=all(r["observed_localization_improvement"] for r in comparisons.values()),
            inference="observed DEV macro improvement for one seed",
            significance_test_performed=False,
        ),
        evaluation_scope=dict(
            split=audit["split"], seed=next(iter(seeds)), categories=audit["categories"],
            tiny_categories=audit["tiny_categories"], boundary_categories=audit["boundary_categories"],
            tiny_excluded=audit["tiny_excluded"], boundary_excluded=audit["boundary_excluded"],
            inference_scope=scope["scope_name"], batch_size=scope["batch_size"],
            precision=scope["precision"], hardware=scope["extra"]["hardware"],
            output_hw=scope["extra"]["output_hw"],
        ),
        aggregation_rules={k: audit[k] for k in (
            "primary_macro", "diagnostic_macro", "runtime_aggregation", "vram_aggregation", "params_aggregation",
        )},
        provenance=dict(
            input_files_sha256=audit["input_files_sha256"], input_roles=audit["input_roles"],
            evaluation_manifest_sha256=audit["evaluation_manifest_sha256"],
            evaluator_sha256=audit["evaluator_sha256"],
            normalization_by_candidate=audit["normalization_by_candidate"],
        ),
    )


def write_lock(lock, output, summary_csv):
    check_lock_output(output)
    summary_csv = Path(summary_csv).resolve()
    files = (summary_csv, summary_csv.with_suffix(".per_category.csv"),
             summary_csv.with_suffix(".provenance.json"))
    lock["provenance"]["aggregate_files_sha256"] = {str(p): sha256(p) for p in files}
    lock["provenance"]["selection_script_sha256"] = sha256(Path(__file__))
    # Serialize before creating the file so a serialization error cannot leave a placeholder lock.
    content = yaml.safe_dump(lock, sort_keys=False, allow_unicode=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)


def load_verified_aggregate(summary_csv):
    """Revalidate original inputs and compare all published aggregation artifacts."""
    summary_csv = Path(summary_csv).resolve()
    saved_audit = read_json(summary_csv.with_suffix(".provenance.json"))
    require("input_roles" in saved_audit,
            "Provenance lacks input_roles; rerun the updated aggregator using new output paths")
    for path, expected in saved_audit["input_files_sha256"].items():
        require(sha256(path) == expected, f"Input changed after aggregation: {path}")
    roles = saved_audit["input_roles"]
    result = aggregate(roles["metrics_json"], roles["tiny_csv"], roles["boundary_csv"], roles["efficiency_csvs"])
    require(result["provenance"] == saved_audit, "Provenance disagrees with validated input reports")
    for path, expected_rows in ((summary_csv, result["summary"]),
                                (summary_csv.with_suffix(".per_category.csv"), result["per_category"])):
        actual = read_csv(path, tuple(expected_rows[0]))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in expected_rows]
        require(actual == expected, f"Published aggregation CSV was changed: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/week06_multiscale.csv"))
    parser.add_argument("--output", type=Path, default=Path("configs/representation_lock.yaml"))
    args = parser.parse_args()
    try:
        check_lock_output(args.output)
        result = load_verified_aggregate(args.summary_csv)
        lock = build_lock(result)
        write_lock(lock, args.output, args.summary_csv)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Representation lock failed: {exc}\n")
    print(f"Selected {lock['selected']} from AU-PRO0.05; wrote {args.output}")
    for name, comparison in lock["comparisons"].items():
        print(f"{name}: delta={comparison['delta_au_pro_0_05']:.6f}, {comparison['primary_direction']}")


if __name__ == "__main__":
    main()
