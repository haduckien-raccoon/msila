#!/usr/bin/env python3
"""Aggregate existing Day-5 metrics; never recompute or renormalize predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

CANDIDATES = ("R0", "R1", "R2")
FIELDS = (
    "candidate", "AU-PRO0.05", "tiny_AU-PRO0.05", "boundary_AU-PRO0.05",
    "params", "runtime_ms_per_batch", "VRAM_MiB",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, label, *, integer=False, score=False):
    require(not isinstance(value, bool) and value not in (None, ""), f"{label}: missing/invalid number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: invalid number") from exc
    require(math.isfinite(result) and result >= 0, f"{label}: must be finite and nonnegative")
    if score:
        require(result <= 1, f"{label}: must be in [0,1], not percent")
    if integer:
        require(result.is_integer(), f"{label}: must be an integer")
        return int(result)
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_hash(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{label}: invalid SHA-256")
    return value


def read_json(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"{path}: expected JSON object")
    return obj


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        require(len(header) == len(set(header)), f"{path}: duplicate CSV columns")
        require(set(required) <= set(header), f"{path}: missing columns {sorted(set(required) - set(header))}")
        rows = list(reader)
    require(rows, f"{path}: empty CSV")
    require(all(None not in r and None not in r.values() for r in rows), f"{path}: malformed CSV row")
    return rows


def index_rows(rows, categories, label):
    result = {}
    for row in rows:
        key = (row["candidate"], row["category"])
        require(key not in result, f"{label}: duplicate candidate/category {key}")
        result[key] = row
    expected = {(c, cat) for c in CANDIDATES for cat in categories}
    require(set(result) == expected,
            f"{label}: candidate/category coverage mismatch; missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}")
    return result


def identical(rows, fields, label):
    for field in fields:
        require(len({str(r[field]) for r in rows}) == 1, f"{label}: inconsistent {field}")


def diagnostic_value(row, column, status_column, count_column):
    count = numeric(row[count_column], count_column, integer=True)
    normals = numeric(row["n_normal_pixels"], "n_normal_pixels", integer=True)
    status, value = row[status_column], row[column]
    if status == "OK":
        require(count > 0 and normals > 0, f"{column}: OK requires GT regions and background")
        return numeric(value, column, score=True)
    require(status in {"NO_REGIONS", "NO_NORMAL_PIXELS"}, f"{column}: unexpected status {status}")
    require(value in (None, ""), f"{column}: undefined metric must be blank, not zero")
    require(count == 0 if status == "NO_REGIONS" else count > 0 and normals == 0,
            f"{column}: status disagrees with GT counts")
    return None


def aggregate(metrics_json, tiny_csv, boundary_csv, efficiency_csvs):
    inputs = {}

    def track(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha256(path)
        return path

    metrics_json, tiny_csv, boundary_csv = map(track, (metrics_json, tiny_csv, boundary_csv))
    primary = read_json(metrics_json)
    categories = primary["categories"]
    require(isinstance(categories, list) and categories
            and all(isinstance(c, str) and c.strip() == c and c for c in categories)
            and len(categories) == len(set(categories)), "Primary: invalid category list")
    require(primary["primary_metric"] == "aupro_0.05" and primary["max_fpr"] == 0.05,
            "Primary: require locked AU-PRO0.05")
    manifest_hash = checked_hash(primary["input_manifest_sha256"], "input_manifest_sha256")
    metric_hash = checked_hash(primary["evaluator_sha256"]["src/metrics/aupro.py"], "AU-PRO source")
    norms = primary["normalization_by_candidate"]
    require(set(norms) == set(CANDIDATES)
            and all(isinstance(n, str) and n.strip() for n in norms.values())
            and len(set(norms.values())) == 1, "Primary: inconsistent normalization")
    main = index_rows(primary["candidate_category"], categories, "Primary")
    for row in main.values():
        numeric(row["aupro_0.05"], "aupro_0.05", score=True)
        require(numeric(row["n_samples"], "n_samples", integer=True) > 0, "Primary: empty category")

    tiny_fields = (
        "candidate", "category", "split", "tiny_area_px", "tiny_rule", "connectivity", "max_fpr",
        "n_images", "n_tiny_regions", "n_non_tiny_regions", "n_normal_pixels",
        "tiny_aupro_0.05", "tiny_status", "normalization_protocol",
        "protocol_sha256", "manifest_sha256", "aupro_source_sha256",
    )
    boundary_fields = (
        "candidate", "category", "split", "boundary_mode", "band_width_px", "selection_rule",
        "connectivity", "prediction_threshold", "tolerance_px", "max_fpr", "n_images",
        "n_gt_boundary_regions", "n_normal_pixels", "boundary_aupro_0.05", "aupro_status",
        "boundary_f1", "boundary_f1_status", "normalization_protocol", "protocol_sha256", "manifest_sha256",
    )
    tiny = index_rows(read_csv(tiny_csv, tiny_fields), categories, "Tiny")
    boundary = index_rows(read_csv(boundary_csv, boundary_fields), categories, "Boundary")
    identical(tiny.values(), ("tiny_area_px", "tiny_rule", "connectivity", "protocol_sha256"), "Tiny")
    identical(boundary.values(), ("boundary_mode", "band_width_px", "selection_rule", "connectivity",
                                  "prediction_threshold", "tolerance_px", "protocol_sha256"), "Boundary")
    tiny_values, boundary_values, boundary_f1_values = {}, {}, {}
    for key in main:
        t, b = tiny[key], boundary[key]
        for label, row in (("Tiny", t), ("Boundary", b)):
            require(row["manifest_sha256"] == manifest_hash, f"{label}: different evaluation manifest")
            checked_hash(row["protocol_sha256"], f"{label} protocol")
            require(row["split"] == primary["split"] and row["normalization_protocol"] == norms[key[0]],
                    f"{label}: split/normalization mismatch")
            require(numeric(row["max_fpr"], "max_fpr") == 0.05
                    and numeric(row["connectivity"], "connectivity", integer=True) == 8,
                    f"{label}: FPR/connectivity mismatch")
            require(numeric(row["n_images"], "n_images", integer=True) == main[key]["n_samples"],
                    f"{label}: sample count mismatch")
        require(t["aupro_source_sha256"] == metric_hash, "Tiny: AU-PRO source differs from E2 lock")
        require(numeric(t["tiny_area_px"], "tiny_area_px", integer=True) > 0
                and t["tiny_rule"] == "area_px <= tiny_area_px", "Tiny: invalid tiny definition")
        require(b["boundary_mode"] in {"image_border_band", "provided_zone_mask"}
                and b["selection_rule"] == "component intersects zone", "Boundary: unsupported rule")
        if b["boundary_mode"] == "image_border_band":
            require(numeric(b["band_width_px"], "band_width_px", integer=True) > 0, "Boundary: invalid band")
        else:
            require(b["band_width_px"] == "", "Boundary: provided zone requires blank band_width_px")
        numeric(b["prediction_threshold"], "prediction_threshold", score=True)
        numeric(b["tolerance_px"], "tolerance_px", integer=True)
        require(numeric(t["n_normal_pixels"], "normal pixels", integer=True)
                == numeric(b["n_normal_pixels"], "normal pixels", integer=True), "Diagnostics: GT background mismatch")
        tiny_values[key] = diagnostic_value(t, "tiny_aupro_0.05", "tiny_status", "n_tiny_regions")
        boundary_values[key] = diagnostic_value(b, "boundary_aupro_0.05", "aupro_status", "n_gt_boundary_regions")
        if b["boundary_f1_status"] == "OK":
            require(numeric(b["n_gt_boundary_regions"], "boundary count", integer=True) > 0, "BF1: no GT")
            boundary_f1_values[key] = numeric(b["boundary_f1"], "boundary_f1", score=True)
        else:
            require(b["boundary_f1_status"] == "NO_GT_BOUNDARY_REGIONS"
                    and b["boundary_f1"] == ""
                    and numeric(b["n_gt_boundary_regions"], "boundary count", integer=True) == 0,
                    "BF1: invalid undefined status")
            boundary_f1_values[key] = None

    for cat in categories:
        identical([main[c, cat] for c in CANDIDATES], ("n_samples",), f"Primary/{cat}")
        identical([tiny[c, cat] for c in CANDIDATES],
                  ("n_tiny_regions", "n_non_tiny_regions", "n_normal_pixels", "tiny_status"), f"Tiny/{cat}")
        identical([boundary[c, cat] for c in CANDIDATES],
                  ("n_gt_boundary_regions", "n_normal_pixels", "aupro_status", "boundary_f1_status"), f"Boundary/{cat}")
        for c in CANDIDATES:
            require(int(boundary[c, cat]["n_gt_boundary_regions"])
                    <= int(tiny[c, cat]["n_tiny_regions"]) + int(tiny[c, cat]["n_non_tiny_regions"]),
                    "Boundary count exceeds total GT regions")

    efficiency_fields = (
        "candidate", "category", "seed", "trainable_params", "inference_ms_per_batch", "peak_vram_MiB",
        "gpu", "tile_resolution", "batch_size", "precision", "latency_warmup", "latency_iterations",
        "latency_rounds", "vram_warmup", "vram_iterations", "scope_sha256", "status",
    )
    efficiency_rows, signatures, scopes, measurement_protocols, efficiency_paths = [], [], [], [], []
    for csv_path in efficiency_csvs:
        csv_path = track(csv_path)
        efficiency_paths.append(str(csv_path))
        rows = read_csv(csv_path, efficiency_fields)
        scope = read_json(track(csv_path.parent / "efficiency_protocol.json"))
        scope_hash = checked_hash(scope["fingerprint_sha256"], "efficiency scope")
        payload = {k: v for k, v in scope.items() if k != "fingerprint_sha256"}
        actual_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
        require(actual_hash == scope_hash, "Efficiency: scope fingerprint is invalid")
        extra, benchmark = scope["extra"], scope["extra"]["benchmark"]
        require(str(scope["device"]).startswith("cuda:") and benchmark["use_inference_mode"] is True,
                "Efficiency: require CUDA inference mode")
        signatures.append({
            k: scope[k] for k in ("scope_name", "device", "batch_size", "precision", "input_signature", "pipeline_stages")
        } | {"extra": {k: v for k, v in extra.items()
                         if k not in {"sample_ids", "cache_manifest_sha256", "val_records_sha256"}}})
        scopes.append(scope)
        for row in rows:
            c = row["candidate"]
            require(c in CANDIDATES and row["status"] == "PASS", "Efficiency: candidate missing or status FAIL")
            require(row["scope_sha256"] == scope_hash, "Efficiency: CSV/scope hash mismatch")
            saved = read_json(track(csv_path.parent / f"{c}_efficiency.json"))
            report, params = saved["efficiency"], saved["parameters"]
            require(report["candidate_id"] == c and params["candidate_id"] == c
                    and report["status"] == "PASS" and report["scope"] == scope
                    and report["scope_fingerprint_sha256"] == scope_hash, "Efficiency: inconsistent candidate report")
            lat, mem = report["latency"], report["peak_vram"]
            measurement_protocols.append({"latency": lat["protocol"], "peak_vram": mem["protocol"]})
            require(lat["status"] == "PASS" and mem["status"] == "PASS"
                    and lat["protocol"]["cuda_sync_before_each_timing"] is True
                    and lat["protocol"]["cuda_sync_after_each_timing"] is True
                    and lat["protocol"]["inference_mode"] is True
                    and mem["protocol"]["inference_mode"] is True,
                    "Efficiency: invalid measurement procedure")
            require(numeric(lat["stability"]["round_median_cv"], "latency CV")
                    <= numeric(benchmark["stability_cv_threshold"], "CV threshold"), "Efficiency: latency unstable")
            for col, protocol_key in (("latency_warmup", "warmup_per_round"),
                                      ("latency_iterations", "iterations_per_round"), ("latency_rounds", "rounds")):
                require(numeric(row[col], col, integer=True) == lat["protocol"][protocol_key] == benchmark[col],
                        f"Efficiency: inconsistent {col}")
            for col, protocol_key in (("vram_warmup", "warmup"), ("vram_iterations", "measured_iterations")):
                require(numeric(row[col], col, integer=True) == mem["protocol"][protocol_key] == benchmark[col],
                        f"Efficiency: inconsistent {col}")
            require(row["gpu"] == extra["hardware"]["gpu"] and row["precision"] == scope["precision"]
                    and numeric(row["batch_size"], "batch_size", integer=True) == scope["batch_size"]
                    and row["tile_resolution"] == "x".join(str(v) for v in extra["output_hw"]),
                    "Efficiency: GPU/resolution/batch/precision mismatch")
            for col, expected in (
                ("trainable_params", params["totals"]["trainable_parameters"]),
                ("inference_ms_per_batch", lat["latency_ms"]["median"]),
                ("peak_vram_MiB", mem["memory"]["peak_allocated"]["MiB"]),
            ):
                actual = numeric(row[col], col, integer=col == "trainable_params")
                require(actual == numeric(expected, col, integer=col == "trainable_params"),
                        f"Efficiency: CSV/report mismatch for {col}")
            efficiency_rows.append(row)
    require(signatures and all(s == signatures[0] for s in signatures),
            "Efficiency: hardware/resolution/procedure differs across benchmark runs")
    require(all(p == measurement_protocols[0] for p in measurement_protocols),
            "Efficiency: actual latency/VRAM procedures differ across candidates/categories")
    eff = index_rows(efficiency_rows, categories, "Efficiency")
    identical(eff.values(), ("seed", "gpu", "tile_resolution", "batch_size", "precision",
                             "latency_warmup", "latency_iterations", "latency_rounds",
                             "vram_warmup", "vram_iterations"), "Efficiency")
    for row in eff.values():
        numeric(row["seed"], "seed", integer=True)
    for cat in categories:
        identical([eff[c, cat] for c in CANDIDATES], ("scope_sha256",), f"Efficiency/{cat}")

    tiny_cats = [cat for cat in categories if tiny_values["R0", cat] is not None]
    boundary_cats = [cat for cat in categories if boundary_values["R0", cat] is not None]
    bf1_cats = [cat for cat in categories if boundary_f1_values["R0", cat] is not None]

    def diagnostic_macro(values, support, candidate):
        return statistics.fmean(values[candidate, cat] for cat in support) if support else None

    summary, per_category = [], []
    for c in CANDIDATES:
        params = {numeric(eff[c, cat]["trainable_params"], "params", integer=True) for cat in categories}
        require(len(params) == 1, f"{c}: params differs across categories; do not average architectures")
        macro = statistics.fmean(numeric(main[c, cat]["aupro_0.05"], "AU-PRO", score=True) for cat in categories)
        if "macro_by_candidate" in primary:
            require(math.isclose(macro, numeric(primary["macro_by_candidate"][c], "E2 macro", score=True),
                                 rel_tol=1e-12, abs_tol=1e-12), "Primary: stored macro disagrees with category rows")
        summary.append(dict(zip(FIELDS, (
            c, macro, diagnostic_macro(tiny_values, tiny_cats, c),
            diagnostic_macro(boundary_values, boundary_cats, c), next(iter(params)),
            statistics.fmean(numeric(eff[c, cat]["inference_ms_per_batch"], "runtime") for cat in categories),
            max(numeric(eff[c, cat]["peak_vram_MiB"], "VRAM") for cat in categories),
        ))))
        for cat in categories:
            per_category.append(dict(
                candidate=c, category=cat, **{
                    "AU-PRO0.05": numeric(main[c, cat]["aupro_0.05"], "AU-PRO", score=True),
                    "tiny_AU-PRO0.05": tiny_values[c, cat], "boundary_AU-PRO0.05": boundary_values[c, cat],
                    "params": numeric(eff[c, cat]["trainable_params"], "params", integer=True),
                    "runtime_ms_per_batch": numeric(eff[c, cat]["inference_ms_per_batch"], "runtime"),
                    "VRAM_MiB": numeric(eff[c, cat]["peak_vram_MiB"], "VRAM"),
                }, tiny_status=tiny[c, cat]["tiny_status"], boundary_status=boundary[c, cat]["aupro_status"],
                n_images=main[c, cat]["n_samples"], n_tiny_regions=int(tiny[c, cat]["n_tiny_regions"]),
                n_boundary_regions=int(boundary[c, cat]["n_gt_boundary_regions"]),
                boundary_f1=boundary_f1_values[c, cat], boundary_f1_status=boundary[c, cat]["boundary_f1_status"],
                seed=int(eff[c, cat]["seed"]), efficiency_scope_sha256=eff[c, cat]["scope_sha256"],
            ))
    require(all(sha256(p) == h for p, h in inputs.items()), "Input files changed during aggregation")
    provenance = dict(
        schema="msila.week06.aggregate.v1", split=primary["split"], categories=categories,
        primary_macro="unweighted category mean over all declared categories",
        diagnostic_macro="unweighted category mean over common GT-supported categories; undefined is never zero",
        tiny_categories=tiny_cats, boundary_categories=boundary_cats, boundary_f1_categories=bf1_cats,
        tiny_excluded={cat: tiny["R0", cat]["tiny_status"] for cat in categories if cat not in tiny_cats},
        boundary_excluded={cat: boundary["R0", cat]["aupro_status"] for cat in categories if cat not in boundary_cats},
        runtime_aggregation="unweighted category mean of batch latency medians",
        vram_aggregation="maximum absolute peak allocated MiB across categories",
        params_aggregation="exact trainable count; require same architecture across categories per candidate",
        metric_unit="fraction [0,1]", input_files_sha256=inputs,
        input_roles=dict(metrics_json=str(metrics_json), tiny_csv=str(tiny_csv),
                         boundary_csv=str(boundary_csv), efficiency_csvs=efficiency_paths),
        evaluation_manifest_sha256=manifest_hash, evaluator_sha256=primary["evaluator_sha256"],
        normalization_by_candidate=norms, efficiency_scopes=scopes,
        efficiency_measurement_protocol=measurement_protocols[0],
        boundary_f1_macro={c: diagnostic_macro(boundary_f1_values, bf1_cats, c) for c in CANDIDATES},
        interpretation="Descriptive evidence for one seed; no significance test or automatic winner selection",
    )
    return {"summary": summary, "per_category": per_category, "provenance": provenance}


def write_outputs(result, output):
    output = Path(output)
    require(output.suffix.lower() == ".csv", "Output must end in .csv")
    detail = output.with_suffix(".per_category.csv")
    provenance = output.with_suffix(".provenance.json")
    require(not any(p.exists() for p in (output, detail, provenance)), "Output exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, rows, fields in ((output, result["summary"], FIELDS),
                               (detail, result["per_category"], tuple(result["per_category"][0]))):
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with provenance.open("x", encoding="utf-8") as handle:
        json.dump(result["provenance"], handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True, help="E2 output metrics.json")
    parser.add_argument("--tiny-csv", type=Path, required=True)
    parser.add_argument("--boundary-csv", type=Path, required=True)
    parser.add_argument("--efficiency-csv", type=Path, action="append", required=True,
                        help="Repeat per category; keep adjacent E8 JSON reports")
    parser.add_argument("--output", type=Path, default=Path("outputs/week06_multiscale.csv"))
    parser.add_argument("--lock-output", type=Path,
                        help="After successful aggregation, create a DEV-based representation lock YAML")
    args = parser.parse_args()
    try:
        result = aggregate(args.metrics_json, args.tiny_csv, args.boundary_csv, args.efficiency_csv)
        lock = None
        if args.lock_output is not None:
            from create_representation_lock import build_lock, check_lock_output, write_lock
            check_lock_output(args.lock_output)
            lock = build_lock(result)
        write_outputs(result, args.output)
        if lock is not None:
            write_lock(lock, args.lock_output, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Aggregation failed: {exc}\n")
    print(" | ".join(FIELDS))
    for row in result["summary"]:
        print(" | ".join("NA" if row[f] is None else f"{row[f]:.6f}" if isinstance(row[f], float)
                         else str(row[f]) for f in FIELDS))
    print(f"Wrote {args.output}")
    if lock is not None:
        print(f"Selected {lock['selected']}; wrote {args.lock_output}")
    print(f"Tiny categories: {len(result['provenance']['tiny_categories'])}/{len(result['provenance']['categories'])}; "
          f"boundary categories: {len(result['provenance']['boundary_categories'])}/{len(result['provenance']['categories'])}")


if __name__ == "__main__":
    main()
