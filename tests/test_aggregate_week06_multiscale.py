"""Synthetic aggregation tests; do not require torch or real metric results."""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_week06_multiscale.py"
spec = importlib.util.spec_from_file_location("week06_aggregate", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def json_write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def csv_read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_scope(scope):
    scope = {k: v for k, v in scope.items() if k != "fingerprint_sha256"}
    scope["fingerprint_sha256"] = hashlib.sha256(json.dumps(
        scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return scope


@pytest.fixture
def artifacts(tmp_path):
    categories = ["fabric", "vial"]
    common_hash = "a" * 64
    primary = dict(
        categories=categories, primary_metric="aupro_0.05", max_fpr=.05,
        input_manifest_sha256=common_hash, evaluator_sha256={"src/metrics/aupro.py": common_hash},
        split="dev_synthetic", normalization_by_candidate={c: "locked_probability" for c in m.CANDIDATES},
        candidate_category=[], macro_by_candidate={},
    )
    tiny, boundary = [], []
    efficiency_paths = []
    benchmark = dict(latency_warmup=10, latency_iterations=50, latency_rounds=3,
                     vram_warmup=10, vram_iterations=1, stability_cv_threshold=.10, use_inference_mode=True)
    for j, category in enumerate(categories):
        directory = tmp_path / category
        directory.mkdir()
        scope = make_scope(dict(
            scope_name="day05_cached_representation_pipeline", device="cuda:0", batch_size=1,
            precision="fp32_tf32_disabled", input_signature="six cached [1,768,32,32] maps",
            pipeline_stages=["adapter", "alignment", "projection", "mean", "decoder"],
            extra=dict(hardware={"gpu": "SYNTHETIC_TEST_GPU", "gpu_uuid": "TEST_ONLY"},
                       input_contract={"tile_size": 512}, output_hw=[512, 512], benchmark=benchmark,
                       sample_ids=[category + "/test"], cache_manifest_sha256=common_hash,
                       val_records_sha256=common_hash, source_hashes={"test.py": common_hash}),
        ))
        json_write(directory / "efficiency_protocol.json", scope)
        efficiency_rows = []
        for i, c in enumerate(m.CANDIDATES):
            primary["candidate_category"].append(dict(candidate=c, category=category,
                                                       **{"aupro_0.05": .5 + .1*i + .02*j}, n_samples=2))
            tiny.append(dict(
                candidate=c, category=category, split="dev_synthetic", tiny_area_px=4,
                tiny_rule="area_px <= tiny_area_px", connectivity=8, max_fpr=.05, n_images=2,
                n_tiny_regions=2, n_non_tiny_regions=3, n_normal_pixels=200,
                **{"tiny_aupro_0.05": .2 + .1*i + .04*j}, tiny_status="OK",
                normalization_protocol="locked_probability", protocol_sha256="b"*64,
                manifest_sha256=common_hash, aupro_source_sha256=common_hash,
            ))
            boundary.append(dict(
                candidate=c, category=category, split="dev_synthetic",
                boundary_mode="image_border_band", band_width_px=2,
                selection_rule="component intersects zone", connectivity=8,
                prediction_threshold=.5, tolerance_px=1, max_fpr=.05, n_images=2,
                n_gt_boundary_regions=1, n_normal_pixels=200,
                **{"boundary_aupro_0.05": .3 + .1*i + .06*j}, aupro_status="OK",
                boundary_f1=.4 + .1*i + .08*j, boundary_f1_status="OK",
                normalization_protocol="locked_probability", protocol_sha256="c"*64,
                manifest_sha256=common_hash,
            ))
            params, runtime, vram = (10 if c == "R0" else 20), 1 + i + .2*j, 100 + 10*i + 5*j
            lat = dict(status="PASS", latency_ms={"median": runtime}, stability={"round_median_cv": .02},
                       protocol=dict(warmup_per_round=10, iterations_per_round=50, rounds=3,
                                     cuda_sync_before_each_timing=True, cuda_sync_after_each_timing=True,
                                     inference_mode=True))
            mem = dict(status="PASS", memory={"peak_allocated": {"MiB": vram}},
                       protocol=dict(warmup=10, measured_iterations=1, inference_mode=True))
            report = dict(candidate_id=c, status="PASS", scope=scope,
                          scope_fingerprint_sha256=scope["fingerprint_sha256"], latency=lat, peak_vram=mem)
            json_write(directory / f"{c}_efficiency.json", dict(
                efficiency=report, parameters=dict(candidate_id=c, totals={"trainable_parameters": params}),
            ))
            efficiency_rows.append(dict(
                candidate=c, category=category, seed=42, trainable_params=params,
                inference_ms_per_batch=runtime, peak_vram_MiB=vram, gpu="SYNTHETIC_TEST_GPU",
                tile_resolution="512x512", batch_size=1, precision="fp32_tf32_disabled",
                latency_warmup=10, latency_iterations=50, latency_rounds=3,
                vram_warmup=10, vram_iterations=1, scope_sha256=scope["fingerprint_sha256"], status="PASS",
            ))
        path = directory / "efficiency_summary.csv"
        csv_write(path, list(reversed(efficiency_rows)))
        efficiency_paths.append(path)
    for c in m.CANDIDATES:
        primary["macro_by_candidate"][c] = sum(r["aupro_0.05"] for r in primary["candidate_category"]
                                                  if r["candidate"] == c) / len(categories)
    json_write(tmp_path / "metrics.json", primary)
    csv_write(tmp_path / "tiny.csv", list(reversed(tiny)))
    csv_write(tmp_path / "boundary.csv", boundary)
    return tmp_path / "metrics.json", tmp_path / "tiny.csv", tmp_path / "boundary.csv", efficiency_paths


def run(paths):
    return m.aggregate(*paths)


def test_exact_macro_params_runtime_and_max_vram(artifacts):
    result = run(artifacts)
    assert [r["candidate"] for r in result["summary"]] == list(m.CANDIDATES)
    assert len(result["per_category"]) == 6
    for i, row in enumerate(result["summary"]):
        assert tuple(row) == m.FIELDS
        assert row["AU-PRO0.05"] == pytest.approx(.51 + .1*i)
        assert row["tiny_AU-PRO0.05"] == pytest.approx(.22 + .1*i)
        assert row["boundary_AU-PRO0.05"] == pytest.approx(.33 + .1*i)
        assert row["params"] == (10 if i == 0 else 20)
        assert row["runtime_ms_per_batch"] == pytest.approx(1.1+i)
        assert row["VRAM_MiB"] == 105+10*i


@pytest.mark.parametrize("all_categories", [False, True])
def test_undefined_tiny_is_never_zero(artifacts, tmp_path, all_categories):
    rows = csv_read(artifacts[1])
    for r in rows:
        if all_categories or r["category"] == "vial":
            r.update(n_tiny_regions="0", tiny_status="NO_REGIONS", **{"tiny_aupro_0.05": ""})
    csv_write(artifacts[1], rows)
    result = run(artifacts)
    if all_categories:
        assert result["provenance"]["tiny_categories"] == []
        assert all(r["tiny_AU-PRO0.05"] is None for r in result["summary"])
    else:
        assert result["provenance"]["tiny_categories"] == ["fabric"]
        assert result["summary"][0]["tiny_AU-PRO0.05"] == pytest.approx(.2)
    output = tmp_path / "output.csv"
    m.write_outputs(result, output)
    if all_categories:
        assert csv_read(output)[0]["tiny_AU-PRO0.05"] == ""


@pytest.mark.parametrize("source", [0, 1, 2, 3])
def test_duplicate_rows_are_rejected(artifacts, source):
    if source == 0:
        obj = m.read_json(artifacts[0])
        obj["candidate_category"].append(obj["candidate_category"][0])
        json_write(artifacts[0], obj)
    else:
        path = artifacts[source] if source < 3 else artifacts[3][0]
        rows = csv_read(path)
        csv_write(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        run(artifacts)


def test_missing_category_candidate_rejected(artifacts):
    rows = csv_read(artifacts[1])
    csv_write(artifacts[1], rows[:-1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        run(artifacts)


@pytest.mark.parametrize("column,value,message", [
    ("tiny_area_px", "8", "inconsistent tiny_area_px"),
    ("max_fpr", "0.1", "FPR/connectivity"),
    ("normalization_protocol", "candidate_minmax", "normalization mismatch"),
    ("manifest_sha256", "d"*64, "different evaluation manifest"),
    ("aupro_source_sha256", "d"*64, "source differs"),
    ("n_images", "3", "sample count"),
    ("n_tiny_regions", "1", "inconsistent n_tiny_regions"),
    ("tiny_aupro_0.05", "nan", "finite"),
    ("tiny_aupro_0.05", "90", "not percent"),
    ("tiny_status", "NO_REGIONS", "must be blank"),
])
def test_tiny_protocol_and_value_mismatches(artifacts, column, value, message):
    rows = csv_read(artifacts[1])
    rows[0][column] = value
    csv_write(artifacts[1], rows)
    with pytest.raises(ValueError, match=message):
        run(artifacts)


@pytest.mark.parametrize("column,value", [
    ("status", "FAIL"), ("batch_size", "2"), ("gpu", "OTHER_GPU"),
    ("tile_resolution", "256x256"), ("latency_warmup", "20"),
    ("precision", "fp16"), ("inference_ms_per_batch", "9"), ("peak_vram_MiB", "999"),
])
def test_efficiency_csv_mismatch_rejected(artifacts, column, value):
    path = artifacts[3][0]
    rows = csv_read(path)
    rows[0][column] = value
    csv_write(path, rows)
    with pytest.raises(ValueError, match="Efficiency:"):
        run(artifacts)


def test_distinct_physical_gpu_rejected_even_with_same_name(artifacts):
    directory = artifacts[3][1].parent
    scope = m.read_json(directory / "efficiency_protocol.json")
    scope["extra"]["hardware"]["gpu_uuid"] = "ANOTHER_TEST_GPU"
    scope = make_scope(scope)
    json_write(directory / "efficiency_protocol.json", scope)
    rows = csv_read(directory / "efficiency_summary.csv")
    for row in rows:
        row["scope_sha256"] = scope["fingerprint_sha256"]
        path = directory / f"{row['candidate']}_efficiency.json"
        obj = m.read_json(path)
        obj["efficiency"].update(scope=scope, scope_fingerprint_sha256=scope["fingerprint_sha256"])
        json_write(path, obj)
    csv_write(directory / "efficiency_summary.csv", rows)
    with pytest.raises(ValueError, match="differs across benchmark runs"):
        run(artifacts)


def test_parameter_counts_are_not_averaged(artifacts):
    directory = artifacts[3][1].parent
    rows = csv_read(directory / "efficiency_summary.csv")
    for r in rows:
        if r["candidate"] == "R0":
            r["trainable_params"] = "11"
    csv_write(directory / "efficiency_summary.csv", rows)
    path = directory / "R0_efficiency.json"
    obj = m.read_json(path)
    obj["parameters"]["totals"]["trainable_parameters"] = 11
    json_write(path, obj)
    with pytest.raises(ValueError, match="params differs across categories"):
        run(artifacts)


def test_mixed_seeds_rejected(artifacts):
    path = artifacts[3][1]
    rows = csv_read(path)
    for r in rows:
        r["seed"] = "43"
    csv_write(path, rows)
    with pytest.raises(ValueError, match="inconsistent seed"):
        run(artifacts)


def test_unstable_json_report_rejected(artifacts):
    path = artifacts[3][0].parent / "R0_efficiency.json"
    obj = m.read_json(path)
    obj["efficiency"]["latency"]["stability"]["round_median_cv"] = .5
    json_write(path, obj)
    with pytest.raises(ValueError, match="latency unstable"):
        run(artifacts)


def test_actual_vram_procedure_drift_rejected(artifacts):
    directory = artifacts[3][0].parent
    for c in m.CANDIDATES:
        path = directory / f"{c}_efficiency.json"
        obj = m.read_json(path)
        obj["efficiency"]["peak_vram"]["protocol"]["empty_cache_before_measurement"] = False
        json_write(path, obj)
    directory = artifacts[3][1].parent
    for c in m.CANDIDATES:
        path = directory / f"{c}_efficiency.json"
        obj = m.read_json(path)
        obj["efficiency"]["peak_vram"]["protocol"]["empty_cache_before_measurement"] = True
        json_write(path, obj)
    with pytest.raises(ValueError, match="actual latency/VRAM procedures differ"):
        run(artifacts)


def test_stored_primary_macro_must_match_categories(artifacts):
    obj = m.read_json(artifacts[0])
    obj["macro_by_candidate"]["R0"] = .99
    json_write(artifacts[0], obj)
    with pytest.raises(ValueError, match="stored macro disagrees"):
        run(artifacts)


def test_boundary_undefined_is_not_imputed(artifacts):
    rows = csv_read(artifacts[2])
    for r in rows:
        if r["category"] == "vial":
            r.update(n_gt_boundary_regions="0", aupro_status="NO_REGIONS", boundary_f1_status="NO_GT_BOUNDARY_REGIONS",
                     boundary_f1="", **{"boundary_aupro_0.05": ""})
    csv_write(artifacts[2], rows)
    result = run(artifacts)
    assert result["provenance"]["boundary_categories"] == ["fabric"]
    assert result["summary"][0]["boundary_AU-PRO0.05"] == pytest.approx(.3)
    assert result["provenance"]["boundary_f1_macro"]["R0"] == pytest.approx(.4)


def test_write_refuses_any_existing_output(artifacts, tmp_path):
    result = run(artifacts)
    output = tmp_path / "week06_multiscale.csv"
    sidecar = output.with_suffix(".per_category.csv")
    sidecar.write_text("existing user content")
    with pytest.raises(ValueError, match="Output exists"):
        m.write_outputs(result, output)
    assert sidecar.read_text() == "existing user content"
    assert not output.exists()


def test_cli_writes_three_candidate_rows_and_audit_files(artifacts, tmp_path):
    output = tmp_path / "final" / "week06_multiscale.csv"
    cmd = [sys.executable, "-B", str(SCRIPT), "--metrics-json", str(artifacts[0]),
           "--tiny-csv", str(artifacts[1]), "--boundary-csv", str(artifacts[2]), "--output", str(output)]
    for path in reversed(artifacts[3]):
        cmd += ["--efficiency-csv", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert [r["candidate"] for r in csv_read(output)] == list(m.CANDIDATES)
    assert len(csv_read(output.with_suffix(".per_category.csv"))) == 6
    audit = m.read_json(output.with_suffix(".provenance.json"))
    assert audit["metric_unit"] == "fraction [0,1]"
    assert "R0 |" in completed.stdout
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 2 and "Output exists" in second.stderr


def test_cli_bad_input_does_not_create_evidence_csv(artifacts, tmp_path):
    rows = csv_read(artifacts[1])
    rows[0]["manifest_sha256"] = "d"*64
    csv_write(artifacts[1], rows)
    output = tmp_path / "must_not_exist.csv"
    cmd = [sys.executable, "-B", str(SCRIPT), "--metrics-json", str(artifacts[0]),
           "--tiny-csv", str(artifacts[1]), "--boundary-csv", str(artifacts[2]), "--output", str(output)]
    for path in artifacts[3]:
        cmd += ["--efficiency-csv", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 2
    assert not output.exists()
