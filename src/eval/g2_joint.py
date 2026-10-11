"""G2 joint-search manifest and strict 45 x 8 reporting; no training/metrics code."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import re

import yaml

from src.data.synthetic_anomaly import validate_native_protocol
from src.eval.evaluator import MVTEC_AD2_CATEGORIES, write_metrics_json
from src.models.backbone_registry import adapter_pairs, backbone_spec
from src.models.contracts import MULTIVIEW_FEATURE_KEYS
from src.train.screen_representation import sha256_json

CATEGORIES = tuple(MVTEC_AD2_CATEGORIES)
FUSIONS = ("mean", "weighted_sum", "concat", "gated", "attention")
METRIC_FIELDS = ("aupro_005", "tiny", "mixed", "dice", "p99")
METRIC_KEYS = (
    "synthetic_dev_aupro_0_05", "tiny_aupro_0_05", "mixed_aupro_0_05",
    "synthetic_dice", "normal_score_p99",
)
STATUSES = ("NOT RUN", "BLOCKED", "FAIL", "PASS", "CPU_PASS", "SMOKE_PASS",
            "READY", "INFEASIBLE_ON_HARDWARE")
FIELDS = (
    "category", "r", "d", "fusion", "seed", "budget", "checkpoint", "config_hash",
    "git_sha", "status", *METRIC_FIELDS, "params", "latency", "peak_vram",
    "run_id", "reason", "evaluation_json", "checkpoint_sha256", "protocol_hash",
    "dev_hash", "benchmark_status",
)
VERSION = "g2_joint_search_v1"
BUDGET_FIELDS = ("epochs", "batch_size", "max_steps", "max_minutes")


class ManifestError(ValueError):
    """Malformed or incompatible search evidence."""


def positive_int(value):
    return type(value) is int and value > 0


def unit_metric(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def run_id(category, r, d, fusion, seed):
    return f"E5/r{r}_d{d}/{fusion}/{category}/seed{seed}"


def validate_budget(budget):
    if not isinstance(budget, dict) or set(budget) != set(BUDGET_FIELDS):
        raise ManifestError(f"budget must declare exactly {BUDGET_FIELDS}")
    if not all(positive_int(budget[k]) for k in ("epochs", "batch_size")):
        raise ManifestError("budget epochs/batch_size must be positive integers")
    if budget["max_steps"] is not None and not positive_int(budget["max_steps"]):
        raise ManifestError("budget max_steps must be null or a positive integer")
    value = budget["max_minutes"]
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise ManifestError("budget max_minutes must be null or positive finite")


def joint_protocol_identity(cfg):
    """Reuse D6 scientific variables and add the fixed six-source architecture."""
    from scripts.eval_g2 import scientific_identity

    identity = scientific_identity(cfg)
    adapter, fusion, views = cfg["adapter"], cfg["fusion"], cfg["paired_views"]
    identity["backbone"]["source_sha256"] = cfg["backbone"].get("source_sha256")
    identity["shared_architecture"] = dict(
        adapter={key: adapter.get(key, default) for key, default in
                 (("kernel_size", 3), ("gamma_init", 0.), ("bias", True))},
        fusion_dim=fusion["dim"], feature_order=list(fusion.get("feature_order", MULTIVIEW_FEATURE_KEYS)),
        context_size=views["context_size"],
        pair_strategy=views.get("pair_strategy", views.get("strategy", "concat")),
        deterministic_sampling=views.get("deterministic_sampling", True),
    )
    return identity


def load_spec(model_path, runner_path, *, root, integration_sha=None):
    """Read TV1 config only. Missing D10 screening budget remains unresolved."""
    root = Path(root)
    model = yaml.safe_load(Path(model_path).read_text())
    runner = yaml.safe_load(Path(runner_path).read_text())
    name = model["backbone"]["name"]
    if runner.get("backbone", name) != name:
        raise ManifestError("Model and runner backbone differ")
    grid = runner["adapter_grids"][name]
    pairs = adapter_pairs(backbone_spec(name).channels, {
        "pairs": [[r, d] for r in grid["r_values"] for d in grid["d_values"]],
    })
    if len(grid["r_values"]) != 3 or len(grid["d_values"]) != 3 or len(pairs) != 9:
        raise ManifestError("Joint search requires exactly 3 x 3 distinct adapter pairs")
    if tuple(model["categories"]) != CATEGORIES or set(model["fusion_methods"]) != set(FUSIONS) or len(model["fusion_methods"]) != 5:
        raise ManifestError("Joint search requires the canonical eight categories and five fusions")
    seed = model["training"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ManifestError("Training seed must be a nonnegative integer")
    budget = model.get("screening_budget", runner.get("screening_budget"))
    if model.get("screening_budget") is not None and runner.get("screening_budget") is not None:
        if model["screening_budget"] != runner["screening_budget"]:
            raise ManifestError("Model and runner screening_budget differ")
    if budget is not None:
        validate_budget(budget)
    prototype = deepcopy(model)
    protocol = prototype["synthetic_protocol"]
    if isinstance(protocol, str):
        path = Path(protocol)
        protocol = yaml.safe_load((path if path.is_absolute() else root / path).read_text())
    prototype["synthetic_protocol"] = validate_native_protocol(protocol)
    prototype.update(category=CATEGORIES[0], sources={"train": [], "dev": []})
    for key in ("max_train_sources", "max_dev_sources"):
        prototype["data"].setdefault(key, None)
    prototype["decoder"].setdefault("deterministic_resize", runner.get("execution", {}).get("deterministic_resize", False))
    for key in ("max_steps", "max_minutes", "max_grad_norm"):
        prototype["training"].setdefault(key, None)
    if budget is not None:
        prototype["training"].update(budget)
    protocol_template = joint_protocol_identity(prototype)
    protocol_template.pop("category")
    protocol_template.pop("sources")
    return dict(
        version=VERSION, categories=list(CATEGORIES), pairs=[list(p) for p in pairs],
        fusions=list(FUSIONS), seed=seed, budget=budget, backbone=name,
        protocol_template=protocol_template,
        model_config_hash=sha256_json(model), runner_config_hash=sha256_json(runner),
        integration_sha=integration_sha,
        budget_status="LOCKED" if budget is not None else "BLOCKED",
        budget_reason=None if budget is not None else "D10 screening_budget not declared; main budget is not a substitute",
        units={"aupro_005": "normalized AU-PRO, FPR <= 0.05",
               "latency": "median milliseconds, native inference on named CUDA GPU",
               "peak_vram": "peak allocated CUDA bytes, including resident model"},
    )


def validate_spec(spec):
    if spec.get("version") != VERSION or tuple(spec.get("categories", ())) != CATEGORIES:
        raise ManifestError("Invalid manifest version/categories")
    pairs = adapter_pairs(backbone_spec(spec["backbone"]).channels, {"pairs": spec["pairs"]})
    r_values, d_values = {p[0] for p in pairs}, {p[1] for p in pairs}
    if (len(pairs) != 9 or len(r_values) != 3 or len(d_values) != 3
            or set(pairs) != {(r, d) for r in r_values for d in d_values}
            or tuple(spec.get("fusions", ())) != FUSIONS):
        raise ManifestError("Expected nine adapter pairs and five canonical fusions")
    if type(spec.get("seed")) is not int or spec["seed"] < 0:
        raise ManifestError("Invalid search seed")
    if spec.get("budget") is not None:
        validate_budget(spec["budget"])
    return pairs


def expected_ids(spec):
    pairs = validate_spec(spec)
    return {
        run_id(c, r, d, f, spec["seed"])
        for r, d in pairs for f in FUSIONS for c in CATEGORIES
    }


def skeleton(spec):
    pairs = validate_spec(spec)
    rows = []
    for r, d in pairs:
        for fusion in FUSIONS:
            for category in CATEGORIES:
                row = dict.fromkeys(FIELDS)
                row.update(category=category, r=r, d=d, fusion=fusion, seed=spec["seed"],
                           budget=deepcopy(spec["budget"]), benchmark_status="NOT RUN",
                           run_id=run_id(category, r, d, fusion, spec["seed"]),
                           status="BLOCKED" if spec["budget"] is None else "NOT RUN",
                           reason=spec["budget_reason"] if spec["budget"] is None else "D10 training not run")
                rows.append(row)
    return rows


def validate_row(row, spec):
    missing = set(FIELDS) - set(row)
    if missing:
        raise ManifestError(f"Missing manifest fields: {sorted(missing)}")
    if set(row) - set(FIELDS):
        raise ManifestError("Unexpected manifest fields")
    if (row["category"] not in CATEGORIES or not positive_int(row["r"]) or not positive_int(row["d"])
            or row["fusion"] not in FUSIONS or row["seed"] != spec["seed"]
            or type(row["seed"]) is not int or row["budget"] != spec["budget"]
            or [row["r"], row["d"]] not in spec["pairs"]):
        raise ManifestError("Manifest candidate/seed/budget differs from declared search")
    if row["run_id"] != run_id(row["category"], row["r"], row["d"], row["fusion"], row["seed"]):
        raise ManifestError("run_id does not match candidate identity")
    if row["status"] not in STATUSES:
        raise ManifestError("Invalid run status")
    if row["benchmark_status"] not in ("NOT RUN", "BLOCKED", "FAIL", "PASS"):
        raise ManifestError("Invalid benchmark status")
    for key, length in (("config_hash", 64), ("git_sha", 40), ("checkpoint_sha256", 64),
                        ("protocol_hash", 64), ("dev_hash", 64)):
        if row[key] is not None and not digest(row[key], length):
            raise ManifestError(f"Invalid {key}")
    for key in METRIC_FIELDS:
        if row[key] is not None and not unit_metric(row[key]):
            raise ManifestError(f"Invalid/nonfinite {key}")
    if row["params"] is not None and not positive_int(row["params"]):
        raise ManifestError("params must be null or a positive integer")
    if row["latency"] is not None and (
            type(row["latency"]) not in (int, float) or not math.isfinite(row["latency"]) or row["latency"] <= 0):
        raise ManifestError("latency must be null or positive finite milliseconds")
    if row["peak_vram"] is not None and not positive_int(row["peak_vram"]):
        raise ManifestError("peak_vram must be null or positive allocated bytes")
    measured = row["latency"] is not None or row["peak_vram"] is not None
    if measured and (row["benchmark_status"] != "PASS" or row["latency"] is None or row["peak_vram"] is None):
        raise ManifestError("GPU measurements require complete PASS benchmark evidence")
    if row["benchmark_status"] == "PASS" and not measured:
        raise ManifestError("PASS benchmark requires measured latency and VRAM")
    if row["status"] == "PASS":
        if (spec["budget"] is None or any(row[key] is None for key in METRIC_FIELDS)
                or any(not row[k] for k in ("checkpoint", "evaluation_json", "config_hash",
                                           "git_sha", "checkpoint_sha256", "protocol_hash", "dev_hash", "params"))):
            raise ManifestError("PASS requires checkpoint, complete metrics, budget and provenance")
    elif any(row[key] is not None for key in (*METRIC_FIELDS, "latency", "peak_vram")):
        raise ManifestError("Non-PASS rows cannot publish scientific metrics/GPU measurements")


def validate_manifest(rows, spec):
    expected = expected_ids(spec)
    for row in rows:
        validate_row(row, spec)
    counts = Counter(row["run_id"] for row in rows)
    duplicate = sorted(k for k, n in counts.items() if n != 1)
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    if duplicate or missing or extra or len(rows) != 360:
        raise ManifestError(f"Expected 360 unique cells: duplicate={duplicate}, missing={missing}, extra={extra}")
    return rows


def write_manifest(path, rows, spec):
    validate_manifest(rows, spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["budget"] = json.dumps(row["budget"], sort_keys=True, allow_nan=False) if row["budget"] is not None else ""
            writer.writerow(encoded)
    temporary.replace(path)


def read_manifest(path, spec):
    path = Path(path).resolve()
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ManifestError("CSV columns differ from the versioned manifest schema")
        for raw in reader:
            row = {k: (v if v != "" else None) for k, v in raw.items()}
            for key in ("r", "d", "seed", "params", "peak_vram"):
                if row[key] is not None:
                    row[key] = int(row[key])
            for key in (*METRIC_FIELDS, "latency"):
                if row[key] is not None:
                    row[key] = float(row[key])
            row["budget"] = json.loads(row["budget"]) if row["budget"] is not None else None
            for key in ("checkpoint", "evaluation_json"):
                if row[key] is not None:
                    file = Path(row[key]).expanduser()
                    row[key] = str(file if file.is_absolute() else path.parent / file)
            rows.append(row)
    return validate_manifest(rows, spec)


def validate_checkpoint_record(row, payload, spec):
    """Bind saved training configuration to the planned row and fixed protocol."""
    validate_row(row, spec)
    if spec["budget"] is None:
        raise ManifestError(spec["budget_reason"])
    cfg, meta = payload["config"], payload["metadata"]
    expected = (row["category"], row["r"], row["d"], row["fusion"], row["seed"])
    actual = (cfg.get("category"), cfg.get("adapter", {}).get("r"), cfg.get("adapter", {}).get("d"),
              cfg.get("fusion", {}).get("method"), cfg.get("training", {}).get("seed"))
    if actual != expected or cfg.get("stage") != "E5" or cfg.get("mode") != "screening":
        raise ManifestError("Checkpoint candidate/stage/mode differs (requires E5 screening, not preflight/smoke)")
    actual_budget = {key: cfg["training"].get(key) for key in BUDGET_FIELDS}
    if actual_budget != row["budget"]:
        raise ManifestError("Checkpoint training budget differs")
    config_hash = sha256_json(cfg)
    if row["config_hash"] != config_hash or meta.get("config_sha256") != config_hash:
        raise ManifestError("Checkpoint config hash differs from manifest/metadata")
    saved_sha = meta.get("git_sha", meta.get("git_commit", cfg.get("git_sha", cfg.get("git_commit"))))
    if saved_sha != row["git_sha"] or not digest(saved_sha, 40):
        raise ManifestError("Checkpoint training Git SHA differs from manifest")
    source_hashes = cfg["backbone"].get("source_sha256")
    if (not digest(cfg["backbone"].get("checkpoint_sha256")) or not isinstance(source_hashes, dict)
            or not source_hashes or any(not isinstance(k, str) or not digest(v) for k, v in source_hashes.items())):
        raise ManifestError("Checkpoint lacks pretrained backbone source-code checksums")
    identity = joint_protocol_identity(cfg)
    identity.pop("category")
    identity.pop("sources")
    template = deepcopy(spec["protocol_template"])
    # The unresolved config may not yet know the pretrained file checksum.
    for key in ("checkpoint_sha256", "source_sha256"):
        if template["backbone"].get(key) is None:
            identity["backbone"].pop(key, None)
            template["backbone"].pop(key, None)
    if identity != template:
        raise ManifestError("Checkpoint fixed backbone/decoder/loss/split/seed/protocol differs")
    return payload


def aggregate(rows, spec):
    """Called only after checkpoint/artifact audit; no partial eight-category macro."""
    validate_manifest(rows, spec)
    macro = []
    for r, d in spec["pairs"]:
        for fusion in FUSIONS:
            group = [row for row in rows if (row["r"], row["d"], row["fusion"]) == (r, d, fusion)]
            valid = [row for row in group if row["status"] == "PASS"]
            entry = dict(r=r, d=d, fusion=fusion, valid_categories=len(valid), expected_categories=8,
                         status="PASS" if len(valid) == 8 else "BLOCKED")
            entry.update({key: math.fsum(row[key] for row in valid) / 8 if len(valid) == 8 else None
                          for key in METRIC_FIELDS})
            macro.append(entry)
    return macro


def write_reports(rows, spec, directory, *, audit, selected_pair=None):
    """Audit every supplied checkpoint and accepted result before aggregation."""
    if selected_pair is not None and list(selected_pair) not in spec["pairs"]:
        raise ManifestError("Selected r/d is outside the declared grid")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validate_manifest(rows, spec)
    audited = deepcopy(rows)
    for row in audited:
        if row["checkpoint"] is not None or row["status"] == "PASS":
            try:
                audit(row)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                row.update(status="BLOCKED", reason=f"Evidence audit: {exc}", benchmark_status="NOT RUN")
                row.update(dict.fromkeys((*METRIC_FIELDS, "latency", "peak_vram")))
    # Candidate comparisons require identical DEV/config identities within each category.
    for category in CATEGORIES:
        valid = [row for row in audited if row["category"] == category and row["status"] == "PASS"]
        if len({(row["protocol_hash"], row["dev_hash"]) for row in valid}) > 1:
            for row in valid:
                row.update(status="BLOCKED", reason="Candidate synthetic DEV/protocol parity differs",
                           benchmark_status="NOT RUN")
                row.update(dict.fromkeys((*METRIC_FIELDS, "latency", "peak_vram")))
    macro = aggregate(audited, spec)
    write_manifest(directory / "joint_search_360.csv", audited, spec)
    write_metrics_json(spec, directory / "joint_search_spec.json")
    write_metrics_json(manifest_schema(), directory / "joint_search_manifest.schema.json")
    with (directory / "joint_search_macro.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    counts = dict(Counter(row["status"] for row in audited))
    coverage = dict(
        version=VERSION, status="PASS" if counts.get("PASS", 0) == 360 else "INCOMPLETE",
        expected_candidates=45, expected_categories=8, expected_cells=360,
        unique_cells=len({row["run_id"] for row in audited}), duplicate_ids=[], missing_ids=[],
        real_pass=counts.get("PASS", 0), status_counts=counts,
        complete_candidates=sum(row["status"] == "PASS" for row in macro),
        gpu_benchmark_pass=sum(row["status"] == "PASS" and row["benchmark_status"] == "PASS" for row in audited),
        gpu_benchmark_status_counts=dict(Counter(row["benchmark_status"] for row in audited)),
        macro_requires_categories=8, budget_status=spec["budget_status"],
        integration_sha=spec.get("integration_sha"), units=spec["units"],
        records=[{k: row[k] for k in ("run_id", "category", "r", "d", "fusion", "status", "reason")}
                 for row in audited],
    )
    if counts.get("BLOCKED") and not counts.get("PASS"):
        coverage["status"] = "BLOCKED"
    if selected_pair is not None:
        table = []
        for fusion in FUSIONS:
            group = {row["category"]: row for row in audited
                     if (row["r"], row["d"], row["fusion"]) == (*selected_pair, fusion)}
            entry = dict(fusion=fusion, r=selected_pair[0], d=selected_pair[1])
            entry.update({cat: group[cat]["aupro_005"] if group[cat]["status"] == "PASS" else None for cat in CATEGORIES})
            entry.update({f"{cat}_status": group[cat]["status"] for cat in CATEGORIES})
            count = sum(group[c]["status"] == "PASS" for c in CATEGORIES)
            entry.update(valid_categories=count, macro_8=math.fsum(entry[c] for c in CATEGORIES) / 8 if count == 8 else None)
            table.append(entry)
        with (directory / "fusion_comparison_selected.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        coverage["displayed_pair"] = list(selected_pair)
        coverage["selection_performed"] = False
    write_metrics_json(coverage, directory / "joint_search_coverage.json")
    return coverage


def manifest_schema():
    """JSON row schema; semantic/full-grid constraints are enforced by the loader."""
    nullable_number = {"type": ["number", "null"]}
    properties = {key: {"type": ["string", "null"]} for key in FIELDS}
    properties.update(category={"enum": list(CATEGORIES)}, fusion={"enum": list(FUSIONS)},
                      status={"enum": list(STATUSES)}, seed={"type": "integer", "minimum": 0},
                      r={"type": "integer", "minimum": 1}, d={"type": "integer", "minimum": 1},
                      run_id={"type": "string", "minLength": 1},
                      budget={"type": ["object", "null"], "required": list(BUDGET_FIELDS),
                              "additionalProperties": False, "properties": {
                                  "epochs": {"type": "integer", "minimum": 1},
                                  "batch_size": {"type": "integer", "minimum": 1},
                                  "max_steps": {"type": ["integer", "null"], "minimum": 1},
                                  "max_minutes": {"type": ["number", "null"], "exclusiveMinimum": 0}}},
                      params={"type": ["integer", "null"], "minimum": 1},
                      latency={**nullable_number, "exclusiveMinimum": 0},
                      peak_vram={"type": ["integer", "null"], "minimum": 1})
    for key in METRIC_FIELDS:
        properties[key] = {**nullable_number, "minimum": 0, "maximum": 1}
    for key, length in (("config_hash", 64), ("git_sha", 40), ("checkpoint_sha256", 64),
                        ("protocol_hash", 64), ("dev_hash", 64)):
        properties[key] = {"type": ["string", "null"], "pattern": f"^[0-9a-f]{{{length}}}$"}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": VERSION,
            "type": "object", "required": list(FIELDS), "additionalProperties": False,
            "properties": properties}
