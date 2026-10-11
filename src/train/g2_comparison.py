"""TV1 artifact export for TV2's adjacent ablations (no metric computation)."""
from __future__ import annotations

from copy import deepcopy
import csv
import os
from pathlib import Path

from src.train.screen_representation import save_json


def comparison_identity(cfg):
    """The scientific variables which must match for a paired E3/E2 result."""
    identity = {key: deepcopy(cfg[key]) for key in (
        "category", "mode", "study_sha256", "selection_sha256", "adapter", "decoder",
        "training", "loss", "data", "sources", "synthetic_protocol", "evaluation", "expected_steps",
    )}
    identity["backbone"] = {k: deepcopy(v) for k, v in cfg["backbone"].items() if k != "feature_blocks"}
    return identity


def write_comparison_inputs(directory, *, categories, rows, study_sha256, selection_sha256, smoke,
                            experiments=("E2", "E3")):
    """Always export eight rows; complete comparisons require eight valid pairs.

    Scalars are synthetic DEV AU-PRO@0.05 inputs, NOT real TEST scores. TV2
    owns differences/aggregation and can load each best checkpoint to evaluate
    TEST_PUBLIC using its own evaluator. Smoke/fixture exports are labelled.
    """
    directory = Path(directory)
    if [row["category"] for row in rows] != list(categories):
        raise ValueError("Comparison export must contain all categories in declared order")
    ready = sum(row["status"] == "READY" for row in rows)
    real = sum(row["status"] == "READY" and row["real_pair"] for row in rows)
    status = ("SMOKE_READY" if ready == 8 else "INCOMPLETE") if smoke else ("PASS" if real == 8 else "INCOMPLETE")
    if experiments not in {("E2", "E3"), ("E3", "E4")}:
        raise ValueError("Only E3-E2 and E4-E3 artifact comparisons are implemented")
    before, after = experiments
    payload = dict(version=f"g2_{after.lower()}_{before.lower()}_inputs_v1", status=status,
                   experiments=list(experiments),
                   mode="smoke" if smoke else "full", split="dev_synthetic",
                   dataset="mvtec_ad2_good_synthetic", metric="synthetic_dev_aupro_0_05", max_fpr=.05,
                   synthetic=True, native_resolution=True,
                   interpretation="Synthetic DEV comparison inputs; TEST evaluation belongs to TV2",
                   study_sha256=study_sha256, selection_sha256=selection_sha256,
                   categories=list(categories), expected_pairs=8, ready_pairs=ready,
                   real_ready_pairs=real, records=rows,
                   checkpoint_restore=f"build_g2_model(config, experiment={before}/{after}); "
                                      "trainable_modules(model, experiment).load_state_dict(payload['model_state'])")
    json_path = directory / f"{after}_minus_{before}_inputs.json"
    csv_path = directory / f"{after}_minus_{before}_inputs.csv"
    save_json(payload, json_path)
    fields = ("category", "status", "real_pair", "r", "d", "seed", "dev_seed", "expected_steps",
              "pair_protocol_sha256") + tuple(f"{stage.lower()}_synthetic_dev_aupro_0_05" for stage in experiments)
    fields += tuple(f"{stage.lower()}_{key}" for key in ("checkpoint", "config", "metrics") for stage in experiments)
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for experiment in (before.lower(), after.lower()):
                for key in ("checkpoint", "config", "metrics"):
                    flat[f"{experiment}_{key}"] = row.get(experiment, {}).get(key)
            writer.writerow(flat)
    os.replace(tmp, csv_path)
    return dict(status=status, ready_pairs=ready, real_ready_pairs=real,
                json=str(json_path.resolve()), csv=str(csv_path.resolve()))
