#!/usr/bin/env python3
"""D6-TV1: one backbone, 72 Adapter screens, then 8 locked E2 runs.

Examples (paths/checkpoints configured in YAML):
  python scripts/run_g2.py --stage adapter_screen --categories all --device cuda --resume
  python scripts/run_g2.py --stage E2 --categories all --device cuda --resume

Smoke uses a separate namespace and never supplies selection evidence. No E3--E5
or fusion grid is scheduled here. Run each study with a single runner process.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
import torch

from src.data.synthetic_anomaly import validate_native_protocol
from src.data.tiling import generate_tile_records
from src.models.adapter_factory import AdapterFactoryConfig, ResidualAdapterFactory
from src.models.backbone_registry import adapter_pairs
from src.models.msila import resolve_g2_config
from src.train.g1_e1 import check_assets, discover_sources
from src.train.g2_e2 import train_e2
from src.train.screen_representation import (enforce_lock, read_yaml, save_json,
                                              sha256_file, sha256_json)
from src.utils.resume import load_checkpoint_payload

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
GRIDS = {
    "dinov3_vits16": ([32, 64, 128], [128, 256, 384]),
    "dinov3_vitb16": ([64, 128, 256], [256, 512, 768]),
    "dinov3_vith16plus": ([128, 256, 384], [384, 768, 1280]),
}
SELECTION_RULE = dict(metric="macro_synthetic_dev_aupro_0_05", direction="max",
                      tie_break=["adapter_trainable_parameters", "r", "d"],
                      expected_screen_runs=72, expected_main_runs=8,
                      require_all_categories=True, require_full_budget=True,
                      checkpoint_tie_break="earliest_epoch")


class G2Blocked(RuntimeError):
    pass


def absolute(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g2_runner.yaml")
    parser.add_argument("--stage", required=True, choices=("adapter_screen", "E2"))
    parser.add_argument("--categories", nargs="+", default=["all"], help="all, names or comma-separated names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", choices=tuple(GRIDS))
    for flag in ("data-root", "repo-dir", "weights", "output-root"):
        parser.add_argument(f"--{flag}")
    return parser.parse_args(argv)


def selected_pairs(cfg, runner):
    name = cfg["backbone"]["name"]
    if name not in GRIDS:
        raise ValueError(f"D6 only declares grids for {tuple(GRIDS)}")
    grid = runner["adapter_grids"][name]
    r_values, d_values = grid["r_values"], grid["d_values"]
    if (r_values, d_values) != GRIDS[name]:
        raise ValueError(f"Expected the declared D6 r/d grid for {name}: {GRIDS[name]}")
    return adapter_pairs(cfg["backbone"]["channels"], {
        "pairs": [[r, d] for r in r_values for d in d_values],
    })


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def load_config(args):
    runner = read_yaml(absolute(args.config))
    if runner.get("version") != "g2_runner_v1" or runner.get("selection") != SELECTION_RULE:
        raise ValueError("Expected g2_runner_v1 and the predeclared D6 selection rule")
    cfg = read_yaml(absolute(runner["model_config"]))
    if cfg["categories"] != list(CATEGORIES):
        raise ValueError("D6 requires the eight canonical categories in declared order")
    cfg["backbone"]["name"] = args.backbone or runner["backbone"]
    for argument, section, key in (("data_root", "data", "root"), ("repo_dir", "backbone", "repo_dir"),
                                    ("weights", "backbone", "weights")):
        value = getattr(args, argument)
        if value:
            cfg[section][key] = value
    cfg = resolve_g2_config(cfg, root=ROOT)
    cfg["data"]["root"] = str(absolute(cfg["data"]["root"]))
    cfg["data"].setdefault("max_train_sources", None)
    cfg["data"].setdefault("max_dev_sources", 4)
    cfg["evaluation"].setdefault("example_limit", 0)
    cfg["training"].update(runner["training"])
    if runner["execution"] != {"full_device": "cuda", "deterministic_resize": True}:
        raise ValueError("Full D6 requires CUDA and deterministic decoder resize")
    cfg["decoder"]["deterministic_resize"] = True
    selected_pairs(cfg, runner)
    t, d, e = cfg["training"], cfg["data"], cfg["evaluation"]
    if (t["optimizer"]["name"] != "AdamW" or t["scheduler"] is not None or t["amp"] is not False
            or t["deterministic_algorithms"] is not True or t["train_synthetic_by_epoch"] is not True
            or t["dev_synthetic_fixed"] is not True or t["seed"] == t["dev_seed"]):
        raise ValueError("D6 requires fixed AdamW/no scheduler/no AMP, strict determinism and independent TRAIN/DEV seeds")
    if (d["tile_size"] != 512 or d["drop_last"] is not False or d["train_split"] != "TRAIN/good"
            or d["dev_split"] != "VALIDATION/good" or not 0 <= d["overlap"] < 512
            or e["max_fpr"] != .05 or e["test_for_selection"] is not False
            or e["score_transform"] != "sigmoid" or e["stitching"] != "hann_native_coordinates"):
        raise ValueError("D6 requires the G2 tile/split and native synthetic DEV protocol")
    if cfg["checkpoint"] != dict(monitor="synthetic_dev_aupro_0_05", mode="max",
                                  tie_break="earliest_epoch", selection_split="VALIDATION/good"):
        raise ValueError("Checkpoint selection must use synthetic DEV, ties earliest epoch")
    if cfg["loss"]["type"] != "bce_plus_positive_mask_dice":
        raise ValueError("D6 requires the existing BCE + positive-mask Dice loss")
    if cfg["adapter"].get("gamma_init", 0) != 0:
        raise ValueError("E2 must initialize the residual gate at zero")
    for key in ("epochs", "batch_size", "checkpoint_interval_steps"):
        positive_int(t[key], f"training.{key}")
    for key in ("seed", "dev_seed"):
        if type(t[key]) is not int or t[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    if t["max_steps"] is not None:
        positive_int(t["max_steps"], "training.max_steps")
    for key in ("max_minutes", "max_grad_norm"):
        if t[key] is not None and (not math.isfinite(t[key]) or t[key] <= 0):
            raise ValueError(f"training.{key} must be positive or null")
    for key in ("train_variants_per_image", "dev_variants_per_image"):
        positive_int(d[key], f"data.{key}")
    for key in ("max_train_sources", "max_dev_sources"):
        if d[key] is not None:
            positive_int(d[key], f"data.{key}")
    if type(d["num_workers"]) is not int or d["num_workers"] < 0:
        raise ValueError("data.num_workers must be a nonnegative integer")
    positive_int(e["tile_batch_size"], "evaluation.tile_batch_size")
    for key, value in runner["smoke"].items():
        positive_int(value, f"smoke.{key}")
    requested = [name for group in args.categories for name in group.split(",")]
    if requested == ["all"]:
        requested = list(CATEGORIES)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(CATEGORIES):
        raise ValueError(f"--categories must be all or distinct names from {CATEGORIES}")
    args.categories = [name for name in CATEGORIES if name in requested]
    root = absolute(args.output_root or runner["output_root"]) / cfg["backbone"]["name"]
    return cfg, runner, root


def prepare_study(cfg, runner, root, *, device):
    check_assets(cfg, device)
    backbone = {k: cfg["backbone"][k] for k in
                ("name", "repo_dir", "weights", "frozen", "norm", "channels", "deepest_block", "patch_size")}
    backbone["checkpoint_sha256"] = sha256_file(backbone["weights"])
    repo = Path(backbone["repo_dir"])
    backbone["source_sha256"] = {str(p.relative_to(repo)): sha256_file(p)
                                 for p in sorted(repo.rglob("*.py"))}
    t = cfg["training"]
    training = {key: t[key] for key in ("seed", "dev_seed", "epochs", "batch_size", "max_steps",
                                       "max_minutes", "max_grad_norm", "checkpoint_interval_steps")}
    training.update(learning_rate=t["optimizer"]["lr"], weight_decay=t["optimizer"]["weight_decay"],
                    optimizer="AdamW", scheduler=None, amp=False, deterministic_algorithms=True,
                    train_synthetic_by_epoch=True, dev_synthetic_fixed=True)
    # Include reused implementations and read-only Data/Evaluator dependencies.
    sources = ["scripts/run_g2.py", "src/train/g2_e2.py", "src/train/g1_e1.py",
               "src/train/screen_adapter.py", "src/train/screen_representation.py",
               "src/train/optimizer.py", "src/train/overfit16.py", "src/models/msila.py",
               "src/models/contracts.py", "src/models/backbone_registry.py",
               "src/models/dinov3_extractor.py", "src/models/residual_adapter.py",
               "src/models/adapter_factory.py", "src/models/basic_decoder.py", "src/models/bilinear_sampling.py",
               "src/utils/checkpoint.py", "src/utils/resume.py", "src/losses/anomaly_loss.py"]
    sources += [str(p.relative_to(ROOT)) for directory in ("src/data", "src/eval")
                for p in sorted((ROOT / directory).rglob("*.py"))]
    protocol = dict(
        version="g2_d6_study_v1", backbone=backbone,
        adapter={k: cfg["adapter"][k] for k in ("kernel_size", "gamma_init", "bias")},
        decoder=cfg["decoder"], data=cfg["data"], training=training,
        loss={k: cfg["loss"][k] for k in ("bce_weight", "dice_weight", "dice_eps")},
        evaluation=cfg["evaluation"],
        synthetic_protocol=validate_native_protocol(read_yaml(absolute(cfg["synthetic_protocol"]))),
        selection=runner["selection"], categories=list(CATEGORIES),
        grid=[list(pair) for pair in selected_pairs(cfg, runner)], smoke=runner["smoke"],
        source_code_sha256={name: sha256_file(ROOT / name) for name in sources},
        runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda},
    )
    return dict(protocol=protocol, sha256=sha256_json(protocol), root=Path(root),
                pairs=selected_pairs(cfg, runner), contexts={})


def make_context(study, category, pair, stage, *, smoke=False, selection_sha256=None):
    cfg = deepcopy(study["protocol"])
    if smoke:
        cfg["training"].update(epochs=cfg["smoke"]["epochs"], max_steps=cfg["smoke"]["max_steps"])
        for key in ("max_train_sources", "max_dev_sources", "dev_variants_per_image"):
            cfg["data"][key] = cfg["smoke"][key]
    cfg["category"] = category
    cache_key = (category, smoke)
    if cache_key not in study["contexts"]:
        pools, manifest = discover_sources(cfg)
        tile_count = 0
        for record in pools["train"]:
            with Image.open(record.image_path) as image:
                tile_count += len(generate_tile_records(image.height, image.width, cfg["data"]["tile_size"],
                                                        cfg["data"]["overlap"], context_size=512))
        batches = math.ceil(tile_count * (cfg["data"]["train_variants_per_image"] + 1)
                            / cfg["training"]["batch_size"])
        expected = batches * cfg["training"]["epochs"]
        if cfg["training"]["max_steps"] is not None:
            expected = min(expected, cfg["training"]["max_steps"])
        positive_int(expected, "expected_steps")
        study["contexts"][cache_key] = pools, manifest, expected
    pools, manifest, expected = study["contexts"][cache_key]
    for key in ("selection", "categories", "grid", "smoke"):
        cfg.pop(key)
    cfg.update(stage=stage, mode="smoke" if smoke else "full", sources=manifest,
               study_sha256=study["sha256"], selection_sha256=selection_sha256, expected_steps=expected)
    cfg["adapter"].update(r=pair[0], d=pair[1])
    return dict(config=cfg, config_sha256=sha256_json(cfg), expected_steps=expected), pools


def run_directory(study, category, pair, stage, smoke=False):
    root = study["root"] / ("smoke" if smoke else "full") / stage
    if stage == "adapter_screen":
        root /= f"adapter_r{pair[0]}_d{pair[1]}"
    return root / category


def read_valid_result(directory, context):
    """A PASS label alone is insufficient: verify budget, DEV and checkpoints."""
    directory = Path(directory)
    try:
        result = json.loads((directory / "metrics.json").read_text())
        cfg = context["config"]
        required = dict(status="PASS", training_complete=True, config_sha256=context["config_sha256"],
                        study_sha256=cfg["study_sha256"], stage=cfg["stage"], category=cfg["category"],
                        mode=cfg["mode"], adapter={"r": cfg["adapter"]["r"], "d": cfg["adapter"]["d"]},
                        expected_steps=context["expected_steps"], global_step=context["expected_steps"],
                        frozen_backbone_unchanged=True, adapter_updated=True, decoder_updated=True)
        if any(result.get(k) != v for k, v in required.items()):
            return None
        dev = result["best_synthetic_dev"]
        score = dev["synthetic_dev_aupro_0_05"]
        if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or dev["qa_status"] != "PASS" or dev["dataset"] != "mvtec_ad2_good_synthetic"
                or dev["split"] != "dev_synthetic" or dev["native_resolution"] is not True
                or dev["aupro_max_fpr"] != .05
                or dev["n_samples"] != len(cfg["sources"]["dev"]) * (cfg["data"]["dev_variants_per_image"] + 1)
                or dev["dev_mixed"]["regions"] < 1 or dev["dev_mixed"]["aupro_0_05"] != score):
            return None
        for name in ("best", "last"):
            path = directory / f"{name}.pt"
            payload, _ = load_checkpoint_payload(path, require_sha256=True)
            if (sha256_file(path) != result[f"{name}_checkpoint_sha256"] or payload["config"] != cfg
                    or payload["metadata"].get("config_sha256") != context["config_sha256"]
                    or not 0 < payload["training_state"]["global_step"] <= context["expected_steps"]):
                return None
            if name == "last" and payload["training_state"]["global_step"] != context["expected_steps"]:
                return None
            state = payload["metadata"]
            if (state["best_metric"] != score or state["best_epoch"] != result["best_epoch"]
                    or state["decoder_initial_sha256"] != result["decoder_initial_sha256"]
                    or state["adapter_updated"] is not True or state["decoder_updated"] is not True):
                return None
            if name == "last" and state["dev_step"] != context["expected_steps"]:
                return None
            weights = payload["model_state"]
            if (any(not torch.isfinite(value).all() for value in weights.values())
                    or sum(value.numel() for key, value in weights.items() if key.startswith("adapter."))
                    != result["adapter_trainable_parameters"]):
                return None
            if name == "best" and (payload["metadata"]["best_metric"] != score
                                   or payload["training_state"]["epoch"] != result["best_epoch"]):
                return None
        result["metrics_sha256"] = sha256_file(directory / "metrics.json")
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError):
        return None


def real_evidence(result):
    return (result["mode"] == "full" and result.get("verification_scope") == "real_pretrained"
            and result.get("device", "").startswith("cuda"))


def collect_results(study, stage, *, pair=None, selection_sha256=None):
    rows, missing = [], []
    pairs = study["pairs"] if stage == "adapter_screen" else [pair]
    for candidate in pairs:
        for category in CATEGORIES:
            try:
                context, _ = make_context(study, category, candidate, stage, selection_sha256=selection_sha256)
                result = read_valid_result(run_directory(study, category, candidate, stage), context)
            except (FileNotFoundError, ValueError):
                result = None
            if result and real_evidence(result):
                rows.append(result)
            else:
                missing.append(dict(category=category, r=candidate[0], d=candidate[1]))
    return rows, missing


def rank_candidates(study, rows):
    fixed = study["protocol"]["adapter"]
    factory = ResidualAdapterFactory(AdapterFactoryConfig(
        in_dim=study["protocol"]["backbone"]["channels"], **fixed))
    ranking, decoder_hashes = [], {}
    for pair in study["pairs"]:
        group = [row for row in rows if row["adapter"] == {"r": pair[0], "d": pair[1]}]
        if len(group) != 8 or {r["category"] for r in group} != set(CATEGORIES):
            raise ValueError("Selection requires exactly eight categories per candidate")
        parameters = factory.build_rd(r=pair[0], d=pair[1]).trainable_params
        scores = {}
        for row in group:
            category = row["category"]
            initial = row["decoder_initial_sha256"]
            if decoder_hashes.setdefault(category, initial) != initial:
                raise ValueError("Decoder initialization drift across Adapter candidates")
            if row["adapter_trainable_parameters"] != parameters:
                raise ValueError("Adapter parameter count does not match the existing factory")
            scores[category] = row["best_synthetic_dev"]["synthetic_dev_aupro_0_05"]
        macro = math.fsum(scores[c] for c in CATEGORIES) / len(CATEGORIES)
        ranking.append(dict(r=pair[0], d=pair[1], adapter_trainable_parameters=parameters,
                            macro_synthetic_dev_aupro_0_05=macro, category_scores=scores))
    return sorted(ranking, key=lambda r: (-r["macro_synthetic_dev_aupro_0_05"],
                                          r["adapter_trainable_parameters"], r["r"], r["d"]))


def selection_payload(study, rows):
    if len(rows) != 72:
        raise G2Blocked(f"BLOCKED: selection requires 72 valid real runs, found {len(rows)}/72")
    ranking = rank_candidates(study, rows)
    fields = ("category", "adapter", "config_sha256", "metrics_sha256",
              "best_checkpoint_sha256", "last_checkpoint_sha256")
    evidence = [{key: row[key] for key in fields} for row in sorted(
        rows, key=lambda r: (r["adapter"]["r"], r["adapter"]["d"], CATEGORIES.index(r["category"])))]
    return dict(version="g2_adapter_selection_v1", study_sha256=study["sha256"],
                backbone=study["protocol"]["backbone"]["name"], categories=list(CATEGORIES),
                selection_rule=SELECTION_RULE, valid_runs=72, ranking=ranking,
                selected_pair={k: ranking[0][k] for k in ("r", "d")}, evidence=evidence)


def publish_selection(study):
    rows, _ = collect_results(study, "adapter_screen")
    if len(rows) != 72:
        return None
    payload = selection_payload(study, rows)
    record = dict(payload=payload, sha256=sha256_json(payload))
    path = study["root"] / "full" / "adapter_selection_lock.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise G2Blocked("BLOCKED: existing selection lock differs from current evidence")
    else:
        save_json(record, path)
    return record


def validate_selection(study):
    path = study["root"] / "full" / "adapter_selection_lock.json"
    if not path.is_file():
        raise G2Blocked("BLOCKED: adapter_selection_lock.json is missing; complete 72 screening runs first")
    record = json.loads(path.read_text())
    if record.get("sha256") != sha256_json(record.get("payload")):
        raise G2Blocked("BLOCKED: selection lock checksum mismatch")
    rows, _ = collect_results(study, "adapter_screen")
    expected = selection_payload(study, rows)
    if record["payload"] != expected:
        raise G2Blocked("BLOCKED: selection lock config/data/checkpoint provenance mismatch")
    return record


def execute_job(study, category, pair, stage, *, smoke, resume, device, selection_sha256=None):
    context, pools = make_context(study, category, pair, stage, smoke=smoke,
                                 selection_sha256=selection_sha256)
    directory = run_directory(study, category, pair, stage, smoke)
    if read_valid_result(directory, context):
        return "SKIP"
    metrics_path = directory / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()).get("status") == "PASS":
        raise G2Blocked(f"BLOCKED: completed run is invalid; inspect {directory} before rerunning")
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "train.log")
    logging.getLogger().addHandler(handler)
    try:
        result = train_e2(context, pools, directory, device=device, resume=resume)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    if result["status"] == "PASS" and read_valid_result(directory, context) is None:
        raise ValueError("Training finished but result/checkpoint validation failed")
    return result["status"]


def summarize(study, stage, outcomes, *, smoke, lock=None):
    screen, _ = collect_results(study, "adapter_screen")
    main = []
    if lock:
        selected = lock["payload"]["selected_pair"]
        main, _ = collect_results(study, "E2", pair=(selected["r"], selected["d"]), selection_sha256=lock["sha256"])
    all_jobs = bool(outcomes) and all(row["status"] in {"PASS", "SKIP"} for row in outcomes)
    full = len(screen) == 72 if stage == "adapter_screen" else len(main) == 8
    status = ("SMOKE_PASS" if all_jobs else "INCOMPLETE") if smoke else ("PASS" if full else "INCOMPLETE")
    return dict(status=status, stage=stage, mode="smoke" if smoke else "full",
                real_adapter_screen_pass=len(screen), expected_adapter_screen=72,
                real_E2_pass=len(main), expected_E2=8, counts_verified=True,
                study_sha256=study["sha256"], outcomes=outcomes,
                selection_lock=None if lock is None else str(study["root"] / "full" / "adapter_selection_lock.json"))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary_path = None
    try:
        cfg, runner, root = load_config(args)
        mode = "smoke" if args.smoke else "full"
        summary_path = root / mode / f"{args.stage}_summary.json"
        if args.stage == "E2" and not (root / "full" / "adapter_selection_lock.json").is_file():
            raise G2Blocked("BLOCKED: adapter_selection_lock.json is missing; complete 72 screening runs first")
        if not args.smoke and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
            raise FileNotFoundError("G2 NOT RUN: full D6 requires a usable CUDA device; smoke may use CPU")
        study = prepare_study(cfg, runner, root, device=args.device)
        enforce_lock(root / mode, study["protocol"])
        lock = validate_selection(study) if args.stage == "E2" else None
        pairs = study["pairs"] if lock is None else [(lock["payload"]["selected_pair"]["r"],
                                                     lock["payload"]["selected_pair"]["d"])]
        outcomes = []
        for pair in pairs:
            for category in args.categories:
                row = dict(category=category, r=pair[0], d=pair[1])
                try:
                    row["status"] = execute_job(study, category, pair, args.stage, smoke=args.smoke,
                                                resume=args.resume, device=args.device,
                                                selection_sha256=None if lock is None else lock["sha256"])
                except (OSError, ValueError, RuntimeError) as exc:
                    row.update(status="BLOCKED" if "BLOCKED" in str(exc) else "FAIL", reason=str(exc))
                    save_json(row, run_directory(study, category, pair, args.stage, args.smoke) / "failure.json")
                    logging.error("%s", row)
                outcomes.append(row)
        if args.stage == "adapter_screen" and not args.smoke:
            lock = publish_selection(study)
        result = summarize(study, args.stage, outcomes, smoke=args.smoke, lock=lock)
        if any(row["status"] == "FAIL" for row in outcomes):
            result["status"] = "FAIL"
        elif any(row["status"] == "BLOCKED" for row in outcomes):
            result["status"] = "BLOCKED"
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        status = "BLOCKED" if "BLOCKED" in str(exc) else ("NOT RUN" if isinstance(exc, FileNotFoundError) else "FAIL")
        result = dict(status=status, stage=args.stage, mode="smoke" if args.smoke else "full", reason=str(exc),
                      real_adapter_screen_pass=0, expected_adapter_screen=72,
                      real_E2_pass=0, expected_E2=8, counts_verified=False)
    if summary_path is not None:
        save_json(result, summary_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"PASS", "SMOKE_PASS"} else (1 if result["status"] == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
