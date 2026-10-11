"""D10 orchestration: existing E5 trainer + unchanged TV2 evaluator/intake.

CPU preparation/fixtures are never experimental evidence. No selection lock.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import gc
import json
import logging
import math
from pathlib import Path
import re
import subprocess

from PIL import Image
import torch

from src.eval import g2_joint as joint
from src.models.backbone_registry import backbone_spec
from src.models.contracts import MULTIVIEW_FEATURE_KEYS
from src.models.fusion_factory import canonical_fusion_method
from src.train.g2_e2 import FrozenE5Factory, train_e5
from src.train.screen_representation import read_yaml, save_json, save_yaml, sha256_file, sha256_json
from src.utils.resume import load_checkpoint_payload

ROOT = Path(__file__).resolve().parents[2]
OWNERS = {"tv1": joint.CATEGORIES[:4], "tv2": joint.CATEGORIES[4:]}
PAIRS = [(r, d) for r in (64, 128, 256) for d in (256, 512, 768)]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def absolute(value):
    path = Path(value).expanduser()
    return (ROOT / path if not path.is_absolute() else path).resolve()


def verify_d9(path):
    """Stop before planning jobs if any audited candidate is missing/invalid."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(r, d, f) for r, d in PAIRS for f in joint.FUSIONS}
    actual = [(int(row["r"]), int(row["d"]), canonical_fusion_method(row["fusion"])) for row in rows]
    require(len(actual) == 45 and set(actual) == expected, "BLOCKED: D9 must contain all 45 unique candidates")
    failed = [f"{key}: {row['status']} {row.get('error', '')}" for key, row in zip(actual, rows) if row["status"] != "PASS"]
    require(not failed, "BLOCKED: repair invalid D9 search space before D10: " + "; ".join(failed))
    capacities = {(r, d): {row["adapter_params"] for row in rows if (int(row["r"]), int(row["d"])) == (r, d)} for r, d in PAIRS}
    require(all(len(v) == 1 for v in capacities.values()) and len({next(iter(v)) for v in capacities.values()}) == 9,
            "BLOCKED: r/d must produce nine distinct Adapter parameter counts")
    return sha256_file(path)


def load_settings(args):
    from scripts import run_g2 as legacy
    path = absolute(args.config)
    if path == ROOT / "configs/g2_runner.yaml":
        path = ROOT / "configs/g2_joint_search.yaml"
    settings = read_yaml(path)
    require(settings.get("version") == "g2_d10_runner_v1", "Expected g2_d10_runner_v1")
    require(settings.get("batch_policy") == "fixed_for_all_candidates"
            and settings.get("oom_policy") == "record_failure_no_automatic_batch_change"
            and settings.get("drive_sync") == "epoch_and_job_end", "D10 requires fixed batch and epoch/job Drive sync")
    joint.validate_budget(settings["screening_budget"])
    require(settings["screening_budget"]["max_minutes"] is None, "Screening budget cannot truncate candidates by wall time")
    base_args = deepcopy(args)
    base_args.config = str(absolute(settings["base_config"]))
    base_args.categories = ["all"]
    cfg, runner, _ = legacy.load_config(base_args)
    require(cfg["backbone"]["name"] == "dinov3_vitb16" and backbone_spec(cfg["backbone"]["name"]).channels == 768,
            "D10 protocol is verified only for ViT-B/16 width 768")
    require(legacy.selected_pairs(cfg, runner) == PAIRS, "BLOCKED: D10 grid differs from D6/D9")
    settings = deepcopy(settings)
    settings["d9_sha256"] = verify_d9(absolute(settings["d9_preflight"]))
    d6 = absolute(settings.get("d6_preflight", "outputs/G2/D6/adapter_preflight.csv"))
    if d6.exists():
        with d6.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 9 and {(int(row["r"]), int(row["d"])) for row in rows} == set(PAIRS)
                and all(row["status"] == "PASS" for row in rows),
                "BLOCKED: D6 reports an invalid/missing pair; repair search space before D10")
        settings["d6_sha256"] = sha256_file(d6)
    cfg["main_budget"] = {key: cfg["training"][key] for key in joint.BUDGET_FIELDS}
    cfg["screening_budget"] = deepcopy(settings["screening_budget"])
    cfg["training"].update(cfg["screening_budget"])
    cfg["paired_views"]["pair_strategy"] = settings.get("paired_strategy", cfg["paired_views"]["pair_strategy"])
    require(cfg["paired_views"]["pair_strategy"] in {"concat", "sequential"}, "Invalid paired strategy")
    workers = settings.get("num_workers", cfg["data"]["num_workers"])
    require(type(workers) is int and workers >= 0, "num_workers must be nonnegative")
    cfg["data"]["num_workers"] = workers
    # Keep evaluation batches no larger than the declared training batch.
    cfg["evaluation"]["tile_batch_size"] = min(cfg["evaluation"]["tile_batch_size"], cfg["training"]["batch_size"])
    owned = OWNERS[args.owner]
    requested = [c for group in args.categories for c in group.split(",")]
    if requested == ["all"]:
        requested = list(owned)
    require(requested and len(set(requested)) == len(requested) and set(requested).issubset(owned),
            f"{args.owner} categories must be a subset of {owned}")
    args.categories = [c for c in owned if c in requested]
    if args.r is not None:
        require((args.r, args.d) in PAIRS, "Requested pair is outside verified D9 grid")
    args.fusions = [canonical_fusion_method(f) for f in args.fusions] if args.fusions else list(joint.FUSIONS)
    require(len(set(args.fusions)) == len(args.fusions), "Duplicate Fusion filter")
    name = args.run_id or settings["run_id"]
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None, "Invalid run_id namespace")
    root = absolute(args.output_root or settings["output_root"]) / name
    return cfg, runner, settings, root


def create_study(cfg, runner, settings, root, *, device, assets):
    from scripts import run_g2 as legacy
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if assets:
        study = legacy.prepare_study(cfg, runner, root / "_new_joint_protocol", device=device)
        protocol = study["protocol"]
        protocol.update(version="g2_d10_study_v1", fusion=deepcopy(cfg["fusion"]), paired_views=deepcopy(cfg["paired_views"]),
                        screening_budget=cfg["screening_budget"], main_budget=cfg["main_budget"], checkpoint=cfg["checkpoint"],
                        git_sha=commit, d9_sha256=settings["d9_sha256"], d6_sha256=settings.get("d6_sha256"))
        protocol["backbone"]["feature_blocks"] = list(backbone_spec("dinov3_vitb16").blocks)
        protocol["training"]["optimizer"] = deepcopy(cfg["training"]["optimizer"])
        extra = ("src/train/g2_joint_search.py", "src/models/fusion_factory.py", "src/models/position_attention_fusion.py",
                 "src/models/fusions/weighted_sum.py", "src/models/fusions/concat.py", "src/models/fusions/gated.py")
        protocol["source_code_sha256"].update({p: sha256_file(ROOT / p) for p in extra})
        from scripts.eval_g2 import source_identity
        datasets = {}
        for category in joint.CATEGORIES:
            context, _ = legacy.make_context(study, category, PAIRS[0], "adapter_screen")
            datasets[category] = sha256_json(source_identity(context["config"]["sources"]))
        protocol["dataset_source_sha256"] = datasets
        study["sha256"] = sha256_json(protocol)
        prototype = deepcopy(cfg)
        prototype["backbone"] = deepcopy(protocol["backbone"])
    else:
        study = dict(protocol=None, sha256=None)
        prototype = deepcopy(cfg)
    # TV2 spec API reads canonical config files; no alternate metric implementation.
    config_root = root / (study["sha256"] or "prepared") / "shared_config"
    save_yaml(prototype, config_root / "model.yaml")
    spec_runner = deepcopy(runner)
    spec_runner["screening_budget"] = cfg["screening_budget"]
    spec_runner["backbone"] = cfg["backbone"]["name"]
    save_yaml(spec_runner, config_root / "runner.yaml")
    spec = joint.load_spec(config_root / "model.yaml", config_root / "runner.yaml", root=ROOT, integration_sha=commit)
    require(spec["pairs"] == [list(p) for p in PAIRS], "Joint spec grid drift")
    study.update(spec=spec, namespace_root=root, config_root=config_root, config=cfg, settings=settings, commit=commit)
    return study


def make_context(study, category, pair, fusion, *, smoke=False):
    from scripts import run_g2 as legacy
    context, pools = legacy.make_context(study, category, pair, "adapter_screen", smoke=smoke)
    cfg = context["config"]
    cfg.update(stage="E5", mode="smoke" if smoke else "screening", selection_status="pending_joint_selection_D10",
               selection_sha256=None, adapter_pair_role="joint_candidate", git_sha=study["commit"],
               run_id=joint.run_id(category, *pair, fusion, cfg["training"]["seed"]))
    cfg["fusion"]["method"] = fusion
    cfg["screening_budget"] = {key: cfg["training"][key] for key in joint.BUDGET_FIELDS}
    context.update(config_sha256=sha256_json(cfg), fairness_root=study["root"] / ("smoke" if smoke else "screening") / "fairness" / f"r{pair[0]}_d{pair[1]}")
    return context, pools


def directory_for(study, row, *, smoke=False):
    return study["root"] / ("smoke" if smoke else "train") / row["run_id"]


def valid_training(directory, context):
    """Strict hashes, candidate, budget, trained finite heads, optimizer/RNG/cursor."""
    try:
        cfg, digest = context["config"], context["config_sha256"]
        report = json.loads((directory / "metrics.json").read_text())
        required = dict(status="PASS", training_complete=True, stage="E5", mode=cfg["mode"],
                        category=cfg["category"], fusion=cfg["fusion"]["method"],
                        adapter={k: cfg["adapter"][k] for k in ("r", "d")},
                        config_sha256=digest, expected_steps=context["expected_steps"], global_step=context["expected_steps"],
                        run_id=cfg["run_id"], git_sha=cfg["git_sha"], screening_budget=cfg["screening_budget"],
                        feature_sources=6, frozen_backbone_unchanged=True, adapter_updated=True,
                        decoder_updated=True, projection_updated=True, context_aligned=True)
        require(all(report.get(k) == v for k, v in required.items()), "Incomplete/mismatched training result")
        require(report["adapters_updated"] == dict.fromkeys(MULTIVIEW_FEATURE_KEYS[:3], True), "Missing source Adapter update")
        metric = report["best_synthetic_dev"]["synthetic_dev_aupro_0_05"]
        require(joint.unit_metric(metric) and report["best_synthetic_dev"]["qa_status"] == "PASS", "Invalid synthetic DEV")
        for name in ("best", "last"):
            path = directory / f"{name}.pt"
            payload, _ = load_checkpoint_payload(path, require_sha256=True)
            meta, state = payload["metadata"], payload["training_state"]
            require(payload["config"] == cfg and meta["config_sha256"] == digest, "Checkpoint config drift")
            require(meta["git_sha"] == cfg["git_sha"] and meta["run_id"] == cfg["run_id"], "Checkpoint candidate provenance drift")
            require(sha256_file(path) == report[f"{name}_checkpoint_sha256"], "Report checkpoint checksum drift")
            require(0 < state["global_step"] <= context["expected_steps"] and meta["history"], "Untrained checkpoint")
            require(meta["best_metric"] == metric and meta["best_epoch"] == report["best_epoch"], "Checkpoint DEV drift")
            require(payload["optimizer_state"]["state"] and payload["rng_state"] and type(meta["next_epoch"]) is int
                    and type(meta["next_batch"]) is int, "Missing optimizer/RNG/cursor")
            require(all(bool(torch.isfinite(value).all()) for state in payload["optimizer_state"]["state"].values()
                        for value in state.values() if isinstance(value, torch.Tensor)), "Nonfinite optimizer state")
            require(all(bool(torch.isfinite(v).all()) for v in payload["model_state"].values()), "Nonfinite checkpoint weights")
            groups = {k: sum(v.numel() for n, v in payload["model_state"].items() if n.startswith(k + "."))
                      for k in ("adapters", "projection", "decoder", "fusion")}
            require(report["parameter_report"]["groups"] == groups
                    and report["parameter_report"]["trainable_total"] == sum(groups.values()), "Parameter report drift")
            require(groups["fusion"] == 0 or meta["fusion_updated"] is True, "Learned Fusion not updated")
            if name == "last":
                require(state["global_step"] == context["expected_steps"] and meta["dev_step"] == state["global_step"], "Uncommitted final DEV/budget")
        return report
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError):
        return None


def initialize_rows(study):
    root = study["root"]
    spec_path, path = root / "joint_search_spec.json", root / "joint_search_360.csv"
    if spec_path.exists():
        require(json.loads(spec_path.read_text()) == study["spec"], "BLOCKED: immutable joint spec drift")
    save_json(study["spec"], spec_path)
    return joint.read_manifest(path, study["spec"]) if path.exists() else joint.skeleton(study["spec"])


def persist(study, rows, owner):
    root, spec = study["root"], study["spec"]
    joint.write_manifest(root / "joint_search_360.csv", rows, spec)
    owned = [row for row in rows if row["category"] in OWNERS[owner]]
    require(len(owned) == 180, "Owner manifest must retain 180 cells")
    path = root / f"{owner}_joint_search_180.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=joint.FIELDS)
        writer.writeheader()
        for row in owned:
            writer.writerow({**row, "budget": json.dumps(row["budget"], sort_keys=True) if row["budget"] is not None else ""})
    temporary.replace(path)
    counts = Counter(row["status"] for row in owned)
    coverage = dict(status="PASS" if counts["PASS"] == 180 else ("NOT RUN" if counts["NOT RUN"] == 180 else "INCOMPLETE"),
                    real_pass=counts["PASS"], expected=180, owner=owner, categories=list(OWNERS[owner]),
                    counts=dict(counts), full_search_cells=360, screening_only=True, selection_performed=False,
                    selection_lock=None, protocol_sha256=study["sha256"], git_sha=study["commit"])
    save_json(coverage, root / "training_coverage.json")
    return coverage


def mirror(study, directory=None):
    """One sync per epoch/job/error; never per train step or on a timer."""
    if study.get("drive_root") is None:
        return
    from scripts.eval_g2 import copy_atomic, sync_tree
    root, target = study["root"], study["drive_root"]
    if directory is not None:
        sync_tree(directory, target / directory.relative_to(root))
    for file in root.iterdir():
        if file.is_file() and not file.name.endswith(".tmp"):
            copy_atomic(file, target / file.name)


def execute(study, row, *, device, resume, smoke=False, model_factory=None):
    from scripts import eval_g2
    context, pools = make_context(study, row["category"], (row["r"], row["d"]), row["fusion"], smoke=smoke)
    directory = directory_for(study, row, smoke=smoke)
    completed = valid_training(directory, context)
    if completed is None:
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(directory / "train.log")
        logging.getLogger().addHandler(handler)
        try:
            def epoch_sync(path, epoch):
                save_json(dict(run_id=row["run_id"], epoch=epoch, status="RUNNING",
                               config_sha256=context["config_sha256"]), path / "epoch_progress.json")
                mirror(study, path)
            completed = train_e5(context, pools, directory, device=device, resume=resume,
                                 model_factory=model_factory, on_epoch=epoch_sync)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    require(valid_training(directory, context) is not None, "Training output failed full-budget validation; resume required")
    if smoke:
        return {**row, "budget": context["config"]["screening_budget"],
                "status": "SMOKE_PASS", "reason": "Technical smoke excluded from screening evidence"}
    require(completed["verification_scope"] == "real_pretrained" and torch.device(device).type == "cuda",
            "CPU/fixture training cannot produce experimental PASS")
    intake = deepcopy(row)
    intake.update(status="READY", reason="Trained checkpoint ready for unchanged TV2 evaluator",
                  checkpoint=str((directory / "best.pt").resolve()), config_hash=context["config_sha256"],
                  git_sha=study["commit"], checkpoint_sha256=sha256_file(directory / "best.pt"))
    eval_dir = study["root"] / "evaluation" / row["run_id"]
    args = eval_g2.parse_args(["--joint-manifest", str(study["root"] / "joint_search_360.csv"),
                              "--device", str(device), "--resume", "--output-root", str(study["root"] / "evaluation")])
    args.joint_record, args.joint_spec, args.evaluation_directory = intake, study["spec"], str(eval_dir)
    # The evaluator owns model restoration. Park the training cache on CPU so
    # two pretrained copies do not reside on GPU during its native evaluation.
    if model_factory is not None:
        model_factory.extractor.cpu()
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    result = eval_g2.evaluate_category(args, "E5", row["category"])
    accepted = eval_g2.joint_result_row(intake, result, eval_dir)
    eval_g2.audit_joint_row(accepted, study["spec"])
    mirror(study, eval_dir)
    return accepted


def run_from_cli(args):
    """Prepare is asset-free; real screening requires Colab CUDA explicitly."""
    study = None
    try:
        cfg, runner, settings, namespace = load_settings(args)
        assets = not args.prepare
        if assets and not args.preflight and not args.report_only:
            require(torch.device(args.device).type == "cuda" and torch.cuda.is_available(),
                    "NOT RUN: real joint screening requires a Colab CUDA runtime; no local CPU experiment")
        study = create_study(cfg, runner, settings, namespace, device=args.device, assets=assets)
        study["runtime"] = dict(device=args.device, torch=str(torch.__version__), cuda=torch.version.cuda,
                                gpu=torch.cuda.get_device_name(torch.device(args.device))
                                if assets and torch.device(args.device).type == "cuda" else None,
                                dtype="float32", amp=False, tf32=False)
        study["root"] = namespace / (study["sha256"] or "prepared") / args.owner
        if args.drive_output:
            study["drive_root"] = absolute(args.drive_output) / namespace.name / (study["sha256"] or "prepared") / args.owner
        rows = initialize_rows(study)
        if assets:
            save_json(study["protocol"], study["root"] / "training_protocol.json")
        save_json(dict(status="PREPARED" if args.prepare else "READY", git_sha=study["commit"],
                       owner=args.owner, settings=settings, screening_budget=cfg["screening_budget"],
                       main_budget=cfg["main_budget"], d9_status="45/45 PASS", scope="E5 screening only",
                       runtime=study["runtime"],
                       verification="NOT RUN" if args.prepare else "assets_verified", gpu_training="NOT RUN",
                       config_sha256=sha256_json(cfg), protocol_sha256=study["sha256"]), study["root"] / "run_manifest.json")
        persist(study, rows, args.owner)
        mirror(study)
        if args.prepare or args.preflight:
            print(json.dumps(dict(status="PREPARED" if args.prepare else "READY", root=str(study["root"]),
                                  real_pass=0, expected=180, experiment_status="NOT RUN"), indent=2))
            return 0
        from scripts.eval_g2 import audit_joint_row
        factory, attempts, smoke_pass = None, 0, False
        for i, row in enumerate(rows):
            if row["category"] not in args.categories or row["fusion"] not in args.fusions or args.r is not None and (row["r"], row["d"]) != (args.r, args.d):
                continue
            directory = directory_for(study, row, smoke=args.smoke)
            touched = False
            try:
                if row["status"] == "PASS" and not args.smoke:
                    context, _ = make_context(study, row["category"], (row["r"], row["d"]), row["fusion"])
                    require(valid_training(directory, context) is not None, "Stored PASS failed training audit")
                    audit_joint_row(row, study["spec"])
                    continue
                if args.report_only:
                    continue
                if args.max_jobs is not None and attempts >= args.max_jobs:
                    break
                if args.smoke and attempts >= 1:
                    break
                touched = True
                attempts += 1
                planned, _ = make_context(study, row["category"], (row["r"], row["d"]), row["fusion"], smoke=args.smoke)
                save_yaml(planned["config"], directory / "planned_config.yaml")
                save_json({"sha256": planned["config_sha256"]}, directory / "planned_config_hash.json")
                if factory is None:
                    factory = FrozenE5Factory(cfg, device=args.device)
                output = execute(study, row, device=args.device, resume=args.resume, smoke=args.smoke, model_factory=factory)
                if args.smoke:
                    save_json(output, directory / "smoke_result.json")
                    smoke_pass = True
                else:
                    rows[i] = output
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                touched = True
                oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                failure = dict(run_id=row["run_id"], category=row["category"], r=row["r"], d=row["d"], fusion=row["fusion"],
                               seed=row["seed"], screening_budget=row["budget"], git_sha=study["commit"], device=args.device,
                               gpu=study["runtime"]["gpu"],
                               protocol_sha256=study["sha256"], status="FAIL", failure_kind="OOM" if oom else "ERROR", error=str(exc))
                count = len(list((directory / "attempts").glob("*.json"))) + 1
                save_json(failure, directory / "attempts" / f"failure_{count:04d}.json")
                save_json(failure, directory / "failure.json")
                if not args.smoke:
                    rows[i] = {**row, "status": "FAIL", "reason": f"{'OOM' if oom else 'ERROR'}: {exc}", "benchmark_status": "NOT RUN",
                               **dict.fromkeys((*joint.METRIC_FIELDS, "latency", "peak_vram"))}
                logging.error("%s", failure)
            finally:
                if touched:
                    if factory is not None:
                        factory.extractor.cpu()
                    gc.collect()
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
                    persist(study, rows, args.owner)
                    mirror(study, directory)
        coverage = persist(study, rows, args.owner)
        manifest_path = study["root"] / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(gpu_training="ATTEMPTED" if attempts else "NOT RUN", experiment_status=coverage["status"],
                        real_pass=coverage["real_pass"], attempted_this_invocation=attempts)
        save_json(manifest, manifest_path)
        mirror(study)
        if args.smoke:
            coverage = {**coverage, "smoke_jobs": attempts, "status": "SMOKE_PASS" if smoke_pass else "FAIL"}
        print(json.dumps({**coverage, "root": str(study["root"]), "attempts": attempts}, indent=2))
        return 0 if coverage["status"] in {"PASS", "SMOKE_PASS"} else 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result = dict(status="NOT RUN" if "NOT RUN" in str(exc) else "BLOCKED", reason=str(exc), real_pass=0, expected=180)
        if study is not None and "root" in study:
            save_json(result, study["root"] / "attempt_status.json")
            mirror(study)
        print(json.dumps(result, indent=2))
        return 2
