#!/usr/bin/env python3
"""G2 TV1: D6 preflight and D7 E3 preparation/debug; main awaits D10.

Examples (paths/checkpoints configured in YAML):
  python scripts/run_g2.py --stage adapter_preflight --device cpu
  python scripts/run_g2.py --stage E2 --smoke --debug-pair 128 512 --device cuda
  python scripts/run_g2.py --stage E3 --prepare --debug-pair 128 512 --device cpu
  python scripts/run_g2.py --stage E3 --smoke --categories rice --debug-pair 128 512 --device cuda

Smoke uses a separate namespace and never supplies selection evidence.
The old adapter_screen CLI is a CPU-preflight alias, never 72 training jobs.
Legacy selection readers remain for historical audit, not new selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
import torch

from src.data.synthetic_anomaly import validate_native_protocol
from src.data.tiling import generate_tile_records
from src.models.adapter_factory import AdapterFactoryConfig, ResidualAdapterFactory
from src.models.backbone_registry import BACKBONES, adapter_pairs, backbone_spec
from src.models.feature_selector import FeatureSelector
from src.models.msila import resolve_g2_config
from src.train.g1_e1 import check_assets, discover_sources
from src.train.g2_e2 import train_e2, train_e3, train_e4, infer_e4
from src.train.g2_comparison import comparison_identity, write_comparison_inputs
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
PENDING_SELECTION = dict(status="pending_joint_selection_D10", method="joint_rd_fusion",
                         legacy_adapter_lock="audit_only")

# Audited D6/D7 implementations whose E2/E3 paths this E4 extension preserves.
# Other dependency/source changes still block reuse; E4 hashes its own sources.
D6_SOURCE_BASELINE = {
    "scripts/run_g2.py": "1151340f61700614eae50cc52d90e59c5c4e3b45aedddf891899e0f6881fa2ad",
    "src/train/g2_e2.py": "b236d460303531eb684860e5494d3826f17c0ee374be4459f166c4f65225d16c",
    "src/models/msila.py": "a0441e6fd79b7db7537c12d362ec87b545b2130507185d03638e21b024a60fe4",
    "src/train/g1_e1.py": "dd7b311e4766abac5499f55606520e72650d5e534f12b48087393fabe7e9dd80",
}
D7_SOURCE_BASELINE = {
    "scripts/run_g2.py": "0acbf56680b4cf235772d3896746be14b19ff73fd20e7245f05ff2d58e39e3b0",
    "src/models/msila.py": "96c9564a9e9337e26a8f85cbc18a8d5bff080dfd74e9690c7fb2c54d9825f9f8",
    "src/train/g2_e2.py": "399f9b365603cc3558f01b75cb9819bd716a0295f4f4b15ec57e069ceacc752a",
    "src/train/g1_e1.py": D6_SOURCE_BASELINE["src/train/g1_e1.py"],
    "src/train/g2_comparison.py": "1ccfd165f5d5feb0173ff37fc93d6939b4f08e29e997dde5e940d6f640e3554f",
}


class G2Blocked(RuntimeError):
    pass


def absolute(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g2_runner.yaml")
    parser.add_argument("--stage", required=True, choices=("adapter_preflight", "adapter_screen", "E2", "E3", "E4"))
    parser.add_argument("--debug-pair", nargs=2, type=int, metavar=("R", "D"),
                        help="E2/E3 debug: explicit provisional widths; no selection evidence")
    parser.add_argument("--prepare", action="store_true",
                        help="E3: write PREPARED manifest without loading data/weights or training")
    parser.add_argument("--inference", action="store_true", help="E4: restore best.pt, export native DEV maps")
    parser.add_argument("--categories", nargs="+", default=["all"], help="all, names or comma-separated names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    for flag in ("data-root", "repo-dir", "weights", "output-root"):
        parser.add_argument(f"--{flag}")
    args = parser.parse_args(argv)
    if args.inference and args.stage != "E4":
        parser.error("--inference is supported for --stage E4")
    if args.prepare and (args.stage != "E3" or args.smoke or args.inference or args.resume):
        parser.error("--prepare requires --stage E3 and cannot train, infer or resume")
    if args.debug_pair and not (args.stage in {"E2", "E3"} and (args.smoke or args.prepare)):
        parser.error("--debug-pair requires --stage E2/E3 --smoke, or --stage E3 --prepare")
    return args


def selected_pairs(cfg, runner):
    name = cfg["backbone"]["name"]
    if name not in runner["adapter_grids"]:
        raise ValueError(f"Declare a 3 x 3 adapter_grids entry for {name} before preflight")
    grid = runner["adapter_grids"][name]
    r_values, d_values = grid["r_values"], grid["d_values"]
    if name in GRIDS and (r_values, d_values) != GRIDS[name]:
        raise ValueError(f"Expected the declared D6 r/d grid for {name}: {GRIDS[name]}")
    if len(r_values) != 3 or len(d_values) != 3:
        raise ValueError("Adapter preflight requires exactly three r and three d values")
    return adapter_pairs(cfg["backbone"]["channels"], {
        "pairs": [[r, d] for r in r_values for d in d_values],
    })


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def load_config(args):
    runner = read_yaml(absolute(args.config))
    if runner.get("version") != "g2_runner_v2" or runner.get("selection") != PENDING_SELECTION:
        raise ValueError("Expected g2_runner_v2: D6 preflight, pending joint selection D10")
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
    if args.debug_pair:
        cfg["adapter"].update(r=args.debug_pair[0], d=args.debug_pair[1])
        cfg = resolve_g2_config(cfg, root=ROOT)
    cfg["data"]["root"] = str(absolute(cfg["data"]["root"]))
    cfg["data"].setdefault("max_train_sources", None)
    cfg["data"].setdefault("max_dev_sources", 4)
    cfg["evaluation"].setdefault("example_limit", 0)
    cfg["training"].update(runner["training"])
    if runner["execution"] != {"full_device": "cuda", "deterministic_resize": True}:
        raise ValueError("Full D6 requires CUDA and deterministic decoder resize")
    cfg["decoder"]["deterministic_resize"] = True
    positive_int(cfg["fusion"]["dim"], "fusion.dim")
    # Old D6/D7 model configs remain usable with the existing Context geometry.
    views = cfg.setdefault("paired_views", {"context_size": cfg["data"]["context_source_fov"][0],
                                           "pair_strategy": "concat", "deterministic_sampling": True})
    if (views.get("context_size") != cfg["data"]["context_source_fov"][0]
            or cfg["data"]["context_source_fov"] != [views.get("context_size")] * 2
            or type(views.get("context_size")) is not int or views["context_size"] < 512
            or (views["context_size"] - 512) % 2
            or views.get("pair_strategy") not in {"concat", "sequential"}
            or views.get("deterministic_sampling") is not True
            or cfg["data"]["context_input_size"] != [512, 512]):
        raise ValueError("E4 requires centered Context FOV >=512, input 512, deterministic alignment")
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
    default_output = runner.get("stage_output_roots", {}).get(args.stage, runner["output_root"])
    root = absolute(args.output_root or default_output) / cfg["backbone"]["name"]
    return cfg, runner, root


def debug_pair(pair):
    r, d = pair
    positive_int(r, "debug_pair.r")
    positive_int(d, "debug_pair.d")
    return dict(name=f"debug_pair_r{r}_d{d}", r=r, d=d, role="technical_only")


def prepare_e3(cfg, root):
    """Prepare resolved E3 provenance without claiming training or GPU evidence."""
    resolved = deepcopy(cfg)
    pair = debug_pair((cfg["adapter"]["r"], cfg["adapter"]["d"]))
    spec = backbone_spec(cfg["backbone"]["name"])
    keys = list(FeatureSelector("multi_local").source_keys)
    resolved.update(stage="E3", mode="prepare", selection_status=PENDING_SELECTION["status"],
                    adapter_pair_role="debug_pair", debug_pair=pair, selection_sha256=None)
    resolved["backbone"]["feature_blocks"] = list(spec.blocks)
    resolved["fusion"].update(method="mean")
    sources = ("scripts/run_g2.py", "src/train/g2_e2.py", "src/models/msila.py",
               "src/models/backbone_registry.py", "src/models/dinov3_extractor.py",
               "src/models/adapter_factory.py", "src/models/residual_adapter.py",
               "src/models/feature_selector.py", "src/models/feature_projection.py",
               "src/models/mean_fusion.py", "src/models/basic_decoder.py",
               "tests/test_g2_tv1_e3.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           check=True, text=True, capture_output=True).stdout
    manifest = dict(version="g2_d7_e3_prepared_v3", status="PREPARED", experiment="E3",
                    config=resolved, config_sha256=sha256_json(resolved), seed=cfg["training"]["seed"],
                    git_commit=commit, git_dirty=bool(dirty), debug_pair=pair,
                    selection_status=PENDING_SELECTION["status"], selection_lock=None,
                    architecture=dict(feature_mode="multilayer", num_sources=3, context=False,
                                      source_blocks=dict(zip(keys, spec.blocks)), block_index_base=1,
                                      adapter_sharing="independent_per_layer", adapter_channels=spec.channels,
                                      projected_shape=[None, cfg["fusion"]["dim"], 32, 32],
                                      fusion="mean", output_shape=[None, 1, 512, 512], frozen_backbone=True),
                    source_code_sha256={name: sha256_file(ROOT / name) for name in sources},
                    gpu_smoke="NOT RUN", peak_vram_if_measured=None, main_training="NOT RUN",
                    E3_minus_E2="NOT RUN: deferred to D11 with real main checkpoints",
                    main_blocker="pending joint selection D10 and D11 protocol integration")
    path = root / pair["name"] / "E3_prepared_manifest.json"
    # Never overwrite a PREPARED artifact when its source/config provenance changes.
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise G2Blocked(f"BLOCKED: prepared provenance changed; use a new --output-root (preserved {path})")
    save_json(manifest, path)
    return dict(status="PREPARED", stage="E3", operation="prepare", manifest=str(path),
                debug_pair=pair, selection_status=PENDING_SELECTION["status"], gpu_smoke="NOT RUN",
                main_training="NOT RUN", E3_minus_E2=manifest["E3_minus_E2"])


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
    protocol = reuse_d6_protocol(protocol, Path(root))
    e3_files = (*sources, "src/models/feature_selector.py", "src/models/feature_projection.py",
                "src/models/mean_fusion.py", "src/train/g2_comparison.py")
    e3 = dict(feature_mode="multilayer", source_blocks=list(backbone_spec(backbone["name"]).blocks),
              source_keys=list(FeatureSelector("multi_local").source_keys),
              adapter_sharing="independent_per_layer", fusion="mean", context=False,
              feature_width=cfg["fusion"]["dim"],
              implementation_sha256={name: sha256_file(ROOT / name) for name in e3_files})
    e4 = {**deepcopy(e3), "source_keys": list(FeatureSelector("multi_local_context").source_keys),
          "adapter_sharing": "shared_across_views_per_layer", "projection_sharing": "shared_across_views_per_layer",
          "context": True, "context_alignment": "before_projection_and_fusion",
          "paired_views": deepcopy(cfg["paired_views"])}
    e4_files = (*e3_files, "src/train/g2_context.py", "src/models/context_alignment.py",
                "src/models/cached_training.py", "src/geometry/view_meta.py")
    e4["implementation_sha256"] = {name: sha256_file(ROOT / name) for name in e4_files}
    e3_smoke = reuse_e3_architecture(e3, Path(root), sha256_json(protocol), mode="smoke")
    e3 = reuse_e3_architecture(e3, Path(root), sha256_json(protocol))
    return dict(protocol=protocol, sha256=sha256_json(protocol), root=Path(root), e3=e3, e4=e4,
                e3_smoke=e3_smoke, pairs=selected_pairs(cfg, runner), contexts={})


def reuse_d6_protocol(current, root):
    """Read completed D6 artifacts against their immutable historical config.

    Reuse is limited to the audited D6/D7 -> D8 TV1 source extension. Scientific
    settings, DINO checkout/checkpoint and Data/Evaluator/loss code stay exact;
    g1_e1's training prediction hook preserves its default E1/E2/E3 path.
    The stored config/metric/checkpoint hashes are never rewritten.
    """
    path = root / "full" / "protocol_lock.json"
    if not path.is_file():
        return current
    record = json.loads(path.read_text())
    old = record["payload"]
    if record.get("sha256") != sha256_json(old):
        raise G2Blocked("BLOCKED: historical protocol checksum mismatch")
    if old == current:
        return current
    old_settings, new_settings = deepcopy(old), deepcopy(current)
    old_code = old_settings.pop("source_code_sha256")
    new_code = new_settings.pop("source_code_sha256")
    changed = {name for name in old_code if old_code[name] != new_code.get(name)}
    if (old_settings != new_settings or old_code.keys() != new_code.keys()
            or any(old_code[name] not in {D6_SOURCE_BASELINE.get(name), D7_SOURCE_BASELINE.get(name)}
                   for name in changed)):
        raise G2Blocked("BLOCKED: scientific protocol or unaudited dependency changed from the saved study")
    logging.info("Reuse immutable D6 protocol %s for E2 baseline; E3 has its own implementation hashes", record["sha256"])
    return old


def reuse_e3_architecture(current, root, study_sha256, *, mode="full"):
    """Preserve signed D7 E3 configs for skip/resume after this E4-only extension."""
    for category in CATEGORIES:
        path = root / mode / "E3" / category / "last.pt"
        if not path.is_file():
            continue
        payload, _ = load_checkpoint_payload(path, require_sha256=True)
        cfg = payload["config"]
        if (cfg.get("stage") != "E3" or cfg.get("mode") != mode
                or cfg.get("study_sha256") != study_sha256
                or payload["metadata"].get("config_sha256") != sha256_json(cfg)):
            raise G2Blocked("BLOCKED: historical E3 checkpoint/config checksum mismatch")
        old = cfg["architecture"]
        old_settings, new_settings = deepcopy(old), deepcopy(current)
        old_code = old_settings.pop("implementation_sha256")
        new_code = new_settings.pop("implementation_sha256")
        changed = {name for name in old_code if old_code[name] != new_code.get(name)}
        if (old_settings != new_settings or old_code.keys() != new_code.keys()
                or any(old_code[name] != D7_SOURCE_BASELINE.get(name) for name in changed)):
            raise G2Blocked("BLOCKED: scientific E3 architecture or unaudited implementation changed")
        return old
    return current


def make_context(study, category, pair, stage, *, smoke=False, selection_sha256=None):
    if stage == "E3" and not smoke:
        raise G2Blocked("BLOCKED: E3 main awaits joint selection D10 and D11 protocol integration")
    if stage == "E3" and selection_sha256 is not None:
        raise G2Blocked("BLOCKED: E3 debug cannot use a legacy Adapter selection lock")
    cfg = deepcopy(study["protocol"])
    if smoke:
        cfg["training"].update(epochs=cfg["smoke"]["epochs"], max_steps=cfg["smoke"]["max_steps"])
        if stage == "E3" and (cfg["training"]["epochs"] != 1 or cfg["training"]["max_steps"] > 2):
            raise G2Blocked("BLOCKED: E3 debug smoke is limited to one epoch and at most two updates")
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
    if stage == "E2" and study["protocol"]["selection"].get("status") == "pending_joint_selection_D10":
        cfg["selection_status"] = "pending_joint_selection_D10"
        cfg["adapter_pair_role"] = "debug_pair"
    if stage in {"E3", "E4"}:
        if stage == "E3":
            cfg.update(selection_status=PENDING_SELECTION["status"], adapter_pair_role="debug_pair",
                       debug_pair=debug_pair(pair))
        elif selection_sha256 is None:
            raise G2Blocked(f"BLOCKED: {stage} requires the Adapter selection lock hash")
        architecture = study["e3_smoke"] if smoke and stage == "E3" else study[stage.lower()]
        cfg["architecture"] = deepcopy(architecture)
        cfg["backbone"]["feature_blocks"] = list(architecture["source_blocks"])
        cfg["fusion"] = {"dim": architecture["feature_width"], "method": "mean"}
        if stage == "E4":
            cfg["paired_views"] = deepcopy(architecture["paired_views"])
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
        if cfg["stage"] == "E3" and cfg.get("adapter_pair_role") == "debug_pair":
            if (result.get("debug_pair") != cfg["debug_pair"]
                    or result.get("adapter_pair_role") != "debug_pair"
                    or result.get("selection_status") != PENDING_SELECTION["status"]
                    or result.get("scientific_evidence") is not False):
                return None
        if cfg["stage"] in {"E3", "E4"}:
            e3 = cfg["architecture"]
            with_context = cfg["stage"] == "E4"
            blocks = e3["source_blocks"] * (2 if with_context else 1)
            if (result.get("feature_sources") != len(e3["source_keys"]) or result.get("source_blocks") != e3["source_blocks"]
                    or result.get("feature_width") != e3["feature_width"] or result.get("fusion") != "mean"
                    or result.get("context") is not with_context or result.get("projection_updated") is not True
                    or result.get("source_block_map") != dict(zip(e3["source_keys"], blocks))
                    or result.get("adapters_updated") != dict.fromkeys(e3["source_keys"][:3], True)):
                return None
            if with_context and (result.get("context_aligned") is not True
                    or result.get("adapter_sharing") != e3["adapter_sharing"]
                    or result.get("projection_sharing") != e3["projection_sharing"]
                    or result.get("pair_strategy") != cfg["paired_views"]["pair_strategy"]):
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
            if cfg["stage"] in {"E3", "E4"} and (state.get("projection_updated") is not True
                    or state.get("adapters_updated") != result["adapters_updated"]):
                return None
            weights = payload["model_state"]
            adapter_prefix = "adapters." if cfg["stage"] in {"E3", "E4"} else "adapter."
            if (any(not torch.isfinite(value).all() for value in weights.values())
                    or sum(value.numel() for key, value in weights.items() if key.startswith(adapter_prefix))
                    != result["adapter_trainable_parameters"]):
                return None
            if cfg["stage"] in {"E3", "E4"}:
                groups = result["parameter_report"]["groups"]
                if (set(groups) != {"adapters", "projection", "decoder"}
                        or groups != {group: sum(v.numel() for k, v in weights.items() if k.startswith(group + "."))
                                      for group in groups}
                        or result["parameter_report"]["trainable_total"] != sum(groups.values())):
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
    raise G2Blocked("BLOCKED: D6 does not publish selection locks; pending joint selection D10")


def validate_selection(study):
    """Legacy/E2-only lock reader for audit, never called by the D6-v3 CLI."""
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


def execute_job(study, category, pair, stage, *, smoke, resume, device, selection_sha256=None, inference=False,
                model_factory=None, on_checkpoint=None):
    if stage != "adapter_screen" and (model_factory is not None or on_checkpoint is not None):
        raise ValueError("Colab callbacks are restricted to adapter_screen")
    context, pools = make_context(study, category, pair, stage, smoke=smoke,
                                 selection_sha256=selection_sha256)
    directory = run_directory(study, category, pair, stage, smoke)
    completed = read_valid_result(directory, context)
    if inference:
        if stage != "E4" or completed is None:
            raise G2Blocked("BLOCKED: E4 inference requires a valid completed training result")
        result = infer_e4(context, pools, directory, device=device, expected_metrics=completed["best_synthetic_dev"])
        return result["status"]
    if completed:
        return "SKIP"
    metrics_path = directory / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()).get("status") == "PASS":
        raise G2Blocked(f"BLOCKED: completed run is invalid; inspect {directory} before rerunning")
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "train.log")
    logging.getLogger().addHandler(handler)
    try:
        train = {"E2": train_e2, "E3": train_e3, "E4": train_e4}.get(stage, train_e2)
        options = {} if model_factory is None and on_checkpoint is None else {
            "model_factory": model_factory, "on_checkpoint": on_checkpoint}
        result = train(context, pools, directory, device=device, resume=resume, **options)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    if result["status"] == "PASS" and read_valid_result(directory, context) is None:
        raise ValueError("Training finished but result/checkpoint validation failed")
    return result["status"]


def export_comparison(study, lock, *, smoke=False, experiments=("E2", "E3")):
    selected = lock["payload"]["selected_pair"]
    pair = selected["r"], selected["d"]
    rows = []
    for category in CATEGORIES:
        row = dict(category=category, status="INCOMPLETE", real_pair=False, r=pair[0], d=pair[1])
        contexts, results = {}, {}
        try:
            for stage in experiments:
                context, _ = make_context(study, category, pair, stage, smoke=smoke,
                                         selection_sha256=lock["sha256"])
                path = run_directory(study, category, pair, stage, smoke)
                contexts[stage] = context
                result = read_valid_result(path, context)
                if result and (smoke or real_evidence(result)):
                    results[stage] = result
                    row[stage.lower() + "_synthetic_dev_aupro_0_05"] = result["best_synthetic_dev"]["synthetic_dev_aupro_0_05"]
                    row[stage.lower()] = dict(
                        checkpoint=str((path / "best.pt").resolve()), checkpoint_sha256=result["best_checkpoint_sha256"],
                        config=str((path / "resolved_config.yaml").resolve()), config_sha256=context["config_sha256"],
                        metrics=str((path / "metrics.json").resolve()), metrics_sha256=result["metrics_sha256"],
                        sources_sha256=sha256_json(context["config"]["sources"]),
                        parameter_report=result.get("parameter_report"),
                    )
            identity = comparison_identity(contexts[experiments[0]]["config"])
            row.update(pair_protocol_sha256=sha256_json(identity), seed=identity["training"]["seed"],
                       dev_seed=identity["training"]["dev_seed"], expected_steps=identity["expected_steps"])
            if identity != comparison_identity(contexts[experiments[1]]["config"]):
                row["status"] = "PROTOCOL_MISMATCH"
            elif len(results) == 2:
                row.update(status="READY", real_pair=all(real_evidence(result) for result in results.values()))
            else:
                row["status"] = "MISSING_" + "_AND_".join(stage for stage in experiments if stage not in results)
        except (OSError, ValueError, RuntimeError) as exc:
            row.update(status="INCOMPLETE", reason=str(exc))
        rows.append(row)
    return write_comparison_inputs(study["root"] / ("smoke" if smoke else "full"), categories=CATEGORIES,
                                   rows=rows, study_sha256=study["sha256"], selection_sha256=lock["sha256"],
                                   smoke=smoke, experiments=experiments)


def summarize(study, stage, outcomes, *, smoke, lock=None):
    if stage in {"E2", "E3"} and lock is None:
        all_jobs = bool(outcomes) and all(row["status"] in {"PASS", "SKIP"} for row in outcomes)
        return dict(status="SMOKE_PASS" if smoke and all_jobs else "INCOMPLETE", stage=stage,
                    mode="smoke" if smoke else "full", outcomes=outcomes, real_E2_pass=0, real_E3_pass=0,
                    selection_status="pending_joint_selection_D10", adapter_pair_role="debug_pair",
                    selection_lock=None, study_sha256=study["sha256"], counts_verified=True,
                    E3_minus_E2="NOT RUN: deferred to D11 with real main checkpoints")
    screen, _ = collect_results(study, "adapter_screen")
    main, e3, e4 = [], [], []
    if lock:
        selected = lock["payload"]["selected_pair"]
        main, _ = collect_results(study, "E2", pair=(selected["r"], selected["d"]), selection_sha256=lock["sha256"])
        e3, _ = collect_results(study, "E3", pair=(selected["r"], selected["d"]), selection_sha256=lock["sha256"])
        e4, _ = collect_results(study, "E4", pair=(selected["r"], selected["d"]), selection_sha256=lock["sha256"])
    all_jobs = bool(outcomes) and all(row["status"] in {"PASS", "SKIP"} for row in outcomes)
    full = len(screen) == 72 if stage == "adapter_screen" else len({"E2": main, "E3": e3, "E4": e4}[stage]) == 8
    status = ("SMOKE_PASS" if all_jobs else "INCOMPLETE") if smoke else ("PASS" if full else "INCOMPLETE")
    return dict(status=status, stage=stage, mode="smoke" if smoke else "full",
                real_adapter_screen_pass=len(screen), expected_adapter_screen=72,
                real_E2_pass=len(main), expected_E2=8, counts_verified=True,
                real_E3_pass=len(e3), expected_E3=8,
                real_E4_pass=len(e4), expected_E4=8,
                study_sha256=study["sha256"], outcomes=outcomes,
                selection_lock=None if lock is None else str(study["root"] / "full" / "adapter_selection_lock.json"))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary_path = None
    try:
        cfg, runner, root = load_config(args)
        if args.prepare:
            result = prepare_e3(cfg, root)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.stage in {"adapter_preflight", "adapter_screen"}:
            from scripts.g2_adapter_preflight import run_cpu
            result = run_cpu(cfg, root.parent)
            result.update(stage="adapter_preflight", operation="technical_check")
            print(json.dumps(result, indent=2))
            return 0 if result["cpu_pass"] == 9 else 1
        mode = "smoke" if args.smoke else "full"
        # Isolate v3 debug runs from immutable historical training artifacts.
        if args.stage == "E2" and args.smoke:
            root = root / "debug_d6_v3"
        elif args.stage == "E3" and args.smoke:
            root = root / "debug_d7_v3" / debug_pair((cfg["adapter"]["r"], cfg["adapter"]["d"]))["name"]
        summary_path = root / mode / f"{args.stage}{'_inference' if args.inference else ''}_summary.json"
        if not (args.stage in {"E2", "E3"} and args.smoke):
            raise G2Blocked("BLOCKED: pending joint selection D10; official main awaits joint_selection_lock.json "
                            "and D11 protocol integration. A legacy D6 Adapter lock is audit-only. "
                            "E2/E3 debug remains available with --smoke [--debug-pair R D].")
        if not args.smoke and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
            raise FileNotFoundError("G2 NOT RUN: full D6 requires a usable CUDA device; smoke may use CPU")
        study = prepare_study(cfg, runner, root, device=args.device)
        enforce_lock(root / mode, study["protocol"])
        lock = None
        pairs = [(cfg["adapter"]["r"], cfg["adapter"]["d"])]
        outcomes = []
        for pair in pairs:
            for category in args.categories:
                row = dict(category=category, r=pair[0], d=pair[1])
                try:
                    row["status"] = execute_job(study, category, pair, args.stage, smoke=args.smoke,
                                                resume=args.resume, device=args.device,
                                                selection_sha256=None if lock is None else lock["sha256"],
                                                inference=args.inference)
                except (OSError, ValueError, RuntimeError) as exc:
                    row.update(status="BLOCKED" if "BLOCKED" in str(exc) else "FAIL", reason=str(exc))
                    save_json(row, run_directory(study, category, pair, args.stage, args.smoke) / "failure.json")
                    logging.error("%s", row)
                outcomes.append(row)
        result = summarize(study, args.stage, outcomes, smoke=args.smoke, lock=lock)
        if lock is not None and args.stage in {"E2", "E3", "E4"}:
            experiments = ("E3", "E4") if args.stage == "E4" else ("E2", "E3")
            result["comparison_inputs"] = export_comparison(study, lock, smoke=args.smoke, experiments=experiments)
        if any(row["status"] == "FAIL" for row in outcomes):
            result["status"] = "FAIL"
        elif any(row["status"] == "BLOCKED" for row in outcomes):
            result["status"] = "BLOCKED"
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        status = "BLOCKED" if "BLOCKED" in str(exc) else ("NOT RUN" if isinstance(exc, FileNotFoundError) else "FAIL")
        result = dict(status=status, stage=args.stage, mode="smoke" if args.smoke else "full", reason=str(exc),
                      real_adapter_screen_pass=0, expected_adapter_screen=0,
                      real_E2_pass=0, expected_E2=8, counts_verified=False)
        result["selection_status"] = "pending_joint_selection_D10"
        result.update(real_E3_pass=0, expected_E3=8)
        result.update(real_E4_pass=0, expected_E4=8)
    result["operation"] = "inference" if args.inference else "train"
    if summary_path is not None:
        save_json(result, summary_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"PASS", "SMOKE_PASS"} else (1 if result["status"] == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
