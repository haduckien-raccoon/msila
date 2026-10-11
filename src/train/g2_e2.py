"""E2--E5 training with existing G1 data, loss, train step and synthetic DEV.

Checkpoints contain Adapter + Decoder + optimizer + RNG; frozen DINO weights
are identified by checksum rather than duplicated in every checkpoint.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loader import G1NativeDataset, G1TileDataset, g1_tile_collate
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.msila import build_g2_model
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.train.g1_e1 import evaluate_dev, file_sha256, module_sha256, predict_native
from src.train.g2_context import PairedG2Tiles, paired_tile_collate, paired_model_forward, predict_native_e4
from src.train.optimizer import build_optimizer
from src.train.overfit16 import Overfit16Trainer
from src.train.screen_adapter import enforce_protocol_lock, seed_everything
from src.train.screen_representation import save_json, save_yaml
from src.utils.checkpoint import save_training_checkpoint
from src.utils.resume import load_checkpoint_payload, resume_training_checkpoint, restore_rng_state


def train_e2(context, pools, output_dir, *, device, resume=False, model_factory=None, on_checkpoint=None):
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume, experiment="E2",
                               model_factory=model_factory, on_checkpoint=on_checkpoint)


def train_e3(context, pools, output_dir, *, device, resume=False):
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume, experiment="E3")


def train_e4(context, pools, output_dir, *, device, resume=False):
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume, experiment="E4")


def train_e5(context, pools, output_dir, *, device, resume=False, model_factory=None, on_epoch=None):
    from src.eval.g2_joint import validate_budget, FUSIONS
    cfg = context["config"]
    validate_budget(cfg["screening_budget"])
    budget = {k: cfg["training"][k] for k in cfg["screening_budget"]}
    if (cfg.get("stage") != "E5" or cfg.get("mode") not in {"screening", "smoke"}
            or cfg.get("adapter_pair_role") != "joint_candidate"
            or cfg.get("selection_status") != "pending_joint_selection_D10"
            or cfg.get("selection_sha256") is not None
            or cfg["fusion"]["method"] not in FUSIONS or budget != cfg["screening_budget"]):
        raise RuntimeError("BLOCKED: E5 requires a declared joint screening protocol; no main training")
    if cfg["mode"] == "smoke" and not (budget["epochs"] == 1 and type(budget["max_steps"]) is int
                                        and 1 <= budget["max_steps"] <= 2 and context["expected_steps"] <= 2):
        raise RuntimeError("BLOCKED: E5 smoke is limited to one epoch and two updates")
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume, experiment="E5",
                               model_factory=model_factory, on_epoch=on_epoch)


def train_main(context, pools, output_dir, *, device, resume=False, model_factory=None, on_epoch=None,
               reuse_checkpoint=None):
    """D11 uses the same loop only after the runner has audited the D10 lock."""
    from src.train.g2_main import require_main_context
    require_main_context(context)
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume,
                               experiment=context["config"]["stage"], model_factory=model_factory,
                               on_epoch=on_epoch, reuse_checkpoint=reuse_checkpoint)


def train_fixed(context, pools, output_dir, *, device, resume=False, model_factory=None,
                on_epoch=None, on_checkpoint=None):
    """Explicit user-fixed E5 experiment; never creates or claims a D10 lock."""
    from src.train.g2_fixed import require_fixed_context
    require_fixed_context(context)
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume,
                               experiment='E5', model_factory=model_factory,
                               on_epoch=on_epoch, on_checkpoint=on_checkpoint)


def _require_debug_context(context, experiment):
    """E3/E4 technical smoke is independent of the obsolete D6 selection lock."""
    cfg = context["config"]
    pair = cfg.get("debug_pair", {})
    r, d = cfg["adapter"]["r"], cfg["adapter"]["d"]
    if not (cfg.get("stage") == experiment and cfg.get("mode") == "smoke"
            and cfg.get("selection_status") == "pending_joint_selection_D10"
            and cfg.get("adapter_pair_role") == "debug_pair"
            and pair == dict(name=f"debug_pair_r{r}_d{d}", r=r, d=d, role="technical_only")
            and cfg.get("selection_sha256") is None
            and cfg["training"]["epochs"] == 1
            and type(cfg["training"].get("max_steps")) is int
            and 1 <= cfg["training"]["max_steps"] <= 2
            and 1 <= context["expected_steps"] <= 2):
        raise RuntimeError(f"BLOCKED: {experiment} accepts only bounded debug_pair smoke; main awaits joint selection D10")


def infer_e4(context, pools, output_dir, *, device, expected_metrics=None):
    """Restore best.pt and export native DEV maps/masks without optimizer steps."""
    cfg, digest = context["config"], context["config_sha256"]
    _require_debug_context(context, "E4")
    output_dir = Path(output_dir)
    path = output_dir / "best.pt"
    if not path.is_file():
        raise RuntimeError("BLOCKED: E4 inference requires a completed best.pt")
    payload, _ = load_checkpoint_payload(path, require_sha256=True)
    if payload["config"] != cfg or payload["metadata"].get("config_sha256") != digest:
        raise ValueError("E4 inference checkpoint/config provenance mismatch")
    seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
    model = build_g2_model(cfg, experiment="E4").to(device)
    trainable_modules(model, "E4").load_state_dict(payload["model_state"], strict=True)
    native = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"],
                             seed=cfg["training"]["dev_seed"], role="dev",
                             variants=cfg["data"]["dev_variants_per_image"], fixed=True)
    directory = output_dir / "inference"
    maps = directory / "predictions"
    maps.mkdir(parents=True, exist_ok=True)
    records = []

    def export_prediction(model, image, config, target_device):
        score = predict_native_e4(model, image, config, target_device)
        sample = native[len(records)]
        stem = f"{len(records):06d}"
        score_path, mask_path = maps / f"{stem}_score.npy", maps / f"{stem}_mask.npy"
        np.save(score_path, score.numpy())
        np.save(mask_path, sample["mask"][0].numpy())
        records.append(dict(sample_id=sample["meta"]["sample_id"], meta=sample["meta"],
                            original_hw=list(score.shape), score=str(score_path.resolve()),
                            mask=str(mask_path.resolve()), score_sha256=file_sha256(score_path),
                            mask_sha256=file_sha256(mask_path)))
        return score

    metrics = evaluate_dev(model, native, cfg, torch.device(device), directory, predict_fn=export_prediction)
    if expected_metrics is not None and metrics != expected_metrics:
        raise ValueError("E4 inference DEV differs from the selected checkpoint result")
    result = dict(status="PASS", operation="inference", stage="E4", category=cfg["category"],
                  mode=cfg["mode"], config_sha256=digest, checkpoint=str(path.resolve()),
                  checkpoint_sha256=file_sha256(path), native_resolution=True,
                  n_samples=len(records), metrics=metrics, predictions=records,
                  split="dev_synthetic", synthetic=True, optimizer_steps=0, device=str(device),
                  debug_pair=cfg["debug_pair"], adapter_pair_role="debug_pair", scientific_evidence=False,
                  selection_status=cfg["selection_status"],
                  verification_scope=("real_pretrained" if model.extractor.backbone.__class__.__module__.startswith("dinov3.")
                                      else "fixture"))
    save_json(result, directory / "metrics.json")
    return result


def trainable_modules(model, experiment):
    """TV2 can restore this ModuleDict directly from best.pt['model_state']."""
    if experiment == "E1":
        return nn.ModuleDict({"decoder": model.decoder})
    if experiment == "E2":
        return nn.ModuleDict({"adapter": model.adapter, "decoder": model.decoder})
    if experiment in {"E3", "E4"}:
        return nn.ModuleDict({"adapters": model.adapters, "projection": model.projection, "decoder": model.decoder})
    if experiment == "E5":
        return nn.ModuleDict({"adapters": model.adapters, "projection": model.projection,
                              "decoder": model.decoder, "fusion": model.fusion})
    raise ValueError(f"Unsupported experiment {experiment}")


class FrozenE2Factory:
    """Reuse only frozen DINO; replay its initialization RNG for fresh E2 heads.

    This preserves the decoder/Adapter initialization of the uncached factory.
    No optimizer or trained head is retained between candidates/categories.
    """

    def __init__(self, cfg, *, device, experiment="E2"):
        if experiment not in {"E1", "E2", "E3", "E4", "E5"}:
            raise ValueError("Frozen factory supports E1--E5")
        self.experiment = experiment
        self.seed = cfg["training"]["seed"]
        b = cfg["backbone"]
        self.identity = {k: b[k] for k in ("name", "repo_dir", "weights", "norm")}
        self.weights_sha256 = file_sha256(b["weights"])
        devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            self.extractor = DINOv3FeatureExtractor(
                b["repo_dir"], b["weights"], model_name=b["name"], norm=b["norm"],
                feature_mode="multilayer" if experiment in {"E3", "E4", "E5"} else "deepest", check_finite=True,
            )
            self.initialization_rng = torch.get_rng_state().clone()
        self.extractor.to(device).eval()

    def __call__(self, cfg):
        b = cfg["backbone"]
        if (cfg["training"]["seed"] != self.seed
                or {k: b[k] for k in self.identity} != self.identity
                or b.get("checkpoint_sha256", self.weights_sha256) != self.weights_sha256):
            raise ValueError("Cached E2 backbone/seed/checkpoint provenance mismatch")
        torch.set_rng_state(self.initialization_rng)
        self.extractor.eval()
        return build_g2_model(cfg, experiment=self.experiment, extractor=self.extractor)


class FrozenE5Factory(FrozenE2Factory):
    """Cache the frozen multilayer backbone only; every job gets fresh heads."""

    def __init__(self, cfg, *, device):
        super().__init__(cfg, device=device, experiment="E5")


def _train_segmentation(context, pools, output_dir, *, device, resume, experiment,
                        model_factory=None, on_checkpoint=None, on_epoch=None, reuse_checkpoint=None):
    """Complete the declared update budget or save INCOMPLETE for --resume.

    Cursor checkpoints keep the final batch of an epoch uncommitted until DEV
    and best.pt are durable. Resuming there evaluates DEV without another step.
    """
    cfg, digest = context["config"], context["config_sha256"]
    main = cfg.get("mode") in {"main", "main_smoke"}
    fixed = cfg.get('mode') in {'fixed', 'fixed_smoke'}
    export_native = main or fixed
    if fixed:
        from src.train.g2_fixed import require_fixed_context
        require_fixed_context(context)
    if main:
        from src.train.g2_main import require_main_context
        require_main_context(context)
    if not main and experiment not in {"E2", "E5"} and (model_factory is not None or on_checkpoint is not None):
        raise ValueError("Colab callbacks are supported only for E2 Adapter screening")
    if not main and experiment in {"E3", "E4"}:
        _require_debug_context(context, experiment)
    t, d = cfg["training"], cfg["data"]
    output_dir, device = Path(output_dir), torch.device(device)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    if last_path.exists():
        if not resume:
            raise RuntimeError("BLOCKED: unfinished checkpoint exists; use --resume")
        payload, _ = load_checkpoint_payload(last_path, require_sha256=True)
        if payload["config"] != cfg or payload["metadata"].get("config_sha256") != digest:
            raise ValueError("Resume config/source/checkpoint provenance mismatch")
    elif (output_dir / "metrics.json").exists():
        raise RuntimeError("BLOCKED: metrics exist without a resumable checkpoint")

    seed_everything(t["seed"], deterministic=t.get('deterministic_algorithms', True) if fixed else True,
                    warn_only=False)
    if experiment == "E5" or main:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = (build_g2_model(cfg, experiment=experiment) if model_factory is None
             else model_factory(cfg)).to(device)
    trainable = trainable_modules(model, experiment)
    adapter_module = model.adapters if experiment in {"E3", "E4", "E5"} else (model.adapter if experiment == "E2" else nn.Identity())
    optimizer_modules = dict(trainable.items())
    if experiment == "E5" and not any(True for _ in model.fusion.parameters()):
        optimizer_modules.pop("fusion")  # Average has no learned parameters.
    optimizer, report = build_optimizer(
        optimizer_modules, frozen_modules={"backbone": model.extractor},
        learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
    )
    decoder_initial_sha = module_sha256(model.decoder)
    # All nine candidates share this category lock, including decoder init.
    fairness_root = output_dir.parent.parent if cfg["stage"] == "adapter_screen" else output_dir.parent
    if experiment == "E5" or main:
        fairness_root = Path(context["fairness_root"])
    enforce_protocol_lock(fairness_root / "_fairness", category=cfg["category"], payload={
        "study_sha256": cfg["study_sha256"], "mode": cfg["mode"],
        "sources": cfg["sources"], "training": t,
        "decoder_initial_sha256": decoder_initial_sha,
    })
    native_train = G1NativeDataset(pools["train"], cfg["synthetic_protocol"],
                                   seed=t["seed"], role="train",
                                   variants=d["train_variants_per_image"], fixed=False)
    if fixed:
        from src.train.g2_fixed_data import IlluminatedNativeDataset
        native_train = IlluminatedNativeDataset(pools['train'], cfg['synthetic_protocol'],
                                                seed=t['seed'], role='train',
                                                variants=d['train_variants_per_image'], fixed=False,
                                                illumination=cfg['illumination'],
                                                illumination_mode=cfg.get('illumination_mode', 'augmentation'),
                                                appearance=cfg.get('synthetic_appearance', 'legacy'))
    tiles = G1TileDataset(native_train, tile_size=d["tile_size"], overlap=d["overlap"],
                          positive_tile_fraction=d.get("train_positive_tile_fraction", 0.0))
    local_tiles = tiles
    if experiment in {"E4", "E5"}:
        tiles = PairedG2Tiles(tiles, context_size=cfg["paired_views"]["context_size"])
    dev_kwargs = {"predict_fn": predict_native_e4} if experiment in {"E4", "E5"} else {}
    if fixed:
        from src.train.g2_fixed import predict_fixed_native, precision_dtype
        dev_kwargs = {'predict_fn': predict_fixed_native}
    native_dev = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"],
                                 seed=t["dev_seed"], role="dev",
                                 variants=d["dev_variants_per_image"], fixed=True)
    trainer = Overfit16Trainer(
        model=model, criterion=AnomalySegmentationLoss(**cfg["loss"]),
        optimizer=optimizer, device=device, frozen_modules={"backbone": model.extractor},
        max_grad_norm=t["max_grad_norm"],
        model_forward=paired_model_forward if experiment in {"E4", "E5"} else None,
        amp_dtype=precision_dtype(t['precision']) if fixed else None,
        illumination_loss_weight=cfg.get('illumination_loss_weight', 0.) if fixed else 0.,
    )
    history, best, best_epoch, step = [], None, None, 0
    start_epoch, start_batch, latest_dev, dev_step = 0, 0, None, 0
    adapter_updated, decoder_updated = False, False
    if reuse_checkpoint is not None:
        if not main or experiment != "E1" or last_path.exists():
            raise ValueError("G1 reuse requires a fresh D11 E1 run")
        from src.train.g2_main import valid_g1_checkpoint
        legacy = valid_g1_checkpoint(reuse_checkpoint, context, pools)
        legacy_last = legacy["_last_payload"]
        model.decoder.load_state_dict(legacy_last["model_state"], strict=True)
        optimizer.load_state_dict(legacy_last["optimizer_state"])
        history = legacy_last["metadata"]["history"]
        step = legacy_last["training_state"]["global_step"]
        best = legacy["metadata"]["best_metric"]
        best_epoch = legacy["training_state"]["epoch"]
        latest_dev, dev_step = legacy_last["metadata"]["dev"], step
        decoder_updated = True
        start_epoch = t["epochs"]
    if last_path.exists():
        restored = resume_training_checkpoint(last_path, model=trainable, optimizer=optimizer,
                                              expected_config=cfg, map_location=device)
        state = restored.metadata
        if fixed:
            trainer.scaler.load_state_dict(state['grad_scaler'])
        history, best, best_epoch = state["history"], state["best_metric"], state["best_epoch"]
        start_epoch, start_batch, step = state["next_epoch"], state["next_batch"], restored.global_step
        latest_dev = state["dev"]
        dev_step = state["dev_step"]
        adapter_updated, decoder_updated = state["adapter_updated"], state["decoder_updated"]
        if state["decoder_initial_sha256"] != decoder_initial_sha:
            raise ValueError("Resume decoder initialization provenance mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "resolved_config.yaml")
    save_json({"sha256": digest}, output_dir / "config_hash.json")
    logging.info("%s category=%s r=%s d=%s optimizer=%s", experiment, cfg["category"],
                 cfg["adapter"]["r"], cfg["adapter"]["d"], report.to_dict())
    frozen_before = module_sha256(model.extractor)
    adapter_before, decoder_before = module_sha256(adapter_module), module_sha256(model.decoder)
    projection_before = module_sha256(model.projection) if experiment in {"E3", "E4", "E5"} else None
    fusion_before = module_sha256(model.fusion) if experiment == "E5" else None
    fusion_updated = False
    projection_updated = False
    adapter_source_before = ({key: module_sha256(module) for key, module in model.adapters.items()}
                             if experiment in {"E3", "E4", "E5"} else {})
    adapters_updated = {key: False for key in adapter_source_before}
    if last_path.exists() and experiment in {"E3", "E4", "E5"}:
        projection_updated = state["projection_updated"]
        adapters_updated = state["adapters_updated"]
        if experiment == "E5":
            fusion_updated = state["fusion_updated"]
    started, stopped = time.monotonic(), False
    stop_reason = "budget_completed"

    def metadata(next_epoch, next_batch):
        result = dict(config_sha256=digest, next_epoch=next_epoch, next_batch=next_batch,
                    history=history, best_metric=best, best_epoch=best_epoch, dev=latest_dev, dev_step=dev_step,
                    decoder_initial_sha256=decoder_initial_sha,
                    adapter_updated=adapter_updated or module_sha256(adapter_module) != adapter_before,
                    decoder_updated=decoder_updated or module_sha256(model.decoder) != decoder_before)
        if experiment in {"E3", "E4", "E5"}:
            result.update(projection_updated=projection_updated or module_sha256(model.projection) != projection_before,
                          adapters_updated={key: adapters_updated[key] or module_sha256(module) != adapter_source_before[key]
                                            for key, module in model.adapters.items()})
        if experiment == "E5":
            result.update(fusion_updated=fusion_updated or module_sha256(model.fusion) != fusion_before,
                          git_sha=cfg["git_sha"], run_id=cfg["run_id"])
        if main:
            result.update(git_sha=cfg["git_sha"], run_id=cfg["run_id"],
                          selection_sha256=cfg["selection_sha256"], main_protocol_sha256=cfg["study_sha256"])
        if fixed:
            result.update(grad_scaler=trainer.scaler.state_dict(), selection_status=cfg['selection_status'],
                          execution_gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                          precision=t['precision'])
        return result

    def checkpoint(path, epoch, next_epoch, next_batch):
        save_training_checkpoint(path, model=trainable, optimizer=optimizer, epoch=epoch,
                                 global_step=step, config=cfg, metadata=metadata(next_epoch, next_batch))
        if on_checkpoint is not None:
            on_checkpoint(path)

    if reuse_checkpoint is not None:
        # Retain the original G1 files. This D11 wrapper records the reuse proof.
        restore_rng_state(legacy_last["rng_state"])
        checkpoint(last_path, legacy_last["training_state"]["epoch"], t["epochs"], 0)
        full_state = history, step, latest_dev, dev_step
        model.decoder.load_state_dict(legacy["model_state"], strict=True)
        optimizer.load_state_dict(legacy["optimizer_state"])
        history, step = legacy["metadata"]["history"], legacy["training_state"]["global_step"]
        latest_dev, dev_step = legacy["metadata"]["dev"], step
        restore_rng_state(legacy["rng_state"])
        checkpoint(best_path, best_epoch, legacy["metadata"]["next_epoch"], legacy["metadata"]["next_batch"])
        history, step, latest_dev, dev_step = full_state
        restore_rng_state(legacy_last["rng_state"])

    for epoch in range(start_epoch, t["epochs"]):
        native_train.set_epoch(epoch)
        sampler = None
        if fixed:
            from src.train.g2_fixed_data import NativeTileSampler
            sampler = NativeTileSampler(local_tiles, t['seed'] + epoch)
        batches = DataLoader(tiles, batch_size=t["batch_size"], shuffle=sampler is None, sampler=sampler,
                             num_workers=d["num_workers"], collate_fn=paired_tile_collate if experiment in {"E4", "E5"} else g1_tile_collate,
                             pin_memory=fixed and device.type == 'cuda',
                             generator=torch.Generator().manual_seed(t["seed"] + epoch))
        next_batch = start_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(batches):
            if batch_index < next_batch:
                continue
            if step >= context["expected_steps"]:
                break
            if t["max_minutes"] is not None and time.monotonic() - started >= t["max_minutes"] * 60:
                stop_reason, stopped = "wall_time_paused", True
                checkpoint(last_path, epoch, epoch, batch_index)
                break
            tick = time.monotonic()
            entry = trainer.train_step(batch, step=step + 1, epoch=epoch).to_dict()
            if model.extractor.training or any(p.requires_grad or p.grad is not None
                                               for p in model.extractor.parameters()):
                raise RuntimeError("Frozen backbone contract violated")
            for name, module in trainable.items():
                # The shared trainer already checks every gradient for NaN/Inf.
                # Avoid the second per-parameter GPU sync in the fixed run.
                if not all(p.grad is not None and (fixed or torch.isfinite(p.grad).all())
                           for p in module.parameters()):
                    raise RuntimeError(f"{name} gradients are absent or nonfinite")
            entry["seconds"] = time.monotonic() - tick
            history.append(entry)
            step += 1
            next_batch = batch_index + 1
            logging.info("epoch=%d step=%d/%d loss=%.6f BCE=%.6f Dice=%.6f positive_tiles=%d positive_pixels=%d p_fg=%s p_bg=%s recall=%.4f",
                         epoch + 1, step, context["expected_steps"], entry["loss"], entry["bce"], entry["dice_loss"],
                         entry['positive_samples'], entry['positive_pixels'], entry['foreground_probability'],
                         entry['background_probability'], entry['recall'])
            if cfg.get('illumination_mode') == 'consistency':
                logging.info('illumination pairs=%d rejected=%d KL=%.6f weighted=%.6f segmentation=%.6f',
                             entry['illumination_pairs'], entry['illumination_rejected'],
                             entry['illumination_loss'], entry['illumination_contribution'], entry['segmentation_loss'])
            if step % t["checkpoint_interval_steps"] == 0 or step == context["expected_steps"]:
                checkpoint(last_path, epoch, epoch, next_batch)
        if stopped:
            break
        # No extra update on resume at an epoch's final batch / declared max_steps.
        latest_dev = evaluate_dev(model, native_dev, cfg, device, **dev_kwargs)
        if fixed:
            from src.train.g2_fixed import save_learning_diagnostics
            save_learning_diagnostics(history, model, output_dir, epoch, cfg['loss'])
        metric = latest_dev["synthetic_dev_aupro_0_05"]
        if not isinstance(metric, (int, float)) or not 0 <= metric <= 1:
            raise ValueError("Synthetic DEV AU-PRO is missing/nonfinite/outside [0,1]")
        dev_step = step
        is_best = best is None or metric > best  # Exact ties keep earliest epoch.
        if is_best:
            best, best_epoch = metric, epoch
        next_epoch = epoch + 1 if next_batch >= len(batches) else epoch
        cursor_batch = 0 if next_epoch > epoch else next_batch
        # Commit best first; last never advertises a best checkpoint not on disk.
        if is_best:
            checkpoint(best_path, epoch, next_epoch, cursor_batch)
        checkpoint(last_path, epoch, next_epoch, cursor_batch)
        if on_epoch is not None:
            save_json(history, output_dir / "loss_history.json")
            on_epoch(output_dir, epoch + 1)
        if step >= context["expected_steps"]:
            break

    frozen_unchanged = module_sha256(model.extractor) == frozen_before
    adapter_updated |= module_sha256(adapter_module) != adapter_before
    decoder_updated |= module_sha256(model.decoder) != decoder_before
    complete = step == context["expected_steps"]
    if not frozen_unchanged or (complete and not (decoder_updated and (adapter_updated or experiment == "E1"))):
        raise RuntimeError("Backbone changed or trainable modules failed to update")
    if experiment in {"E3", "E4", "E5"}:
        projection_updated |= module_sha256(model.projection) != projection_before
        adapters_updated = {key: adapters_updated[key] or module_sha256(module) != adapter_source_before[key]
                            for key, module in model.adapters.items()}
        if complete and (not projection_updated or not all(adapters_updated.values())):
            raise RuntimeError(f"{experiment} projection or a source Adapter failed to update")
    if experiment == "E5":
        fusion_updated |= module_sha256(model.fusion) != fusion_before
        if complete and any(p.requires_grad for p in model.fusion.parameters()) and not fusion_updated:
            raise RuntimeError("E5 learned Fusion failed to update")
    best_dev = None
    if best_path.is_file():
        payload, _ = load_checkpoint_payload(best_path, require_sha256=True)
        trainable.load_state_dict(payload["model_state"], strict=True)
        if export_native:
            # Observe the existing native inference/metric pass; no metric reimplementation.
            from scripts.eval_g2 import CaptureNative, dice_from_exports
            native_dir = output_dir / "native_dev"
            capture = CaptureNative(native_dev, native_dir,
                                    predict_fn=(predict_fixed_native if fixed else
                                                predict_native_e4 if experiment in {"E4", "E5"} else predict_native))
            best_dev = evaluate_dev(model, native_dev, cfg, device, native_dir, predict_fn=capture)
            save_json(capture.rows, native_dir / "native_maps.json")
            best_dev.update(synthetic_dice=dice_from_exports(native_dir, capture.rows, cfg["category"],
                                                           cfg["synthetic_protocol"]["prediction_threshold"])["f1"],
                            dice_threshold=cfg["synthetic_protocol"]["prediction_threshold"])
        else:
            best_dev = evaluate_dev(model, native_dev, cfg, device, output_dir, **dev_kwargs)
    if complete and (best_dev is None or best_dev["synthetic_dev_aupro_0_05"] != best):
        raise ValueError("Best checkpoint DEV does not reproduce its selected score")
    result = dict(
        status="PASS" if complete else "INCOMPLETE", training_complete=complete,
        mode=cfg["mode"], stage=cfg["stage"], category=cfg["category"],
        adapter=None if experiment == "E1" else {"r": cfg["adapter"]["r"], "d": cfg["adapter"]["d"]},
        study_sha256=cfg["study_sha256"], config_sha256=digest,
        expected_steps=context["expected_steps"], global_step=step, best_epoch=best_epoch,
        best_synthetic_dev=best_dev, last_synthetic_dev=latest_dev,
        decoder_initial_sha256=decoder_initial_sha,
        adapter_trainable_parameters=sum(p.numel() for p in adapter_module.parameters() if p.requires_grad),
        frozen_backbone_unchanged=frozen_unchanged, adapter_updated=adapter_updated,
        decoder_updated=decoder_updated,
        verification_scope=("real_pretrained" if model.extractor.backbone.__class__.__module__.startswith("dinov3.")
                            else "fixture"), device=str(device),
        elapsed_seconds=time.monotonic() - started, stop_reason=stop_reason,
        best_checkpoint_sha256=file_sha256(best_path) if best_path.exists() else None,
        last_checkpoint_sha256=file_sha256(last_path),
    )
    groups = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
              for name, module in trainable.items()}
    result.update(parameter_report=dict(groups=groups, trainable_total=sum(groups.values()),
                                        frozen_backbone=sum(p.numel() for p in model.extractor.parameters())),
                  feature_sources=model.num_sources if experiment in {"E3", "E4", "E5"} else 1,
                  source_blocks=list(model.source_blocks) if experiment in {"E3", "E4", "E5"} else [model.extractor.depth],
                  feature_width=model.decoder.head[0].in_channels)
    if experiment in {"E3", "E4", "E5"}:
        result.update(projection_updated=projection_updated, adapters_updated=adapters_updated,
                      source_block_map=model.source_block_map,
                      fusion=cfg["fusion"]["method"] if experiment == "E5" else "mean", context=experiment in {"E4", "E5"})
        result.update(selection_status=cfg["selection_status"], adapter_pair_role=cfg["adapter_pair_role"],
                      scientific_evidence=experiment == "E5" and cfg["mode"] == "screening"
                      and result["verification_scope"] == "real_pretrained" and device.type == "cuda")
        if experiment != "E5" and not main:
            result["debug_pair"] = cfg["debug_pair"]
        elif experiment == "E5":
            result.update(fusion_updated=fusion_updated, git_sha=cfg["git_sha"], run_id=cfg["run_id"],
                          **({'fixed_budget': cfg['fixed_budget']} if fixed else
                             {'screening_budget': cfg['screening_budget']}))
            save_json(history, output_dir / "loss_history.json")
        if experiment in {"E4", "E5"}:
            result.update(context_aligned=True, adapter_sharing="shared_across_views_per_layer",
                          projection_sharing="shared_across_views_per_layer",
                          pair_strategy=model.pair_strategy)
    if export_native:
        save_json(history, output_dir / "loss_history.json")
        result.update(git_sha=cfg["git_sha"], run_id=cfg["run_id"],
                      **({'fixed_budget': cfg['fixed_budget'], 'illumination': cfg['illumination']} if fixed else
                         {'main_budget': cfg['main_budget']}),
                      selection_sha256=cfg["selection_sha256"], selection_status=cfg['selection_status'],
                      scientific_evidence=complete and cfg["mode"] in {'main', 'fixed'}
                      and result["verification_scope"] == "real_pretrained" and device.type == "cuda",
                      reuse_g1_checkpoint=str(reuse_checkpoint) if reuse_checkpoint else None,
                      reuse_g1_sha256=file_sha256(reuse_checkpoint) if reuse_checkpoint else None,
                      native_exports={"directory": "native_dev", "count": len(capture.rows),
                                      "manifest_sha256": file_sha256(native_dir / "native_maps.json"),
                                      "qa_sha256": file_sha256(native_dir / "qa_report.json")} if best_dev is not None else None)
    save_json(result, output_dir / "metrics.json")
    if export_native:
        (output_dir / "metrics.json.sha256").write_text(file_sha256(output_dir / "metrics.json") + "\n")
    return result
