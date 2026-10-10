"""E2/E3 training with existing G1 data, loss, train step and synthetic DEV.

Checkpoints contain Adapter + Decoder + optimizer + RNG; frozen DINO weights
are identified by checksum rather than duplicated in every checkpoint.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loader import G1NativeDataset, G1TileDataset, g1_tile_collate
from src.losses.anomaly_loss import AnomalySegmentationLoss
from src.models.msila import build_g2_model
from src.train.g1_e1 import evaluate_dev, file_sha256, module_sha256
from src.train.optimizer import build_optimizer
from src.train.overfit16 import Overfit16Trainer
from src.train.screen_adapter import enforce_protocol_lock, seed_everything
from src.train.screen_representation import save_json, save_yaml
from src.utils.checkpoint import save_training_checkpoint
from src.utils.resume import load_checkpoint_payload, resume_training_checkpoint


def train_e2(context, pools, output_dir, *, device, resume=False):
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume, experiment="E2")


def train_e3(context, pools, output_dir, *, device, resume=False):
    return _train_segmentation(context, pools, output_dir, device=device, resume=resume, experiment="E3")


def trainable_modules(model, experiment):
    """TV2 can restore this ModuleDict directly from best.pt['model_state']."""
    if experiment == "E2":
        return nn.ModuleDict({"adapter": model.adapter, "decoder": model.decoder})
    if experiment == "E3":
        return nn.ModuleDict({"adapters": model.adapters, "projection": model.projection, "decoder": model.decoder})
    raise ValueError(f"Unsupported experiment {experiment}")


def _train_segmentation(context, pools, output_dir, *, device, resume, experiment):
    """Complete the declared update budget or save INCOMPLETE for --resume.

    Cursor checkpoints keep the final batch of an epoch uncommitted until DEV
    and best.pt are durable. Resuming there evaluates DEV without another step.
    """
    cfg, digest = context["config"], context["config_sha256"]
    if experiment == "E3" and (cfg["stage"] != "E3" or not cfg.get("selection_sha256")):
        raise RuntimeError("BLOCKED: E3 training requires its locked Adapter selection context")
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

    seed_everything(t["seed"], deterministic=True, warn_only=False)
    model = build_g2_model(cfg, experiment=experiment).to(device)
    trainable = trainable_modules(model, experiment)
    adapter_module = model.adapters if experiment == "E3" else model.adapter
    optimizer, report = build_optimizer(
        dict(trainable.items()), frozen_modules={"backbone": model.extractor},
        learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
    )
    decoder_initial_sha = module_sha256(model.decoder)
    # All nine candidates share this category lock, including decoder init.
    fairness_root = output_dir.parent.parent if cfg["stage"] == "adapter_screen" else output_dir.parent
    enforce_protocol_lock(fairness_root / "_fairness", category=cfg["category"], payload={
        "study_sha256": cfg["study_sha256"], "mode": cfg["mode"],
        "sources": cfg["sources"], "training": t,
        "decoder_initial_sha256": decoder_initial_sha,
    })
    native_train = G1NativeDataset(pools["train"], cfg["synthetic_protocol"],
                                   seed=t["seed"], role="train",
                                   variants=d["train_variants_per_image"], fixed=False)
    tiles = G1TileDataset(native_train, tile_size=d["tile_size"], overlap=d["overlap"])
    native_dev = G1NativeDataset(pools["dev"], cfg["synthetic_protocol"],
                                 seed=t["dev_seed"], role="dev",
                                 variants=d["dev_variants_per_image"], fixed=True)
    trainer = Overfit16Trainer(
        model=model, criterion=AnomalySegmentationLoss(**cfg["loss"]),
        optimizer=optimizer, device=device, frozen_modules={"backbone": model.extractor},
        max_grad_norm=t["max_grad_norm"],
    )
    history, best, best_epoch, step = [], None, None, 0
    start_epoch, start_batch, latest_dev, dev_step = 0, 0, None, 0
    adapter_updated, decoder_updated = False, False
    if last_path.exists():
        restored = resume_training_checkpoint(last_path, model=trainable, optimizer=optimizer,
                                              expected_config=cfg, map_location=device)
        state = restored.metadata
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
    projection_before = module_sha256(model.projection) if experiment == "E3" else None
    projection_updated = False
    adapter_source_before = ({key: module_sha256(module) for key, module in model.adapters.items()}
                             if experiment == "E3" else {})
    adapters_updated = {key: False for key in adapter_source_before}
    if last_path.exists() and experiment == "E3":
        projection_updated = state["projection_updated"]
        adapters_updated = state["adapters_updated"]
    started, stopped = time.monotonic(), False
    stop_reason = "budget_completed"

    def metadata(next_epoch, next_batch):
        result = dict(config_sha256=digest, next_epoch=next_epoch, next_batch=next_batch,
                    history=history, best_metric=best, best_epoch=best_epoch, dev=latest_dev, dev_step=dev_step,
                    decoder_initial_sha256=decoder_initial_sha,
                    adapter_updated=adapter_updated or module_sha256(adapter_module) != adapter_before,
                    decoder_updated=decoder_updated or module_sha256(model.decoder) != decoder_before)
        if experiment == "E3":
            result.update(projection_updated=projection_updated or module_sha256(model.projection) != projection_before,
                          adapters_updated={key: adapters_updated[key] or module_sha256(module) != adapter_source_before[key]
                                            for key, module in model.adapters.items()})
        return result

    def checkpoint(path, epoch, next_epoch, next_batch):
        save_training_checkpoint(path, model=trainable, optimizer=optimizer, epoch=epoch,
                                 global_step=step, config=cfg, metadata=metadata(next_epoch, next_batch))

    for epoch in range(start_epoch, t["epochs"]):
        native_train.set_epoch(epoch)
        batches = DataLoader(tiles, batch_size=t["batch_size"], shuffle=True,
                             num_workers=d["num_workers"], collate_fn=g1_tile_collate,
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
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()):
                    raise RuntimeError(f"{name} gradients are absent or nonfinite")
            entry["seconds"] = time.monotonic() - tick
            history.append(entry)
            step += 1
            next_batch = batch_index + 1
            logging.info("epoch=%d step=%d/%d loss=%.6f BCE=%.6f Dice=%.6f",
                         epoch + 1, step, context["expected_steps"], entry["loss"], entry["bce"], entry["dice_loss"])
            if step % t["checkpoint_interval_steps"] == 0 or step == context["expected_steps"]:
                checkpoint(last_path, epoch, epoch, next_batch)
        if stopped:
            break
        # No extra update on resume at an epoch's final batch / declared max_steps.
        latest_dev = evaluate_dev(model, native_dev, cfg, device)
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
        if step >= context["expected_steps"]:
            break

    frozen_unchanged = module_sha256(model.extractor) == frozen_before
    adapter_updated |= module_sha256(adapter_module) != adapter_before
    decoder_updated |= module_sha256(model.decoder) != decoder_before
    complete = step == context["expected_steps"]
    if not frozen_unchanged or (complete and not (adapter_updated and decoder_updated)):
        raise RuntimeError("Backbone changed or trainable modules failed to update")
    if experiment == "E3":
        projection_updated |= module_sha256(model.projection) != projection_before
        adapters_updated = {key: adapters_updated[key] or module_sha256(module) != adapter_source_before[key]
                            for key, module in model.adapters.items()}
        if complete and (not projection_updated or not all(adapters_updated.values())):
            raise RuntimeError("E3 projection or a source Adapter failed to update")
    best_dev = None
    if best_path.is_file():
        payload, _ = load_checkpoint_payload(best_path, require_sha256=True)
        trainable.load_state_dict(payload["model_state"], strict=True)
        best_dev = evaluate_dev(model, native_dev, cfg, device, output_dir)
    if complete and (best_dev is None or best_dev["synthetic_dev_aupro_0_05"] != best):
        raise ValueError("Best checkpoint DEV does not reproduce its selected score")
    result = dict(
        status="PASS" if complete else "INCOMPLETE", training_complete=complete,
        mode=cfg["mode"], stage=cfg["stage"], category=cfg["category"],
        adapter={"r": cfg["adapter"]["r"], "d": cfg["adapter"]["d"]},
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
                  feature_sources=3 if experiment == "E3" else 1,
                  source_blocks=list(model.source_blocks) if experiment == "E3" else [model.extractor.depth],
                  feature_width=model.decoder.head[0].in_channels)
    if experiment == "E3":
        result.update(projection_updated=projection_updated, adapters_updated=adapters_updated,
                      source_block_map=model.source_block_map, fusion="mean", context=False)
    save_json(result, output_dir / "metrics.json")
    return result
