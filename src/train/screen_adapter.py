"""Day-04 controlled r×d Adapter screening runner.

Purpose
-------
Train exactly one Adapter candidate while enforcing a controlled ablation:
only

    r = bottleneck_dim
    d = projection_dim

may change across candidates in the same category screen.

One Day-04 candidate is instantiated as THREE independent block-specific
Adapters:

    b4  -> ResidualAdapter2d(r, d)
    b8  -> ResidualAdapter2d(r, d)
    b12 -> ResidualAdapter2d(r, d)

The three modules share the same architecture hyperparameters (r,d) but DO NOT
share trainable Parameter objects. This matches CachedFeatureTrainingModel,
which is the source-of-truth for the current cached-feature architecture.

The runner intentionally does NOT reimplement Fusion, Decoder, or the project
loss. Those components already belong to the project and must remain fixed.
Instead, a small hook module provides:

    build_model(adapters=..., config=...) -> torch.nn.Module
    step(model=..., batch=..., stage=..., config=...) -> Mapping

This avoids silently inventing a second training pipeline.

Required stage contract
-----------------------
For stage in {"train", "val"}:
    {
        "loss": scalar Tensor,
        "metrics": {optional scalar metrics}
    }

For stage == "predict":
    {
        "prediction": Tensor | nested Mapping/Tuple/List of Tensors
    }

Scientific controls implemented here
------------------------------------
1. Candidate must exist in the locked r×d grid.
2. The training protocol is hashed and locked per category.
3. Train/val record contents are hashed; overlap is rejected.
4. The same seed is used for every candidate in one locked screen.
5. The b4/b8/b12 Adapter initializations are deterministic, independent, and
   isolated from fixed-model initialization.
6. Non-adapter parameter structure AND initialization are hashed. Therefore
   changing Fusion/Decoder shape or initialization across candidates fails.
7. Output run names are deterministic: adapter_r{r}_d{d}.
8. Existing non-empty run directories are never overwritten.
9. Every run saves resolved config, best checkpoint, epoch CSV log, and raw
   validation predictions.

No DINO/DINOv3 extraction is performed here. Data must come from the Day-03
feature cache via data.cached_dataset.CachedFeatureDataset.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import inspect
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn


class ScreenConfigError(RuntimeError):
    """Raised when the Day-04 screen would no longer be a controlled ablation."""


@dataclass(frozen=True, slots=True)
class Candidate:
    r: int
    d: int
    run_name: str


@dataclass(frozen=True, slots=True)
class ProjectDeps:
    """Late-loaded project classes/functions to keep this runner testable."""

    CachedFeatureDataset: Any
    load_training_records: Any
    make_cached_dataloader: Any
    AdapterFactoryConfig: Any
    ResidualAdapterFactory: Any


def _load_project_dependencies() -> ProjectDeps:
    from src.data.cached_dataset import (
        CachedFeatureDataset,
        load_training_records,
        make_cached_dataloader,
    )
    from src.models.adapter_factory import (
        AdapterFactoryConfig,
        ResidualAdapterFactory,
    )

    return ProjectDeps(
        CachedFeatureDataset=CachedFeatureDataset,
        load_training_records=load_training_records,
        make_cached_dataloader=make_cached_dataloader,
        AdapterFactoryConfig=AdapterFactoryConfig,
        ResidualAdapterFactory=ResidualAdapterFactory,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one locked Day-04 Adapter r×d candidate."
    )
    p.add_argument("--r", type=int, required=True, help="Adapter bottleneck_dim.")
    p.add_argument("--d", type=int, required=True, help="Adapter projection_dim.")
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--grid",
        type=Path,
        default=Path("configs/day04_adapter_grid.yaml"),
        help="Locked candidate grid from the Adapter-screen setup.",
    )
    p.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/day04_train_protocol.yaml"),
        help="All fixed training/model/data settings.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', 'cuda', 'cuda:0', ...",
    )
    return p.parse_args(argv)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ScreenConfigError(f"{path} must contain one YAML mapping.")
    return data


def canonical_json(value: Any) -> str:
    """Stable serialization used for experiment fingerprints."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ScreenConfigError(f"Missing required protocol field: {path}")
        current = current[part]
    if current is None:
        raise ScreenConfigError(f"Protocol field must not be null: {path}")
    if isinstance(current, str) and current.strip().upper() in {
        "",
        "REPLACE_ME",
        "TODO",
        "REQUIRED",
    }:
        raise ScreenConfigError(f"Protocol field is not configured: {path}")
    return current


def _format_category(value: Any, category: str) -> Any:
    """Recursively replace literal {category} in string config values."""
    if isinstance(value, str):
        return value.replace("{category}", category)
    if isinstance(value, list):
        return [_format_category(x, category) for x in value]
    if isinstance(value, tuple):
        return tuple(_format_category(x, category) for x in value)
    if isinstance(value, dict):
        return {k: _format_category(v, category) for k, v in value.items()}
    return value


def resolve_protocol(
    protocol: Mapping[str, Any],
    *,
    category: str,
) -> dict[str, Any]:
    out = _format_category(dict(protocol), category)

    # Required instead of guessed: the note demands SAME values but does not
    # provide the project's actual LR/epochs/Fusion/Decoder/loss/paths.
    for path in (
        "hooks_module",
        "output_root",
        "data.cache_dir",
        "data.train_records",
        "data.val_records",
        "training.epochs",
        "training.batch_size",
        "training.optimizer.name",
        "training.optimizer.lr",
        "model.loss",
        "model.fusion",
        "model.decoder",
        "frozen_backbone.checkpoint",
        "frozen_backbone.blocks",
        "frozen_backbone.local_context",
        "frozen_backbone.alignment",
    ):
        _require(out, path)

    if not bool(_require(out, "frozen_backbone.frozen")):
        raise ScreenConfigError(
            "Day-04 cached-feature screen requires frozen_backbone.frozen=true."
        )

    epochs = int(_require(out, "training.epochs"))
    batch_size = int(_require(out, "training.batch_size"))
    if epochs <= 0:
        raise ScreenConfigError("training.epochs must be > 0")
    if batch_size <= 0:
        raise ScreenConfigError("training.batch_size must be > 0")

    monitor = str(out.get("checkpoint", {}).get("monitor", "val_loss"))
    mode = str(out.get("checkpoint", {}).get("mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ScreenConfigError("checkpoint.mode must be 'min' or 'max'")
    out.setdefault("checkpoint", {})
    out["checkpoint"]["monitor"] = monitor
    out["checkpoint"]["mode"] = mode

    return out


def load_candidate_from_grid(
    grid_path: str | Path,
    *,
    r: int,
    d: int,
) -> tuple[Candidate, dict[str, Any]]:
    grid = load_yaml(grid_path)
    candidates = grid.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ScreenConfigError("Grid must contain a non-empty 'candidates' list.")

    seen: set[str] = set()
    match: Candidate | None = None

    for i, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ScreenConfigError(f"grid.candidates[{i}] must be a mapping.")
        rr = int(_candidate_value(raw, "bottleneck_dim", "r"))
        dd = int(_candidate_value(raw, "projection_dim", "d"))
        expected_name = f"adapter_r{rr}_d{dd}"
        supplied = raw.get("run_name", expected_name)
        if supplied != expected_name:
            raise ScreenConfigError(
                f"Non-deterministic run_name in grid: {supplied!r}; "
                f"expected {expected_name!r}."
            )
        if expected_name in seen:
            raise ScreenConfigError(f"Duplicate candidate in grid: {expected_name}")
        seen.add(expected_name)

        if rr == r and dd == d:
            match = Candidate(r=rr, d=dd, run_name=expected_name)

    expected_n = grid.get("expected_num_candidates")
    if expected_n is not None and int(expected_n) != len(candidates):
        raise ScreenConfigError(
            f"Grid expected_num_candidates={expected_n}, actual={len(candidates)}."
        )

    if match is None:
        raise ScreenConfigError(
            f"(r={r}, d={d}) is not in the locked grid. "
            "Do not create ad-hoc candidates during Day 04."
        )

    defaults = grid.get("adapter_defaults")
    if not isinstance(defaults, Mapping):
        raise ScreenConfigError("Grid must contain adapter_defaults.")
    return match, dict(defaults)


def _candidate_value(raw: Mapping[str, Any], canonical: str, alias: str) -> Any:
    if canonical in raw:
        return raw[canonical]
    if alias in raw:
        return raw[alias]
    raise ScreenConfigError(f"Candidate missing {canonical}/{alias}.")


def seed_everything(seed: int, *, deterministic: bool, warn_only: bool) -> None:
    """Seed process RNGs and configure PyTorch deterministic execution."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ScreenConfigError(f"Requested {device}, but CUDA is unavailable.")
    return device


def import_hooks(module_name: str):
    module = importlib.import_module(module_name)
    for name in ("build_model", "step"):
        fn = getattr(module, name, None)
        if not callable(fn):
            raise ScreenConfigError(
                f"Hook module {module_name!r} must define callable {name}()."
            )
    return module


def module_source_sha256(module: Any) -> str | None:
    try:
        source = inspect.getsourcefile(module)
    except (TypeError, OSError):
        source = None
    if source is None:
        return None
    path = Path(source)
    return sha256_file(path) if path.is_file() else None


def filtered_records(
    load_training_records: Any,
    source: str | Path,
    *,
    category: str,
) -> list[dict[str, Any]]:
    records = load_training_records(source)
    selected = [dict(r) for r in records if str(r["category"]) == category]
    if not selected:
        raise ScreenConfigError(
            f"No records for category={category!r} in {source!s}."
        )
    return selected


def audit_split_disjoint(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
) -> None:
    def keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
        return {(str(r["category"]), str(r["image_id"])) for r in records}

    overlap = keys(train_records) & keys(val_records)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ScreenConfigError(
            f"Train/val leakage: {len(overlap)} sample(s) overlap, e.g. {preview}."
        )


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [_jsonable_record(r) for r in records]
    return sha256_json(normalized)


def _jsonable_record(value: Any) -> Any:
    if isinstance(value, Tensor):
        t = value.detach().cpu().contiguous()
        h = hashlib.sha256(t.numpy().tobytes()).hexdigest()
        return {
            "__tensor__": True,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256": h,
        }
    if isinstance(value, Mapping):
        return {str(k): _jsonable_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_record(v) for v in value]
    if isinstance(value, Path):
        return str(value)

    # Normalize scalar subclasses to built-in Python types before YAML dump.
    # Newer PyTorch versions expose torch.__version__ as TorchVersion, a str
    # subclass that PyYAML SafeDumper may reject even though isinstance(..., str)
    # is True.  Returning the original object therefore is not safe.
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return repr(value)


def build_dataset(
    deps: ProjectDeps,
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
):
    data_cfg = protocol["data"]
    kwargs: dict[str, Any] = {
        "cache_dir": data_cfg["cache_dir"],
        "records": records,
        "expected_producer_signature": data_cfg.get(
            "expected_producer_signature"
        ),
        "mask_root": data_cfg.get("mask_root"),
        "mask_threshold": int(data_cfg.get("mask_threshold", 0)),
        "mask_hw_source": str(data_cfg.get("mask_hw_source", "record")),
        "allow_zero_mask_for_normal": bool(
            data_cfg.get("allow_zero_mask_for_normal", True)
        ),
        "squeeze_cached_batch_dim": bool(
            data_cfg.get("squeeze_cached_batch_dim", True)
        ),
        "feature_dtype": None,
        "mmap": bool(data_cfg.get("mmap", True)),
        "shard_cache_size": int(data_cfg.get("shard_cache_size", 2)),
    }
    return deps.CachedFeatureDataset(**kwargs)


def build_loader(
    deps: ProjectDeps,
    dataset: Any,
    protocol: Mapping[str, Any],
    *,
    shuffle: bool,
    seed: int,
):
    data_cfg = protocol["data"]
    train_cfg = protocol["training"]
    return deps.make_cached_dataloader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=data_cfg.get("pin_memory"),
        persistent_workers=data_cfg.get("persistent_workers"),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        drop_last=bool(data_cfg.get("drop_last", False)) if shuffle else False,
        seed=int(seed),
    )


def build_adapters(
    deps: ProjectDeps,
    adapter_defaults: Mapping[str, Any],
    candidate: Candidate,
    *,
    seed: int,
) -> tuple[nn.ModuleDict, dict[str, dict[str, Any]]]:
    """Build the Day-04 candidate as three independent b4/b8/b12 Adapters.

    All three blocks use the same candidate hyperparameters ``(r, d)`` and the
    same fixed Adapter settings.  Their Parameter objects are deliberately
    independent, matching ``CachedFeatureTrainingModel.build_default()``.

    A deterministic block-specific seed is used only inside a forked CPU RNG
    context, so Adapter construction cannot perturb initialization of the fixed
    Projection/Fusion/Decoder modules built afterwards.
    """
    fixed = deps.AdapterFactoryConfig.from_mapping(adapter_defaults)
    factory = deps.ResidualAdapterFactory(fixed)

    modules: dict[str, nn.Module] = {}
    records: dict[str, dict[str, Any]] = {}

    for block in (4, 8, 12):
        key = f"b{block}"

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + int(block))
            build = factory.build_rd(r=candidate.r, d=candidate.d)

        if build.run_name != candidate.run_name:
            raise ScreenConfigError(
                f"Factory run_name drift for {key}: "
                f"{build.run_name} != {candidate.run_name}"
            )

        modules[key] = build.model
        records[key] = build.record()

    adapters = nn.ModuleDict(modules)

    if set(adapters.keys()) != {"b4", "b8", "b12"}:
        raise ScreenConfigError(
            "Day-04 candidate must contain exactly b4/b8/b12 Adapters."
        )

    # Strong non-sharing gate: neither module objects nor Parameter objects may
    # be reused across DINO blocks.
    if (
        adapters["b4"] is adapters["b8"]
        or adapters["b4"] is adapters["b12"]
        or adapters["b8"] is adapters["b12"]
    ):
        raise ScreenConfigError(
            "b4/b8/b12 must be independent Adapter module instances."
        )

    parameter_ids = {
        key: {id(p) for p in module.parameters()}
        for key, module in adapters.items()
    }
    for left, right in (("b4", "b8"), ("b4", "b12"), ("b8", "b12")):
        if parameter_ids[left] & parameter_ids[right]:
            raise ScreenConfigError(
                f"{left} and {right} share Adapter Parameter objects; "
                "Day-04 requires independent block-specific Adapters."
            )

    return adapters, records


def build_project_model(
    hooks: Any,
    *,
    adapters: nn.ModuleDict,
    protocol: Mapping[str, Any],
    seed: int,
    deterministic: bool,
    warn_only: bool,
) -> nn.Module:
    """Build the fixed downstream model around the exact supplied Adapters."""
    if not isinstance(adapters, nn.ModuleDict):
        raise ScreenConfigError("adapters must be torch.nn.ModuleDict.")
    if set(adapters.keys()) != {"b4", "b8", "b12"}:
        raise ScreenConfigError("adapters must contain exactly b4/b8/b12.")

    seed_everything(seed, deterministic=deterministic, warn_only=warn_only)
    model = hooks.build_model(adapters=adapters, config=protocol)

    if not isinstance(model, nn.Module):
        raise ScreenConfigError("hooks.build_model() must return torch.nn.Module.")

    supplied_param_ids = {id(p) for p in adapters.parameters()}
    model_param_ids = {id(p) for p in model.parameters()}

    if not supplied_param_ids:
        raise ScreenConfigError("Supplied Adapter ModuleDict has no parameters.")

    if not supplied_param_ids.issubset(model_param_ids):
        raise ScreenConfigError(
            "build_model() did not register the exact b4/b8/b12 Adapter "
            "Parameter objects supplied by the runner. Do not silently "
            "rebuild/copy the candidate Adapters."
        )

    return model


def non_adapter_model_fingerprint(
    model: nn.Module,
    adapters: nn.Module,
) -> dict[str, Any]:
    """Fingerprint fixed model state while excluding all screenable Adapters."""
    adapter_param_ids = {id(p) for p in adapters.parameters()}
    adapter_buffer_ids = {id(b) for b in adapters.buffers()}

    structure: list[dict[str, Any]] = []
    h = hashlib.sha256()

    def add_tensor(kind: str, name: str, tensor: Tensor) -> None:
        t = tensor.detach().cpu().contiguous()
        record = {
            "kind": kind,
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        }
        structure.append(record)
        h.update(canonical_json(record).encode("utf-8"))
        if t.dtype == torch.bfloat16:
            raw = t.view(torch.uint16).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        h.update(raw)

    for name, p in model.named_parameters():
        if id(p) not in adapter_param_ids:
            add_tensor("parameter", name, p)

    for name, b in model.named_buffers():
        if id(b) not in adapter_buffer_ids:
            add_tensor("buffer", name, b)

    return {
        "structure_sha256": sha256_json(structure),
        "initial_state_sha256": h.hexdigest(),
        "num_fixed_tensors": len(structure),
        "structure": structure,
    }


def build_optimizer(model: nn.Module, protocol: Mapping[str, Any]):
    cfg = protocol["training"]["optimizer"]
    name = str(cfg["name"])
    lr = float(cfg["lr"])
    if lr <= 0:
        raise ScreenConfigError("training.optimizer.lr must be > 0.")

    cls = getattr(torch.optim, name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(
        cls, torch.optim.Optimizer
    ):
        raise ScreenConfigError(f"Unknown torch.optim optimizer: {name!r}")

    kwargs = dict(cfg.get("kwargs", {}))
    if "lr" in kwargs or "params" in kwargs:
        raise ScreenConfigError(
            "optimizer.kwargs must not override params or lr."
        )
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ScreenConfigError("Model has no trainable parameters.")
    return cls(params, lr=lr, **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, protocol: Mapping[str, Any]):
    cfg = protocol["training"].get("scheduler")
    if cfg in (None, False):
        return None, None

    if not isinstance(cfg, Mapping):
        raise ScreenConfigError("training.scheduler must be null or a mapping.")
    name = str(_require(cfg, "name"))
    cls = getattr(torch.optim.lr_scheduler, name, None)
    if cls is None or not isinstance(cls, type):
        raise ScreenConfigError(f"Unknown torch scheduler: {name!r}")
    kwargs = dict(cfg.get("kwargs", {}))
    step_on = str(cfg.get("step_on", "epoch"))
    if step_on not in {"epoch", "metric"}:
        raise ScreenConfigError("scheduler.step_on must be epoch or metric")
    return cls(optimizer, **kwargs), step_on


def _amp_settings(protocol: Mapping[str, Any], device: torch.device):
    cfg = protocol["training"].get("amp", {}) or {}
    enabled = bool(cfg.get("enabled", False))
    dtype_name = str(cfg.get("dtype", "bfloat16")).lower()

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ScreenConfigError("AMP dtype must be float16/fp16 or bfloat16/bf16.")
    dtype = dtype_map[dtype_name]

    if enabled and device.type not in {"cuda", "cpu"}:
        raise ScreenConfigError(
            f"AMP in this runner supports cuda/cpu, got {device.type}."
        )
    if enabled and device.type == "cpu" and dtype is torch.float16:
        raise ScreenConfigError("CPU AMP float16 is not supported here; use bfloat16.")

    use_scaler = enabled and device.type == "cuda" and dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    return enabled, dtype, scaler


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=(device.type == "cuda"))
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def _batch_size(batch: Mapping[str, Any]) -> int:
    mask = batch.get("mask")
    if isinstance(mask, Tensor) and mask.ndim >= 1:
        return int(mask.shape[0])
    meta = batch.get("meta")
    if isinstance(meta, Sequence):
        return len(meta)
    raise ScreenConfigError("Cannot infer batch size from mask/meta.")


def _scalar(value: Any, *, name: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ScreenConfigError(f"Metric {name!r} must be scalar.")
        value = value.detach().item()
    value = float(value)
    if not np.isfinite(value):
        raise ScreenConfigError(f"Metric {name!r} is NaN/Inf.")
    return value


def normalize_step_output(
    output: Any,
    *,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise ScreenConfigError("hooks.step() must return a mapping.")
    out = dict(output)

    if stage in {"train", "val"}:
        loss = out.get("loss")
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise ScreenConfigError(
                f"{stage} step must return scalar Tensor under key 'loss'."
            )
        if not torch.isfinite(loss.detach()).item():
            raise ScreenConfigError(f"{stage} loss is NaN/Inf.")
        metrics = out.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ScreenConfigError("'metrics' must be a mapping.")
    elif stage == "predict":
        if "prediction" not in out:
            raise ScreenConfigError(
                "predict step must return key 'prediction'."
            )
    else:
        raise ValueError(stage)
    return out


def _autocast_context(
    *,
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype,
):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def run_epoch(
    *,
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    hooks: Any,
    protocol: Mapping[str, Any],
    device: torch.device,
    stage: str,
    optimizer: torch.optim.Optimizer | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    train = stage == "train"
    model.train(train)

    sums: dict[str, float] = {}
    total_n = 0
    clip_norm = protocol["training"].get("gradient_clip_norm")

    grad_context = contextlib.nullcontext() if train else torch.no_grad()
    with grad_context:
        for batch_cpu in loader:
            batch = move_to_device(batch_cpu, device)
            n = _batch_size(batch)
            total_n += n

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            with _autocast_context(
                device=device,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                raw = hooks.step(
                    model=model,
                    batch=batch,
                    stage=stage,
                    config=protocol,
                )
                out = normalize_step_output(raw, stage=stage)
                loss: Tensor = out["loss"]

            if train:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(clip_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(clip_norm)
                        )
                    optimizer.step()

            values = {"loss": _scalar(loss, name="loss")}
            for name, value in dict(out.get("metrics", {})).items():
                values[str(name)] = _scalar(value, name=str(name))

            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + value * n

    if total_n == 0:
        raise ScreenConfigError(f"{stage} loader produced zero samples.")
    return {name: value / total_n for name, value in sums.items()}


def is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return value < best if mode == "min" else value > best


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def save_yaml(data: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            _jsonable_record(data),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "sample"


def _slice_prediction(value: Any, i: int, batch_size: int) -> Any:
    if isinstance(value, Tensor):
        v = value.detach().cpu()
        if v.ndim >= 1 and v.shape[0] == batch_size:
            return v[i]
        return v
    if isinstance(value, Mapping):
        return {k: _slice_prediction(v, i, batch_size) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_prediction(v, i, batch_size) for v in value)
    if isinstance(value, list):
        if len(value) == batch_size:
            return _slice_prediction(value[i], 0, 1)
        return [_slice_prediction(v, i, batch_size) for v in value]
    return value


def export_predictions(
    *,
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    hooks: Any,
    protocol: Mapping[str, Any],
    device: torch.device,
    predictions_dir: Path,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = predictions_dir / "manifest.jsonl"

    model.eval()
    row_index = 0
    with manifest_path.open("w", encoding="utf-8") as manifest, torch.no_grad():
        for batch_cpu in loader:
            batch = move_to_device(batch_cpu, device)
            n = _batch_size(batch)
            with _autocast_context(
                device=device,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                raw = hooks.step(
                    model=model,
                    batch=batch,
                    stage="predict",
                    config=protocol,
                )
                out = normalize_step_output(raw, stage="predict")

            metas = batch_cpu.get("meta")
            if not isinstance(metas, Sequence) or len(metas) != n:
                raise ScreenConfigError(
                    "Prediction export requires batch['meta'] with one item/sample."
                )

            for i, meta in enumerate(metas):
                meta = dict(meta)
                image_id = str(meta.get("image_id", f"sample_{row_index:06d}"))
                category = str(meta.get("category", "unknown"))
                stem = _sanitize_filename(
                    f"{row_index:06d}__{category}__{image_id}"
                )
                filename = stem + ".pt"
                payload = {
                    "prediction": _slice_prediction(
                        out["prediction"], i, n
                    ),
                    "meta": meta,
                }
                torch.save(payload, predictions_dir / filename)
                manifest.write(
                    json.dumps(
                        {
                            "index": row_index,
                            "file": filename,
                            "category": category,
                            "image_id": image_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                row_index += 1


def _git_commit_or_none() -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return p.stdout.strip() or None
    except Exception:
        return None


def protocol_lock_payload(
    *,
    protocol: Mapping[str, Any],
    category: str,
    seed: int,
    grid_sha256: str,
    train_records_sha256: str,
    val_records_sha256: str,
    hooks_module: str,
    hooks_source_sha256: str | None,
    fixed_model_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    fp = dict(fixed_model_fingerprint)
    fp.pop("structure", None)
    return {
        "category": category,
        "seed": int(seed),
        "grid_sha256": grid_sha256,
        "train_records_sha256": train_records_sha256,
        "val_records_sha256": val_records_sha256,
        "hooks_module": hooks_module,
        "hooks_source_sha256": hooks_source_sha256,
        "fixed_model": fp,
        "protocol": protocol,
    }


def enforce_protocol_lock(
    output_root: Path,
    *,
    category: str,
    payload: Mapping[str, Any],
) -> str:
    lock_dir = output_root / "_protocol_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    category_name = _sanitize_filename(category)
    lock_path = lock_dir / f"{category_name}.yaml"
    hash_path = lock_dir / f"{category_name}.sha256"

    fingerprint = sha256_json(payload)
    if lock_path.exists():
        existing = load_yaml(lock_path)
        existing_hash = sha256_json(existing)
        if existing_hash != fingerprint:
            raise ScreenConfigError(
                "Day-04 protocol drift detected for this category. "
                "At least one supposedly fixed variable changed "
                "(seed/data split/cache/model/loss/optimizer/LR/epochs/"
                "batch size/Fusion/Decoder/etc.). Start a new experiment "
                "instead of mixing it into the existing screen."
            )
    else:
        save_yaml(dict(payload), lock_path)
        hash_path.write_text(fingerprint + "\n", encoding="utf-8")

    return fingerprint


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: Mapping[str, Any],
    candidate: Candidate,
    protocol_sha256: str,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "epoch": int(epoch),
            "candidate": {
                "r": candidate.r,
                "d": candidate.d,
                "run_name": candidate.run_name,
            },
            "protocol_sha256": protocol_sha256,
            "metrics": dict(metrics),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "torch_version": torch.__version__,
        },
        path,
    )



def _format_duration(seconds: float) -> str:
    """Compact human-readable duration used only for live console progress."""
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _safe_len(value: Any) -> int | None:
    """Return len(value) when available; progress reporting must never break training."""
    try:
        return int(len(value))
    except (TypeError, AttributeError):
        return None


def run_candidate(args: argparse.Namespace) -> Path:
    run_t0 = time.perf_counter()
    candidate, adapter_defaults = load_candidate_from_grid(
        args.grid, r=args.r, d=args.d
    )
    print(
        f"[DAY04][SETUP] category={args.category} | "
        f"candidate={candidate.run_name} | seed={args.seed} | device_request={args.device}",
        flush=True,
    )
    raw_protocol = load_yaml(args.protocol)
    protocol = resolve_protocol(raw_protocol, category=args.category)

    deps = _load_project_dependencies()
    train_records = filtered_records(
        deps.load_training_records,
        protocol["data"]["train_records"],
        category=args.category,
    )
    val_records = filtered_records(
        deps.load_training_records,
        protocol["data"]["val_records"],
        category=args.category,
    )
    audit_split_disjoint(train_records, val_records)
    print(
        f"[DAY04][DATA] category={args.category} | "
        f"train_samples={len(train_records)} | val_samples={len(val_records)} | "
        f"cache_dir={protocol['data']['cache_dir']}",
        flush=True,
    )

    deterministic = bool(
        protocol["training"].get("deterministic_algorithms", True)
    )
    warn_only = bool(
        protocol["training"].get("deterministic_warn_only", False)
    )
    seed_everything(
        args.seed,
        deterministic=deterministic,
        warn_only=warn_only,
    )

    hooks_module = str(protocol["hooks_module"])
    hooks = import_hooks(hooks_module)

    adapters, adapter_records = build_adapters(
        deps,
        adapter_defaults,
        candidate,
        seed=args.seed,
    )

    model = build_project_model(
        hooks,
        adapters=adapters,
        protocol=protocol,
        seed=args.seed,
        deterministic=deterministic,
        warn_only=warn_only,
    )

    fixed_fp = non_adapter_model_fingerprint(model, adapters)
    output_root = Path(protocol["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    lock_payload = protocol_lock_payload(
        protocol=protocol,
        category=args.category,
        seed=args.seed,
        grid_sha256=sha256_file(args.grid),
        train_records_sha256=records_sha256(train_records),
        val_records_sha256=records_sha256(val_records),
        hooks_module=hooks_module,
        hooks_source_sha256=module_source_sha256(hooks),
        fixed_model_fingerprint=fixed_fp,
    )
    protocol_sha = enforce_protocol_lock(
        output_root,
        category=args.category,
        payload=lock_payload,
    )

    run_dir = output_root / candidate.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ScreenConfigError(
            f"Refusing to overwrite existing run directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_total_params = sum(
        int(record["trainable_params"])
        for record in adapter_records.values()
    )

    config_record = {
        "candidate": {
            "r": candidate.r,
            "d": candidate.d,
            "run_name": candidate.run_name,
            "adapter_blocks": ["b4", "b8", "b12"],
            "trainable_adapter_params": adapter_total_params,
            "per_block": adapter_records,
        },
        "category": args.category,
        "seed": int(args.seed),

        # Authoritative E9 subtree.  Within one category/seed screen, the
        # fairness audit may remove ONLY adapter.bottleneck_dim and
        # adapter.projection_dim; all remaining scientific settings must hash
        # identically across candidates.
        "scientific_config": {
            "adapter": {
                "bottleneck_dim": candidate.r,
                "projection_dim": candidate.d,
                "kernel_size": int(adapter_defaults["kernel_size"]),
                "gamma_init": float(adapter_defaults["gamma_init"]),
                "bias": bool(adapter_defaults["bias"]),
                "blocks": [4, 8, 12],
                "sharing": "independent_across_blocks_shared_local_context_within_block",
            },
            "frozen_backbone": protocol["frozen_backbone"],
            "data": protocol["data"],
            "model": protocol["model"],
            "training": protocol["training"],
            "checkpoint": protocol["checkpoint"],
        },

        "protocol_sha256": protocol_sha,
        "grid_sha256": sha256_file(args.grid),
        "train_records_sha256": records_sha256(train_records),
        "val_records_sha256": records_sha256(val_records),
        "hooks_source_sha256": module_source_sha256(hooks),
        "fixed_model_fingerprint": fixed_fp,
        "protocol": protocol,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_request": args.device,
            "git_commit": _git_commit_or_none(),
        },
    }
    expected_one = next(iter(adapter_records.values()))["trainable_params"]
    if any(
        int(record["trainable_params"]) != int(expected_one)
        for record in adapter_records.values()
    ):
        raise ScreenConfigError(
            "b4/b8/b12 Adapter parameter counts differ for the same (r,d)."
        )
    if adapter_total_params != 3 * int(expected_one):
        raise ScreenConfigError(
            "Total Adapter parameter count must equal 3 × per-block count."
        )

    save_yaml(config_record, run_dir / "config.yaml")

    train_ds = build_dataset(deps, protocol, train_records)
    val_ds = build_dataset(deps, protocol, val_records)
    train_loader = build_loader(
        deps, train_ds, protocol, shuffle=True, seed=args.seed
    )
    val_loader = build_loader(
        deps, val_ds, protocol, shuffle=False, seed=args.seed
    )

    device = resolve_device(args.device)
    model = model.to(device)

    optimizer = build_optimizer(model, protocol)
    scheduler, scheduler_step_on = build_scheduler(optimizer, protocol)
    amp_enabled, amp_dtype, scaler = _amp_settings(protocol, device)

    epochs = int(protocol["training"]["epochs"])
    monitor = str(protocol["checkpoint"]["monitor"])
    mode = str(protocol["checkpoint"]["mode"])
    best: float | None = None
    rows: list[dict[str, Any]] = []
    best_path = run_dir / "best.pt"

    train_steps = _safe_len(train_loader)
    val_steps = _safe_len(val_loader)
    batch_size = int(protocol["training"]["batch_size"])
    amp_label = str(amp_dtype).replace("torch.", "") if amp_enabled else "off"
    print(
        f"[DAY04][TRAIN START] category={args.category} | "
        f"candidate={candidate.run_name} | seed={args.seed} | "
        f"epochs={epochs} | batch_size={batch_size} | "
        f"train_steps={train_steps if train_steps is not None else '?'} | "
        f"val_steps={val_steps if val_steps is not None else '?'} | "
        f"device={device} | amp={amp_label}",
        flush=True,
    )

    train_t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            hooks=hooks,
            protocol=protocol,
            device=device,
            stage="train",
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            hooks=hooks,
            protocol=protocol,
            device=device,
            stage="val",
            optimizer=None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        rows.append(row)
        write_csv(rows, run_dir / "train_log.csv")

        if monitor not in row:
            raise ScreenConfigError(
                f"checkpoint.monitor={monitor!r} not produced. "
                f"Available epoch metrics: {sorted(row)}"
            )
        score = float(row[monitor])

        if is_better(score, best, mode):
            best = score
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=row,
                candidate=candidate,
                protocol_sha256=protocol_sha,
            )

        if scheduler is not None:
            if scheduler_step_on == "metric":
                scheduler.step(score)
            else:
                scheduler.step()

        # Live progress only: no effect on optimization, checkpointing, or metrics.
        epoch_sec = time.perf_counter() - epoch_t0
        train_elapsed = time.perf_counter() - train_t0
        avg_epoch_sec = train_elapsed / float(epoch)
        eta_sec = avg_epoch_sec * float(epochs - epoch)
        best_text = "nan" if best is None else f"{best:.6f}"
        train_loss = float(row.get("train_loss", float("nan")))
        val_loss = float(row.get("val_loss", float("nan")))
        pct = 100.0 * float(epoch) / float(epochs)
        print(
            f"[DAY04][TRAIN] {pct:6.2f}% | "
            f"category={args.category} | candidate={candidate.run_name} | seed={args.seed} | "
            f"epoch={epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"best_{monitor}={best_text} | "
            f"epoch_time={_format_duration(epoch_sec)} | ETA={_format_duration(eta_sec)}",
            flush=True,
        )

    print(
        f"[DAY04][TRAIN DONE] category={args.category} | "
        f"candidate={candidate.run_name} | seed={args.seed} | "
        f"epochs={epochs} | elapsed={_format_duration(time.perf_counter() - train_t0)} | "
        f"best_{monitor}={'nan' if best is None else f'{best:.6f}'}",
        flush=True,
    )

    if not best_path.is_file():
        raise ScreenConfigError("Training ended without a best checkpoint.")

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,  # trusted checkpoint produced by this process
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    print(
        f"[DAY04][PREDICT START] category={args.category} | "
        f"candidate={candidate.run_name} | val_samples={len(val_ds)}",
        flush=True,
    )
    predict_t0 = time.perf_counter()
    export_predictions(
        model=model,
        loader=val_loader,
        hooks=hooks,
        protocol=protocol,
        device=device,
        predictions_dir=run_dir / "predictions",
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    print(
        f"[DAY04][PREDICT DONE] category={args.category} | "
        f"candidate={candidate.run_name} | elapsed={_format_duration(time.perf_counter() - predict_t0)}",
        flush=True,
    )
    print(
        f"[DAY04][RUN DONE] category={args.category} | candidate={candidate.run_name} | "
        f"seed={args.seed} | total_elapsed={_format_duration(time.perf_counter() - run_t0)} | "
        f"run_dir={run_dir}",
        flush=True,
    )

    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = run_candidate(args)
    except Exception as exc:
        # Keep the exact root cause visible through notebook log filters.
        print(
            f"[DAY04 ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    print(f"[DAY04][PASS] completed: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
