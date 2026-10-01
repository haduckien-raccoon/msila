"""
MS-ILA Day-3 — Task 12: checkpoint writing.

This module ONLY serializes training state. It does not restore a checkpoint
into a model/optimizer; restoration belongs to Task 13 (Resume).

Checkpoint contents
-------------------
- full model state_dict
- optimizer state_dict
- epoch and global_step
- experiment config
- RNG states: Python, NumPy, PyTorch CPU/CUDA
- optional named torch.Generator states (e.g. DataLoader shuffle generator)
- optional scheduler/scaler/other state_dict-capable objects
- lightweight runtime metadata

Writes are atomic (temporary file -> os.replace) and may emit a SHA-256
sidecar for file-integrity verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_SCHEMA_NAME = "msila_training_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when the Task-12 checkpoint contract is violated."""


@dataclass(frozen=True)
class CheckpointWriteResult:
    path: Path
    sha256: str
    size_bytes: int
    epoch: int
    global_step: int
    schema_name: str = CHECKPOINT_SCHEMA_NAME
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


def _jsonable(value: Any, *, path: str = "config") -> Any:
    """Convert common config values to deterministic JSON-compatible objects."""
    if is_dataclass(value):
        value = asdict(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            raise CheckpointError(f"{path}: non-finite float is not allowed")
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _jsonable(value.item(), path=path)

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointError(
                    f"{path}: config mapping keys must be strings, got {type(key)!r}"
                )
            out[key] = _jsonable(item, path=f"{path}.{key}")
        return out

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise CheckpointError(
        f"{path}: unsupported config value type {type(value)!r}. "
        "Use JSON-compatible values, pathlib.Path, dataclasses, or NumPy scalars."
    )


def capture_rng_state(
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    """Capture RNG states without advancing them."""
    named_generators: dict[str, torch.Tensor] = {}
    if generators:
        for name, generator in generators.items():
            if not isinstance(name, str) or not name:
                raise CheckpointError("Generator names must be non-empty strings")
            if not isinstance(generator, torch.Generator):
                raise TypeError(
                    f"generators[{name!r}] must be torch.Generator, "
                    f"got {type(generator)!r}"
                )
            named_generators[name] = generator.get_state().clone()

    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_states,
        "generators": named_generators,
    }


def _runtime_metadata() -> dict[str, Any]:
    cudnn_version = None
    if torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()

    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": cudnn_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _named_state_dicts(
    objects: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not objects:
        return {}

    out: dict[str, Any] = {}
    for name, obj in objects.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("stateful object names must be non-empty strings")
        state_dict = getattr(obj, "state_dict", None)
        if not callable(state_dict):
            raise CheckpointError(
                f"stateful_objects[{name!r}] must expose callable state_dict()"
            )
        out[name] = state_dict()
    return out


def build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    generators: Mapping[str, torch.Generator] | None = None,
    stateful_objects: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Task-12 payload in memory without writing or restoring state."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)!r}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"optimizer must be torch.optim.Optimizer, got {type(optimizer)!r}"
        )

    epoch = int(epoch)
    global_step = int(global_step)
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    normalized_config = _jsonable(config)
    normalized_metadata = _jsonable(metadata or {}, path="metadata")

    trainable_parameter_names = tuple(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    frozen_parameter_names = tuple(
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )

    payload = {
        "schema_name": CHECKPOINT_SCHEMA_NAME,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_state": {
            "epoch": epoch,
            "global_step": global_step,
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": normalized_config,
        "rng_state": capture_rng_state(generators=generators),
        "stateful_states": _named_state_dicts(stateful_objects),
        "model_contract": {
            "class_name": model.__class__.__qualname__,
            "trainable_parameter_names": trainable_parameter_names,
            "frozen_parameter_names": frozen_parameter_names,
        },
        "runtime": _runtime_metadata(),
        "metadata": normalized_metadata,
    }

    return payload


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    generators: Mapping[str, torch.Generator] | None = None,
    stateful_objects: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    write_sha256: bool = True,
) -> CheckpointWriteResult:
    """Atomically serialize one training checkpoint.

    This function never calls ``load_state_dict`` and therefore does not perform
    resume. Task 13 is responsible for restoring and validating saved state.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        config=config,
        generators=generators,
        stateful_objects=stateful_objects,
        metadata=metadata,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    sha256 = _sha256_file(destination)
    if write_sha256:
        sidecar = destination.with_suffix(destination.suffix + ".sha256")
        _atomic_text_write(sidecar, f"{sha256}  {destination.name}\n")

    return CheckpointWriteResult(
        path=destination.resolve(),
        sha256=sha256,
        size_bytes=int(destination.stat().st_size),
        epoch=int(epoch),
        global_step=int(global_step),
    )
