"""
MS-ILA Day-3 — Task 13: resume training state from a Task-12 checkpoint.

Scope
-----
This module restores:
- model state
- optimizer state
- epoch/global_step
- optional scheduler/scaler-like stateful objects
- Python/NumPy/PyTorch RNG states
- optional named torch.Generator states

It does NOT implement:
- the training loop (Task 10),
- checkpoint writing (Task 12),
- full integration QA (Task 14),
- Day-03 reporting (Task 15).

Security note
-------------
Task-12 checkpoints contain Python/NumPy RNG objects and are loaded with
``torch.load(..., weights_only=False)``. Only load checkpoints produced by this
project or another trusted source. PyTorch pickle-based checkpoints are not a
safe format for untrusted files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from .checkpoint import (
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
)


class ResumeError(RuntimeError):
    """Raised when a checkpoint cannot be resumed safely."""


@dataclass(frozen=True)
class ResumeResult:
    path: Path
    epoch: int
    global_step: int
    config: dict[str, Any]
    metadata: dict[str, Any]
    sha256_verified: bool
    rng_restored: bool
    restored_generators: tuple[str, ...]
    restored_stateful_objects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(
    path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    require_sidecar: bool = True,
) -> bool:
    """Verify the Task-12 ``.sha256`` sidecar before deserialization."""
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sidecar = (
        Path(sidecar_path).expanduser()
        if sidecar_path is not None
        else checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    )

    if not sidecar.is_file():
        if require_sidecar:
            raise ResumeError(f"SHA-256 sidecar not found: {sidecar}")
        return False

    raw = sidecar.read_text(encoding="utf-8").strip()
    parts = raw.split()
    if not parts:
        raise ResumeError(f"Malformed SHA-256 sidecar: {sidecar}")

    expected = parts[0].lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ResumeError(f"Malformed SHA-256 digest in {sidecar}")

    if len(parts) >= 2:
        recorded_name = parts[-1]
        if recorded_name != checkpoint.name:
            raise ResumeError(
                "SHA-256 sidecar filename does not match checkpoint: "
                f"{recorded_name!r} != {checkpoint.name!r}"
            )

    observed = _sha256_file(checkpoint)
    if observed != expected:
        raise ResumeError(
            "Checkpoint SHA-256 mismatch. The file may be corrupted or modified."
        )

    return True


def _normalize_jsonlike(value: Any, *, path: str = "config") -> Any:
    """Normalize expected config for comparison with Task-12 serialized config."""
    if is_dataclass(value):
        value = asdict(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ResumeError(f"{path}: non-finite float is not allowed")
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _normalize_jsonlike(value.item(), path=path)

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResumeError(
                    f"{path}: mapping keys must be strings, got {type(key)!r}"
                )
            out[key] = _normalize_jsonlike(item, path=f"{path}.{key}")
        return out

    if isinstance(value, (list, tuple)):
        return [
            _normalize_jsonlike(item, path=f"{path}[{i}]")
            for i, item in enumerate(value)
        ]

    raise ResumeError(
        f"{path}: unsupported value type {type(value)!r} for config comparison"
    )


def _required_payload_keys() -> set[str]:
    return {
        "schema_name",
        "schema_version",
        "training_state",
        "model_state",
        "optimizer_state",
        "config",
        "rng_state",
        "stateful_states",
        "model_contract",
        "runtime",
        "metadata",
    }


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ResumeError(
            f"Checkpoint payload must be dict, got {type(payload)!r}"
        )

    missing = _required_payload_keys() - set(payload)
    if missing:
        raise ResumeError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    if payload["schema_name"] != CHECKPOINT_SCHEMA_NAME:
        raise ResumeError(
            f"schema_name mismatch: expected {CHECKPOINT_SCHEMA_NAME!r}, "
            f"got {payload['schema_name']!r}"
        )
    if int(payload["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ResumeError(
            f"schema_version mismatch: expected {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {payload['schema_version']!r}"
        )

    training_state = payload["training_state"]
    if not isinstance(training_state, Mapping):
        raise ResumeError("training_state must be a mapping")
    if set(training_state) != {"epoch", "global_step"}:
        raise ResumeError(
            "training_state must contain exactly {'epoch','global_step'}"
        )

    epoch = int(training_state["epoch"])
    global_step = int(training_state["global_step"])
    if epoch < 0 or global_step < 0:
        raise ResumeError("epoch/global_step must be non-negative")

    if not isinstance(payload["model_state"], Mapping):
        raise ResumeError("model_state must be a mapping")
    if not isinstance(payload["optimizer_state"], Mapping):
        raise ResumeError("optimizer_state must be a mapping")
    if not isinstance(payload["config"], Mapping):
        raise ResumeError("config must be a mapping")
    if not isinstance(payload["rng_state"], Mapping):
        raise ResumeError("rng_state must be a mapping")
    if not isinstance(payload["stateful_states"], Mapping):
        raise ResumeError("stateful_states must be a mapping")
    if not isinstance(payload["model_contract"], Mapping):
        raise ResumeError("model_contract must be a mapping")
    if not isinstance(payload["metadata"], Mapping):
        raise ResumeError("metadata must be a mapping")

    return payload


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    verify_sha256: bool = True,
    require_sha256: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Verify, deserialize and validate a Task-12 checkpoint without mutating objects."""
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sha_verified = False
    if verify_sha256:
        sha_verified = verify_checkpoint_sha256(
            checkpoint,
            require_sidecar=require_sha256,
        )

    # Task-12 stores NumPy/Python RNG objects. This requires trusted pickle load.
    payload = torch.load(
        checkpoint,
        map_location=map_location,
        weights_only=False,
    )
    return _validate_payload(payload), bool(sha_verified)


def _validate_model_contract(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    strict_class_name: bool,
    strict_trainability: bool,
) -> None:
    contract = payload["model_contract"]

    if strict_class_name:
        expected_class = contract.get("class_name")
        actual_class = model.__class__.__qualname__
        if expected_class != actual_class:
            raise ResumeError(
                f"Model class mismatch: checkpoint={expected_class!r}, "
                f"current={actual_class!r}"
            )

    if strict_trainability:
        expected_trainable = tuple(contract.get("trainable_parameter_names", ()))
        expected_frozen = tuple(contract.get("frozen_parameter_names", ()))

        actual_trainable = tuple(
            name for name, p in model.named_parameters()
            if p.requires_grad
        )
        actual_frozen = tuple(
            name for name, p in model.named_parameters()
            if not p.requires_grad
        )

        if actual_trainable != expected_trainable:
            raise ResumeError(
                "Trainable parameter-name contract changed between checkpoint "
                "and current model."
            )
        if actual_frozen != expected_frozen:
            raise ResumeError(
                "Frozen parameter-name contract changed between checkpoint "
                "and current model."
            )


def _validate_optimizer_topology(
    optimizer: Optimizer,
    optimizer_state: Mapping[str, Any],
) -> None:
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list):
        raise ResumeError("optimizer_state.param_groups must be a list")

    current_groups = optimizer.param_groups
    if len(saved_groups) != len(current_groups):
        raise ResumeError(
            "Optimizer parameter-group count mismatch: "
            f"checkpoint={len(saved_groups)}, current={len(current_groups)}"
        )

    for i, (saved, current) in enumerate(zip(saved_groups, current_groups)):
        saved_params = saved.get("params")
        current_params = current.get("params")
        if not isinstance(saved_params, list) or not isinstance(current_params, list):
            raise ResumeError(f"Optimizer group {i} has invalid params field")
        if len(saved_params) != len(current_params):
            raise ResumeError(
                f"Optimizer group {i} parameter count mismatch: "
                f"checkpoint={len(saved_params)}, current={len(current_params)}"
            )


def _validate_named_state_targets(
    saved_states: Mapping[str, Any],
    current_objects: Mapping[str, Any] | None,
    *,
    strict: bool,
    label: str,
) -> dict[str, Any]:
    current = dict(current_objects or {})
    saved_names = set(saved_states)
    current_names = set(current)

    if strict and saved_names != current_names:
        raise ResumeError(
            f"{label} name mismatch: checkpoint={sorted(saved_names)}, "
            f"current={sorted(current_names)}"
        )

    missing = saved_names - current_names
    if missing:
        raise ResumeError(
            f"Missing {label} required by checkpoint: {sorted(missing)}"
        )

    return current


def _validate_stateful_objects(
    saved_states: Mapping[str, Any],
    objects: Mapping[str, Any] | None,
    *,
    strict: bool,
) -> dict[str, Any]:
    current = _validate_named_state_targets(
        saved_states,
        objects,
        strict=strict,
        label="stateful object",
    )
    for name, obj in current.items():
        load_state_dict = getattr(obj, "load_state_dict", None)
        if not callable(load_state_dict):
            raise ResumeError(
                f"stateful_objects[{name!r}] must expose load_state_dict()"
            )
    return current


def _validate_generators(
    saved_rng_state: Mapping[str, Any],
    generators: Mapping[str, torch.Generator] | None,
    *,
    strict: bool,
) -> dict[str, torch.Generator]:
    saved = saved_rng_state.get("generators", {})
    if not isinstance(saved, Mapping):
        raise ResumeError("rng_state.generators must be a mapping")

    current = _validate_named_state_targets(
        saved,
        generators,
        strict=strict,
        label="torch.Generator",
    )
    for name, generator in current.items():
        if not isinstance(generator, torch.Generator):
            raise ResumeError(
                f"generators[{name!r}] must be torch.Generator, "
                f"got {type(generator)!r}"
            )
    return current  # type: ignore[return-value]


def restore_rng_state(
    rng_state: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
    strict_generators: bool = True,
    strict_cuda: bool = True,
) -> tuple[str, ...]:
    """Restore Python/NumPy/PyTorch RNG state and named generators."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "generators"}
    missing = required - set(rng_state)
    if missing:
        raise ResumeError(
            f"rng_state missing required keys: {sorted(missing)}"
        )

    current_generators = _validate_generators(
        rng_state,
        generators,
        strict=strict_generators,
    )

    saved_cuda = rng_state["torch_cuda"]
    if not isinstance(saved_cuda, list):
        raise ResumeError("rng_state.torch_cuda must be a list")

    if saved_cuda:
        if not torch.cuda.is_available():
            if strict_cuda:
                raise ResumeError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
        elif len(saved_cuda) != torch.cuda.device_count():
            if strict_cuda:
                raise ResumeError(
                    "CUDA device-count mismatch for RNG restore: "
                    f"checkpoint={len(saved_cuda)}, current={torch.cuda.device_count()}"
                )

    # Restore global RNG states only after all validation has passed.
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"].cpu())

    if saved_cuda and torch.cuda.is_available():
        if len(saved_cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda])
        elif not strict_cuda:
            # Best-effort restore of overlapping devices only.
            for device_index, state in enumerate(
                saved_cuda[: torch.cuda.device_count()]
            ):
                torch.cuda.set_rng_state(state.cpu(), device=device_index)

    saved_generators = rng_state["generators"]
    for name, saved_state in saved_generators.items():
        current_generators[name].set_state(saved_state.cpu())

    return tuple(sorted(saved_generators))


def _restore_stateful_objects(
    saved_states: Mapping[str, Any],
    objects: Mapping[str, Any],
) -> tuple[str, ...]:
    for name, saved_state in saved_states.items():
        objects[name].load_state_dict(saved_state)
    return tuple(sorted(saved_states))


def resume_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_config: Mapping[str, Any] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
    stateful_objects: Mapping[str, Any] | None = None,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    verify_sha256: bool = True,
    require_sha256: bool = True,
    strict_model_state: bool = True,
    strict_model_class: bool = True,
    strict_trainability: bool = True,
    strict_generators: bool = True,
    strict_stateful_objects: bool = True,
    restore_rng: bool = True,
    strict_cuda_rng: bool = True,
) -> ResumeResult:
    """Restore a complete Task-12 checkpoint into live training objects.

    Validation is performed before mutation whenever possible. RNG is restored
    last so validation/loading work cannot consume the resumed random sequence.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)!r}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"optimizer must be torch.optim.Optimizer, got {type(optimizer)!r}"
        )

    checkpoint = Path(path).expanduser()

    payload, sha_verified = load_checkpoint_payload(
        checkpoint,
        map_location=map_location,
        verify_sha256=verify_sha256,
        require_sha256=require_sha256,
    )

    if expected_config is not None:
        normalized_expected = _normalize_jsonlike(expected_config)
        if normalized_expected != payload["config"]:
            raise ResumeError(
                "Experiment config mismatch. Refusing to resume into a different "
                "configuration."
            )

    _validate_model_contract(
        model,
        payload,
        strict_class_name=strict_model_class,
        strict_trainability=strict_trainability,
    )
    _validate_optimizer_topology(
        optimizer,
        payload["optimizer_state"],
    )

    current_stateful = _validate_stateful_objects(
        payload["stateful_states"],
        stateful_objects,
        strict=strict_stateful_objects,
    )

    if restore_rng:
        current_generators = _validate_generators(
            payload["rng_state"],
            generators,
            strict=strict_generators,
        )
    else:
        current_generators = dict(generators or {})

    # Mutating stage.
    incompatible = model.load_state_dict(
        payload["model_state"],
        strict=bool(strict_model_state),
    )
    if strict_model_state:
        # PyTorch should already raise under strict=True; this is defensive.
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ResumeError(
                "Model state mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    optimizer.load_state_dict(payload["optimizer_state"])
    restored_stateful = _restore_stateful_objects(
        payload["stateful_states"],
        current_stateful,
    )

    restored_generators: tuple[str, ...] = ()
    if restore_rng:
        restored_generators = restore_rng_state(
            payload["rng_state"],
            generators=current_generators,
            strict_generators=strict_generators,
            strict_cuda=strict_cuda_rng,
        )

    training_state = payload["training_state"]

    return ResumeResult(
        path=checkpoint.resolve(),
        epoch=int(training_state["epoch"]),
        global_step=int(training_state["global_step"]),
        config=dict(payload["config"]),
        metadata=dict(payload["metadata"]),
        sha256_verified=bool(sha_verified),
        rng_restored=bool(restore_rng),
        restored_generators=restored_generators,
        restored_stateful_objects=restored_stateful,
    )
