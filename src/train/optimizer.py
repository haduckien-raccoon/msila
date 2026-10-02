"""
Day-3 Task 9: strict optimizer setup for MS-ILA.

Scope
-----
Build an optimizer for exactly the trainable Day-3 modules:

    Adapter + Projection + Fusion + Decoder

and enforce that forbidden/frozen modules (especially DINOv3) are not part of
the optimizer.

The module deliberately does NOT implement a trainer, scheduler, checkpoint,
resume logic, visualization, or loss.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import torch
from torch import nn
from torch.optim import Optimizer


OptimizerName = Literal["adamw", "adam", "sgd"]

class OptimizerContractError(RuntimeError):
    """Raised when the Day-3 optimizer contract is violated."""


@dataclass(frozen=True)
class ParameterGroupSummary:
    name: str
    parameter_tensors: int
    parameter_elements: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizerSetupReport:
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    trainable_parameter_tensors: int
    trainable_parameter_elements: int
    frozen_parameter_tensors: int
    frozen_parameter_elements: int
    trainable_groups: tuple[ParameterGroupSummary, ...]
    frozen_groups: tuple[ParameterGroupSummary, ...]

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["trainable_groups"] = [g.to_dict() for g in self.trainable_groups]
        out["frozen_groups"] = [g.to_dict() for g in self.frozen_groups]
        return out


def _validate_hyperparameters(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    momentum: float,
) -> None:
    if not (learning_rate > 0.0):
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
    if len(betas) != 2:
        raise ValueError("betas must contain exactly two values.")
    beta1, beta2 = float(betas[0]), float(betas[1])
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError(f"betas must be in [0,1), got {betas}.")
    if not (eps > 0.0):
        raise ValueError(f"eps must be > 0, got {eps}.")
    if momentum < 0.0:
        raise ValueError(f"momentum must be >= 0, got {momentum}.")


def _module_parameters(module: nn.Module) -> list[nn.Parameter]:
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(module)!r}.")
    return list(module.parameters())


def _summarize_group(name: str, parameters: list[nn.Parameter]) -> ParameterGroupSummary:
    return ParameterGroupSummary(
        name=name,
        parameter_tensors=len(parameters),
        parameter_elements=sum(int(p.numel()) for p in parameters),
    )


def _collect_strict_trainable_groups(
    modules: Mapping[str, nn.Module],
) -> tuple[list[dict[str, object]], tuple[ParameterGroupSummary, ...], set[int]]:
    """
    Collect one optimizer group per logical module.

    Strict Day-3 rule:
      - each requested group must contain parameters;
      - every parameter in a requested trainable module must require gradients;
      - one Parameter object may not occur in two logical groups.
    """
    if not modules:
        raise OptimizerContractError("At least one trainable module is required.")

    optimizer_groups: list[dict[str, object]] = []
    summaries: list[ParameterGroupSummary] = []
    seen: dict[int, str] = {}

    for group_name, module in modules.items():
        name = str(group_name)
        params = _module_parameters(module)

        if not params:
            raise OptimizerContractError(
                f"Trainable group {name!r} has no parameters."
            )

        frozen_names = [
            param_name
            for param_name, p in module.named_parameters()
            if not p.requires_grad
        ]
        if frozen_names:
            raise OptimizerContractError(
                f"Trainable group {name!r} contains requires_grad=False parameters: "
                f"{frozen_names}. Day-3 expects the whole requested group to be trainable."
            )

        for p in params:
            pid = id(p)
            if pid in seen:
                raise OptimizerContractError(
                    f"Parameter sharing across optimizer groups is ambiguous: "
                    f"{name!r} overlaps with {seen[pid]!r}."
                )
            seen[pid] = name

        optimizer_groups.append({"params": params, "group_name": name})
        summaries.append(_summarize_group(name, params))

    return optimizer_groups, tuple(summaries), set(seen)


def _validate_frozen_modules(
    modules: Mapping[str, nn.Module] | None,
    trainable_ids: set[int],
) -> tuple[tuple[ParameterGroupSummary, ...], set[int]]:
    """
    Validate forbidden/frozen modules.

    In particular, DINOv3 should be supplied here when it exists in the training
    process. Every parameter must already have requires_grad=False.
    """
    if not modules:
        return (), set()

    summaries: list[ParameterGroupSummary] = []
    forbidden_ids: set[int] = set()

    for group_name, module in modules.items():
        name = str(group_name)
        params = _module_parameters(module)

        trainable_names = [
            param_name
            for param_name, p in module.named_parameters()
            if p.requires_grad
        ]
        if trainable_names:
            raise OptimizerContractError(
                f"Frozen/forbidden group {name!r} contains trainable parameters: "
                f"{trainable_names}. Freeze the module before building the optimizer."
            )

        ids = {id(p) for p in params}
        overlap = ids & trainable_ids
        if overlap:
            raise OptimizerContractError(
                f"Frozen/forbidden group {name!r} overlaps with the trainable set."
            )

        forbidden_ids |= ids
        summaries.append(_summarize_group(name, params))

    return tuple(summaries), forbidden_ids


def optimizer_parameter_ids(optimizer: Optimizer) -> set[int]:
    """Return the exact set of Parameter object IDs held by an optimizer."""
    ids: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            pid = id(p)
            if pid in ids:
                raise OptimizerContractError(
                    "The same Parameter object appears more than once in the optimizer."
                )
            ids.add(pid)
    return ids


def assert_optimizer_contract(
    optimizer: Optimizer,
    *,
    trainable_modules: Mapping[str, nn.Module],
    frozen_modules: Mapping[str, nn.Module] | None = None,
) -> None:
    """
    Hard QA gate.

    The optimizer must contain exactly all parameters from trainable_modules and
    no parameter from frozen_modules.
    """
    expected_ids: set[int] = set()
    for name, module in trainable_modules.items():
        for param_name, p in module.named_parameters():
            if not p.requires_grad:
                raise OptimizerContractError(
                    f"Expected trainable parameter {name}.{param_name} has "
                    "requires_grad=False."
                )
            expected_ids.add(id(p))

    actual_ids = optimizer_parameter_ids(optimizer)

    if actual_ids != expected_ids:
        missing = len(expected_ids - actual_ids)
        unexpected = len(actual_ids - expected_ids)
        raise OptimizerContractError(
            "Optimizer parameter set does not exactly match the requested trainable "
            f"modules: missing={missing}, unexpected={unexpected}."
        )

    if frozen_modules:
        for name, module in frozen_modules.items():
            frozen_ids = {id(p) for p in module.parameters()}
            overlap = actual_ids & frozen_ids
            if overlap:
                raise OptimizerContractError(
                    f"Forbidden module {name!r} is present in the optimizer."
                )
            bad = [
                param_name
                for param_name, p in module.named_parameters()
                if p.requires_grad
            ]
            if bad:
                raise OptimizerContractError(
                    f"Forbidden module {name!r} is not fully frozen: {bad}."
                )


def build_optimizer(
    trainable_modules: Mapping[str, nn.Module],
    *,
    frozen_modules: Mapping[str, nn.Module] | None = None,
    optimizer_name: OptimizerName = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> tuple[Optimizer, OptimizerSetupReport]:
    """
    Build a strict optimizer and return a reproducibility report.

    All logical trainable groups use the same learning rate and weight decay.
    Differential learning rates are intentionally not introduced in Day-3 QA.
    """
    _validate_hyperparameters(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        momentum=momentum,
    )

    param_groups, trainable_summary, trainable_ids = _collect_strict_trainable_groups(
        trainable_modules
    )
    frozen_summary, _ = _validate_frozen_modules(
        frozen_modules,
        trainable_ids,
    )

    common_groups: list[dict[str, object]] = []
    for group in param_groups:
        common_groups.append(
            {
                **group,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )

    name = str(optimizer_name).lower()
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            common_groups,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=tuple(float(v) for v in betas),
            eps=float(eps),
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            common_groups,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=tuple(float(v) for v in betas),
            eps=float(eps),
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            common_groups,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
        )
    else:
        raise ValueError(
            f"Unsupported optimizer_name={optimizer_name!r}; "
            "use 'adamw', 'adam', or 'sgd'."
        )

    assert_optimizer_contract(
        optimizer,
        trainable_modules=trainable_modules,
        frozen_modules=frozen_modules,
    )

    trainable_tensors = sum(g.parameter_tensors for g in trainable_summary)
    trainable_elements = sum(g.parameter_elements for g in trainable_summary)
    frozen_tensors = sum(g.parameter_tensors for g in frozen_summary)
    frozen_elements = sum(g.parameter_elements for g in frozen_summary)

    report = OptimizerSetupReport(
        optimizer_name=name,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        trainable_parameter_tensors=trainable_tensors,
        trainable_parameter_elements=trainable_elements,
        frozen_parameter_tensors=frozen_tensors,
        frozen_parameter_elements=frozen_elements,
        trainable_groups=trainable_summary,
        frozen_groups=frozen_summary,
    )

    return optimizer, report


def build_day3_msila_optimizer(
    *,
    adapters: nn.Module,
    projection: nn.Module,
    fusion: nn.Module,
    decoder: nn.Module,
    dino: nn.Module | None = None,
    optimizer_name: OptimizerName = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> tuple[Optimizer, OptimizerSetupReport]:
    """
    Project-specific Day-3 convenience wrapper.

    Optimized:
        adapters + projection + fusion + decoder

    Forbidden:
        dino (when supplied)

    `adapters` may be one adapter or an nn.ModuleDict containing all adapters.
    """
    trainable_modules = OrderedDict(
        [
            ("adapter", adapters),
            ("projection", projection),
            ("fusion", fusion),
            ("decoder", decoder),
        ]
    )

    frozen_modules = None
    if dino is not None:
        frozen_modules = OrderedDict([("dino", dino)])

    return build_optimizer(
        trainable_modules,
        frozen_modules=frozen_modules,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        momentum=momentum,
    )
