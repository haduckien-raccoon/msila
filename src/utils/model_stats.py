"""Parameter accounting utilities for Day-04 Adapter r×d screening.

The purpose of this module is deliberately narrow:

1. count UNIQUE registered parameters correctly;
2. report trainable/frozen parameter counts;
3. audit the ResidualAdapter2d closed-form parameter count;
4. verify that the r×d candidate grid changes capacity exactly as expected.

No training logic lives here.

Why unique parameters?
----------------------
PyTorch modules may share/tie the same ``nn.Parameter`` object through multiple
module paths. Counting every path would over-count the model. We therefore use
``named_parameters(remove_duplicate=True)`` when available and keep a defensive
identity-based fallback.

ResidualAdapter2d architecture assumed by the audit
---------------------------------------------------
For the project adapter

    C -> r -> DWConv(k×k) -> d -> C
    F' = F + gamma * ΔF

the trainable parameter count is

    P = C*r + r*k^2 + r*d + d*C + bias_terms + 1

where

    bias_terms = 2*r + d + C    if bias=True
               = 0              if bias=False

and the final ``+1`` is the learnable scalar residual gate ``gamma``.

This formula is a direct parameter-count derivation from the layer shapes; it is
not a formula quoted from a research paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from torch import nn


class ParameterCountError(RuntimeError):
    """Raised when an Adapter parameter audit fails."""


@dataclass(frozen=True, slots=True)
class ParameterStats:
    """Unique parameter statistics for one ``nn.Module``."""

    total_params: int
    trainable_params: int
    frozen_params: int
    unique_parameter_tensors: int

    @property
    def trainable_fraction(self) -> float:
        return (
            self.trainable_params / self.total_params
            if self.total_params > 0
            else 0.0
        )

    @property
    def trainable_percent(self) -> float:
        return 100.0 * self.trainable_fraction

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "trainable_fraction": self.trainable_fraction,
            "trainable_percent": self.trainable_percent,
            "unique_parameter_tensors": self.unique_parameter_tensors,
        }


def _unique_named_parameters(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """Return each registered ``nn.Parameter`` object exactly once."""

    if not isinstance(module, nn.Module):
        raise TypeError(
            f"module must be torch.nn.Module, got {type(module).__name__}"
        )

    try:
        # Current PyTorch API (2026): duplicate Parameter objects are removed
        # when remove_duplicate=True.
        return list(
            module.named_parameters(
                recurse=True,
                remove_duplicate=True,
            )
        )
    except TypeError:
        # Defensive compatibility fallback for older PyTorch releases.
        unique: list[tuple[str, nn.Parameter]] = []
        seen: set[int] = set()
        for name, parameter in module.named_parameters(recurse=True):
            key = id(parameter)
            if key not in seen:
                seen.add(key)
                unique.append((name, parameter))
        return unique


def parameter_stats(module: nn.Module) -> ParameterStats:
    """Count total/trainable/frozen UNIQUE model parameters.

    Buffers, activations, gradients, optimizer states and cached features are
    intentionally excluded because they are not ``nn.Parameter`` objects.
    """

    named = _unique_named_parameters(module)

    total = sum(int(p.numel()) for _, p in named)
    trainable = sum(
        int(p.numel())
        for _, p in named
        if bool(p.requires_grad)
    )
    frozen = total - trainable

    if min(total, trainable, frozen) < 0:
        raise ParameterCountError("Negative parameter count is impossible.")
    if total != trainable + frozen:
        raise ParameterCountError("Invariant failed: total != trainable + frozen.")

    return ParameterStats(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        unique_parameter_tensors=len(named),
    )


def count_parameters(
    module: nn.Module,
    *,
    trainable_only: bool = False,
) -> int:
    """Return one scalar parameter count."""

    stats = parameter_stats(module)
    return (
        stats.trainable_params
        if trainable_only
        else stats.total_params
    )


def expected_residual_adapter_params(
    *,
    in_dim: int,
    bottleneck_dim: int,
    projection_dim: int,
    kernel_size: int = 3,
    bias: bool = True,
) -> int:
    """Closed-form parameter count for the project ResidualAdapter2d.

    Architecture:
        C -> r -> DWConv(k×k) -> d -> C, plus scalar gamma.

    Weight terms:
        down_proj : C*r
        DWConv    : r*k^2
        mid_proj  : r*d
        out_proj  : d*C

    Bias terms when ``bias=True``:
        down_proj : r
        DWConv    : r
        mid_proj  : d
        out_proj  : C

    Gate:
        gamma     : 1
    """

    c = _positive_int("in_dim", in_dim)
    r = _positive_int("bottleneck_dim", bottleneck_dim)
    d = _positive_int("projection_dim", projection_dim)
    k = _positive_int("kernel_size", kernel_size)

    if k % 2 == 0:
        raise ValueError(
            "kernel_size must be odd for the current same-H×W Adapter contract"
        )
    if not isinstance(bias, bool):
        raise TypeError(f"bias must be bool, got {type(bias).__name__}")

    weights = c * r + r * (k * k) + r * d + d * c
    biases = (2 * r + d + c) if bias else 0
    gamma = 1
    return int(weights + biases + gamma)


def residual_adapter_breakdown(
    *,
    in_dim: int,
    bottleneck_dim: int,
    projection_dim: int,
    kernel_size: int = 3,
    bias: bool = True,
) -> dict[str, int]:
    """Return the analytical contribution of each Adapter component."""

    c = _positive_int("in_dim", in_dim)
    r = _positive_int("bottleneck_dim", bottleneck_dim)
    d = _positive_int("projection_dim", projection_dim)
    k = _positive_int("kernel_size", kernel_size)

    if not isinstance(bias, bool):
        raise TypeError(f"bias must be bool, got {type(bias).__name__}")

    out = {
        "down_proj": c * r + (r if bias else 0),
        "dwconv": r * (k * k) + (r if bias else 0),
        "mid_proj": r * d + (d if bias else 0),
        "out_proj": d * c + (c if bias else 0),
        "gamma": 1,
    }
    out["total"] = sum(out.values())
    return out


def adapter_parameter_record(
    adapter: nn.Module,
    *,
    run_name: str | None = None,
    require_all_trainable: bool = True,
) -> dict[str, Any]:
    """Audit one ResidualAdapter-like module and return a logging record.

    Required public attributes:
        in_dim, bottleneck_dim, projection_dim, kernel_size, bias

    The utility intentionally uses attributes instead of importing the concrete
    Adapter class, so it does not create a circular dependency.
    """

    required = (
        "in_dim",
        "bottleneck_dim",
        "projection_dim",
        "kernel_size",
        "bias",
    )
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise ParameterCountError(
            f"Adapter is missing required attributes: {missing}"
        )

    c = int(getattr(adapter, "in_dim"))
    r = int(getattr(adapter, "bottleneck_dim"))
    d = int(getattr(adapter, "projection_dim"))
    k = int(getattr(adapter, "kernel_size"))
    bias = bool(getattr(adapter, "bias"))

    expected = expected_residual_adapter_params(
        in_dim=c,
        bottleneck_dim=r,
        projection_dim=d,
        kernel_size=k,
        bias=bias,
    )
    stats = parameter_stats(adapter)

    if stats.total_params != expected:
        raise ParameterCountError(
            "ResidualAdapter parameter-count mismatch: "
            f"actual_total={stats.total_params}, expected={expected}, "
            f"C={c}, r={r}, d={d}, k={k}, bias={bias}"
        )

    if require_all_trainable and stats.trainable_params != expected:
        raise ParameterCountError(
            "Day-04 Adapter is not fully trainable: "
            f"trainable={stats.trainable_params}, total={stats.total_params}. "
            "If this is intentional, call with require_all_trainable=False."
        )

    expected_name = f"adapter_r{r}_d{d}"
    if run_name is None:
        run_name = expected_name
    elif run_name != expected_name:
        raise ParameterCountError(
            f"run_name={run_name!r} does not match deterministic "
            f"name {expected_name!r}."
        )

    return {
        "run_name": run_name,
        "in_dim": c,
        "bottleneck_dim": r,
        "projection_dim": d,
        "kernel_size": k,
        "bias": bias,
        "total_params": stats.total_params,
        "trainable_params": stats.trainable_params,
        "frozen_params": stats.frozen_params,
        "expected_params": expected,
        "audit_pass": True,
    }


def validate_rd_screen_growth(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate exact parameter growth across an r×d screen.

    Checks:
    - no duplicate (r,d);
    - actual count equals analytical count for every record;
    - for fixed d, increasing r gives the exact expected positive delta;
    - for fixed r, increasing d gives the exact expected positive delta.

    Returns normalized records sorted by ``(r,d)`` for CSV/YAML logging.
    """

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for raw in records:
        c = _positive_int("in_dim", int(raw["in_dim"]))
        r = _positive_int("bottleneck_dim", int(raw["bottleneck_dim"]))
        d = _positive_int("projection_dim", int(raw["projection_dim"]))
        k = _positive_int("kernel_size", int(raw.get("kernel_size", 3)))
        bias = bool(raw.get("bias", True))
        actual = int(raw["trainable_params"])

        key = (r, d)
        if key in seen:
            raise ParameterCountError(f"Duplicate Adapter candidate: {key}")
        seen.add(key)

        expected = expected_residual_adapter_params(
            in_dim=c,
            bottleneck_dim=r,
            projection_dim=d,
            kernel_size=k,
            bias=bias,
        )
        if actual != expected:
            raise ParameterCountError(
                f"Candidate r={r}, d={d}: trainable_params={actual}, "
                f"expected={expected}."
            )

        normalized.append(
            {
                **dict(raw),
                "in_dim": c,
                "bottleneck_dim": r,
                "projection_dim": d,
                "kernel_size": k,
                "bias": bias,
                "trainable_params": actual,
                "expected_params": expected,
            }
        )

    normalized.sort(
        key=lambda x: (
            int(x["bottleneck_dim"]),
            int(x["projection_dim"]),
        )
    )

    # A scientifically controlled r×d grid must keep these structural fields
    # fixed; otherwise parameter deltas cannot be attributed only to r,d.
    fixed_signatures = {
        (
            int(row["in_dim"]),
            int(row["kernel_size"]),
            bool(row["bias"]),
        )
        for row in normalized
    }
    if len(fixed_signatures) > 1:
        raise ParameterCountError(
            "Screen mixes different in_dim/kernel_size/bias values. "
            "Only r and d may vary."
        )

    if not normalized:
        return tuple()

    c, k, bias = next(iter(fixed_signatures))

    # Fixed d: exact slope with respect to r.
    by_d: dict[int, list[dict[str, Any]]] = {}
    for row in normalized:
        by_d.setdefault(int(row["projection_dim"]), []).append(row)

    for d, rows in by_d.items():
        rows.sort(key=lambda x: int(x["bottleneck_dim"]))
        for left, right in zip(rows, rows[1:]):
            r1 = int(left["bottleneck_dim"])
            r2 = int(right["bottleneck_dim"])
            p1 = int(left["trainable_params"])
            p2 = int(right["trainable_params"])
            expected_delta = (r2 - r1) * (
                c + k * k + d + (2 if bias else 0)
            )
            if p2 - p1 != expected_delta or p2 <= p1:
                raise ParameterCountError(
                    "Unexpected parameter growth with r: "
                    f"d={d}, r:{r1}->{r2}, actual_delta={p2-p1}, "
                    f"expected_delta={expected_delta}."
                )

    # Fixed r: exact slope with respect to d.
    by_r: dict[int, list[dict[str, Any]]] = {}
    for row in normalized:
        by_r.setdefault(int(row["bottleneck_dim"]), []).append(row)

    for r, rows in by_r.items():
        rows.sort(key=lambda x: int(x["projection_dim"]))
        for left, right in zip(rows, rows[1:]):
            d1 = int(left["projection_dim"])
            d2 = int(right["projection_dim"])
            p1 = int(left["trainable_params"])
            p2 = int(right["trainable_params"])
            expected_delta = (d2 - d1) * (
                r + c + (1 if bias else 0)
            )
            if p2 - p1 != expected_delta or p2 <= p1:
                raise ParameterCountError(
                    "Unexpected parameter growth with d: "
                    f"r={r}, d:{d1}->{d2}, actual_delta={p2-p1}, "
                    f"expected_delta={expected_delta}."
                )

    return tuple(normalized)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value
