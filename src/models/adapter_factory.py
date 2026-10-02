"""Factory for reproducible ResidualAdapter2d r×d screening.

This module DOES NOT redefine the adapter architecture. It builds the existing
``models.residual_adapter.ResidualAdapter2d`` while keeping every structural
setting fixed and allowing only the two Day-04 screening variables to change:

    r = bottleneck_dim
    d = projection_dim

The adapter itself implements the project-specific branch

    F -> C -> r -> DWConv -> d -> C -> ΔF
    F' = F + gamma * ΔF

Why a factory?
--------------
A single immutable factory config prevents accidental per-candidate drift in
``in_dim``, ``kernel_size``, ``gamma_init`` or ``bias``. Candidate objects carry
only r and d, so the construction API mirrors a controlled ablation.

Important scope
---------------
This factory can lock only ADAPTER-construction settings. The training runner
must separately lock/check the DINO checkpoint/frozen state, selected blocks,
Local/Context pipeline, alignment/cache, Fusion, Decoder, loss, optimizer,
learning rate, epochs, batch size, seed, split and augmentation.

Scientific background
---------------------
The factory pattern itself is software engineering, not a paper contribution.
The architecture it instantiates is grounded in residual learning and
parameter-efficient adapters (He et al., 2016; Houlsby et al., 2019;
AdaptFormer, 2022), while the vision-specific convolutional bypass idea is
related to ConvPass (arXiv 2022; peer-reviewed ECAI 2024). DINOv3 (2025)
explicitly supports frozen-backbone downstream use with lightweight adapters.
No cited work establishes a universal optimal pair (r, d); those values must be
selected empirically on a held-out validation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from torch import nn

from .residual_adapter import ResidualAdapter2d, make_adapter_run_name


@dataclass(frozen=True, slots=True)
class AdapterCandidate:
    """The ONLY two variables allowed to change during the adapter screen."""

    bottleneck_dim: int  # r
    projection_dim: int  # d

    def __post_init__(self) -> None:
        _validate_positive_int("bottleneck_dim", self.bottleneck_dim)
        _validate_positive_int("projection_dim", self.projection_dim)

    @property
    def r(self) -> int:
        return self.bottleneck_dim

    @property
    def d(self) -> int:
        return self.projection_dim

    @property
    def run_name(self) -> str:
        """Deterministic label: ``adapter_r{r}_d{d}``."""

        return make_adapter_run_name(self.r, self.d)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AdapterCandidate":
        """Parse a strict candidate mapping.

        Accepted forms:

        ``{"bottleneck_dim": 64, "projection_dim": 256}``
        ``{"r": 64, "d": 256}``

        An optional ``run_name`` is accepted only when it exactly matches the
        deterministic name derived from r and d. Mixing aliases with conflicting
        values is rejected instead of silently choosing one.
        """

        allowed = {
            "bottleneck_dim",
            "projection_dim",
            "r",
            "d",
            "run_name",
        }
        unknown = set(config) - allowed
        if unknown:
            raise KeyError(
                f"Unknown candidate keys: {sorted(unknown)}. "
                "Only r/d (or bottleneck_dim/projection_dim) may vary."
            )

        r = _resolve_alias(
            config,
            canonical="bottleneck_dim",
            alias="r",
        )
        d = _resolve_alias(
            config,
            canonical="projection_dim",
            alias="d",
        )

        candidate = cls(bottleneck_dim=r, projection_dim=d)

        supplied_name = config.get("run_name")
        if supplied_name is not None and supplied_name != candidate.run_name:
            raise ValueError(
                f"run_name={supplied_name!r} does not match deterministic "
                f"name {candidate.run_name!r}."
            )

        return candidate


@dataclass(frozen=True, slots=True)
class AdapterFactoryConfig:
    """Adapter settings that MUST remain fixed across all r×d candidates."""

    in_dim: int
    kernel_size: int = 3
    gamma_init: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int("in_dim", self.in_dim)
        _validate_positive_int("kernel_size", self.kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve H×W")
        if not math.isfinite(float(self.gamma_init)):
            raise ValueError("gamma_init must be finite")
        if not isinstance(self.bias, bool):
            raise TypeError(f"bias must be bool, got {type(self.bias).__name__}")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AdapterFactoryConfig":
        allowed = {"in_dim", "kernel_size", "gamma_init", "bias"}
        unknown = set(config) - allowed
        if unknown:
            raise KeyError(
                f"Unknown fixed adapter keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}"
            )
        if "in_dim" not in config:
            raise KeyError("Missing required fixed adapter key: 'in_dim'")
        return cls(**dict(config))

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "kernel_size": self.kernel_size,
            "gamma_init": self.gamma_init,
            "bias": self.bias,
        }


@dataclass(frozen=True, slots=True)
class AdapterBuild:
    """Auditable result returned by :class:`ResidualAdapterFactory`."""

    candidate: AdapterCandidate
    fixed_config: AdapterFactoryConfig
    model: ResidualAdapter2d = field(repr=False, compare=False)

    @property
    def run_name(self) -> str:
        return self.candidate.run_name

    @property
    def trainable_params(self) -> int:
        return self.model.num_trainable_parameters

    def record(self) -> dict[str, Any]:
        """Small metadata record suitable for YAML/CSV/JSON logging."""

        return {
            "run_name": self.run_name,
            "bottleneck_dim": self.candidate.r,
            "projection_dim": self.candidate.d,
            **self.fixed_config.as_dict(),
            "trainable_params": self.trainable_params,
        }


class ResidualAdapterFactory:
    """Build screen candidates while freezing all non-r/d adapter settings.

    Examples
    --------
    >>> fixed = AdapterFactoryConfig(in_dim=384, kernel_size=3, gamma_init=0.0)
    >>> factory = ResidualAdapterFactory(fixed)
    >>> build = factory.build_rd(r=64, d=256)
    >>> build.run_name
    'adapter_r64_d256'
    >>> build.model.bottleneck_dim, build.model.projection_dim
    (64, 256)
    """

    def __init__(
        self,
        fixed_config: AdapterFactoryConfig | Mapping[str, Any],
    ) -> None:
        self._fixed = (
            fixed_config
            if isinstance(fixed_config, AdapterFactoryConfig)
            else AdapterFactoryConfig.from_mapping(fixed_config)
            if isinstance(fixed_config, Mapping)
            else None
        )
        if self._fixed is None:
            raise TypeError(
                "fixed_config must be AdapterFactoryConfig or Mapping, "
                f"got {type(fixed_config).__name__}"
            )

    @property
    def fixed_config(self) -> AdapterFactoryConfig:
        return self._fixed

    def build(
        self,
        candidate: AdapterCandidate | Mapping[str, Any],
    ) -> AdapterBuild:
        """Instantiate exactly one candidate."""

        cand = (
            candidate
            if isinstance(candidate, AdapterCandidate)
            else AdapterCandidate.from_mapping(candidate)
            if isinstance(candidate, Mapping)
            else None
        )
        if cand is None:
            raise TypeError(
                "candidate must be AdapterCandidate or Mapping, "
                f"got {type(candidate).__name__}"
            )

        model = ResidualAdapter2d(
            in_dim=self._fixed.in_dim,
            bottleneck_dim=cand.r,
            projection_dim=cand.d,
            kernel_size=self._fixed.kernel_size,
            gamma_init=self._fixed.gamma_init,
            bias=self._fixed.bias,
        )

        # Audit construction immediately. If ResidualAdapter2d changes later,
        # the factory fails early instead of silently screening the wrong model.
        _audit_model(model=model, candidate=cand, fixed=self._fixed)

        return AdapterBuild(
            candidate=cand,
            fixed_config=self._fixed,
            model=model,
        )

    def build_rd(self, *, r: int, d: int) -> AdapterBuild:
        """CLI-friendly convenience wrapper for ``--r`` and ``--d``."""

        return self.build(AdapterCandidate(r, d))

    def build_many(
        self,
        candidates: Iterable[AdapterCandidate | Mapping[str, Any]],
    ) -> tuple[AdapterBuild, ...]:
        """Build candidates in input order and reject duplicate run names."""

        builds: list[AdapterBuild] = []
        seen: set[str] = set()

        for candidate in candidates:
            build = self.build(candidate)
            if build.run_name in seen:
                raise ValueError(f"Duplicate candidate: {build.run_name}")
            seen.add(build.run_name)
            builds.append(build)

        return tuple(builds)


# Short project alias.
AdapterFactory = ResidualAdapterFactory


def _audit_model(
    *,
    model: ResidualAdapter2d,
    candidate: AdapterCandidate,
    fixed: AdapterFactoryConfig,
) -> None:
    """Verify the created module exactly reflects factory + candidate config."""

    expected = {
        "in_dim": fixed.in_dim,
        "bottleneck_dim": candidate.r,
        "projection_dim": candidate.d,
        "kernel_size": fixed.kernel_size,
        "gamma_init": fixed.gamma_init,
        "bias": fixed.bias,
    }
    actual = {key: getattr(model, key) for key in expected}

    if actual != expected:
        raise RuntimeError(
            "ResidualAdapter2d construction drift detected: "
            f"expected={expected}, actual={actual}"
        )

    if model.num_trainable_parameters != model.expected_parameter_count():
        raise RuntimeError(
            "ResidualAdapter2d parameter-count audit failed: "
            f"actual={model.num_trainable_parameters}, "
            f"expected={model.expected_parameter_count()}"
        )


def _resolve_alias(
    config: Mapping[str, Any],
    *,
    canonical: str,
    alias: str,
) -> int:
    has_canonical = canonical in config
    has_alias = alias in config

    if not has_canonical and not has_alias:
        raise KeyError(
            f"Missing candidate value: provide '{canonical}' or '{alias}'"
        )

    if has_canonical and has_alias and config[canonical] != config[alias]:
        raise ValueError(
            f"Conflicting values for '{canonical}'={config[canonical]!r} "
            f"and '{alias}'={config[alias]!r}"
        )

    value = config[canonical] if has_canonical else config[alias]
    _validate_positive_int(canonical, value)
    return value


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
