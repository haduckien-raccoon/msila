"""Day-04 integration tests for every locked Adapter candidate.

This file is intended to REPLACE ``tests/test_residual_adapter_screening.py``.
It keeps the previous screening coverage and adds a strict 9/9 gradient audit.

Run:
    pytest -q tests/test_adapter_candidates.py

Show the gradient report:
    pytest -q -s tests/test_adapter_candidates.py

Expected scientific behavior of the zero-initialized residual gate
------------------------------------------------------------------
The Adapter is

    F_out = F + gamma * R(F; theta)

At gamma = 0:

    dL/dtheta = gamma * (...) = 0

so the residual-branch parameters are expected to receive zero-valued
gradients on the first backward pass. This is NOT a failure.

However,

    dL/dgamma = <dL/dF_out, R(F; theta)>

can be non-zero, so gamma can move away from zero. After gamma != 0,
all residual-branch parameters must receive finite, non-zero gradients.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Iterable

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adapter_factory import (  # noqa: E402
    AdapterCandidate,
    AdapterFactoryConfig,
    ResidualAdapterFactory,
)
from src.models.residual_adapter import (  # noqa: E402
    ResidualAdapter2d,
    ResidualAdapterConfig,
    make_adapter_run_name,
)


GRID_PATH = ROOT / "configs" / "day04_adapter_grid.yaml"

EXPECTED_PAIRS = (
    (32, 128),
    (32, 256),
    (32, 384),
    (64, 128),
    (64, 256),
    (64, 384),
    (128, 128),
    (128, 256),
    (128, 384),
)
EXPECTED_NAMES = tuple(
    f"adapter_r{r}_d{d}"
    for r, d in EXPECTED_PAIRS
)

# Small spatial shape keeps this unit test fast while still exercising DWConv.
TEST_HW = (4, 5)


def load_grid() -> dict:
    if not GRID_PATH.is_file():
        raise FileNotFoundError(
            f"Missing locked Day-04 grid: {GRID_PATH}"
        )
    with GRID_PATH.open("r", encoding="utf-8") as f:
        grid = yaml.safe_load(f)
    assert isinstance(grid, dict)
    return grid


def build_factory() -> ResidualAdapterFactory:
    grid = load_grid()
    return ResidualAdapterFactory(
        AdapterFactoryConfig.from_mapping(grid["adapter_defaults"])
    )


def build_candidate(r: int, d: int):
    return build_factory().build_rd(r=r, d=d)


def explicit_expected_parameter_count(
    *,
    c: int,
    r: int,
    d: int,
    k: int,
    bias: bool,
) -> int:
    """Independent closed-form audit of the project's Adapter parameter count.

    C -> r -> DWConv(k×k) -> d -> C plus scalar gamma.

    Weights:
        C*r + r*k^2 + r*d + d*C

    Biases when enabled:
        r + r + d + C

    Gate:
        +1 for gamma
    """

    weights = c * r + r * (k * k) + r * d + d * c
    biases = (2 * r + d + c) if bias else 0
    return weights + biases + 1


def branch_named_parameters(
    model: ResidualAdapter2d,
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """Parameters belonging to R(F), excluding the scalar residual gate gamma."""

    prefixes = ("down_proj.", "dwconv.", "mid_proj.", "out_proj.")
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            yield name, parameter


def grad_norm(parameter: torch.nn.Parameter) -> float:
    assert parameter.grad is not None
    return float(parameter.grad.detach().norm().item())


# ---------------------------------------------------------------------------
# [PASS] grid đúng 9 candidate
# ---------------------------------------------------------------------------

def test_locked_grid_is_exactly_the_expected_3x3_screen() -> None:
    grid = load_grid()
    candidates = grid["candidates"]

    assert grid["expected_num_candidates"] == 9
    assert len(candidates) == 9

    pairs = tuple(
        (int(c["bottleneck_dim"]), int(c["projection_dim"]))
        for c in candidates
    )
    names = tuple(str(c["run_name"]) for c in candidates)

    assert pairs == EXPECTED_PAIRS
    assert names == EXPECTED_NAMES
    assert len(set(pairs)) == 9
    assert len(set(names)) == 9

    assert {r for r, _ in pairs} == {32, 64, 128}
    assert {d for _, d in pairs} == {128, 256, 384}


# ---------------------------------------------------------------------------
# [PASS] run_name deterministic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_run_name_is_deterministic(r: int, d: int) -> None:
    expected = f"adapter_r{r}_d{d}"

    assert make_adapter_run_name(r, d) == expected
    assert AdapterCandidate(r, d).run_name == expected
    assert build_candidate(r, d).run_name == expected


# ---------------------------------------------------------------------------
# [PASS] 9/9 forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_all_9_candidates_forward(r: int, d: int) -> None:
    torch.manual_seed(1000 + r + d)
    build = build_candidate(r, d)
    model = build.model

    x = torch.randn(
        2,
        model.in_dim,
        *TEST_HW,
        dtype=torch.float32,
    )

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# [PASS] r,d map đúng layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_r_d_map_to_exact_intended_layers(r: int, d: int) -> None:
    model = build_candidate(r, d).model
    c = model.in_dim

    # C -> r
    assert model.down_proj.in_channels == c
    assert model.down_proj.out_channels == r

    # r -> r depthwise spatial mixing
    assert model.dwconv.in_channels == r
    assert model.dwconv.out_channels == r
    assert model.dwconv.groups == r

    # r -> d
    assert model.mid_proj.in_channels == r
    assert model.mid_proj.out_channels == d

    # d -> C
    assert model.out_proj.in_channels == d
    assert model.out_proj.out_channels == c

    assert model.r == r
    assert model.d == d


# ---------------------------------------------------------------------------
# [PASS] gamma=0 identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_gamma_zero_is_exact_identity_for_all_candidates(
    r: int,
    d: int,
) -> None:
    torch.manual_seed(2000 + r + d)
    model = build_candidate(r, d).model

    assert model.gamma.item() == 0.0

    x = torch.randn(2, model.in_dim, *TEST_HW)

    with torch.no_grad():
        y = model(x)

    torch.testing.assert_close(
        y,
        x,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# [PASS] 9/9 gamma gradient ở gamma=0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_gamma_gets_finite_nonzero_gradient_at_zero_for_all_candidates(
    r: int,
    d: int,
) -> None:
    """Probe gamma at the exact identity initialization.

    The loss is deliberately a gradient-probe objective, not the project loss:

        loss = mean(F_out * stopgrad(R(F)))

    At gamma=0 this gives

        dloss/dgamma = mean(R(F)^2) > 0

    for a non-zero residual branch, making this unit test deterministic and
    avoiding accidental cancellation from an arbitrary random target.
    """

    torch.manual_seed(3000 + r + d)
    model = build_candidate(r, d).model
    model.zero_grad(set_to_none=True)

    x = torch.randn(2, model.in_dim, *TEST_HW)

    # Probe target is detached so it does not itself send gradients into R(F).
    residual_probe = model.residual(x).detach()
    assert torch.isfinite(residual_probe).all()
    assert residual_probe.abs().max().item() > 0.0

    y = model(x)
    loss = (y * residual_probe).mean()
    loss.backward()

    assert model.gamma.grad is not None
    assert torch.isfinite(model.gamma.grad).all()
    assert abs(float(model.gamma.grad.item())) > 0.0

    # Scientifically expected at gamma=0:
    # branch gradients exist in the graph but evaluate to exactly zero.
    for name, parameter in branch_named_parameters(model):
        assert parameter.grad is not None, f"{r=}, {d=}: missing {name}.grad"
        assert torch.isfinite(parameter.grad).all(), (
            f"{r=}, {d=}: non-finite gradient in {name}"
        )
        assert torch.count_nonzero(parameter.grad).item() == 0, (
            f"{r=}, {d=}: {name} should have zero gradient at gamma=0"
        )


# ---------------------------------------------------------------------------
# [PASS] 9/9 toàn branch gradient khi gamma!=0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_every_branch_parameter_gets_finite_nonzero_gradient_when_gate_open(
    r: int,
    d: int,
) -> None:
    torch.manual_seed(4000 + r + d)
    model = build_candidate(r, d).model

    # Open the residual path without changing any architecture.
    with torch.no_grad():
        model.gamma.fill_(0.1)

    model.zero_grad(set_to_none=True)

    x = torch.randn(2, model.in_dim, *TEST_HW)
    probe = torch.randn_like(x)

    # Fixed linear probe avoids dependence on the project's training loss while
    # exercising the entire residual branch.
    y = model(x)
    loss = (y * probe).mean()
    loss.backward()

    assert model.gamma.grad is not None
    assert torch.isfinite(model.gamma.grad).all()

    branch = list(branch_named_parameters(model))
    assert branch, "No residual-branch parameters were discovered."

    for name, parameter in branch:
        assert parameter.grad is not None, f"{r=}, {d=}: missing {name}.grad"
        assert torch.isfinite(parameter.grad).all(), (
            f"{r=}, {d=}: non-finite gradient in {name}"
        )

        norm = grad_norm(parameter)
        assert norm > 0.0, (
            f"{r=}, {d=}: zero gradient norm in {name}; "
            "the candidate is not fully connected to the loss."
        )


# ---------------------------------------------------------------------------
# Human-readable grad report. Use pytest -s to display.
# This also jointly checks BOTH required gradient phases for all 9 candidates.
# ---------------------------------------------------------------------------

def test_gradient_report_all_9_candidates() -> None:
    rows: list[tuple[str, float, float]] = []

    for index, (r, d) in enumerate(EXPECTED_PAIRS):
        # Phase A: gamma=0 -> gamma must receive a finite non-zero gradient.
        torch.manual_seed(5000 + index)
        model_zero = build_candidate(r, d).model
        x0 = torch.randn(1, model_zero.in_dim, *TEST_HW)
        residual_probe = model_zero.residual(x0).detach()

        model_zero.zero_grad(set_to_none=True)
        loss0 = (model_zero(x0) * residual_probe).mean()
        loss0.backward()

        assert model_zero.gamma.grad is not None
        gamma_grad = abs(float(model_zero.gamma.grad.item()))
        assert math.isfinite(gamma_grad)
        assert gamma_grad > 0.0

        # Phase B: gamma!=0 -> every branch parameter must get non-zero grad.
        torch.manual_seed(6000 + index)
        model_open = build_candidate(r, d).model
        with torch.no_grad():
            model_open.gamma.fill_(0.1)

        x1 = torch.randn(1, model_open.in_dim, *TEST_HW)
        probe = torch.randn_like(x1)

        model_open.zero_grad(set_to_none=True)
        loss1 = (model_open(x1) * probe).mean()
        loss1.backward()

        norms = []
        for name, parameter in branch_named_parameters(model_open):
            assert parameter.grad is not None, (
                f"adapter_r{r}_d{d}: missing grad for {name}"
            )
            assert torch.isfinite(parameter.grad).all(), (
                f"adapter_r{r}_d{d}: non-finite grad for {name}"
            )
            norm = grad_norm(parameter)
            assert norm > 0.0, (
                f"adapter_r{r}_d{d}: zero grad for {name}"
            )
            norms.append(norm)

        min_branch_grad = min(norms)
        rows.append(
            (
                f"adapter_r{r}_d{d}",
                gamma_grad,
                min_branch_grad,
            )
        )

    assert len(rows) == 9

    print("\nDay-04 Adapter gradient report")
    print(
        f"{'candidate':<22} "
        f"{'|gamma.grad| @ gamma=0':>24} "
        f"{'min branch grad @ gamma=0.1':>30}"
    )
    print("-" * 82)
    for name, gamma_grad, min_branch_grad in rows:
        print(
            f"{name:<22} "
            f"{gamma_grad:>24.6e} "
            f"{min_branch_grad:>30.6e}"
        )


# ---------------------------------------------------------------------------
# [PASS] parameter count đúng
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_parameter_count_matches_independent_closed_form(
    r: int,
    d: int,
) -> None:
    model = build_candidate(r, d).model

    actual = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    expected = explicit_expected_parameter_count(
        c=model.in_dim,
        r=r,
        d=d,
        k=model.kernel_size,
        bias=model.bias,
    )

    assert actual == expected
    assert model.num_trainable_parameters == expected
    assert model.expected_parameter_count() == expected


# ---------------------------------------------------------------------------
# [PASS] invalid config bị reject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        (
            {
                "in_dim": 0,
                "bottleneck_dim": 32,
                "projection_dim": 128,
            },
            ValueError,
        ),
        (
            {
                "in_dim": 384,
                "bottleneck_dim": 0,
                "projection_dim": 128,
            },
            ValueError,
        ),
        (
            {
                "in_dim": 384,
                "bottleneck_dim": 32,
                "projection_dim": 0,
            },
            ValueError,
        ),
        (
            {
                "in_dim": 384,
                "bottleneck_dim": 32,
                "projection_dim": 128,
                "kernel_size": 4,
            },
            ValueError,
        ),
        (
            {
                "in_dim": 384,
                "bottleneck_dim": 32,
                "projection_dim": 128,
                "gamma_init": float("inf"),
            },
            ValueError,
        ),
        (
            {
                "in_dim": 384,
                "bottleneck_dim": True,
                "projection_dim": 128,
            },
            TypeError,
        ),
    ],
)
def test_invalid_adapter_config_is_rejected(
    kwargs: dict,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ResidualAdapter2d(**kwargs)


@pytest.mark.parametrize(
    "candidate",
    [
        {"r": 0, "d": 128},
        {"r": 32, "d": 0},
        {"r": -1, "d": 128},
        {"r": 32, "d": -1},
        {"r": True, "d": 128},
        {"r": 32, "d": 128, "run_name": "wrong_name"},
        {"r": 32, "d": 128, "learning_rate": 1e-4},
    ],
)
def test_invalid_factory_candidate_is_rejected(candidate: dict) -> None:
    factory = build_factory()
    with pytest.raises((TypeError, ValueError, KeyError)):
        factory.build(candidate)


# ---------------------------------------------------------------------------
# [PASS] legacy config bị reject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "legacy_key,legacy_value",
    [
        ("reduction", 4),
        ("hidden_dim", 256),
        ("bottleneck_channels", 64),
        ("in_channels", 384),
    ],
)
def test_legacy_adapter_config_keys_are_rejected(
    legacy_key: str,
    legacy_value,
) -> None:
    config = {
        "in_dim": 384,
        "bottleneck_dim": 64,
        "projection_dim": 256,
        legacy_key: legacy_value,
    }

    with pytest.raises(KeyError):
        ResidualAdapterConfig.from_mapping(config)


def test_conflicting_aliases_are_rejected() -> None:
    with pytest.raises(ValueError):
        AdapterCandidate.from_mapping(
            {
                "bottleneck_dim": 64,
                "r": 32,
                "projection_dim": 256,
                "d": 256,
            }
        )


# ---------------------------------------------------------------------------
# Additional regression: strict config + factory can build every YAML candidate.
# This preserves the useful integration coverage from the previous test file.
# ---------------------------------------------------------------------------

def test_all_yaml_candidates_build_through_factory_and_strict_config() -> None:
    grid = load_grid()
    factory = build_factory()

    for raw in grid["candidates"]:
        r = int(raw["bottleneck_dim"])
        d = int(raw["projection_dim"])

        strict_cfg = ResidualAdapterConfig.from_mapping(
            {
                **grid["adapter_defaults"],
                "bottleneck_dim": r,
                "projection_dim": d,
            }
        )
        direct = ResidualAdapter2d.from_config(strict_cfg)
        built = factory.build(raw)

        assert direct.r == built.model.r == r
        assert direct.d == built.model.d == d
        assert built.run_name == raw["run_name"]
