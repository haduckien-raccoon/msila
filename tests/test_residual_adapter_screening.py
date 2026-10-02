"""Unit tests for residual_adapter.py and the locked Day-04 r×d grid.

Run from repository root:
    pytest -q tests/test_residual_adapter_screening.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.residual_adapter import (  # noqa: E402
    ResidualAdapter2d,
    ResidualAdapterConfig,
    make_adapter_run_name,
)


GRID_PATH = ROOT / "configs" / "day04_adapter_grid.yaml"

EXPECTED_PAIRS = [
    (32, 128),
    (32, 256),
    (32, 384),
    (64, 128),
    (64, 256),
    (64, 384),
    (128, 128),
    (128, 256),
    (128, 384),
]
EXPECTED_NAMES = [f"adapter_r{r}_d{d}" for r, d in EXPECTED_PAIRS]


def load_grid() -> dict:
    with GRID_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merged_candidate_config(grid: dict, candidate: dict) -> dict:
    return {
        **grid["adapter_defaults"],
        "bottleneck_dim": candidate["bottleneck_dim"],
        "projection_dim": candidate["projection_dim"],
    }


def test_locked_grid_has_exactly_nine_candidates() -> None:
    grid = load_grid()
    candidates = grid["candidates"]

    assert grid["expected_num_candidates"] == 9
    assert len(candidates) == 9

    pairs = [
        (c["bottleneck_dim"], c["projection_dim"])
        for c in candidates
    ]
    names = [c["run_name"] for c in candidates]

    assert pairs == EXPECTED_PAIRS
    assert names == EXPECTED_NAMES
    assert len(set(pairs)) == 9
    assert len(set(names)) == 9


def test_grid_run_names_are_deterministic() -> None:
    grid = load_grid()
    for candidate in grid["candidates"]:
        expected = make_adapter_run_name(
            candidate["bottleneck_dim"],
            candidate["projection_dim"],
        )
        assert candidate["run_name"] == expected


@pytest.mark.parametrize("r,d", EXPECTED_PAIRS)
def test_every_locked_candidate_forwards(r: int, d: int) -> None:
    torch.manual_seed(0)
    grid = load_grid()
    defaults = grid["adapter_defaults"]

    model = ResidualAdapter2d(
        in_dim=defaults["in_dim"],
        bottleneck_dim=r,
        projection_dim=d,
        kernel_size=defaults["kernel_size"],
        gamma_init=defaults["gamma_init"],
        bias=defaults["bias"],
    )
    x = torch.randn(1, defaults["in_dim"], 5, 7)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert torch.isfinite(y).all()


def test_all_yaml_candidates_build_through_strict_config() -> None:
    grid = load_grid()

    for candidate in grid["candidates"]:
        cfg = ResidualAdapterConfig.from_mapping(
            merged_candidate_config(grid, candidate)
        )
        model = ResidualAdapter2d.from_config(cfg)

        assert model.r == candidate["bottleneck_dim"]
        assert model.d == candidate["projection_dim"]


def test_r_and_d_reach_the_intended_layers() -> None:
    model = ResidualAdapter2d(
        in_dim=384,
        bottleneck_dim=64,
        projection_dim=256,
    )

    assert model.down_proj.in_channels == 384
    assert model.down_proj.out_channels == 64

    assert model.dwconv.in_channels == 64
    assert model.dwconv.out_channels == 64
    assert model.dwconv.groups == 64

    assert model.mid_proj.in_channels == 64
    assert model.mid_proj.out_channels == 256

    assert model.out_proj.in_channels == 256
    assert model.out_proj.out_channels == 384


def test_gamma_zero_is_exact_identity() -> None:
    torch.manual_seed(1)
    model = ResidualAdapter2d(
        in_dim=96,
        bottleneck_dim=13,
        projection_dim=29,
        gamma_init=0.0,
    )
    x = torch.randn(2, 96, 9, 7)

    with torch.no_grad():
        y = model(x)

    torch.testing.assert_close(y, x, rtol=0.0, atol=0.0)


def test_gamma_gets_gradient_at_identity_initialization() -> None:
    torch.manual_seed(2)
    model = ResidualAdapter2d(
        in_dim=32,
        bottleneck_dim=8,
        projection_dim=16,
        gamma_init=0.0,
    )
    x = torch.randn(2, 32, 5, 5)

    model(x).square().mean().backward()

    assert model.gamma.grad is not None
    assert torch.isfinite(model.gamma.grad).all()


def test_branch_gradients_are_finite_when_gate_is_nonzero() -> None:
    torch.manual_seed(3)
    model = ResidualAdapter2d(
        in_dim=48,
        bottleneck_dim=12,
        projection_dim=20,
        gamma_init=0.1,
    )
    x = torch.randn(2, 48, 6, 6, requires_grad=True)

    model(x).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"


def test_parameter_count_matches_closed_form() -> None:
    for bias in (False, True):
        model = ResidualAdapter2d(
            in_dim=384,
            bottleneck_dim=64,
            projection_dim=256,
            kernel_size=3,
            bias=bias,
        )
        assert model.num_trainable_parameters == model.expected_parameter_count()


def test_non_power_of_two_dimensions_are_supported() -> None:
    model = ResidualAdapter2d(
        in_dim=37,
        bottleneck_dim=7,
        projection_dim=11,
    )
    x = torch.randn(1, 37, 4, 6)
    assert model(x).shape == x.shape


@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        ({"in_dim": 0, "bottleneck_dim": 8, "projection_dim": 16}, ValueError),
        ({"in_dim": 32, "bottleneck_dim": 0, "projection_dim": 16}, ValueError),
        ({"in_dim": 32, "bottleneck_dim": 8, "projection_dim": 0}, ValueError),
        (
            {
                "in_dim": 32,
                "bottleneck_dim": 8,
                "projection_dim": 16,
                "kernel_size": 4,
            },
            ValueError,
        ),
        (
            {
                "in_dim": 32,
                "bottleneck_dim": 8,
                "projection_dim": 16,
                "gamma_init": float("inf"),
            },
            ValueError,
        ),
    ],
)
def test_invalid_constructor_config_is_rejected(kwargs, error_type) -> None:
    with pytest.raises(error_type):
        ResidualAdapter2d(**kwargs)


def test_stale_legacy_keys_are_rejected() -> None:
    with pytest.raises(KeyError):
        ResidualAdapterConfig.from_mapping(
            {
                "in_dim": 384,
                "bottleneck_dim": 64,
                "projection_dim": 256,
                "reduction": 4,
            }
        )
