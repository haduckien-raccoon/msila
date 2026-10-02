"""Tests for models/adapter_factory.py.

Assumes the project already contains:
    models/residual_adapter.py

Run:
    pytest -q tests/test_adapter_factory.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adapter_factory import (  # noqa: E402
    AdapterCandidate,
    AdapterFactoryConfig,
    ResidualAdapterFactory,
)


GRID = [
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

EXPECTED_NAMES = [f"adapter_r{r}_d{d}" for r, d in GRID]


@pytest.fixture
def factory() -> ResidualAdapterFactory:
    return ResidualAdapterFactory(
        AdapterFactoryConfig(
            in_dim=384,
            kernel_size=3,
            gamma_init=0.0,
            bias=True,
        )
    )


def test_builds_exact_nine_coarse_grid_candidates(factory) -> None:
    builds = factory.build_many(
        AdapterCandidate(r, d) for r, d in GRID
    )

    assert len(builds) == 9
    assert [b.run_name for b in builds] == EXPECTED_NAMES


@pytest.mark.parametrize("r,d", GRID)
def test_each_candidate_has_exact_requested_dimensions(factory, r, d) -> None:
    build = factory.build_rd(r=r, d=d)
    model = build.model

    assert model.bottleneck_dim == r
    assert model.projection_dim == d

    assert model.down_proj.in_channels == 384
    assert model.down_proj.out_channels == r

    assert model.dwconv.in_channels == r
    assert model.dwconv.out_channels == r
    assert model.dwconv.groups == r

    assert model.mid_proj.in_channels == r
    assert model.mid_proj.out_channels == d

    assert model.out_proj.in_channels == d
    assert model.out_proj.out_channels == 384


def test_run_name_is_deterministic(factory) -> None:
    a = factory.build_rd(r=64, d=256)
    b = factory.build({"r": 64, "d": 256})
    c = factory.build(
        {
            "bottleneck_dim": 64,
            "projection_dim": 256,
            "run_name": "adapter_r64_d256",
        }
    )

    assert a.run_name == b.run_name == c.run_name == "adapter_r64_d256"


def test_factory_keeps_all_non_rd_adapter_settings_fixed(factory) -> None:
    builds = factory.build_many(
        [
            {"r": 32, "d": 128},
            {"r": 64, "d": 256},
            {"r": 128, "d": 384},
        ]
    )

    for build in builds:
        model = build.model
        assert model.in_dim == 384
        assert model.kernel_size == 3
        assert model.gamma_init == 0.0
        assert model.bias is True
        assert build.fixed_config == factory.fixed_config


def test_candidate_mapping_cannot_override_fixed_settings(factory) -> None:
    with pytest.raises(KeyError):
        factory.build(
            {
                "r": 64,
                "d": 256,
                "kernel_size": 5,
            }
        )


def test_forward_shape_for_every_candidate(factory) -> None:
    torch.manual_seed(42)
    x = torch.randn(1, 384, 5, 7)

    for r, d in GRID:
        model = factory.build_rd(r=r, d=d).model
        with torch.no_grad():
            y = model(x)

        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()


def test_parameter_count_is_audited_for_every_candidate(factory) -> None:
    counts = {}

    for r, d in GRID:
        build = factory.build_rd(r=r, d=d)
        model = build.model

        assert build.trainable_params == model.expected_parameter_count()
        assert build.record()["trainable_params"] == model.expected_parameter_count()
        counts[(r, d)] = build.trainable_params

    # The screen really changes capacity; it is not nine aliases of one model.
    assert len(set(counts.values())) == 9


def test_factory_does_not_hard_code_the_day04_grid(factory) -> None:
    build = factory.build_rd(r=17, d=29)

    assert build.run_name == "adapter_r17_d29"
    assert build.model.bottleneck_dim == 17
    assert build.model.projection_dim == 29


def test_candidate_aliases_must_not_conflict(factory) -> None:
    with pytest.raises(ValueError):
        factory.build(
            {
                "r": 64,
                "bottleneck_dim": 32,
                "d": 256,
                "projection_dim": 256,
            }
        )


def test_supplied_run_name_must_match_candidate(factory) -> None:
    with pytest.raises(ValueError):
        factory.build(
            {
                "r": 64,
                "d": 256,
                "run_name": "adapter_r32_d128",
            }
        )


def test_duplicate_candidates_are_rejected(factory) -> None:
    with pytest.raises(ValueError, match="Duplicate candidate"):
        factory.build_many(
            [
                {"r": 64, "d": 256},
                {"bottleneck_dim": 64, "projection_dim": 256},
            ]
        )


def test_build_metadata_contains_only_resolved_values(factory) -> None:
    record = factory.build_rd(r=64, d=256).record()

    assert record["run_name"] == "adapter_r64_d256"
    assert record["bottleneck_dim"] == 64
    assert record["projection_dim"] == 256
    assert record["in_dim"] == 384
    assert record["kernel_size"] == 3
    assert record["gamma_init"] == 0.0
    assert record["bias"] is True
    assert isinstance(record["trainable_params"], int)
    assert record["trainable_params"] > 0


@pytest.mark.parametrize(
    "mapping,error",
    [
        ({"r": 0, "d": 256}, ValueError),
        ({"r": 64, "d": 0}, ValueError),
        ({"r": True, "d": 256}, TypeError),
        ({"r": 64}, KeyError),
        ({"d": 256}, KeyError),
        ({"r": 64, "d": 256, "dropout": 0.1}, KeyError),
    ],
)
def test_invalid_candidate_config_is_rejected(factory, mapping, error) -> None:
    with pytest.raises(error):
        factory.build(mapping)


def test_fixed_config_is_strict() -> None:
    with pytest.raises(KeyError):
        AdapterFactoryConfig.from_mapping(
            {
                "in_dim": 384,
                "kernel_size": 3,
                "gamma_init": 0.0,
                "bias": True,
                "learning_rate": 1e-4,
            }
        )


def test_fixed_config_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        AdapterFactoryConfig(in_dim=384, kernel_size=4)
