from __future__ import annotations

import pytest
import torch
from torch import nn

from src.utils.model_stats import (
    ParameterCountError,
    adapter_parameter_record,
    count_parameters,
    expected_residual_adapter_params,
    parameter_stats,
    residual_adapter_breakdown,
    validate_rd_screen_growth,
)

from src.models.residual_adapter import ResidualAdapter2d


def test_parameter_stats_total_trainable_frozen_and_buffers():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = nn.Parameter(torch.zeros(3, 4))
            self.frozen = nn.Parameter(torch.zeros(5), requires_grad=False)
            self.register_buffer("not_a_parameter", torch.zeros(100))

    model = Tiny()
    stats = parameter_stats(model)

    assert stats.total_params == 12 + 5
    assert stats.trainable_params == 12
    assert stats.frozen_params == 5
    assert count_parameters(model) == 17
    assert count_parameters(model, trainable_only=True) == 12


def test_shared_parameter_is_counted_once():
    shared = nn.Linear(4, 4, bias=False)

    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = shared
            self.b = shared

    model = SharedModel()
    assert count_parameters(model) == 16


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("r", [32, 64, 128])
@pytest.mark.parametrize("d", [128, 256, 384])
def test_formula_matches_real_residual_adapter(bias, r, d):
    model = ResidualAdapter2d(
        in_dim=384,
        bottleneck_dim=r,
        projection_dim=d,
        kernel_size=3,
        gamma_init=0.0,
        bias=bias,
    )

    expected = expected_residual_adapter_params(
        in_dim=384,
        bottleneck_dim=r,
        projection_dim=d,
        kernel_size=3,
        bias=bias,
    )

    assert count_parameters(model) == expected
    assert count_parameters(model, trainable_only=True) == expected


def test_exact_day04_counts_for_c384_k3_bias_true():
    expected = {
        (32, 128): 66401,
        (32, 256): 119777,
        (32, 384): 173153,
        (64, 128): 83137,
        (64, 256): 140609,
        (64, 384): 198081,
        (128, 128): 116609,
        (128, 256): 182273,
        (128, 384): 247937,
    }

    for (r, d), n in expected.items():
        assert expected_residual_adapter_params(
            in_dim=384,
            bottleneck_dim=r,
            projection_dim=d,
            kernel_size=3,
            bias=True,
        ) == n


def test_breakdown_sums_to_total():
    breakdown = residual_adapter_breakdown(
        in_dim=384,
        bottleneck_dim=64,
        projection_dim=256,
        kernel_size=3,
        bias=True,
    )

    assert breakdown["total"] == 140609
    assert breakdown["total"] == sum(
        v for k, v in breakdown.items() if k != "total"
    )


def test_adapter_record_audits_run_name_and_trainability():
    model = ResidualAdapter2d(
        in_dim=384,
        bottleneck_dim=64,
        projection_dim=256,
        kernel_size=3,
        gamma_init=0.0,
        bias=True,
    )

    record = adapter_parameter_record(
        model,
        run_name="adapter_r64_d256",
    )

    assert record["trainable_params"] == 140609
    assert record["expected_params"] == 140609
    assert record["audit_pass"] is True

    with pytest.raises(ParameterCountError, match="run_name"):
        adapter_parameter_record(
            model,
            run_name="adapter_r32_d128",
        )


def test_adapter_record_detects_accidental_freeze():
    model = ResidualAdapter2d(
        in_dim=384,
        bottleneck_dim=64,
        projection_dim=256,
    )
    model.down_proj.weight.requires_grad_(False)

    with pytest.raises(ParameterCountError, match="not fully trainable"):
        adapter_parameter_record(model)


def test_full_grid_has_exact_expected_growth():
    records = []

    for r in (32, 64, 128):
        for d in (128, 256, 384):
            model = ResidualAdapter2d(
                in_dim=384,
                bottleneck_dim=r,
                projection_dim=d,
                kernel_size=3,
                gamma_init=0.0,
                bias=True,
            )
            records.append(
                adapter_parameter_record(
                    model,
                    run_name=f"adapter_r{r}_d{d}",
                )
            )

    validated = validate_rd_screen_growth(records)

    assert len(validated) == 9
    assert validated[0]["run_name"] == "adapter_r32_d128"
    assert validated[-1]["run_name"] == "adapter_r128_d384"


def test_growth_audit_rejects_non_rd_structural_drift():
    a = {
        "run_name": "adapter_r32_d128",
        "in_dim": 384,
        "bottleneck_dim": 32,
        "projection_dim": 128,
        "kernel_size": 3,
        "bias": True,
        "trainable_params": expected_residual_adapter_params(
            in_dim=384,
            bottleneck_dim=32,
            projection_dim=128,
        ),
    }
    b = {
        "run_name": "adapter_r64_d128",
        "in_dim": 384,
        "bottleneck_dim": 64,
        "projection_dim": 128,
        "kernel_size": 5,  # forbidden Day-04 drift
        "bias": True,
        "trainable_params": expected_residual_adapter_params(
            in_dim=384,
            bottleneck_dim=64,
            projection_dim=128,
            kernel_size=5,
        ),
    }

    with pytest.raises(ParameterCountError, match="Only r and d may vary"):
        validate_rd_screen_growth([a, b])
