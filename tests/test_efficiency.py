from __future__ import annotations

import contextlib

import pytest
import torch
from torch import nn

from src.eval import efficiency as efficiency


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d: 4*3*3*3 + 4 = 112 parameters.
        self.conv = nn.Conv2d(3, 4, kernel_size=3, bias=True)
        # Linear: 2*4 + 2 = 10 parameters; frozen.
        self.fc = nn.Linear(4, 2, bias=True)
        for p in self.fc.parameters():
            p.requires_grad_(False)


def test_parameter_count_matches_manual_count():
    model = _ToyModel()

    counts = efficiency.count_parameters(model)

    assert counts["total_parameters"] == 122
    assert counts["trainable_parameters"] == 112
    assert counts["frozen_parameters"] == 10
    assert counts["trainable_fraction"] == pytest.approx(112 / 122)


def test_shared_parameter_object_is_counted_once():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(4, 3, bias=False)  # 12 scalar parameters
            self.a = shared
            self.b = shared

    report = efficiency.parameter_report(
        Shared(),
        candidate_id="shared",
        scope="unit_test",
    )

    assert report["totals"]["total_parameters"] == 12
    assert report["totals"]["trainable_parameters"] == 12
    assert len(report["shared_parameter_aliases"]) == 1
    assert report["shared_parameter_aliases"][0]["aliases"]


def test_manual_cached_msila_formula_regression():
    manual = efficiency.manual_cached_msila_parameter_formula(
        in_channels=384,
        fusion_dim=128,
        adapter_reduction=4,
        adapter_kernel_size=3,
        num_blocks=3,
        share_projection_across_views=True,
    )

    assert manual["adapter_bottleneck_channels"] == 96
    assert manual["components"] == {
        "adapters": 225_507,
        "aligner": 0,
        "projection": 147_840,
        "fusion": 134,
        "decoder": 73_857,
    }
    assert manual["total_parameters"] == 447_338


def test_assert_parameter_count_rejects_mismatch():
    report = efficiency.parameter_report(
        _ToyModel(),
        candidate_id="toy",
        scope="unit_test",
    )

    with pytest.raises(
        efficiency.ParameterCountError,
        match="FAILED",
    ):
        efficiency.assert_parameter_count(
            report,
            expected_total=123,
            expected_trainable=112,
        )


def test_benchmark_scope_fingerprint_is_deterministic_and_sensitive():
    a = efficiency.make_benchmark_scope(
        scope_name="day04_cached_head",
        device="cpu",
        batch_size=1,
        precision="fp32",
        input_signature="six cached feature maps",
        pipeline_stages=(
            "adapter",
            "context_alignment",
            "projection",
            "attention_fusion",
            "decoder",
        ),
        extra={"output_size": [512, 512], "tag": "locked"},
    )

    # Mapping insertion order must not change the fingerprint.
    b = efficiency.make_benchmark_scope(
        scope_name="day04_cached_head",
        device="cpu",
        batch_size=1,
        precision="fp32",
        input_signature="six cached feature maps",
        pipeline_stages=(
            "adapter",
            "context_alignment",
            "projection",
            "attention_fusion",
            "decoder",
        ),
        extra={"tag": "locked", "output_size": [512, 512]},
    )

    changed = efficiency.make_benchmark_scope(
        scope_name="day04_cached_head",
        device="cpu",
        batch_size=2,
        precision="fp32",
        input_signature="six cached feature maps",
        pipeline_stages=(
            "adapter",
            "context_alignment",
            "projection",
            "attention_fusion",
            "decoder",
        ),
        extra={"output_size": [512, 512], "tag": "locked"},
    )

    assert a["fingerprint_sha256"] == b["fingerprint_sha256"]
    assert a["fingerprint_sha256"] != changed["fingerprint_sha256"]


def test_latency_runs_exact_warmup_and_measurement_counts(monkeypatch):
    rounds = 3
    warmup = 2
    iterations = 4
    calls = {"n": 0, "inference_mode": []}

    def fn():
        calls["n"] += 1
        calls["inference_mode"].append(torch.is_inference_mode_enabled())
        return torch.tensor(1.0)

    # Every measured call is exactly 2 ms.
    times = []
    base = 0
    for _ in range(rounds * iterations):
        times.extend([base, base + 2_000_000])
        base += 10_000_000
    iterator = iter(times)
    monkeypatch.setattr(
        efficiency.time,
        "perf_counter_ns",
        lambda: next(iterator),
    )

    report = efficiency.benchmark_latency(
        fn,
        device="cpu",
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
        use_inference_mode=True,
        stability_cv_threshold=0.01,
    )

    assert calls["n"] == rounds * (warmup + iterations)
    assert all(calls["inference_mode"])

    assert report["status"] == "PASS"
    assert report["protocol"]["total_timed_iterations"] == rounds * iterations
    assert report["latency_ms"]["mean"] == pytest.approx(2.0)
    assert report["latency_ms"]["median"] == pytest.approx(2.0)
    assert report["latency_ms"]["p95"] == pytest.approx(2.0)
    assert report["latency_ms"]["std"] == pytest.approx(0.0)
    assert report["stability"]["round_medians_ms"] == pytest.approx([2.0, 2.0, 2.0])
    assert report["stability"]["round_median_cv"] == pytest.approx(0.0)


def test_latency_stability_gate_detects_round_drift(monkeypatch):
    # One sample per round. Durations: 1 ms, 2 ms, 4 ms -> intentionally unstable.
    durations_ns = [1_000_000, 2_000_000, 4_000_000]
    times = []
    base = 0
    for duration in durations_ns:
        times.extend([base, base + duration])
        base += 10_000_000
    iterator = iter(times)
    monkeypatch.setattr(
        efficiency.time,
        "perf_counter_ns",
        lambda: next(iterator),
    )

    report = efficiency.benchmark_latency(
        lambda: None,
        device="cpu",
        warmup=0,
        iterations=1,
        rounds=3,
        stability_cv_threshold=0.10,
    )

    assert report["status"] == "FAIL"
    assert report["stability"]["status"] == "FAIL"
    assert report["stability"]["round_median_cv"] > 0.10


@pytest.mark.parametrize(
    ("warmup", "iterations", "rounds"),
    [
        (-1, 1, 2),
        (0, 0, 2),
        (0, 1, 1),
    ],
)
def test_latency_rejects_invalid_protocol(warmup, iterations, rounds):
    with pytest.raises(ValueError):
        efficiency.benchmark_latency(
            lambda: None,
            device="cpu",
            warmup=warmup,
            iterations=iterations,
            rounds=rounds,
        )


def test_peak_vram_rejects_cpu_measurement():
    with pytest.raises(
        efficiency.CUDAUnavailableError,
        match="CUDA-only",
    ):
        efficiency.benchmark_peak_vram(
            lambda: None,
            device="cpu",
            warmup=0,
            iterations=1,
        )


def test_peak_vram_protocol_reset_then_measure_is_regression_tested(monkeypatch):
    events: list[str] = []
    calls = {"n": 0}

    # Unit-test the CUDA protocol without requiring physical CUDA hardware.
    monkeypatch.setattr(
        efficiency,
        "_resolve_device",
        lambda device: torch.device("cuda:0"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        efficiency,
        "_sync_device",
        lambda device: events.append("sync"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: events.append("reset"),
    )

    def memory_allocated(device):
        events.append("memory_allocated")
        return 100

    def memory_reserved(device):
        events.append("memory_reserved")
        return 200

    def max_memory_allocated(device):
        events.append("max_memory_allocated")
        return 180

    def max_memory_reserved(device):
        events.append("max_memory_reserved")
        return 280

    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", memory_reserved)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_memory_allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", max_memory_reserved)

    def fn():
        calls["n"] += 1
        events.append("fn")
        return object()

    report = efficiency.benchmark_peak_vram(
        fn,
        device="cuda:0",
        warmup=2,
        iterations=3,
        use_inference_mode=True,
    )

    assert calls["n"] == 5

    # Critical ordering: warm-up completes, then sync, then reset, then baseline.
    reset_index = events.index("reset")
    assert events[:reset_index] == ["fn", "sync", "fn", "sync", "sync"]
    assert events[reset_index + 1 : reset_index + 3] == [
        "memory_allocated",
        "memory_reserved",
    ]

    assert report["status"] == "PASS"
    assert report["memory"]["baseline_allocated"]["bytes"] == 100
    assert report["memory"]["peak_allocated"]["bytes"] == 180
    assert report["memory"]["incremental_peak_allocated"]["bytes"] == 80
    assert report["memory"]["baseline_reserved"]["bytes"] == 200
    assert report["memory"]["peak_reserved"]["bytes"] == 280
    assert report["memory"]["incremental_peak_reserved"]["bytes"] == 80
    assert report["protocol"]["empty_cache_before_measurement"] is False


def test_peak_vram_rejects_impossible_peak_smaller_than_baseline(monkeypatch):
    monkeypatch.setattr(
        efficiency,
        "_resolve_device",
        lambda device: torch.device("cuda:0"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: contextlib.nullcontext(),
    )
    monkeypatch.setattr(efficiency, "_sync_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 200)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 300)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 150)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 350)

    with pytest.raises(
        efficiency.BenchmarkProtocolError,
        match="peak allocated",
    ):
        efficiency.benchmark_peak_vram(
            lambda: None,
            device="cuda:0",
            warmup=0,
            iterations=1,
        )
