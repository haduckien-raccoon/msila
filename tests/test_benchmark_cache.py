from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_cache import FEATURE_KEYS, FeatureCacheReader, FeatureCacheWriter

MODULE_PATH = PROJECT_ROOT / "tools" / "benchmark_cache.py"
spec = importlib.util.spec_from_file_location("benchmark_cache", MODULE_PATH)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
sys.modules["benchmark_cache"] = bench
spec.loader.exec_module(bench)


def make_feature_dict(value: float = 0.0):
    return {
        key: torch.full((1, 4, 4, 4), float(value), dtype=torch.float32)
        for key in FEATURE_KEYS
    }


def make_geometry():
    return {
        "image_hw": [64, 64],
        "local_hw": [32, 32],
        "context_hw": [48, 48],
        "local_box": [16, 16, 48, 48],
        "context_box": [8, 8, 56, 56],
        "context_to_local": [
            [1.0, 0.0, -8.0],
            [0.0, 1.0, -8.0],
            [0.0, 0.0, 1.0],
        ],
    }


class MockOnlineExtractor(torch.nn.Module):
    """Deterministic mock of the online cache-schema extractor."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.eval()

    def extract_online_cache_features(
        self,
        x_local,
        x_context,
        *,
        strategy="concat",
        to_cpu=False,
    ):
        assert strategy == "concat"

        local_value = x_local.mean()
        context_value = x_context.mean()

        out = {}
        for key in FEATURE_KEYS:
            base = local_value if key.startswith("local_") else context_value
            if key.endswith("b4"):
                offset = 4.0
            elif key.endswith("b8"):
                offset = 8.0
            else:
                offset = 12.0

            t = torch.ones(
                (1, 4, 4, 4),
                device=x_local.device,
                dtype=torch.float32,
            ) * (base + offset)

            if to_cpu:
                t = t.detach().cpu().contiguous()
            out[key] = t

        return out


def make_real_sample(i: int = 0):
    return {
        "image_id": f"fabric/sample_{i:03d}.png",
        "category": "fabric",
        "x_local": torch.full((1, 3, 32, 32), float(i + 1), dtype=torch.float32),
        "x_context": torch.full((1, 3, 32, 32), float(i + 2), dtype=torch.float32),
        "geometry": make_geometry(),
    }


def expected_features(sample):
    return MockOnlineExtractor().extract_online_cache_features(
        sample["x_local"],
        sample["x_context"],
    )


def build_matching_cache(root: Path, samples):
    signature = {
        "backbone": "mock",
        "preprocess_version": "test-v1",
        "logical_layers_1based": [4, 8, 12],
    }

    with FeatureCacheWriter(
        root,
        producer_signature=signature,
        target_shard_bytes=1024 * 1024,
    ) as writer:
        for sample in samples:
            writer.add({
                "image_id": sample["image_id"],
                "category": sample["category"],
                **expected_features(sample),
                "geometry": sample["geometry"],
            })

    return signature


def test_summarize_latencies_uses_median_and_percentiles():
    stats = bench.summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])

    assert stats.iterations == 5
    assert stats.median_ms == 3.0
    assert stats.mean_ms == pytest.approx(22.0)
    assert stats.min_ms == 1.0
    assert stats.max_ms == 100.0
    assert stats.p90_ms >= stats.median_ms
    assert stats.p95_ms >= stats.p90_ms


def test_benchmark_callable_runs_warmup_plus_iterations():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]

    stats = bench.benchmark_callable(
        fn,
        device=torch.device("cpu"),
        warmup=2,
        iterations=5,
    )

    assert calls["n"] == 7
    assert stats.iterations == 5
    assert stats.median_ms >= 0.0


def test_compare_feature_dicts_passes_exact_match():
    online = make_feature_dict(2.0)
    cached = {key: value.clone() for key, value in online.items()}

    errors = bench.compare_feature_dicts(online, cached, tolerance=1e-5)

    assert set(errors) == set(FEATURE_KEYS)
    assert all(v == 0.0 for v in errors.values())


def test_compare_feature_dicts_rejects_error_above_tolerance():
    online = make_feature_dict(0.0)
    cached = {key: value.clone() for key, value in online.items()}
    cached["local_b8"][0, 0, 0, 0] = 1.1e-5

    with pytest.raises(bench.BenchmarkCorrectnessError, match="local_b8"):
        bench.compare_feature_dicts(online, cached, tolerance=1e-5)


def test_verify_cached_pipeline_passes_independent_online_vs_cache(tmp_path):
    samples = [make_real_sample(0), make_real_sample(1)]
    signature = build_matching_cache(tmp_path, samples)

    reader = FeatureCacheReader(
        tmp_path,
        expected_producer_signature=signature,
        mmap=True,
    )

    report = bench.verify_cached_pipeline(
        samples=samples,
        extractor=MockOnlineExtractor(),
        reader=reader,
        device=torch.device("cpu"),
        tolerance=1e-5,
    )

    assert len(report) == 2
    assert all(
        err == 0.0
        for sample_report in report.values()
        for err in sample_report.values()
    )


def test_verify_cached_pipeline_rejects_wrong_cache(tmp_path):
    samples = [make_real_sample(0)]
    signature = build_matching_cache(tmp_path / "good", samples)

    good_reader = FeatureCacheReader(
        tmp_path / "good",
        expected_producer_signature=signature,
        mmap=False,
    )
    cached = good_reader.get(
        image_id=samples[0]["image_id"],
        category=samples[0]["category"],
    )

    wrong_dir = tmp_path / "wrong"
    with FeatureCacheWriter(wrong_dir, producer_signature=signature) as writer:
        wrong = {
            "image_id": cached["image_id"],
            "category": cached["category"],
            **{key: cached[key].clone() for key in FEATURE_KEYS},
            "geometry": cached["geometry"],
        }
        wrong["local_b4"][0, 0, 0, 0] += 1e-3
        writer.add(wrong)

    wrong_reader = FeatureCacheReader(
        wrong_dir,
        expected_producer_signature=signature,
        mmap=True,
    )

    with pytest.raises(bench.BenchmarkCorrectnessError, match="local_b4"):
        bench.verify_cached_pipeline(
            samples=samples,
            extractor=MockOnlineExtractor(),
            reader=wrong_reader,
            device=torch.device("cpu"),
            tolerance=1e-5,
        )


def test_run_benchmark_reports_three_paths_and_speedup(tmp_path):
    samples = [make_real_sample(0), make_real_sample(1)]
    signature = build_matching_cache(tmp_path, samples)

    reader = FeatureCacheReader(
        tmp_path,
        expected_producer_signature=signature,
        mmap=True,
        shard_cache_size=2,
    )

    report = bench.run_benchmark(
        samples=samples,
        extractor=MockOnlineExtractor(),
        reader=reader,
        device=torch.device("cpu"),
        warmup=1,
        iterations=3,
        tolerance=1e-5,
        verify=True,
    )

    assert report["correctness_pass"] is True
    assert report["num_samples"] == 2

    for key in ("online_compute", "online_e2e", "cached_e2e"):
        assert report[key]["iterations"] == 3
        assert report[key]["median_ms"] >= 0.0
        assert report[key]["p95_ms"] >= 0.0

    assert report["speedup_cached_vs_online_e2e"] > 0.0


def test_load_real_samples_accepts_single_and_samples_dict(tmp_path):
    s0 = make_real_sample(0)
    s1 = make_real_sample(1)

    one = tmp_path / "one.pt"
    torch.save(s0, one)
    loaded_one = bench.load_real_samples(one)
    assert len(loaded_one) == 1
    assert loaded_one[0]["image_id"] == s0["image_id"]

    many = tmp_path / "many.pt"
    torch.save({"samples": [s0, s1]}, many)
    loaded_many = bench.load_real_samples(many)
    assert len(loaded_many) == 2
