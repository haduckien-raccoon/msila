#!/usr/bin/env python3
"""
tools/benchmark_cache.py
========================

Benchmark frozen DINOv3 online feature extraction vs MS-ILA feature cache.

Primary comparison
------------------
ONLINE E2E:
    CPU preprocessed x_local/x_context
        -> H2D (if CUDA)
        -> frozen DINOv3
        -> six features on training device

CACHED E2E:
    FeatureCacheReader
        -> six cached tensors on CPU/mmap
        -> materialize / H2D
        -> six features on training device

Correctness gate before timing:
    max |F_online - F_cache| < tolerance
for all six sources:
    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

References:
- PyTorch CUDA semantics:
  https://docs.pytorch.org/docs/stable/notes/cuda.html
- torch.cuda.synchronize:
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html
- torch.utils.benchmark:
  https://docs.pytorch.org/docs/stable/benchmark_utils.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_cache import (  # noqa: E402
    FEATURE_KEYS,
    FeatureCacheReader,
    validate_geometry,
)
from src.models.dinov3_extractor import build_online_extractor  # noqa: E402

DEFAULT_TOLERANCE = 1e-5


class BenchmarkCacheError(RuntimeError):
    """Base error for benchmark/cache validation."""


class BenchmarkCorrectnessError(BenchmarkCacheError):
    """Raised when cached features do not reproduce online extraction."""


@dataclass(frozen=True)
class LatencyStats:
    """Robust latency summary in milliseconds."""

    iterations: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float

    @property
    def samples_per_second_from_median(self) -> float:
        return 1000.0 / self.median_ms if self.median_ms > 0 else math.inf


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")

    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]

    alpha = pos - lo
    return xs[lo] * (1.0 - alpha) + xs[hi] * alpha


def summarize_latencies(samples_ms: Sequence[float]) -> LatencyStats:
    if not samples_ms:
        raise ValueError("samples_ms must be non-empty")

    values = [float(v) for v in samples_ms]
    if any((not math.isfinite(v)) or v < 0 for v in values):
        raise ValueError("latencies must be finite and >= 0")

    return LatencyStats(
        iterations=len(values),
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        std_ms=statistics.pstdev(values),
        min_ms=min(values),
        max_ms=max(values),
        p90_ms=_percentile(values, 0.90),
        p95_ms=_percentile(values, 0.95),
    )


def _sync_if_needed(device: torch.device) -> None:
    """Synchronize asynchronous CUDA work so wall-clock timing is valid."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> LatencyStats:
    """Benchmark one operation with warm-up and synchronized repetitions."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iterations <= 0:
        raise ValueError("iterations must be > 0")

    for _ in range(warmup):
        fn()
    _sync_if_needed(device)

    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync_if_needed(device)
        start_ns = time.perf_counter_ns()
        fn()
        _sync_if_needed(device)
        end_ns = time.perf_counter_ns()
        samples_ms.append((end_ns - start_ns) / 1_000_000.0)

    return summarize_latencies(samples_ms)


def load_real_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load one or more deterministic PREPROCESSED real samples.

    Accepted .pt payload:
      - one dict
      - list/tuple of dicts
      - {"samples": [dict, ...]}

    Each sample must contain:
        image_id, category, x_local, x_context, geometry
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    obj = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(obj, Mapping) and "samples" in obj:
        raw = obj["samples"]
    elif isinstance(obj, Mapping):
        raw = [obj]
    elif isinstance(obj, (list, tuple)):
        raw = list(obj)
    else:
        raise BenchmarkCacheError(
            f"{path}: expected dict/list/tuple payload, got {type(obj)!r}"
        )

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        raw = raw[:max_samples]

    if not raw:
        raise BenchmarkCacheError("No benchmark samples found")

    out: list[dict[str, Any]] = []
    required = {"image_id", "category", "x_local", "x_context", "geometry"}

    for i, sample in enumerate(raw):
        if not isinstance(sample, Mapping):
            raise BenchmarkCacheError(f"sample[{i}] must be a mapping")

        missing = sorted(required.difference(sample.keys()))
        if missing:
            raise BenchmarkCacheError(f"sample[{i}] missing required fields: {missing}")

        item = dict(sample)
        item["image_id"] = str(item["image_id"])
        item["category"] = str(item["category"])

        if not item["image_id"] or not item["category"]:
            raise BenchmarkCacheError(
                f"sample[{i}] image_id/category must be non-empty"
            )

        x_local = item["x_local"]
        x_context = item["x_context"]

        for name, x in (("x_local", x_local), ("x_context", x_context)):
            if not isinstance(x, torch.Tensor):
                raise BenchmarkCacheError(f"sample[{i}].{name} must be torch.Tensor")
            if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
                raise BenchmarkCacheError(
                    f"sample[{i}].{name} must be [1,3,H,W], got {tuple(x.shape)}"
                )
            if not torch.is_floating_point(x):
                raise BenchmarkCacheError(f"sample[{i}].{name} must be floating point")
            if not torch.isfinite(x).all().item():
                raise BenchmarkCacheError(f"sample[{i}].{name} contains NaN/Inf")

        if x_local.shape != x_context.shape:
            raise BenchmarkCacheError(
                f"sample[{i}] local/context shape mismatch: "
                f"{tuple(x_local.shape)} vs {tuple(x_context.shape)}"
            )

        if not isinstance(item["geometry"], Mapping):
            raise BenchmarkCacheError(f"sample[{i}].geometry must be a mapping")

        out.append(item)

    return out


def compare_feature_dicts(
    online: Mapping[str, torch.Tensor],
    cached: Mapping[str, torch.Tensor],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """
    Numerical correctness gate.

    For each feature F:
        e_max(F) = max_j |F_online[j] - F_cache[j]|

    PASS iff e_max(F) < tolerance for all six FEATURE_KEYS.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")

    errors: dict[str, float] = {}

    for key in FEATURE_KEYS:
        if key not in online:
            raise BenchmarkCorrectnessError(f"online output missing feature {key}")
        if key not in cached:
            raise BenchmarkCorrectnessError(f"cached output missing feature {key}")

        a = online[key].detach().cpu().float()
        b = cached[key].detach().cpu().float()

        if a.shape != b.shape:
            raise BenchmarkCorrectnessError(
                f"{key}: shape mismatch online={tuple(a.shape)} cached={tuple(b.shape)}"
            )

        if not torch.isfinite(a).all().item():
            raise BenchmarkCorrectnessError(f"{key}: online feature contains NaN/Inf")
        if not torch.isfinite(b).all().item():
            raise BenchmarkCorrectnessError(f"{key}: cached feature contains NaN/Inf")

        max_abs_error = (a - b).abs().max().item()
        errors[key] = max_abs_error

        if not max_abs_error < tolerance:
            raise BenchmarkCorrectnessError(
                f"{key}: max_abs_error={max_abs_error:.8e} "
                f">= tolerance={tolerance:.1e}"
            )

    return errors


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def verify_cached_pipeline(
    *,
    samples: Sequence[Mapping[str, Any]],
    extractor: Any,
    reader: FeatureCacheReader,
    device: torch.device,
    tolerance: float,
) -> dict[str, dict[str, float]]:
    """Verify fresh online DINOv3 against the previously builder-produced cache."""
    extractor.eval()
    report: dict[str, dict[str, float]] = {}

    for sample in samples:
        image_id = str(sample["image_id"])
        category = str(sample["category"])

        if (image_id, category) not in reader:
            raise BenchmarkCorrectnessError(f"cache miss for {category}/{image_id}")

        x_local = sample["x_local"].to(device=device)
        x_context = sample["x_context"].to(device=device)

        with torch.inference_mode():
            online = extractor.extract_online_cache_features(
                x_local,
                x_context,
                strategy="concat",
                to_cpu=False,
            )

        cached = reader.get(image_id=image_id, category=category)
        errors = compare_feature_dicts(online, cached, tolerance=tolerance)

        cached_geometry = json.dumps(
            _jsonable(cached["geometry"]),
            sort_keys=True,
            separators=(",", ":"),
        )
        sample_geometry = json.dumps(
            _jsonable(validate_geometry(sample["geometry"])),
            sort_keys=True,
            separators=(",", ":"),
        )
        if cached_geometry != sample_geometry:
            raise BenchmarkCorrectnessError(
                f"geometry mismatch for {category}/{image_id}"
            )

        report[f"{category}/{image_id}"] = errors

    return report


class _Cycler:
    def __init__(self, samples: Sequence[Any]) -> None:
        if not samples:
            raise ValueError("samples must be non-empty")
        self.samples = list(samples)
        self.index = 0

    def next(self) -> Any:
        item = self.samples[self.index % len(self.samples)]
        self.index += 1
        return item


def _materialize_cached_features_on_device(
    cached: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Force all six cached tensor payloads to be consumed."""
    if device.type == "cpu":
        return {key: cached[key].clone() for key in FEATURE_KEYS}

    return {
        key: cached[key].to(device=device, non_blocking=False)
        for key in FEATURE_KEYS
    }


def run_benchmark(
    *,
    samples: Sequence[Mapping[str, Any]],
    extractor: Any,
    reader: FeatureCacheReader,
    device: torch.device,
    warmup: int = 5,
    iterations: int = 30,
    tolerance: float = DEFAULT_TOLERANCE,
    verify: bool = True,
) -> dict[str, Any]:
    """
    Run correctness gate followed by online-vs-cache latency benchmark.

    Primary speedup:
        S = median(online_e2e) / median(cached_e2e)
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    if verify:
        correctness = verify_cached_pipeline(
            samples=samples,
            extractor=extractor,
            reader=reader,
            device=device,
            tolerance=tolerance,
        )
    else:
        correctness = {}

    device_samples = [
        {
            "image_id": str(s["image_id"]),
            "category": str(s["category"]),
            "x_local": s["x_local"].to(device=device),
            "x_context": s["x_context"].to(device=device),
        }
        for s in samples
    ]
    _sync_if_needed(device)

    online_compute_cycle = _Cycler(device_samples)
    online_e2e_cycle = _Cycler(samples)
    cached_cycle = _Cycler(samples)

    def online_compute_once():
        sample = online_compute_cycle.next()
        with torch.inference_mode():
            return extractor.extract_online_cache_features(
                sample["x_local"],
                sample["x_context"],
                strategy="concat",
                to_cpu=False,
            )

    def online_e2e_once():
        sample = online_e2e_cycle.next()
        x_local = sample["x_local"].to(device=device, non_blocking=False)
        x_context = sample["x_context"].to(device=device, non_blocking=False)
        with torch.inference_mode():
            return extractor.extract_online_cache_features(
                x_local,
                x_context,
                strategy="concat",
                to_cpu=False,
            )

    def cached_e2e_once():
        sample = cached_cycle.next()
        cached = reader.get(
            image_id=str(sample["image_id"]),
            category=str(sample["category"]),
        )
        return _materialize_cached_features_on_device(cached, device=device)

    online_compute = benchmark_callable(
        online_compute_once,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    online_e2e = benchmark_callable(
        online_e2e_once,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    cached_e2e = benchmark_callable(
        cached_e2e_once,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )

    speedup = (
        online_e2e.median_ms / cached_e2e.median_ms
        if cached_e2e.median_ms > 0
        else math.inf
    )

    return {
        "correctness_pass": True if verify else None,
        "tolerance": tolerance,
        "num_samples": len(samples),
        "warmup": warmup,
        "iterations": iterations,
        "device": str(device),
        "online_compute": {
            **asdict(online_compute),
            "samples_per_second_from_median": online_compute.samples_per_second_from_median,
        },
        "online_e2e": {
            **asdict(online_e2e),
            "samples_per_second_from_median": online_e2e.samples_per_second_from_median,
        },
        "cached_e2e": {
            **asdict(cached_e2e),
            "samples_per_second_from_median": cached_e2e.samples_per_second_from_median,
        },
        "speedup_cached_vs_online_e2e": speedup,
        "max_abs_error_by_sample": correctness,
    }


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        meta["cuda_device_name"] = torch.cuda.get_device_name(device)
        meta["cuda_version"] = torch.version.cuda
    return meta


def _fmt_stats(name: str, stats: Mapping[str, Any]) -> str:
    return (
        f"{name:<18}"
        f" median={stats['median_ms']:>9.3f} ms"
        f"  p95={stats['p95_ms']:>9.3f} ms"
        f"  mean={stats['mean_ms']:>9.3f} ms"
        f"  throughput~={stats['samples_per_second_from_median']:>9.2f} sample/s"
    )


def print_report(report: Mapping[str, Any]) -> None:
    print("\n=== MS-ILA FEATURE CACHE BENCHMARK ===")
    print(f"device      : {report['device']}")
    print(f"samples     : {report['num_samples']}")
    print(f"warmup/iter : {report['warmup']}/{report['iterations']}")
    print(f"tolerance   : {report['tolerance']:.1e}")
    print(f"correctness : {report['correctness_pass']}")
    print()
    print(_fmt_stats("online_compute", report["online_compute"]))
    print(_fmt_stats("online_e2e", report["online_e2e"]))
    print(_fmt_stats("cached_e2e", report["cached_e2e"]))
    print(
        f"\nPRIMARY speedup (online_e2e / cached_e2e): "
        f"{report['speedup_cached_vs_online_e2e']:.3f}x"
    )

    if report.get("max_abs_error_by_sample"):
        max_error = max(
            err
            for sample_errors in report["max_abs_error_by_sample"].values()
            for err in sample_errors.values()
        )
        print(f"max numerical error across verified samples: {max_error:.8e}")


def _resolve_required_arg(
    cli_value: str | None,
    env_name: str,
    parser: argparse.ArgumentParser,
) -> str:
    value = cli_value or os.getenv(env_name)
    if not value:
        parser.error(f"missing required value: pass CLI argument or set {env_name}")
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real online frozen DINOv3 feature extraction against "
            "the prebuilt MS-ILA feature cache."
        )
    )
    parser.add_argument("--sample-pt", help="Preprocessed samples; env REAL_SAMPLE_PT.")
    parser.add_argument("--cache-dir", help="Prebuilt cache; env FEATURE_CACHE_DIR.")
    parser.add_argument("--dinov3-repo", help="Pinned local repo; env DINOV3_REPO.")
    parser.add_argument("--dinov3-weights", help="Checkpoint; env DINOV3_WEIGHTS.")
    parser.add_argument("--model", default=os.getenv("DINOV3_MODEL", "dinov3_vits16"))
    parser.add_argument(
        "--device",
        default=os.getenv(
            "DINOV3_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical gate; only for repeated profiling, not final results.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    sample_pt = _resolve_required_arg(args.sample_pt, "REAL_SAMPLE_PT", parser)
    cache_dir = _resolve_required_arg(args.cache_dir, "FEATURE_CACHE_DIR", parser)
    dinov3_repo = _resolve_required_arg(args.dinov3_repo, "DINOV3_REPO", parser)
    dinov3_weights = _resolve_required_arg(
        args.dinov3_weights,
        "DINOV3_WEIGHTS",
        parser,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but torch.cuda.is_available() is False")

    samples = load_real_samples(sample_pt, max_samples=args.max_samples)
    extractor = build_online_extractor(
        repo_dir=dinov3_repo,
        weights=dinov3_weights,
        device=device,
        model_name=args.model,
        blocks=(4, 8, 12),
        norm=True,
        check_finite=True,
    )
    reader = FeatureCacheReader(cache_dir, mmap=True, shard_cache_size=2)

    report = run_benchmark(
        samples=samples,
        extractor=extractor,
        reader=reader,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        tolerance=args.tolerance,
        verify=not args.no_verify,
    )

    report["environment"] = _environment_metadata(device)
    report["model"] = args.model
    report["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    report["sample_pt"] = str(Path(sample_pt).expanduser().resolve())

    print_report(report)

    if args.json_out is not None:
        out = args.json_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nJSON report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
