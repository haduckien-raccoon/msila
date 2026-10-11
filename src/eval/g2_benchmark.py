"""Measured CUDA native-inference benchmark; CPU execution supplies no numbers."""
from __future__ import annotations

import math
import statistics
import time

import torch


class BenchmarkUnavailable(RuntimeError):
    """A real CUDA device is required for latency/VRAM measurements."""


def benchmark_native(model, image, cfg, device, *, predict_fn, warmup=1, repeats=5):
    """Measure synchronized end-to-end native inference, including Hann stitching.

    Latency is median wall-clock milliseconds, not a claim about a particular
    GPU model. Peak VRAM is allocated CUDA bytes, including the resident model.
    The named GPU, image shape and tile batch size accompany every measurement.
    """
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 must be integers")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise BenchmarkUnavailable("NOT RUN: CUDA GPU required; CPU timings are not GPU measurements")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    if device.index >= torch.cuda.device_count():
        raise BenchmarkUnavailable(f"CUDA device unavailable: {device}")
    if not any(p.device == device for p in model.parameters()):
        raise BenchmarkUnavailable("Benchmark model must already reside on the requested CUDA device")
    model.eval()
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            predict_fn(model, image, cfg, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            score = predict_fn(model, image, cfg, device)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000
            if (not math.isfinite(elapsed) or elapsed <= 0 or score.shape != image.shape[-2:]
                    or not torch.isfinite(score).all()):
                raise ValueError("Invalid benchmark duration/native map")
            samples.append(elapsed)
    peak = torch.cuda.max_memory_allocated(device)
    if peak <= 0:
        raise ValueError("CUDA benchmark has no allocated-memory evidence")
    return dict(status="PASS", device=str(device), gpu=torch.cuda.get_device_name(device),
                torch=str(torch.__version__), cuda=torch.version.cuda,
                latency=statistics.median(samples), latency_unit="milliseconds",
                peak_vram=peak, peak_vram_unit="allocated_bytes",
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                native_hw=list(image.shape[-2:]), tile_batch_size=cfg["evaluation"]["tile_batch_size"],
                scope="native inference: feature extraction, decoder, sigmoid, transfers, Hann stitching",
                warmup=warmup, repeats=repeats, samples_ms=samples,
                cpu_fixture=False)
