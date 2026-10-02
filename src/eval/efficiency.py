"""
MS-ILA E4–E6 — Efficiency evaluation utilities.

Implemented scopes
------------------
E4:
    - total/trainable parameter counting
    - architecture-specific manual count verification

E5:
    - inference latency after warm-up
    - explicit CUDA synchronization
    - repeated rounds and iterations
    - mean / median / p95
    - repeated-run stability diagnostic

E6:
    - CUDA peak tensor memory via PyTorch's CUDA allocator statistics
    - reset peak -> inference -> query peak
    - absolute peak and incremental peak above the pre-inference baseline
    - allocated memory is primary; reserved allocator memory is diagnostic

Not implemented here:
    - E7 candidate aggregation
    - E8 model selection
    - E9 fairness audit
    - E10 regression-test suite

Scientific conventions
----------------------
Parameter count:
    N_total = sum(p.numel()) over unique nn.Parameter objects
    N_train = sum(p.numel()) where p.requires_grad=True

Latency:
    warm-up -> device sync -> repeated wall-clock measurement
    with synchronization before/after each CUDA-timed call.

Peak VRAM:
    warm-up -> synchronize -> reset_peak_memory_stats()
    -> inference -> synchronize -> max_memory_allocated().

The caller must set the model to eval() before benchmarking.  Inference mode is
used by default to remove autograd bookkeeping, but torch.inference_mode() does
not itself switch a model to evaluation mode.
"""


from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


__all__ = [
    "ParameterCountError",
    "count_parameters",
    "parameter_report",
    "save_parameter_report",
    "evaluate_candidate_parameters",
    "assert_parameter_count",
    "manual_cached_msila_parameter_formula",
    "verify_cached_msila_parameter_count",
    "BenchmarkProtocolError",
    "CUDAUnavailableError",
    "make_benchmark_scope",
    "benchmark_latency",
    "benchmark_peak_vram",
    "benchmark_inference_efficiency",
]


class ParameterCountError(RuntimeError):
    """Raised when parameter accounting violates the locked E4 contract."""


def _require_nonempty_string(value: str, *, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _collect_unique_parameters(model: nn.Module) -> list[dict[str, Any]]:
    """Collect unique Parameter objects and record all registered aliases.

    PyTorch normally removes duplicated Parameter objects from
    ``named_parameters()``.  E4 explicitly asks for auditable counting, so we
    request duplicates and then deduplicate by Python object identity.

    This handles the normal weight-sharing case where two module paths point to
    the same ``nn.Parameter`` object.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}")

    try:
        named = list(model.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError:  # defensive fallback for old PyTorch releases
        named = list(model.named_parameters(recurse=True))

    by_identity: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    for name, parameter in named:
        if not isinstance(parameter, nn.Parameter):
            raise ParameterCountError(
                f"named_parameters returned non-Parameter object for {name!r}"
            )

        key = id(parameter)
        if key not in by_identity:
            by_identity[key] = {
                "parameter": parameter,
                "names": [str(name)],
            }
            order.append(key)
        else:
            by_identity[key]["names"].append(str(name))

    rows: list[dict[str, Any]] = []
    for key in order:
        item = by_identity[key]
        p: nn.Parameter = item["parameter"]
        names = list(dict.fromkeys(item["names"]))
        canonical_name = names[0] if names else "<unnamed>"

        rows.append(
            {
                "canonical_name": canonical_name,
                "aliases": names[1:],
                "shape": [int(v) for v in p.shape],
                "numel": int(p.numel()),
                "requires_grad": bool(p.requires_grad),
                "dtype": str(p.dtype).replace("torch.", ""),
                "device": str(p.device),
            }
        )

    return rows


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Return the locked E4 scalar parameter counts.

    Buffers, activations, gradients, optimizer states, and cached features are
    deliberately excluded because they are not model parameters.
    """
    rows = _collect_unique_parameters(model)

    total = sum(int(row["numel"]) for row in rows)
    trainable = sum(
        int(row["numel"])
        for row in rows
        if bool(row["requires_grad"])
    )
    frozen = total - trainable

    if total < 0 or trainable < 0 or frozen < 0:
        raise ParameterCountError("negative parameter count is impossible")
    if total != trainable + frozen:
        raise ParameterCountError("total != trainable + frozen")

    fraction = (trainable / total) if total > 0 else 0.0

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "trainable_fraction": float(fraction),
        "trainable_percent": float(100.0 * fraction),
        "unique_parameter_tensors": int(len(rows)),
    }


def _group_rows_by_top_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Diagnostic breakdown by the canonical parameter's top-level module."""
    groups: dict[str, dict[str, int]] = {}

    for row in rows:
        name = str(row["canonical_name"])
        group = name.split(".", 1)[0] if "." in name else "__root__"
        item = groups.setdefault(
            group,
            {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "frozen_parameters": 0,
                "unique_parameter_tensors": 0,
            },
        )

        n = int(row["numel"])
        item["total_parameters"] += n
        if bool(row["requires_grad"]):
            item["trainable_parameters"] += n
        else:
            item["frozen_parameters"] += n
        item["unique_parameter_tensors"] += 1

    return dict(sorted(groups.items()))


def parameter_report(
    model: nn.Module,
    *,
    candidate_id: str,
    scope: str = "model_object",
) -> dict[str, Any]:
    """Build an auditable E4 parameter report for one candidate.

    Parameters
    ----------
    model:
        The exact ``nn.Module`` whose parameters are being reported.

    candidate_id:
        Stable experiment/candidate identifier, e.g. ``r4_d128``.

    scope:
        Explicit interpretation of ``model``. Recommended project values:
        ``cached_trainable_pipeline`` or ``full_inference_model``.

        This matters because ``CachedFeatureTrainingModel`` intentionally does
        not instantiate frozen DINOv3; therefore its ``total_parameters`` must
        not be mislabeled as the total parameters of the full deployment model.
    """
    candidate_id = _require_nonempty_string(candidate_id, name="candidate_id")
    scope = _require_nonempty_string(scope, name="scope")

    rows = _collect_unique_parameters(model)
    totals = count_parameters(model)

    duplicate_aliases = [
        {
            "canonical_name": row["canonical_name"],
            "aliases": list(row["aliases"]),
        }
        for row in rows
        if row["aliases"]
    ]

    # Independent consistency check from the detailed rows.
    detail_total = sum(int(row["numel"]) for row in rows)
    detail_trainable = sum(
        int(row["numel"])
        for row in rows
        if bool(row["requires_grad"])
    )

    if detail_total != int(totals["total_parameters"]):
        raise ParameterCountError("detail rows do not reproduce total_parameters")
    if detail_trainable != int(totals["trainable_parameters"]):
        raise ParameterCountError("detail rows do not reproduce trainable_parameters")

    return {
        "schema_version": "msila.e4.params.v1",
        "candidate_id": candidate_id,
        "scope": scope,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "counting_rule": {
            "unit": "scalar_parameter_elements",
            "total": "sum(p.numel()) over unique nn.Parameter objects",
            "trainable": "sum(p.numel()) where p.requires_grad is True",
            "shared_parameter_policy": "same nn.Parameter object counted once",
            "buffers_included": False,
            "optimizer_membership_inspected": False,
        },
        "totals": totals,
        "by_top_level_module": _group_rows_by_top_level(rows),
        "shared_parameter_aliases": duplicate_aliases,
        "parameters": rows,
        "internal_consistency": {
            "detail_total_equals_total": True,
            "detail_trainable_equals_trainable": True,
            "status": "PASS",
        },
    }


def _atomic_json_dump(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")

    os.replace(tmp, path)
    return path


def save_parameter_report(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write one candidate's E4 report as JSON."""
    if "totals" not in report or "candidate_id" not in report:
        raise ValueError("report does not look like an E4 parameter report")
    return _atomic_json_dump(report, output_path)


def evaluate_candidate_parameters(
    model: nn.Module,
    *,
    candidate_id: str,
    output_path: str | Path,
    scope: str = "model_object",
) -> dict[str, Any]:
    """Convenience entry point: count -> audit -> write ``params.json``."""
    report = parameter_report(
        model,
        candidate_id=candidate_id,
        scope=scope,
    )
    save_parameter_report(report, output_path)
    return report


def assert_parameter_count(
    report: Mapping[str, Any],
    *,
    expected_total: int,
    expected_trainable: int,
) -> None:
    """Hard PASS gate against an independently derived manual count."""
    if isinstance(expected_total, bool) or not isinstance(expected_total, int):
        raise TypeError("expected_total must be int")
    if isinstance(expected_trainable, bool) or not isinstance(expected_trainable, int):
        raise TypeError("expected_trainable must be int")
    if expected_total < 0 or expected_trainable < 0:
        raise ValueError("expected counts must be >= 0")
    if expected_trainable > expected_total:
        raise ValueError("expected_trainable cannot exceed expected_total")

    try:
        totals = report["totals"]
        actual_total = int(totals["total_parameters"])
        actual_trainable = int(totals["trainable_parameters"])
    except Exception as exc:
        raise ParameterCountError("invalid E4 report structure") from exc

    problems: list[str] = []
    if actual_total != expected_total:
        problems.append(
            f"total_parameters actual={actual_total} expected={expected_total}"
        )
    if actual_trainable != expected_trainable:
        problems.append(
            "trainable_parameters "
            f"actual={actual_trainable} expected={expected_trainable}"
        )

    if problems:
        raise ParameterCountError("Manual parameter-count check FAILED: " + "; ".join(problems))


def manual_cached_msila_parameter_formula(
    *,
    in_channels: int,
    fusion_dim: int,
    adapter_reduction: int = 4,
    adapter_bottleneck_channels: int | None = None,
    adapter_kernel_size: int = 3,
    num_blocks: int = 3,
    share_projection_across_views: bool = True,
    decoder_hidden_channels: int | None = None,
) -> dict[str, Any]:
    """Closed-form count for the current ``CachedFeatureTrainingModel``.

    This formula is independent of traversing ``model.parameters()`` and is
    therefore suitable as the manual/reference side of the E4 PASS criterion.

    Current project architecture assumptions
    ----------------------------------------
    - one ResidualAdapter2d per DINO block, shared by Local/Context;
    - ContextToLocalAligner is parameter-free;
    - one 1x1 projector per block when Local/Context projection is shared,
      otherwise two projectors per block;
    - AttentionFusion has a shared d->1 bias-free Linear and one learnable
      source bias per Local/Context source;
    - BasicDecoder: Conv3x3(d->h, bias) + Conv1x1(h->1, bias).

    Frozen DINOv3 is NOT included, because CachedFeatureTrainingModel does not
    instantiate the backbone.
    """
    for name, value in (
        ("in_channels", in_channels),
        ("fusion_dim", fusion_dim),
        ("adapter_reduction", adapter_reduction),
        ("adapter_kernel_size", adapter_kernel_size),
        ("num_blocks", num_blocks),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be int")
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    if adapter_kernel_size % 2 == 0:
        raise ValueError("adapter_kernel_size must be odd")

    c = int(in_channels)
    d = int(fusion_dim)
    k = int(adapter_kernel_size)
    bcount = int(num_blocks)

    if adapter_bottleneck_channels is None:
        bottleneck = max(1, c // int(adapter_reduction))
    else:
        if (
            isinstance(adapter_bottleneck_channels, bool)
            or not isinstance(adapter_bottleneck_channels, int)
        ):
            raise TypeError("adapter_bottleneck_channels must be int or None")
        if adapter_bottleneck_channels <= 0:
            raise ValueError("adapter_bottleneck_channels must be > 0")
        bottleneck = int(adapter_bottleneck_channels)

    if decoder_hidden_channels is None:
        hidden = max(d // 2, 1)
    else:
        if isinstance(decoder_hidden_channels, bool) or not isinstance(
            decoder_hidden_channels, int
        ):
            raise TypeError("decoder_hidden_channels must be int or None")
        if decoder_hidden_channels <= 0:
            raise ValueError("decoder_hidden_channels must be > 0")
        hidden = int(decoder_hidden_channels)

    # ResidualAdapter2d(C -> b -> C):
    # down 1x1: C*b weights + b bias
    # depthwise kxk: b*k^2 weights + b bias
    # up 1x1: b*C weights + C bias
    # gamma: 1 scalar
    adapter_one = (
        c * bottleneck
        + bottleneck
        + bottleneck * k * k
        + bottleneck
        + bottleneck * c
        + c
        + 1
    )
    adapters = bcount * adapter_one

    # SixFeatureProjection:
    # shared views -> one C->d 1x1 per block
    # unshared     -> Local + Context projectors per block
    n_projectors = bcount if share_projection_across_views else 2 * bcount
    projection_one = c * d + d
    projection = n_projectors * projection_one

    # AttentionFusion:
    # score_proj Linear(d,1,bias=False): d
    # source_bias: 2 * num_blocks
    num_sources = 2 * bcount
    fusion = d + num_sources

    # BasicDecoder:
    # Conv3x3 d->h: 9*d*h + h
    # Conv1x1 h->1: h + 1
    decoder = 9 * d * hidden + 2 * hidden + 1

    components = {
        "adapters": int(adapters),
        "aligner": 0,
        "projection": int(projection),
        "fusion": int(fusion),
        "decoder": int(decoder),
    }
    total = sum(components.values())

    return {
        "scope": "cached_trainable_pipeline",
        "in_channels": c,
        "fusion_dim": d,
        "adapter_reduction": int(adapter_reduction),
        "adapter_bottleneck_channels": int(bottleneck),
        "adapter_kernel_size": k,
        "num_blocks": bcount,
        "num_sources": num_sources,
        "share_projection_across_views": bool(share_projection_across_views),
        "decoder_hidden_channels": hidden,
        "components": components,
        "total_parameters": int(total),
        "expected_trainable_parameters": int(total),
        "backbone_included": False,
    }


def _actual_cached_msila_components(report: Mapping[str, Any]) -> dict[str, int]:
    """Map current CachedFeatureTrainingModel names to manual components."""
    out = {
        "adapters": 0,
        "aligner": 0,
        "projection": 0,
        "fusion": 0,
        "decoder": 0,
        "other": 0,
    }

    for row in report.get("parameters", []):
        name = str(row["canonical_name"])
        n = int(row["numel"])

        if name.startswith("adapters."):
            out["adapters"] += n
        elif name.startswith("aligner."):
            out["aligner"] += n
        elif name.startswith("projection."):
            out["projection"] += n
        elif name.startswith("head.fusion."):
            out["fusion"] += n
        elif name.startswith("head.decoder."):
            out["decoder"] += n
        else:
            out["other"] += n

    return out


def verify_cached_msila_parameter_count(
    model: nn.Module,
    *,
    candidate_id: str,
    in_channels: int,
    fusion_dim: int,
    adapter_reduction: int = 4,
    adapter_bottleneck_channels: int | None = None,
    adapter_kernel_size: int = 3,
    num_blocks: int = 3,
    share_projection_across_views: bool = True,
    decoder_hidden_channels: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hard E4 verification for the current cached MS-ILA trainable pipeline.

    PASS requires:
      1. model total == closed-form total;
      2. model trainable == closed-form total;
      3. every named architecture component matches the closed-form component;
      4. no unexpected parameter-bearing component exists.
    """
    report = parameter_report(
        model,
        candidate_id=candidate_id,
        scope="cached_trainable_pipeline",
    )
    manual = manual_cached_msila_parameter_formula(
        in_channels=in_channels,
        fusion_dim=fusion_dim,
        adapter_reduction=adapter_reduction,
        adapter_bottleneck_channels=adapter_bottleneck_channels,
        adapter_kernel_size=adapter_kernel_size,
        num_blocks=num_blocks,
        share_projection_across_views=share_projection_across_views,
        decoder_hidden_channels=decoder_hidden_channels,
    )

    assert_parameter_count(
        report,
        expected_total=int(manual["total_parameters"]),
        expected_trainable=int(manual["expected_trainable_parameters"]),
    )

    actual_components = _actual_cached_msila_components(report)
    expected_components = dict(manual["components"])

    mismatches: list[str] = []
    for name, expected in expected_components.items():
        actual = int(actual_components[name])
        if actual != int(expected):
            mismatches.append(f"{name}: actual={actual}, expected={expected}")

    if actual_components["other"] != 0:
        mismatches.append(
            f"other: unexpected parameter count={actual_components['other']}"
        )

    if mismatches:
        raise ParameterCountError(
            "Component-level manual parameter check FAILED: "
            + "; ".join(mismatches)
        )

    result = {
        "schema_version": "msila.e4.manual_check.v1",
        "candidate_id": str(candidate_id),
        "status": "PASS",
        "report": report,
        "manual_formula": manual,
        "actual_components": actual_components,
        "manual_check": {
            "total_match": True,
            "trainable_match": True,
            "component_match": True,
            "unexpected_parameters": 0,
        },
    }

    if output_path is not None:
        _atomic_json_dump(result, output_path)

    return result

# =====================================================================
# E5 — LATENCY
# E6 — PEAK VRAM
# =====================================================================

class BenchmarkProtocolError(RuntimeError):
    """Raised when E5/E6 benchmark metadata or protocol is invalid."""


class CUDAUnavailableError(BenchmarkProtocolError):
    """Raised when a CUDA-only E6 measurement is requested without CUDA."""


def _resolve_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)

    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise CUDAUnavailableError(
                f"CUDA benchmark requested for {dev}, but torch.cuda.is_available() is False"
            )
        index = torch.cuda.current_device() if dev.index is None else int(dev.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise BenchmarkProtocolError(
                f"CUDA device index {index} outside available range "
                f"[0, {torch.cuda.device_count() - 1}]"
            )
        return torch.device(f"cuda:{index}")

    if dev.type != "cpu":
        raise BenchmarkProtocolError(
            "E5 supports CPU/CUDA timing; E6 requires CUDA. "
            f"Received device type {dev.type!r}."
        )

    return dev


def _sync_device(device: torch.device) -> None:
    """Synchronize CUDA work; CPU execution is already synchronous here."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile_linear(values: list[float], q: float) -> float:
    """Linear percentile with the same interpolation used by the repo benchmark."""
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")

    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]

    alpha = pos - lo
    return xs[lo] * (1.0 - alpha) + xs[hi] * alpha


def _coefficient_of_variation(values: list[float]) -> float:
    """Population standard deviation / mean; 0 for one identical run."""
    if not values:
        raise ValueError("values must be non-empty")
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return 0.0 if all(v == 0.0 for v in values) else math.inf
    return statistics.pstdev(values) / mean


def _canonical_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _canonical_jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "Benchmark scope metadata must be JSON-serializable; "
        f"unsupported type {type(value)!r}"
    )


def make_benchmark_scope(
    *,
    scope_name: str,
    device: torch.device | str,
    batch_size: int,
    precision: str,
    input_signature: str,
    pipeline_stages: tuple[str, ...] | list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create immutable-style benchmark provenance plus a SHA-256 fingerprint.

    The fingerprint intentionally excludes candidate-specific hyperparameters.
    Two candidates are directly comparable only when the same benchmark scope
    fingerprint is used.
    """
    scope_name = _require_nonempty_string(scope_name, name="scope_name")
    precision = _require_nonempty_string(precision, name="precision")
    input_signature = _require_nonempty_string(
        input_signature, name="input_signature"
    )

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be int")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if not isinstance(pipeline_stages, (tuple, list)) or not pipeline_stages:
        raise ValueError("pipeline_stages must be a non-empty tuple/list")

    stages = [
        _require_nonempty_string(stage, name="pipeline stage")
        for stage in pipeline_stages
    ]

    dev = _resolve_device(device)
    payload: dict[str, Any] = {
        "scope_name": scope_name,
        "device": str(dev),
        "batch_size": int(batch_size),
        "precision": precision,
        "input_signature": input_signature,
        "pipeline_stages": stages,
        "extra": {} if extra is None else _canonical_jsonable(extra),
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _inference_context(enabled: bool):
    return torch.inference_mode() if enabled else contextlib.nullcontext()


def _call_and_release(fn):
    result = fn()
    return result


def benchmark_latency(
    fn,
    *,
    device: torch.device | str,
    warmup: int = 10,
    iterations: int = 50,
    rounds: int = 3,
    use_inference_mode: bool = True,
    stability_cv_threshold: float = 0.10,
) -> dict[str, Any]:
    """Measure E5 latency using warm-up + synchronized repeated timing.

    Stability PASS is a project QA convention, not a universal hardware law:
    coefficient of variation of the *round medians* must be <= the pre-declared
    threshold.  The default is 10%; projects may lock a stricter threshold
    before seeing candidate results.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be an integer >= 0")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be an integer > 0")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2:
        raise ValueError("rounds must be an integer >= 2 for repeated-run stability")
    if (
        not isinstance(stability_cv_threshold, (int, float))
        or isinstance(stability_cv_threshold, bool)
        or not math.isfinite(float(stability_cv_threshold))
        or float(stability_cv_threshold) <= 0.0
    ):
        raise ValueError("stability_cv_threshold must be finite and > 0")

    dev = _resolve_device(device)
    all_samples_ms: list[float] = []
    round_medians_ms: list[float] = []

    with _inference_context(bool(use_inference_mode)):
        for _round in range(rounds):
            # Warm-up is deliberately outside the timed region.
            for _ in range(warmup):
                warm = _call_and_release(fn)
                del warm
            _sync_device(dev)

            samples_ms: list[float] = []
            for _ in range(iterations):
                # CUDA launches are asynchronous.  Synchronizing both sides makes
                # the wall-clock interval correspond to completed device work.
                _sync_device(dev)
                start_ns = time.perf_counter_ns()

                result = _call_and_release(fn)

                _sync_device(dev)
                end_ns = time.perf_counter_ns()
                del result

                elapsed_ms = (end_ns - start_ns) / 1_000_000.0
                if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
                    raise BenchmarkProtocolError(
                        f"invalid latency sample: {elapsed_ms}"
                    )
                samples_ms.append(float(elapsed_ms))

            all_samples_ms.extend(samples_ms)
            round_medians_ms.append(float(statistics.median(samples_ms)))

    mean_ms = float(statistics.fmean(all_samples_ms))
    median_ms = float(statistics.median(all_samples_ms))
    std_ms = float(statistics.pstdev(all_samples_ms))
    p95_ms = float(_percentile_linear(all_samples_ms, 0.95))
    round_median_cv = float(_coefficient_of_variation(round_medians_ms))
    stable = bool(round_median_cv <= float(stability_cv_threshold))

    return {
        "schema_version": "msila.e5.latency.v1",
        "status": "PASS" if stable else "FAIL",
        "device": str(dev),
        "protocol": {
            "warmup_per_round": int(warmup),
            "iterations_per_round": int(iterations),
            "rounds": int(rounds),
            "total_timed_iterations": int(rounds * iterations),
            "clock": "time.perf_counter_ns",
            "cuda_sync_before_each_timing": bool(dev.type == "cuda"),
            "cuda_sync_after_each_timing": bool(dev.type == "cuda"),
            "inference_mode": bool(use_inference_mode),
            "stability_rule": "CV(round_medians) <= threshold",
            "stability_cv_threshold": float(stability_cv_threshold),
        },
        "latency_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "p95": p95_ms,
            "std": std_ms,
            "min": float(min(all_samples_ms)),
            "max": float(max(all_samples_ms)),
        },
        "stability": {
            "round_medians_ms": round_medians_ms,
            "round_median_cv": round_median_cv,
            "status": "PASS" if stable else "FAIL",
        },
    }


def _bytes_report(value: int) -> dict[str, float | int]:
    value = int(value)
    return {
        "bytes": value,
        "MiB": float(value / (1024 ** 2)),
        "GiB": float(value / (1024 ** 3)),
    }


def benchmark_peak_vram(
    fn,
    *,
    device: torch.device | str,
    warmup: int = 10,
    iterations: int = 1,
    use_inference_mode: bool = True,
) -> dict[str, Any]:
    """Measure E6 peak CUDA tensor memory for one benchmark scope.

    Primary metric:
        ``peak_allocated`` = torch.cuda.max_memory_allocated(device)

    We also record ``peak_reserved`` because PyTorch uses a caching allocator,
    but reserved memory is allocator state and is not used as the primary E6
    model-VRAM number.

    ``incremental_peak_allocated`` is:
        peak_allocated - allocated_baseline_after_warmup

    Absolute ``peak_allocated`` includes model/input tensors already resident on
    the device at the reset point, which is generally the useful "peak occupied
    by PyTorch tensors during inference" number for candidate comparison.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be an integer >= 0")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be an integer > 0")

    dev = _resolve_device(device)
    if dev.type != "cuda":
        raise CUDAUnavailableError(
            "E6 peak VRAM is CUDA-only because it uses torch.cuda allocator "
            f"statistics; received device={dev}"
        )

    with torch.cuda.device(dev):
        with _inference_context(bool(use_inference_mode)):
            # Warm-up creates lazy CUDA state/kernels/temporary allocator state
            # before the actual peak-tracking window.
            for _ in range(warmup):
                warm = _call_and_release(fn)
                _sync_device(dev)
                del warm

            _sync_device(dev)

            # Do not call empty_cache() here: the benchmark should observe the
            # allocator/model state reached after the same warm-up protocol.
            torch.cuda.reset_peak_memory_stats(dev)

            baseline_allocated = int(torch.cuda.memory_allocated(dev))
            baseline_reserved = int(torch.cuda.memory_reserved(dev))

            for _ in range(iterations):
                result = _call_and_release(fn)
                _sync_device(dev)
                del result

            _sync_device(dev)

            peak_allocated = int(torch.cuda.max_memory_allocated(dev))
            peak_reserved = int(torch.cuda.max_memory_reserved(dev))
            end_allocated = int(torch.cuda.memory_allocated(dev))
            end_reserved = int(torch.cuda.memory_reserved(dev))

    if peak_allocated < baseline_allocated:
        raise BenchmarkProtocolError(
            "peak allocated memory cannot be smaller than baseline allocated memory"
        )
    if peak_reserved < baseline_reserved:
        raise BenchmarkProtocolError(
            "peak reserved memory cannot be smaller than baseline reserved memory"
        )

    return {
        "schema_version": "msila.e6.peak_vram.v1",
        "status": "PASS",
        "device": str(dev),
        "protocol": {
            "warmup": int(warmup),
            "measured_iterations": int(iterations),
            "inference_mode": bool(use_inference_mode),
            "peak_reset": "torch.cuda.reset_peak_memory_stats",
            "primary_metric": "torch.cuda.max_memory_allocated",
            "empty_cache_before_measurement": False,
        },
        "memory": {
            "baseline_allocated": _bytes_report(baseline_allocated),
            "peak_allocated": _bytes_report(peak_allocated),
            "incremental_peak_allocated": _bytes_report(
                peak_allocated - baseline_allocated
            ),
            "end_allocated": _bytes_report(end_allocated),
            "baseline_reserved": _bytes_report(baseline_reserved),
            "peak_reserved": _bytes_report(peak_reserved),
            "incremental_peak_reserved": _bytes_report(
                peak_reserved - baseline_reserved
            ),
            "end_reserved": _bytes_report(end_reserved),
        },
        "interpretation": {
            "primary_peak_vram_field": "memory.peak_allocated",
            "allocated_means": "GPU memory occupied by PyTorch tensors",
            "reserved_means": "GPU memory managed by PyTorch caching allocator",
        },
    }


def benchmark_inference_efficiency(
    fn,
    *,
    candidate_id: str,
    scope: Mapping[str, Any],
    output_path: str | Path | None = None,
    latency_warmup: int = 10,
    latency_iterations: int = 50,
    latency_rounds: int = 3,
    stability_cv_threshold: float = 0.10,
    vram_warmup: int = 10,
    vram_iterations: int = 1,
    use_inference_mode: bool = True,
) -> dict[str, Any]:
    """Run E5 and E6 over the exact same callable and scope metadata.

    This is the preferred Day-04 API when a CUDA GPU is available.
    """
    candidate_id = _require_nonempty_string(candidate_id, name="candidate_id")
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not isinstance(scope, Mapping):
        raise TypeError("scope must be a mapping returned by make_benchmark_scope")

    required_scope = {
        "scope_name",
        "device",
        "batch_size",
        "precision",
        "input_signature",
        "pipeline_stages",
        "fingerprint_sha256",
    }
    missing = sorted(required_scope.difference(scope.keys()))
    if missing:
        raise BenchmarkProtocolError(f"scope missing required fields: {missing}")

    scope_copy = _canonical_jsonable(dict(scope))
    dev = _resolve_device(scope_copy["device"])

    if dev.type != "cuda":
        raise CUDAUnavailableError(
            "Combined E5+E6 benchmark requires CUDA. "
            "Use benchmark_latency() separately for CPU timing."
        )

    latency = benchmark_latency(
        fn,
        device=dev,
        warmup=latency_warmup,
        iterations=latency_iterations,
        rounds=latency_rounds,
        use_inference_mode=use_inference_mode,
        stability_cv_threshold=stability_cv_threshold,
    )

    vram = benchmark_peak_vram(
        fn,
        device=dev,
        warmup=vram_warmup,
        iterations=vram_iterations,
        use_inference_mode=use_inference_mode,
    )

    overall = (
        "PASS"
        if latency["status"] == "PASS" and vram["status"] == "PASS"
        else "FAIL"
    )

    report = {
        "schema_version": "msila.e5_e6.efficiency.v1",
        "candidate_id": candidate_id,
        "status": overall,
        "scope": scope_copy,
        "scope_fingerprint_sha256": scope_copy["fingerprint_sha256"],
        "latency": latency,
        "peak_vram": vram,
        "comparability_rule": (
            "Compare candidates only when scope_fingerprint_sha256 and "
            "benchmark protocol are identical."
        ),
    }

    if output_path is not None:
        _atomic_json_dump(report, output_path)

    return report
