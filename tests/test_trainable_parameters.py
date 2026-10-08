"""MS-ILA Day-05 trainable-parameter / gradient ownership gate.

Project-relative path:
    msila/tests/test_trainable_parameters.py

Purpose
-------
Audit the ACTUAL DINOv3 checkpoint plus the current Day-05 downstream modules:

    frozen DINOv3
      -> ResidualAdapter2d
      -> ContextToLocalAligner (R2 only)
      -> SixFeatureProjection
      -> FeatureSelector
      -> parameter-free MeanFusion
      -> BasicDecoder
      -> AnomalySegmentationLoss
      -> backward

This test answers a different question from representation quality. Random
normalized image tensors are acceptable here because the scientific gate is
parameter ownership / autograd connectivity, not anomaly-localization quality.
The backbone itself is real and loaded from the real local DINOv3 repository and
checkpoint.

Required environment
--------------------
DINOV3_REPO=/content/dinov3_repo
DINOV3_WEIGHTS=/path/to/dinov3_vits16_checkpoint.pth
MSILA_DAY04_ADAPTER_R=<locked Day-04 bottleneck_dim>
MSILA_DAY04_ADAPTER_D=<locked Day-04 projection_dim>

The r,d environment variables are intentionally required when the numeric Day-04
winner is not stored in ``configs/day05_representation.yaml``. We do NOT guess a
winner from an old QA/example value.

Output
------
    msila/tests/artifacts/trainable_parameter_report.json

Run
---
    pytest -q -s tests/test_trainable_parameters.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.anomaly_loss import AnomalySegmentationLoss  # noqa: E402
from src.models.basic_decoder import BasicDecoder  # noqa: E402
from src.models.context_alignment import ContextToLocalAligner  # noqa: E402
from src.models.dinov3_extractor import build_online_extractor  # noqa: E402
from src.models.feature_projection import SixFeatureProjection  # noqa: E402
from src.models.feature_selector import FeatureSelector  # noqa: E402
from src.models.mean_fusion import MeanFusion  # noqa: E402
from src.models.residual_adapter import ResidualAdapter2d  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DAY05_CONFIG = ROOT / "configs" / "day05_representation.yaml"
REPORT_PATH = ROOT / "tests" / "artifacts" / "trainable_parameter_report.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        pytest.fail(f"Missing required config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        pytest.fail(f"YAML root must be a mapping: {path}")
    return payload


def _resolve_dinov3_repo() -> Path:
    explicit = os.getenv("DINOV3_REPO")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "hubconf.py").is_file():
            pytest.fail(f"DINOV3_REPO is not a DINOv3 checkout: {path}")
        return path

    candidates = [Path("/content/dinov3_repo"), Path("/content/dinov3")]
    valid = [p.resolve() for p in candidates if (p / "hubconf.py").is_file()]
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        pytest.fail(
            "Multiple DINOv3 repositories found. Set DINOV3_REPO explicitly: "
            + ", ".join(str(p) for p in valid)
        )
    pytest.skip("Real DINO source missing; set DINOV3_REPO")


def _resolve_dinov3_weights(model_name: str) -> Path:
    explicit = os.getenv("DINOV3_WEIGHTS")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            pytest.fail(f"DINOV3_WEIGHTS does not exist: {path}")
        return path

    roots = [ROOT / "weights", Path("/content/checkpoints")]
    patterns = [f"*{model_name}*.pth", "*dinov3*vits16*.pth"]
    found: set[Path] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for pattern in patterns:
            found.update(p.resolve() for p in base.rglob(pattern) if p.is_file())

    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        pytest.fail(
            "Multiple DINOv3 checkpoints found. Set DINOV3_WEIGHTS explicitly:\n"
            + "\n".join(str(p) for p in sorted(found))
        )
    pytest.fail("Set DINOV3_WEIGHTS=/path/to/dinov3_vits16_checkpoint.pth.")


def _resolve_adapter_dims(day05_cfg: dict[str, Any]) -> tuple[int, int, str]:
    """Resolve the already-selected Day-04 adapter without inventing r,d."""

    adapter_cfg = day05_cfg["locked"]["adapter"]
    if "bottleneck_dim" in adapter_cfg and "projection_dim" in adapter_cfg:
        r = int(adapter_cfg["bottleneck_dim"])
        d = int(adapter_cfg["projection_dim"])
        source = "configs/day05_representation.yaml"
    else:
        raw_r = os.getenv("MSILA_DAY04_ADAPTER_R")
        raw_d = os.getenv("MSILA_DAY04_ADAPTER_D")
        if raw_r is None or raw_d is None:
            pytest.fail(
                "Day-05 config references 'day04_locked_selected_candidate' but "
                "does not store its numeric r,d. Set both "
                "MSILA_DAY04_ADAPTER_R and MSILA_DAY04_ADAPTER_D to the locked "
                "Day-04 winner. The test intentionally refuses to guess values."
            )
        try:
            r, d = int(raw_r), int(raw_d)
        except ValueError as exc:
            pytest.fail("MSILA_DAY04_ADAPTER_R/D must be positive integers.")
            raise AssertionError from exc
        source = "environment"

    if r <= 0 or d <= 0:
        pytest.fail(f"Adapter dimensions must be positive, got r={r}, d={d}.")
    return r, d, source


def _parameter_stats(module: torch.nn.Module) -> dict[str, Any]:
    named = list(module.named_parameters())
    trainable = [(n, p) for n, p in named if p.requires_grad]
    grads = [(n, p) for n, p in trainable if p.grad is not None]

    finite_grad = True
    nonzero_grad_params = 0
    for _, p in grads:
        finite_grad = finite_grad and bool(torch.isfinite(p.grad).all())
        if bool((p.grad != 0).any()):
            nonzero_grad_params += 1

    return {
        "parameter_tensors": len(named),
        "total_params": int(sum(p.numel() for _, p in named)),
        "trainable_parameter_tensors": len(trainable),
        "trainable_params": int(sum(p.numel() for _, p in trainable)),
        "params_with_grad": len(grads),
        "params_without_grad": len(trainable) - len(grads),
        "nonzero_grad_parameter_tensors": nonzero_grad_params,
        "finite_grad": bool(finite_grad),
    }


def _assert_trainable(module: torch.nn.Module, *, name: str) -> None:
    params = list(module.parameters())
    assert params, f"{name} unexpectedly has no parameters."
    assert all(p.requires_grad for p in params), f"{name} has frozen parameters."


def _assert_parameter_free(module: torch.nn.Module, *, name: str) -> None:
    assert list(module.parameters()) == [], f"{name} must be parameter-free."


def _assert_active_grads(params: Iterable[torch.nn.Parameter], *, name: str) -> None:
    params = list(params)
    assert params, f"{name}: expected trainable parameters."
    for p in params:
        assert p.requires_grad, f"{name}: active parameter is frozen."
        assert p.grad is not None, f"{name}: active parameter is disconnected."
        assert torch.isfinite(p.grad).all(), f"{name}: NaN/Inf gradient."


def _nested_context_geometry(
    *,
    batch: int,
    local_input_size: int,
    context_source_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Centered Local 512 inside Context 768, both fed to DINO at 512.

    If Local source coordinate is x_L, Context-model coordinate is
        x_C = scale * (x_L + margin),
    where scale = model_input/context_source.
    """

    local_source_size = local_input_size
    margin = (context_source_size - local_source_size) / 2.0
    scale = local_input_size / float(context_source_size)
    offset = scale * margin

    matrix = torch.tensor(
        [
            [scale, 0.0, offset],
            [0.0, scale, offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return {
        "local_to_context": matrix.unsqueeze(0).repeat(batch, 1, 1),
        "local_input_hw": torch.tensor(
            [[float(local_input_size), float(local_input_size)]],
            device=device,
        ).repeat(batch, 1),
        "context_input_hw": torch.tensor(
            [[float(local_input_size), float(local_input_size)]],
            device=device,
        ).repeat(batch, 1),
    }


def _active_blocks(candidate_id: str) -> tuple[int, ...]:
    return (12,) if candidate_id == "R0" else (4, 8, 12)


def _audit_candidate(
    *,
    candidate_id: str,
    online: dict[str, torch.Tensor],
    in_channels: int,
    fusion_dim: int,
    adapter_r: int,
    adapter_d: int,
    adapter_kernel: int,
    gamma_init: float,
    context_source_size: int,
    model_input_size: int,
    device: torch.device,
    day05_cfg: dict[str, Any],
) -> dict[str, Any]:
    selector = FeatureSelector.from_config(day05_cfg, candidate_id)
    active_blocks = _active_blocks(candidate_id)

    adapters = torch.nn.ModuleDict(
        {
            f"b{block}": ResidualAdapter2d(
                in_dim=in_channels,
                bottleneck_dim=adapter_r,
                projection_dim=adapter_d,
                kernel_size=adapter_kernel,
                gamma_init=gamma_init,
                bias=True,
            )
            for block in (4, 8, 12)
        }
    ).to(device)
    projection = SixFeatureProjection(
        in_channels=in_channels,
        fusion_dim=fusion_dim,
        blocks=(4, 8, 12),
        share_across_views=True,
        check_finite=True,
    ).to(device)
    aligner = ContextToLocalAligner(
        check_finite=True,
        check_bounds=True,
    ).to(device)
    fusion = MeanFusion(validate=True, return_weights=True).to(device)
    decoder = BasicDecoder(in_channels=fusion_dim).to(device)
    criterion = AnomalySegmentationLoss(
        bce_weight=1.0,
        dice_weight=1.0,
    ).to(device)

    _assert_trainable(adapters, name="adapters")
    _assert_trainable(projection, name="projection")
    _assert_trainable(decoder, name="decoder")
    _assert_parameter_free(aligner, name="ContextToLocalAligner")
    _assert_parameter_free(fusion, name="MeanFusion")
    _assert_parameter_free(selector, name="FeatureSelector")

    local = {
        f"L{block}": adapters[f"b{block}"](online[f"L{block}"])
        for block in active_blocks
    }

    aligned_context = None
    if candidate_id == "R2":
        context = {
            f"C{block}": adapters[f"b{block}"](online[f"C{block}"])
            for block in active_blocks
        }
        h, w = local["L4"].shape[-2:]
        geometry = _nested_context_geometry(
            batch=int(local["L4"].shape[0]),
            local_input_size=model_input_size,
            context_source_size=context_source_size,
            device=device,
        )
        aligned_context = aligner(
            context,
            geometry,
            target_hw=(int(h), int(w)),
        )

    projected = projection.project_sources(
        local,
        aligned_context,
        source_keys=selector.source_keys,
    )
    for tensor in projected.values():
        tensor.retain_grad()

    selected = selector(projected)
    fused, mean_weights = fusion(selected)
    logits = decoder(fused, output_size=(model_input_size, model_input_size))

    target = torch.zeros_like(logits)
    s0 = model_input_size // 3
    s1 = 2 * model_input_size // 3
    target[..., s0:s1, s0:s1] = 1.0

    loss_out = criterion(logits, target)
    loss = loss_out["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    # MeanFusion has no parameters, so its correct gradient contract is to pass
    # gradients to every selected source.
    source_gradient_report: dict[str, Any] = {}
    for key, tensor in projected.items():
        assert tensor.grad is not None, f"{candidate_id}:{key} has no gradient."
        assert torch.isfinite(tensor.grad).all(), f"{candidate_id}:{key} grad NaN/Inf."
        source_gradient_report[key] = {
            "grad_present": True,
            "finite": True,
            "nonzero_elements": int((tensor.grad != 0).sum().item()),
        }

    # Adapter: only blocks used by this representation are connected.
    active_adapter_report: dict[str, Any] = {}
    inactive_adapter_report: dict[str, Any] = {}
    for block in (4, 8, 12):
        module = adapters[f"b{block}"]
        params = list(module.parameters())
        if block in active_blocks:
            _assert_active_grads(params, name=f"adapter.b{block}")
            active_adapter_report[f"b{block}"] = _parameter_stats(module)
        else:
            assert all(p.grad is None for p in params)
            inactive_adapter_report[f"b{block}"] = _parameter_stats(module)

    # Projection has a common projector per DINO block; only selected blocks are
    # expected to receive gradients for a given representation.
    assert projection.projectors is not None
    selected_blocks = {int(key.rsplit("b", 1)[1]) for key in selector.source_keys}
    projection_branches: dict[str, Any] = {}
    for block in (4, 8, 12):
        module = projection.projectors[f"b{block}"]
        params = list(module.parameters())
        if block in selected_blocks:
            _assert_active_grads(params, name=f"projection.b{block}")
            state = "active"
        else:
            assert all(p.grad is None for p in params)
            state = "inactive"
        projection_branches[f"b{block}"] = {
            "state": state,
            **_parameter_stats(module),
        }

    _assert_active_grads(decoder.parameters(), name="decoder")

    return {
        "status": "PASS",
        "mode": selector.mode,
        "sources": list(selector.source_keys),
        "num_sources": selector.num_sources,
        "loss": float(loss.detach().cpu()),
        "loss_terms": {
            "bce": float(loss_out["bce"].detach().cpu()),
            "dice": float(loss_out["dice"].detach().cpu()),
            "positive_samples": int(loss_out["positive_samples"].detach().cpu()),
        },
        "adapter": {
            "all_parameters_require_grad": all(
                p.requires_grad for p in adapters.parameters()
            ),
            "active_blocks": list(active_blocks),
            "active": active_adapter_report,
            "inactive": inactive_adapter_report,
        },
        "projection": {
            "all_parameters_require_grad": all(
                p.requires_grad for p in projection.parameters()
            ),
            "branches": projection_branches,
            "overall": _parameter_stats(projection),
        },
        "mean_fusion": {
            "parameter_free": len(list(fusion.parameters())) == 0,
            "fixed_weights": [float(v) for v in mean_weights.detach().cpu()],
            "gradient_passthrough": source_gradient_report,
        },
        "decoder": {
            "all_parameters_require_grad": all(
                p.requires_grad for p in decoder.parameters()
            ),
            **_parameter_stats(decoder),
        },
        "alignment": {
            "parameter_free": len(list(aligner.parameters())) == 0,
            "used": candidate_id == "R2",
        },
        "feature_selector": {
            "parameter_free": len(list(selector.parameters())) == 0,
        },
    }


@pytest.mark.integration
def test_real_backbone_frozen_and_day05_gradient_ownership() -> None:
    default_cfg = _load_yaml(DEFAULT_CONFIG)
    day05_cfg = _load_yaml(DAY05_CONFIG)

    backbone_cfg = default_cfg["backbone"]
    input_cfg = default_cfg["input"]
    assert bool(backbone_cfg["frozen"]) is True

    model_name = str(backbone_cfg["name"])
    blocks = tuple(int(v) for v in backbone_cfg["feature_blocks"])
    assert blocks == (4, 8, 12)

    model_input_size = int(input_cfg["tile_size"])
    context_source_size = int(input_cfg["context_size"])
    fusion_dim = int(day05_cfg["locked"]["projection"]["fusion_dim"])
    adapter_kernel = int(day05_cfg["locked"]["adapter"]["kernel_size"])
    gamma_init = float(day05_cfg["locked"]["adapter"]["gamma_init"])
    # Check real assets before requiring numeric selection fields. Missing
    # assets are SKIP; explicitly supplied bad paths/configs still FAIL.
    repo = _resolve_dinov3_repo()
    weights = _resolve_dinov3_weights(model_name)
    adapter_r, adapter_d, adapter_dim_source = _resolve_adapter_dims(day05_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = _resolve_dinov3_repo()
    weights = _resolve_dinov3_weights(model_name)

    extractor = build_online_extractor(
        repo_dir=repo,
        weights=weights,
        device=device,
        model_name=model_name,
        blocks=blocks,
        norm=True,
        check_finite=True,
    )

    # Hard backbone freeze gate.
    backbone_params = list(extractor.backbone.parameters())
    assert backbone_params, "DINOv3 backbone unexpectedly exposes no parameters."
    assert all(not p.requires_grad for p in backbone_params)
    assert extractor.backbone.training is False

    # Parent .train() must never switch the DINO backbone back to training mode.
    extractor.train(True)
    assert extractor.training is True
    assert extractor.backbone.training is False
    assert all(not p.requires_grad for p in backbone_params)

    # One deterministic synthetic normalized pair is sufficient for autograd
    # ownership. The backbone/checkpoint is real; no claim about representation
    # quality is made by this test.
    torch.manual_seed(int(day05_cfg["locked"]["training"]["seed"]))
    x_local = torch.randn(
        1,
        3,
        model_input_size,
        model_input_size,
        device=device,
    )
    x_context = torch.randn_like(x_local)

    online = extractor.extract_local_context(
        x_local,
        x_context,
        strategy="concat",
    )
    assert set(online) == {"L4", "L8", "L12", "C4", "C8", "C12"}
    assert all(not tensor.requires_grad for tensor in online.values())
    assert all(torch.isfinite(tensor).all() for tensor in online.values())

    in_channels = int(online["L4"].shape[1])
    assert all(int(online[f"L{b}"].shape[1]) == in_channels for b in blocks)

    candidate_reports: dict[str, Any] = {}
    for candidate_id in ("R0", "R1", "R2"):
        # No downstream backward is allowed to populate frozen backbone grads.
        for p in backbone_params:
            p.grad = None

        candidate_reports[candidate_id] = _audit_candidate(
            candidate_id=candidate_id,
            online=online,
            in_channels=in_channels,
            fusion_dim=fusion_dim,
            adapter_r=adapter_r,
            adapter_d=adapter_d,
            adapter_kernel=adapter_kernel,
            gamma_init=gamma_init,
            context_source_size=context_source_size,
            model_input_size=model_input_size,
            device=device,
            day05_cfg=day05_cfg,
        )

        assert all(p.grad is None for p in backbone_params), (
            f"{candidate_id}: frozen DINOv3 received a gradient."
        )

    backbone_report = _parameter_stats(extractor.backbone)
    assert backbone_report["trainable_params"] == 0
    assert backbone_report["params_with_grad"] == 0

    report = {
        "gate": "MS-ILA Day05 trainable-parameter / gradient ownership",
        "status": "PASS",
        "default_config": "msila/configs/default.yaml",
        "day05_config": "msila/configs/day05_representation.yaml",
        "device": str(device),
        "backbone": {
            "name": model_name,
            "blocks": list(blocks),
            "repo": str(repo),
            "weights": str(weights),
            "training_mode": bool(extractor.backbone.training),
            "frozen": extractor.backbone_is_frozen(),
            **backbone_report,
        },
        "adapter_lock": {
            "r_bottleneck_dim": adapter_r,
            "d_projection_dim": adapter_d,
            "kernel_size": adapter_kernel,
            "gamma_init": gamma_init,
            "resolved_from": adapter_dim_source,
        },
        "projection": {
            "fusion_dim": fusion_dim,
            "share_across_views": bool(
                day05_cfg["locked"]["projection"]["share_across_views"]
            ),
        },
        "fusion": {
            "type": day05_cfg["locked"]["fusion"]["type"],
            "trainable": day05_cfg["locked"]["fusion"]["trainable"],
            "interpretation": (
                "MeanFusion is parameter-free; the gate checks gradient "
                "passthrough to every selected source rather than parameter grads."
            ),
        },
        "candidates": candidate_reports,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REPORT_PATH)

    assert REPORT_PATH.is_file()
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"
    assert saved["backbone"]["frozen"] is True
    assert saved["backbone"]["trainable_params"] == 0
    assert saved["backbone"]["params_with_grad"] == 0
    assert all(saved["candidates"][c]["status"] == "PASS" for c in ("R0", "R1", "R2"))
