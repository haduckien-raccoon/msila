"""Cumulative acceptance tests for MS-ILA Day 05.

Project-relative path:
    msila/tests/test_day05.py

Purpose
-------
This is the single cumulative Day-05 test file.  Keep adding new test sections
here as Day-05 modules are implemented; do not replace earlier acceptance
checks unless the locked scientific protocol itself changes.

Current coverage
----------------
TV1-A Framework:
    1. configs/day05_representation.yaml
    2. src/models/feature_selector.py
    3. src/models/context_alignment.py (existing Day-2 module, regression gate)
    4. src/models/feature_projection.py (Day-05 flexible source mapping)
    5. src/models/mean_fusion.py (same fusion for 1/3/6 sources)
    6. tests/test_representation_shapes.py (real-batch integration gate)

Run from the ``msila`` repository root:

    pytest -q tests/test_day05.py

Scientific contract
-------------------
Day 05 has exactly three representation candidates and only the representation
source may change:

    R0 = {L12}
    R1 = {L4, L8, L12}
    R2 = {L4, L8, L12, C4, C8, C12}

All other locked components (input geometry, backbone, Adapter choice,
projection policy, mean fusion, decoder and loss) must be common to R0/R1/R2.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Project imports / paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # .../msila
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.basic_decoder import BasicDecoder  # noqa: E402
from src.models.context_alignment import ContextToLocalAligner  # noqa: E402
from src.models.dinov3_extractor import DINOv3FeatureExtractor  # noqa: E402
from src.models.feature_projection import (  # noqa: E402
    CANONICAL_SOURCE_ORDER,
    SOURCE_TO_RAW,
    SixFeatureProjection,
)
from src.models.mean_fusion import MeanFusion  # noqa: E402
from src.models.residual_adapter import ResidualAdapter2d  # noqa: E402
from src.losses.anomaly_loss import AnomalySegmentationLoss  # noqa: E402
from src.models.feature_selector import (  # noqa: E402
    CANDIDATE_TO_MODE,
    MODE_TO_KEYS,
    FeatureSelectionError,
    FeatureSelector,
)


DAY05_CONFIG = ROOT / "configs" / "day05_representation.yaml"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

EXPECTED = {
    "R0": {
        "mode": "deep_only",
        "sources": ("local_b12",),
    },
    "R1": {
        "mode": "multi_local",
        "sources": ("local_b4", "local_b8", "local_b12"),
    },
    "R2": {
        "mode": "multi_local_context",
        "sources": (
            "local_b4",
            "local_b8",
            "local_b12",
            "context_b4",
            "context_b8",
            "context_b12",
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Missing required config: {path}"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"YAML root must be a mapping: {path}"
    return payload


@pytest.fixture(scope="module")
def day05_cfg() -> dict[str, Any]:
    return _load_yaml(DAY05_CONFIG)


@pytest.fixture(scope="module")
def default_cfg() -> dict[str, Any]:
    return _load_yaml(DEFAULT_CONFIG)


def _feature_bank(
    *,
    batch: int = 2,
    channels: int = 8,
    height: int = 5,
    width: int = 7,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build six deterministic-shape tensors for selector contract tests."""

    return {
        key: torch.randn(
            batch,
            channels,
            height,
            width,
            dtype=dtype,
            device=device,
        )
        for key in EXPECTED["R2"]["sources"]
    }


# ===========================================================================
# A1. CONFIG — configs/day05_representation.yaml
# ===========================================================================


def test_day05_config_exists_and_is_mapping(day05_cfg: dict[str, Any]) -> None:
    assert DAY05_CONFIG.is_file()
    assert day05_cfg["version"] == 1


def test_day05_points_to_canonical_default_config(
    day05_cfg: dict[str, Any],
) -> None:
    experiment = day05_cfg["experiment"]

    # Keep the project-facing path explicit exactly as the Day-05 config says.
    assert experiment["base_config"] == "msila/configs/default.yaml"
    assert experiment["only_variable"] == "representation_source"
    assert experiment["candidate_ids"] == ["R0", "R1", "R2"]

    # In the repository itself the same file is configs/default.yaml.
    assert DEFAULT_CONFIG.is_file(), (
        "day05_representation.yaml declares msila/configs/default.yaml but "
        "configs/default.yaml is missing from the msila repository."
    )


def test_day05_has_exactly_three_locked_candidates(
    day05_cfg: dict[str, Any],
) -> None:
    representations = day05_cfg["representations"]
    assert list(representations.keys()) == ["R0", "R1", "R2"]
    assert set(representations) == set(EXPECTED)


@pytest.mark.parametrize("candidate_id", ["R0", "R1", "R2"])
def test_candidate_definition_is_exact(
    day05_cfg: dict[str, Any],
    candidate_id: str,
) -> None:
    raw = day05_cfg["representations"][candidate_id]
    expected = EXPECTED[candidate_id]

    assert raw["mode"] == expected["mode"]
    assert tuple(raw["sources"]) == expected["sources"]
    assert raw["num_sources"] == len(expected["sources"])


def test_candidate_entries_contain_no_model_specific_override(
    day05_cfg: dict[str, Any],
) -> None:
    """R0/R1/R2 may describe representation only, not change model/training."""

    allowed_candidate_fields = {
        "name",
        "mode",
        "sources",
        "num_sources",
        "hypothesis",
    }

    for candidate_id, raw in day05_cfg["representations"].items():
        unexpected = set(raw) - allowed_candidate_fields
        assert not unexpected, (
            f"{candidate_id} contains candidate-specific overrides {unexpected}. "
            "Day-05 R0/R1/R2 may differ only by representation source."
        )


def test_hard_exclusions_are_enabled(day05_cfg: dict[str, Any]) -> None:
    forbidden = day05_cfg["forbidden_changes"]
    expected_keys = {
        "brute_force_layer_combinations",
        "attention_fusion",
        "illumination_loss",
        "candidate_specific_adapter",
        "candidate_specific_decoder",
        "candidate_specific_loss",
        "candidate_specific_tile_or_context_size",
    }

    assert set(forbidden) == expected_keys
    assert all(forbidden[key] is True for key in expected_keys)


def test_default_locked_input_and_backbone_are_preserved(
    day05_cfg: dict[str, Any],
    default_cfg: dict[str, Any],
) -> None:
    """Audit the fields that are explicitly locked by default.yaml."""

    locked = day05_cfg["locked"]

    for key in (
        "preserve_aspect_ratio",
        "tile_size",
        "context_size",
        "overlap",
    ):
        assert locked["input"][key] == default_cfg["input"][key]

    for key in ("name", "frozen", "feature_blocks"):
        assert locked["backbone"][key] == default_cfg["backbone"][key]


def test_locked_day05_protocol_is_clean(day05_cfg: dict[str, Any]) -> None:
    locked = day05_cfg["locked"]

    assert locked["geometry"]["context_to_local_alignment"] == "required"

    assert locked["adapter"]["source"] == "day04_locked_selected_candidate"
    assert locked["adapter"]["trainable"] is True
    assert locked["adapter"]["kernel_size"] == 3
    assert float(locked["adapter"]["gamma_init"]) == 0.0

    assert locked["projection"]["enabled"] is True
    assert locked["projection"]["fusion_dim"] == 64
    assert locked["projection"]["share_across_views"] is True

    assert locked["fusion"]["type"] == "mean"
    assert locked["fusion"]["trainable"] is False
    assert locked["fusion"]["allowed_num_sources"] == [1, 3, 6]
    assert locked["fusion"]["return_weights"] is True

    assert locked["decoder"]["type"] == "basic_decoder"
    assert locked["decoder"]["trainable"] is True

    assert locked["loss"]["type"] == "anomaly_segmentation"
    assert float(locked["loss"]["bce_weight"]) == 1.0
    assert float(locked["loss"]["dice_weight"]) == 1.0
    assert float(locked["loss"]["illumination_weight"]) == 0.0

    assert locked["training"]["seed"] == 42
    assert locked["evaluation"]["primary_metric"] == "au_pro_0.05"


# ===========================================================================
# A2. FEATURE SELECTOR — src/models/feature_selector.py
# ===========================================================================


def test_python_contract_matches_scientific_candidate_contract() -> None:
    assert CANDIDATE_TO_MODE == {
        "R0": "deep_only",
        "R1": "multi_local",
        "R2": "multi_local_context",
    }

    for candidate_id, expected in EXPECTED.items():
        mode = expected["mode"]
        assert MODE_TO_KEYS[mode] == expected["sources"]


@pytest.mark.parametrize("candidate_id", ["R0", "R1", "R2"])
def test_feature_selector_from_candidate_returns_exact_sources(
    candidate_id: str,
) -> None:
    features = _feature_bank()
    selector = FeatureSelector.from_candidate(candidate_id)

    selected = selector(features)
    expected_keys = EXPECTED[candidate_id]["sources"]

    assert selector.mode == EXPECTED[candidate_id]["mode"]
    assert selector.source_keys == expected_keys
    assert selector.num_sources == len(expected_keys)
    assert len(selected) == len(expected_keys)

    # Exact object identity proves the selector did not clone, transform,
    # interpolate, align, project, adapt, or otherwise modify feature tensors.
    for tensor, key in zip(selected, expected_keys):
        assert tensor is features[key]


@pytest.mark.parametrize("candidate_id", ["R0", "R1", "R2"])
def test_feature_selector_from_yaml_matches_candidate_contract(
    day05_cfg: dict[str, Any],
    candidate_id: str,
) -> None:
    selector = FeatureSelector.from_config(day05_cfg, candidate_id)
    assert selector.mode == EXPECTED[candidate_id]["mode"]
    assert selector.source_keys == EXPECTED[candidate_id]["sources"]
    assert selector.num_sources == len(EXPECTED[candidate_id]["sources"])


def test_r0_needs_only_local_b12() -> None:
    x = torch.randn(2, 8, 5, 7)
    selector = FeatureSelector("R0")

    selected = selector({"local_b12": x})
    assert selected == [x]


def test_r1_does_not_require_context_branch() -> None:
    features = _feature_bank()
    local_only = {k: v for k, v in features.items() if k.startswith("local_")}

    selected = FeatureSelector("R1")(local_only)
    assert len(selected) == 3


def test_unused_extra_sources_do_not_change_r0() -> None:
    features = _feature_bank()
    selected = FeatureSelector("R0")(features)

    assert len(selected) == 1
    assert selected[0] is features["local_b12"]


@pytest.mark.parametrize("candidate_id", ["R0", "R1", "R2"])
def test_missing_required_source_fails(candidate_id: str) -> None:
    features = _feature_bank()
    missing_key = EXPECTED[candidate_id]["sources"][-1]
    features.pop(missing_key)

    with pytest.raises(FeatureSelectionError, match="Missing feature source"):
        FeatureSelector(candidate_id)(features)


def test_unknown_candidate_or_mode_fails() -> None:
    with pytest.raises(FeatureSelectionError, match="Unknown representation mode"):
        FeatureSelector("R3")

    with pytest.raises(FeatureSelectionError, match="Unknown candidate"):
        FeatureSelector.from_candidate("R3")


def test_non_mapping_input_fails() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        FeatureSelector("R0")([torch.randn(1, 8, 5, 7)])  # type: ignore[arg-type]


def test_non_tensor_required_source_fails() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        FeatureSelector("R0")({"local_b12": "not-a-tensor"})  # type: ignore[dict-item]


def test_non_bchw_feature_fails() -> None:
    bad = torch.randn(2, 35, 8)  # [B,N,C], deliberately not BCHW

    with pytest.raises(FeatureSelectionError, match="BCHW"):
        FeatureSelector("R0")({"local_b12": bad})


def test_r1_shape_mismatch_fails_before_mean_fusion() -> None:
    features = _feature_bank()
    features["local_b8"] = torch.randn(2, 8, 6, 7)

    with pytest.raises(FeatureSelectionError, match="identical shapes"):
        FeatureSelector("R1")(features)


def test_r2_unaligned_context_shape_fails() -> None:
    """Scientific gate: Context must already be aligned/projected to Local."""

    features = _feature_bank(height=5, width=7)
    for key in ("context_b4", "context_b8", "context_b12"):
        features[key] = torch.randn(2, 8, 9, 11)

    with pytest.raises(FeatureSelectionError, match="Context->Local alignment"):
        FeatureSelector("R2")(features)


def test_dtype_mismatch_fails() -> None:
    features = _feature_bank(dtype=torch.float32)
    features["local_b8"] = features["local_b8"].double()

    with pytest.raises(FeatureSelectionError, match="dtype mismatch"):
        FeatureSelector("R1")(features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_mismatch_fails_when_cuda_is_available() -> None:
    features = _feature_bank(device="cpu")
    features["local_b8"] = features["local_b8"].cuda()

    with pytest.raises(FeatureSelectionError, match="device mismatch"):
        FeatureSelector("R1")(features)


def test_validate_false_only_selects_without_shape_audit() -> None:
    features = _feature_bank()
    features["local_b8"] = torch.randn(2, 8, 6, 7)

    selected = FeatureSelector("R1", validate=False)(features)
    assert len(selected) == 3
    assert selected[1] is features["local_b8"]


def test_yaml_source_drift_is_rejected(day05_cfg: dict[str, Any]) -> None:
    drifted = copy.deepcopy(day05_cfg)
    drifted["representations"]["R1"]["sources"] = [
        "local_b4",
        "local_b12",
        "local_b8",
    ]

    with pytest.raises(FeatureSelectionError, match="Config/code drift"):
        FeatureSelector.from_config(drifted, "R1")


def test_yaml_missing_representations_fails(day05_cfg: dict[str, Any]) -> None:
    drifted = copy.deepcopy(day05_cfg)
    drifted.pop("representations")

    with pytest.raises(FeatureSelectionError, match="representations"):
        FeatureSelector.from_config(drifted, "R0")


def test_yaml_candidate_mode_must_be_string(day05_cfg: dict[str, Any]) -> None:
    drifted = copy.deepcopy(day05_cfg)
    drifted["representations"]["R0"]["mode"] = None

    with pytest.raises(FeatureSelectionError, match="mode must be a string"):
        FeatureSelector.from_config(drifted, "R0")



# ===========================================================================
# A3. ALIGNMENT + PROJECTION MAPPING — existing modules, no new geometry module
# ===========================================================================


def _raw_local_context(
    *,
    batch: int = 2,
    channels: int = 12,
    height: int = 5,
    width: int = 7,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    local = {
        f"L{b}": torch.randn(batch, channels, height, width)
        for b in (4, 8, 12)
    }
    context = {
        f"C{b}": torch.randn(batch, channels, height, width)
        for b in (4, 8, 12)
    }
    return local, context


def _identity_geometry(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "local_to_context": torch.eye(3).unsqueeze(0).repeat(batch, 1, 1),
        "local_input_hw": torch.tensor([[512.0, 512.0]]).repeat(batch, 1),
        "context_input_hw": torch.tensor([[512.0, 512.0]]).repeat(batch, 1),
    }


def test_day05_projection_mapping_is_exact() -> None:
    assert CANONICAL_SOURCE_ORDER == EXPECTED["R2"]["sources"]
    assert SOURCE_TO_RAW == {
        "local_b4": ("local", 4, "L4"),
        "local_b8": ("local", 8, "L8"),
        "local_b12": ("local", 12, "L12"),
        "context_b4": ("context", 4, "C4_to_L"),
        "context_b8": ("context", 8, "C8_to_L"),
        "context_b12": ("context", 12, "C12_to_L"),
    }


def test_existing_alignment_places_context_on_local_grid() -> None:
    _, context = _raw_local_context()
    aligner = ContextToLocalAligner(check_finite=True, check_bounds=True)
    aligned = aligner(context, _identity_geometry(), target_hw=(5, 7))

    assert tuple(aligned) == ("C4_to_L", "C8_to_L", "C12_to_L")
    for x in aligned.values():
        assert x.shape == (2, 12, 5, 7)
        assert torch.isfinite(x).all()


@pytest.mark.parametrize("candidate_id", ["R0", "R1", "R2"])
def test_one_projection_module_supports_r0_r1_r2(candidate_id: str) -> None:
    local, context = _raw_local_context()
    aligner = ContextToLocalAligner(check_finite=True, check_bounds=True)
    aligned = aligner(context, _identity_geometry(), target_hw=(5, 7))

    projection = SixFeatureProjection(
        in_channels=12,
        fusion_dim=6,
        share_across_views=True,
        check_finite=True,
    )
    selector = FeatureSelector.from_candidate(candidate_id)
    projected = projection.project_sources(
        local,
        aligned if candidate_id == "R2" else None,
        source_keys=selector.source_keys,
    )

    assert tuple(projected) == EXPECTED[candidate_id]["sources"]
    assert len(projected) == len(EXPECTED[candidate_id]["sources"])
    for x in projected.values():
        assert x.shape == (2, 6, 5, 7)
        assert torch.isfinite(x).all()


def test_r0_projection_does_not_require_context() -> None:
    local, _ = _raw_local_context()
    projection = SixFeatureProjection(in_channels=12, fusion_dim=6)
    out = projection.project_sources(
        local,
        None,
        source_keys=("local_b12",),
    )
    assert tuple(out) == ("local_b12",)


def test_r1_projection_does_not_require_context() -> None:
    local, _ = _raw_local_context()
    projection = SixFeatureProjection(in_channels=12, fusion_dim=6)
    out = projection.project_sources(
        local,
        None,
        source_keys=("local_b4", "local_b8", "local_b12"),
    )
    assert len(out) == 3


def test_r2_projection_rejects_missing_alignment() -> None:
    local, _ = _raw_local_context()
    projection = SixFeatureProjection(in_channels=12, fusion_dim=6)
    with pytest.raises(KeyError, match="ContextToLocalAligner"):
        projection.project_sources(
            local,
            None,
            source_keys=EXPECTED["R2"]["sources"],
        )


def test_projection_rejects_unaligned_spatial_context() -> None:
    local, _ = _raw_local_context(height=5, width=7)
    bad_aligned = {
        f"C{b}_to_L": torch.randn(2, 12, 9, 11)
        for b in (4, 8, 12)
    }
    projection = SixFeatureProjection(in_channels=12, fusion_dim=6)
    with pytest.raises(ValueError, match="share B/H/W"):
        projection.project_sources(
            local,
            bad_aligned,
            source_keys=EXPECTED["R2"]["sources"],
        )


def test_projection_rejects_non_candidate_layer_bruteforce() -> None:
    local, _ = _raw_local_context()
    projection = SixFeatureProjection(in_channels=12, fusion_dim=6)
    with pytest.raises(ValueError, match="only accepts R0/R1/R2"):
        projection.project_sources(
            local,
            None,
            source_keys=("local_b4", "local_b12"),
        )


# ===========================================================================
# A4. FLEXIBLE MEAN FUSION — same code path for 1 / 3 / 6 sources
# ===========================================================================


@pytest.mark.parametrize("num_sources", [1, 3, 6])
def test_mean_fusion_accepts_exact_day05_source_counts(num_sources: int) -> None:
    xs = [torch.randn(2, 6, 5, 7) for _ in range(num_sources)]
    fusion = MeanFusion(validate=True, return_weights=True)
    fused, weights = fusion(xs)

    assert fused.shape == (2, 6, 5, 7)
    assert weights.shape == (num_sources,)
    assert torch.isfinite(fused).all()
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / num_sources))


def test_r0_mean_fusion_is_exact_identity() -> None:
    x = torch.randn(2, 6, 5, 7)
    fused, weights = MeanFusion(validate=True, return_weights=True)([x])
    assert fused is x
    assert weights.tolist() == [1.0]


def test_mean_fusion_rejects_shape_drift() -> None:
    xs = [torch.randn(2, 6, 5, 7), torch.randn(2, 6, 6, 7), torch.randn(2, 6, 5, 7)]
    with pytest.raises(ValueError, match="identical BCHW"):
        MeanFusion(validate=True)(xs)


def test_mean_fusion_rejects_nan_and_inf() -> None:
    for bad_value in (float("nan"), float("inf")):
        xs = [torch.randn(2, 6, 5, 7) for _ in range(3)]
        xs[1][0, 0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="NaN/Inf"):
            MeanFusion(validate=True)(xs)


# ===========================================================================
# A5. REAL-BATCH INTEGRATION TEST CONTRACT
# ===========================================================================


def test_real_batch_representation_gate_file_exists() -> None:
    path = ROOT / "tests" / "test_representation_shapes.py"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "torch.randn" not in source
    assert "NestedMultiViewTransform" in source
    assert "build_online_extractor" in source
    assert "ContextToLocalAligner" in source
    assert "SixFeatureProjection" in source
    assert "FeatureSelector" in source
    assert "MeanFusion" in source
    assert "representation_shape_report.json" in source


# ===========================================================================
# A6. TRAINABILITY / GRADIENT CONTRACT
# ===========================================================================


def _trainable_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _assert_all_grads_finite(params: list[torch.nn.Parameter]) -> None:
    assert params, "Expected at least one trainable parameter."
    for p in params:
        assert p.grad is not None, "Trainable parameter is disconnected from loss."
        assert torch.isfinite(p.grad).all(), "Gradient contains NaN/Inf."


def test_day05_parameter_ownership_contract() -> None:
    """Only learned Day-05 components own trainable parameters.

    MeanFusion/FeatureSelector/ContextToLocalAligner are intentionally
    parameter-free. The real DINO freeze invariant is covered below and again
    by tests/test_trainable_parameters.py with the actual checkpoint.
    """

    adapter = ResidualAdapter2d(
        in_dim=8,
        bottleneck_dim=4,
        projection_dim=6,
        kernel_size=3,
        gamma_init=0.0,
    )
    projection = SixFeatureProjection(
        in_channels=8,
        fusion_dim=5,
        blocks=(4, 8, 12),
        share_across_views=True,
    )
    decoder = BasicDecoder(in_channels=5)
    fusion = MeanFusion(validate=True, return_weights=True)
    selector = FeatureSelector("R0")
    aligner = ContextToLocalAligner()

    assert _trainable_params(adapter)
    assert _trainable_params(projection)
    assert _trainable_params(decoder)

    assert list(fusion.parameters()) == []
    assert list(selector.parameters()) == []
    assert list(aligner.parameters()) == []


def _candidate_backward_smoke(candidate_id: str) -> None:
    """Exercise the exact Day-05 source path without a DINO checkpoint.

    Raw tensors deliberately have ``requires_grad=False`` to emulate outputs of
    the frozen/no_grad DINO extractor. The real-backbone test lives in the
    separate integration file.
    """

    torch.manual_seed(42)
    c = 8
    fusion_dim = 5
    h = w = 4

    raw_local = {
        f"L{b}": torch.randn(1, c, h, w, requires_grad=False)
        for b in (4, 8, 12)
    }
    raw_context = {
        f"C{b}": torch.randn(1, c, h, w, requires_grad=False)
        for b in (4, 8, 12)
    }

    adapters = torch.nn.ModuleDict(
        {
            f"b{b}": ResidualAdapter2d(
                in_dim=c,
                bottleneck_dim=4,
                projection_dim=6,
                kernel_size=3,
                gamma_init=0.0,
            )
            for b in (4, 8, 12)
        }
    )
    projection = SixFeatureProjection(
        in_channels=c,
        fusion_dim=fusion_dim,
        blocks=(4, 8, 12),
        share_across_views=True,
        check_finite=True,
    )
    fusion = MeanFusion(validate=True, return_weights=True)
    decoder = BasicDecoder(in_channels=fusion_dim)
    criterion = AnomalySegmentationLoss(bce_weight=1.0, dice_weight=1.0)
    selector = FeatureSelector.from_candidate(candidate_id)

    if candidate_id == "R0":
        active_blocks = (12,)
    else:
        active_blocks = (4, 8, 12)

    local = {
        f"L{b}": adapters[f"b{b}"](raw_local[f"L{b}"])
        for b in active_blocks
    }

    aligned_context = None
    if candidate_id == "R2":
        context = {
            f"C{b}": adapters[f"b{b}"](raw_context[f"C{b}"])
            for b in active_blocks
        }
        geometry = {
            "local_to_context": torch.eye(3).unsqueeze(0),
            "local_input_hw": torch.tensor([[512.0, 512.0]]),
            "context_input_hw": torch.tensor([[512.0, 512.0]]),
        }
        aligned_context = ContextToLocalAligner(
            check_finite=True,
            check_bounds=True,
        )(context, geometry, target_hw=(h, w))

    projected = projection.project_sources(
        local,
        aligned_context,
        source_keys=selector.source_keys,
    )
    for value in projected.values():
        value.retain_grad()

    selected = selector(projected)
    fused, weights = fusion(selected)
    logits = decoder(fused, output_size=(16, 16))

    target = torch.zeros_like(logits)
    target[..., 4:12, 4:12] = 1.0
    loss = criterion(logits, target)["loss"]
    loss.backward()

    # Frozen-backbone surrogate inputs never receive gradients.
    assert all(x.grad is None for x in raw_local.values())
    assert all(x.grad is None for x in raw_context.values())

    # Every selected source receives a finite gradient through MeanFusion.
    assert tuple(projected.keys()) == selector.source_keys
    assert weights.shape == (selector.num_sources,)
    for value in projected.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()

    # Active Adapter branches participate in the graph. gamma_init=0 can make
    # some first-step branch gradients exactly zero, but they must not be None.
    for block in active_blocks:
        _assert_all_grads_finite(_trainable_params(adapters[f"b{block}"]))

    inactive_blocks = set((4, 8, 12)) - set(active_blocks)
    for block in inactive_blocks:
        assert all(p.grad is None for p in adapters[f"b{block}"].parameters())

    # Only projectors for selected DINO blocks should be active.
    selected_blocks = {int(key.split("b")[-1]) for key in selector.source_keys}
    assert projection.projectors is not None
    for block in (4, 8, 12):
        params = list(projection.projectors[f"b{block}"].parameters())
        if block in selected_blocks:
            _assert_all_grads_finite(params)
        else:
            assert all(p.grad is None for p in params)

    _assert_all_grads_finite(_trainable_params(decoder))
    assert list(fusion.parameters()) == []


@pytest.mark.parametrize("candidate_id", ["R0", "R1", "R2"])
def test_backward_routes_gradients_only_through_active_day05_path(
    candidate_id: str,
) -> None:
    _candidate_backward_smoke(candidate_id)


def test_dinov3_extractor_enforces_frozen_eval_mode(monkeypatch, tmp_path) -> None:
    """Unit-level regression for DINO freeze semantics without real weights."""

    class FakeBackbone(torch.nn.Module):
        patch_size = 16
        embed_dim = 384  # The mocked architecture must match the official registry.

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(1, 1) for _ in range(12)]
            )

        def get_intermediate_layers(
            self,
            x,
            n,
            reshape,
            return_class_token,
            return_extra_tokens,
            norm,
        ):
            b, _, h, w = x.shape
            oh, ow = h // self.patch_size, w // self.patch_size
            base = x.mean(dim=1, keepdim=True)
            base = torch.nn.functional.interpolate(
                base,
                size=(oh, ow),
                mode="bilinear",
                align_corners=False,
            )
            return tuple(base.repeat(1, self.embed_dim, 1, 1) for _ in n)

    repo = tmp_path / "dinov3"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# fake hubconf\n", encoding="utf-8")
    fake = FakeBackbone()
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: fake)

    extractor = DINOv3FeatureExtractor(
        repo_dir=repo,
        weights="unused.pth",
        model_name="dinov3_vits16",
        blocks=(4, 8, 12),
        check_finite=True,
    )

    assert extractor.backbone_is_frozen()
    assert all(not p.requires_grad for p in extractor.backbone.parameters())
    assert extractor.backbone.training is False

    extractor.train(True)
    assert extractor.training is True
    assert extractor.backbone.training is False
    assert all(not p.requires_grad for p in extractor.backbone.parameters())

    x = torch.randn(1, 3, 32, 32)
    out = extractor(x)
    assert all(not value.requires_grad for value in out.values())


# ===========================================================================
# A7. TRAINABLE-PARAMETER REPORT INTEGRATION TEST CONTRACT
# ===========================================================================


def test_trainable_parameter_report_gate_file_exists() -> None:
    path = ROOT / "tests" / "test_trainable_parameters.py"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "trainable_parameter_report.json" in source
    assert "build_online_extractor" in source
    assert "ResidualAdapter2d" in source
    assert "SixFeatureProjection" in source
    assert "MeanFusion" in source
    assert "BasicDecoder" in source
    assert "AnomalySegmentationLoss" in source
    assert "loss.backward()" in source
    assert "MSILA_DAY04_ADAPTER_R" in source
    assert "MSILA_DAY04_ADAPTER_D" in source


# ===========================================================================
# NEXT DAY-05 MODULES
# ===========================================================================
# Append future tests BELOW this line. Keep all tests above as regression gates
# so later modules cannot silently break the R0/R1/R2 causal ablation.
