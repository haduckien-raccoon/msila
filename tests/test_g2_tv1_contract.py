"""G2 shared batch/feature/fusion and experiment-manifest acceptance gates."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.models.contracts import (
    ContractError, G2_FUSION_METHODS, G2_OUTPUT_SIZE, MULTIVIEW_FEATURE_KEYS,
    g2_input_image, validate_g2_batch, validate_g2_fused_feature,
    validate_multiview_features,
)
from src.models.backbone_registry import backbone_spec
from src.models.msila import build_g2_model, resolve_g2_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def batch():
    return dict(image=torch.zeros(2, 3, 512, 512), mask=torch.zeros(2, 1, 512, 512),
                meta=[{}, {}])


@pytest.fixture
def features():
    # Reverse insertion order deliberately; source order comes from the tuple.
    return {key: torch.full((2, 64, 32, 32), float(i))
            for i, key in reversed(list(enumerate(MULTIVIEW_FEATURE_KEYS)))}


def test_batch_matches_existing_g1_collator_and_allows_inference(batch):
    from src.data.loader import g1_tile_collate

    collated = g1_tile_collate([dict(image=batch["image"][i], mask=batch["mask"][i],
                                  meta=batch["meta"][i]) for i in range(2)])
    validate_g2_batch(collated)
    assert g2_input_image(collated) is collated["image"]
    validate_g2_batch({"image": batch["image"]}, require_target=False)
    assert g2_input_image(batch["image"]) is batch["image"]
    batch.update(context=torch.zeros_like(batch["image"]), view_meta=[{}, {}])
    validate_g2_batch(batch)


@pytest.mark.parametrize("case", [
    "missing_image", "missing_target", "bad_rgb", "bad_image_size", "integer_image",
    "nan_image", "bad_mask_size", "soft_mask", "nan_mask", "bad_meta_count",
    "bad_meta_type", "bad_context_shape", "bad_context_dtype", "bad_view_meta",
])
def test_invalid_batch_fails_at_model_boundary(batch, case):
    if case == "missing_image":
        del batch["image"]
    elif case == "missing_target":
        del batch["mask"]
    elif case == "bad_rgb":
        batch["image"] = batch["image"][:, :1]
    elif case == "bad_image_size":
        batch["image"] = batch["image"][..., :256, :256]
    elif case == "integer_image":
        batch["image"] = batch["image"].to(torch.uint8)
    elif case == "nan_image":
        batch["image"][0, 0, 0, 0] = float("nan")
    elif case == "bad_mask_size":
        batch["mask"] = batch["mask"][:, :, :256]
    elif case == "soft_mask":
        batch["mask"][0, 0, 0, 0] = 0.5
    elif case == "nan_mask":
        batch["mask"][0, 0, 0, 0] = float("nan")
    elif case == "bad_meta_count":
        batch["meta"] = [{}]
    elif case == "bad_meta_type":
        batch["meta"] = [{}, "bad"]
    elif case == "bad_context_shape":
        batch["context"] = batch["image"][:1]
    elif case == "bad_context_dtype":
        batch["context"] = batch["image"].double()
    elif case == "bad_view_meta":
        batch["view_meta"] = [{}]
    with pytest.raises(ContractError):
        validate_g2_batch(batch)


def test_canonical_six_source_order_and_fusion_tensor(features):
    assert MULTIVIEW_FEATURE_KEYS == (
        "local_b4", "local_b8", "local_b12", "context_b4", "context_b8", "context_b12",
    )
    ordered = torch.stack([features[key] for key in MULTIVIEW_FEATURE_KEYS], dim=1)
    assert ordered[:, :, 0, 0, 0].tolist() == [[0., 1., 2., 3., 4., 5.]] * 2
    validate_multiview_features(features, expected_channels=64)
    fused = ordered.mean(dim=1)
    validate_g2_fused_feature(fused, features, expected_channels=64)
    assert fused.shape == (2, 64, 32, 32)


@pytest.mark.parametrize("case", ["concat_width", "spatial", "nan", "dtype", "dict", "missing_source"])
def test_invalid_fusion_handoff_rejected(features, case):
    fused = torch.zeros(2, 64, 32, 32)
    if case == "concat_width":
        fused = torch.zeros(2, 6*64, 32, 32)
    elif case == "spatial":
        fused = fused[..., :16, :16]
    elif case == "nan":
        fused[0, 0, 0, 0] = float("nan")
    elif case == "dtype":
        fused = fused.double()
    elif case == "dict":
        fused = {"feature": fused}  # TV2 must unpack the old attention API.
    elif case == "missing_source":
        features.pop("context_b12")
    with pytest.raises(ContractError):
        validate_g2_fused_feature(fused, features, expected_channels=64)


def test_manifest_locks_eight_categories_five_fusions_and_g1_training_protocol():
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    g1 = yaml.safe_load((ROOT / "configs/g1_e1.yaml").read_text())
    assert cfg["categories"] == ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
    assert tuple(cfg["fusion_methods"]) == G2_FUSION_METHODS and len(G2_FUSION_METHODS) == 5
    assert tuple(cfg["fusion"]["feature_order"]) == MULTIVIEW_FEATURE_KEYS
    assert cfg["backbone"]["name"] == "dinov3_vitb16"
    for name, checkpoint in g1["backbone"]["checkpoints"].items():
        assert cfg["backbone"]["checkpoints"][name] == checkpoint
    assert cfg["backbone"]["frozen"] is True
    resolved = resolve_g2_config(cfg)
    assert resolved["backbone"]["feature_blocks"] == [4, 8, 12]
    assert (resolved["adapter"]["r"], resolved["adapter"]["d"]) == (128, 512)
    assert cfg["adapter"]["gamma_init"] == 0
    assert cfg["fusion"]["dim"] != resolved["adapter"]["d"]
    assert cfg["decoder"]["output_size"] == list(G2_OUTPUT_SIZE)
    assert cfg["decoder"]["hidden_channels"] == g1["decoder"]["hidden_channels"]
    assert cfg["decoder"]["output_channels"] == 1 and cfg["decoder"]["output"] == "raw_logits"
    assert list(cfg["experiments"]) == ["E1", "E2", "E3", "E4", "E5"]
    for name in ("E1", "E2"):
        assert cfg["experiments"][name]["feature_mode"] == "deepest"
        assert cfg["experiments"][name]["context"] is False
        assert cfg["experiments"][name]["fusion"] is None
    assert cfg["experiments"]["E3"]["feature_mode"] == "multilayer"
    assert cfg["experiments"]["E3"]["context"] is False
    assert cfg["experiments"]["E3"]["fusion"] == "mean"
    assert cfg["experiments"]["E4"]["implementation"] == "src.models.msila.E4"
    assert cfg["experiments"]["E4"]["feature_sources"] == 6
    assert cfg["experiments"]["E4"]["context"] is True
    assert cfg["experiments"]["E4"]["fusion"] == "mean"
    assert cfg["experiments"]["E5"]["implementation"] is None
    for key in ("seed", "dev_seed", "epochs", "batch_size", "max_grad_norm", "max_steps", "max_minutes"):
        assert cfg["training"][key] == g1["training"][key]
    assert cfg["training"]["optimizer"]["lr"] == g1["training"]["learning_rate"]
    assert cfg["training"]["optimizer"]["weight_decay"] == g1["training"]["weight_decay"]
    assert cfg["synthetic_protocol"] == g1["synthetic_protocol"]
    assert cfg["loss"]["bce_weight"] == g1["loss"]["bce_weight"]
    assert cfg["loss"]["dice_weight"] == g1["loss"]["dice_weight"]
    assert cfg["checkpoint"]["selection_split"] == "VALIDATION/good"
    assert cfg["evaluation"]["test_for_selection"] is False


@pytest.mark.parametrize("name,c,depth,r,d", [
    ("dinov3_vits16", 384, 12, 64, 256),
    ("dinov3_vits16plus", 384, 12, 64, 256),
    ("dinov3_vitb16", 768, 12, 128, 512),
    ("dinov3_vitl16", 1024, 24, 176, 688),
    ("dinov3_vith16plus", 1280, 32, 216, 856),
])
def test_one_backbone_selector_resolves_metadata_checkpoint_and_adapter(tmp_path, name, c, depth, r, d):
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["backbone"]["name"] = name
    original = deepcopy(cfg)
    resolved = resolve_g2_config(cfg, root=tmp_path)
    assert cfg == original  # Safe to reuse one manifest for independent runs.
    backbone = resolved["backbone"]
    assert (backbone["channels"], backbone["deepest_block"], backbone["patch_size"]) == (c, depth, 16)
    assert backbone["feature_blocks"] == list(backbone_spec(name).blocks)
    assert backbone["weights"] == str(tmp_path / cfg["backbone"]["checkpoints"][name])
    assert (resolved["adapter"]["r"], resolved["adapter"]["d"]) == (r, d)
    assert resolved["fusion"]["dim"] == 64
    assert resolved["decoder"]["hidden_channels"] == 64
    assert resolved["decoder"]["output_size"] == [512, 512]


@pytest.mark.parametrize("r,d,expected", [(64, 256, (64, 256)), (32, None, (32, 512)), (None, 96, (128, 96))])
def test_explicit_adapter_widths_override_scaling(r, d, expected):
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["adapter"].update(r=r, d=d)
    resolved = resolve_g2_config(cfg)
    assert (resolved["adapter"]["r"], resolved["adapter"]["d"]) == expected


def test_weights_override_and_missing_assets_fail_clearly(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    cfg["backbone"]["weights"] = "custom/vitb.pth"
    assert resolve_g2_config(cfg, root=tmp_path)["backbone"]["weights"] == str(tmp_path / "custom/vitb.pth")
    with pytest.raises(FileNotFoundError, match="DINOv3 checkpoint not found"):
        build_g2_model(cfg, root=tmp_path)
    cfg["backbone"].pop("weights")
    cfg["backbone"]["checkpoints"].pop(cfg["backbone"]["name"])
    with pytest.raises(KeyError, match="No checkpoint configured"):
        resolve_g2_config(cfg)


@pytest.mark.parametrize("case,error", [("unknown_backbone", ValueError), ("unfrozen", ValueError),
                                       ("invalid_ratio", ValueError), ("invalid_width", TypeError)])
def test_invalid_backbone_or_adapter_resolution_rejected(case, error):
    cfg = yaml.safe_load((ROOT / "configs/g2_experiments.yaml").read_text())
    if case == "unknown_backbone":
        cfg["backbone"]["name"] = "unofficial_model"
    elif case == "unfrozen":
        cfg["backbone"]["frozen"] = False
    elif case == "invalid_ratio":
        cfg["adapter"]["r_ratio"] = -1
    else:
        cfg["adapter"]["r"] = True
    with pytest.raises(error):
        resolve_g2_config(cfg)


def test_g2_builder_keeps_e5_unimplemented():
    with pytest.raises(NotImplementedError, match="only E1, E2, E3 and E4"):
        build_g2_model({}, experiment="E5")
