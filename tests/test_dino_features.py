import pytest
import torch

from src.models.dinov3_extractor import DINOv3FeatureExtractor


DINOV3_REPO = "/content/dinov3"

CHECKPOINT = (
    "/content/drive/MyDrive/[Q3-4] 2026/[S7] Computer Vision/CV-Nhóm 9/weights/"
    "dinov3_vits16_pretrain_lvd1689m.pth"
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def extractor():
    """
    Load DINOv3 only once for the whole test module.

    Important for Colab/CPU:
    avoids loading the backbone again for every test.
    """
    model = DINOv3FeatureExtractor(
        repo_dir=DINOV3_REPO,
        weights=CHECKPOINT,
        model_name="dinov3_vits16",
        blocks=(4, 8, 12),
        norm=True,
    )

    model.eval()

    return model


@pytest.fixture(scope="module")
def features(extractor):
    """
    Run DINOv3 only once.

    Random input is sufficient here because these tests verify:
        - interface
        - shape
        - numerical validity

    They do NOT evaluate feature quality.
    """
    torch.manual_seed(0)

    x = torch.randn(
        1,
        3,
        224,
        224,
    )

    with torch.no_grad():
        outputs = extractor(x)

    return outputs


# ============================================================
# Test 1 — Correct block contract
# ============================================================

def test_correct_blocks(extractor):

    assert extractor.blocks == (4, 8, 12)

    assert extractor.block_indices == (
        3,
        7,
        11,
    )


# ============================================================
# Test 2 — Backbone must be frozen
# ============================================================

def test_backbone_frozen(extractor):

    assert extractor.backbone_is_frozen()

    assert all(
        p.requires_grad is False
        for p in extractor.backbone.parameters()
    )


# ============================================================
# Test 3 — Exactly three features
# ============================================================

def test_exact_three_features(features):

    assert len(features) == 3


# ============================================================
# Test 4 — Feature shapes
# ============================================================

def test_feature_shapes(extractor, features):

    B = 1
    H = 224
    W = 224

    C = extractor.out_channels
    P = extractor.patch_size

    expected = (
        B,
        C,
        H // P,
        W // P,
    )

    assert features["b4"].shape == expected
    assert features["b8"].shape == expected
    assert features["b12"].shape == expected

# ============================================================
# Test 5 — No NaN / Inf
# ============================================================

def test_features_are_finite(features):

     for name in ("b4", "b8", "b12"):

        feature = features[name]

        assert torch.isfinite(feature).all(), (
            f"{name} contains NaN or Inf."
        )


# ============================================================
# Test 6 — model.train() must NOT unfreeze DINO
# ============================================================

def test_train_does_not_unfreeze_dino(extractor):

    extractor.train()

    # Parent module enters training mode.
    assert extractor.training is True

    # Frozen DINO must remain eval.
    assert extractor.backbone.training is False

    assert extractor.backbone_is_frozen()

    # Restore state because fixture is shared.
    extractor.eval()

def test_feature_keys(features):

    assert set(features.keys()) == {
        "b4",
        "b8",
        "b12",
    }


def test_feature_contract(extractor, features):

    for name in ("b4", "b8", "b12"):

        f = features[name]

        # [B,C,h,w]
        assert f.ndim == 4

        assert f.shape[1] == extractor.out_channels

        # no NaN / Inf
        assert torch.isfinite(f).all()


def test_dynamic_spatial_shape(
    extractor,
    features,
):

    H = 224
    W = 224

    P = extractor.patch_size
    C = extractor.out_channels

    expected = (
        1,
        C,
        H // P,
        W // P,
    )

    for f in features.values():
        assert f.shape == expected