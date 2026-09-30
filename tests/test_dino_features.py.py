import pytest
import torch

from models.dinov3_extractor import DINOv3FeatureExtractor


DINOV3_REPO = "/content/dinov3"

CHECKPOINT = (
    "/content/checkpoints/"
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

def test_feature_shapes(features):

    f4, f8, f12 = features

    expected = (
        1,
        384,
        14,
        14,
    )

    print("\nDINOv3 feature shapes")
    print("Block 4 :", tuple(f4.shape))
    print("Block 8 :", tuple(f8.shape))
    print("Block 12:", tuple(f12.shape))

    assert f4.shape == expected
    assert f8.shape == expected
    assert f12.shape == expected


# ============================================================
# Test 5 — No NaN / Inf
# ============================================================

def test_features_are_finite(features):

    f4, f8, f12 = features

    for block, feature in zip(
        (4, 8, 12),
        (f4, f8, f12),
    ):

        assert torch.isfinite(feature).all(), (
            f"Block {block} contains NaN or Inf."
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