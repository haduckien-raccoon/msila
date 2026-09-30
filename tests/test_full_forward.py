"""Day-1 full-forward integration test.

The real DINOv3 test uses the official local repository and checkpoint.
Paths are configurable with environment variables:
    DINOV3_REPO
    DINOV3_CHECKPOINT

Defaults match the Colab layout used by the existing Day-1 tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.models.contracts import DINO_FEATURE_KEYS
from src.models.msila import MSILA


DINOV3_REPO = Path(os.getenv("DINOV3_REPO", "/content/dinov3"))
DINOV3_CHECKPOINT = Path(
    os.getenv(
        "DINOV3_CHECKPOINT",
        "/content/drive/MyDrive/[Q3-4] 2026/[S7] Computer Vision/CV-Nhóm 9/weights/dinov3_vits16_pretrain_lvd1689m.pth",
    )
)


@pytest.fixture(scope="module")
def model() -> MSILA:
    if not (DINOV3_REPO / "hubconf.py").exists():
        pytest.skip(f"DINOv3 repository not found: {DINOV3_REPO}")
    if not DINOV3_CHECKPOINT.exists():
        pytest.skip(f"DINOv3 checkpoint not found: {DINOV3_CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = MSILA.from_dinov3(
        repo_dir=DINOV3_REPO,
        weights=DINOV3_CHECKPOINT,
        model_name="dinov3_vits16",
        blocks=(4, 8, 12),
        norm=True,
        adapter_reduction=4,
        adapter_kernel_size=3,
        gamma_init=0.0,
        validate=True,
    ).to(device)

    net.eval()
    return net


def test_full_forward(model: MSILA) -> None:
    """Dummy RGB image must pass through the complete Day-1 pipeline."""
    torch.manual_seed(0)
    device = next(model.parameters()).device

    x = torch.randn(2, 3, 512, 512, device=device)

    with torch.no_grad():
        y, trace = model(x, return_trace=True)

    assert y.shape == (2, 1, 512, 512)
    assert torch.isfinite(y).all()

    dino = trace["dino"]
    adapted = trace["adapted"]
    fused = trace["fused"]

    assert set(dino.keys()) == set(DINO_FEATURE_KEYS)
    assert set(adapted.keys()) == set(DINO_FEATURE_KEYS)

    for key in DINO_FEATURE_KEYS:
        assert dino[key].ndim == 4
        assert adapted[key].shape == dino[key].shape
        assert torch.isfinite(dino[key]).all()
        assert torch.isfinite(adapted[key]).all()

        # gamma=0 -> exact identity at initialization.
        assert model.adapters[key].gamma.item() == 0.0
        torch.testing.assert_close(
            adapted[key],
            dino[key],
            rtol=0.0,
            atol=0.0,
        )

    assert fused.shape == dino["b4"].shape
    assert torch.isfinite(fused).all()

    assert model.extractor.backbone_is_frozen()
    assert all(
        p.requires_grad is False
        for p in model.extractor.backbone.parameters()
    )


def test_parent_train_keeps_dino_frozen(model: MSILA) -> None:
    """Training downstream modules must not switch the frozen DINO backbone."""
    model.train()

    assert model.training is True
    assert model.extractor.backbone.training is False
    assert model.extractor.backbone_is_frozen()

    # Restore shared fixture state.
    model.eval()
