from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Works both when copied into repo/tests/ and when run next to the artifact.
try:
    from src.tools.build_feature_cache import (
        FEATURE_KEYS,
        FeatureCacheBuilder,
        FeatureCacheStore,
        SignatureMismatchError,
        apply_homogeneous_2d,
        assert_cache_matches,
        context_to_local_from_boxes,
    )
except ModuleNotFoundError:
    HERE = Path(__file__).resolve().parent
    CANDIDATES = [HERE, HERE.parent / "tools", HERE.parent]
    for p in CANDIDATES:
        if p.exists():
            sys.path.insert(0, str(p))
    from build_feature_cache import (  # type: ignore
        FEATURE_KEYS,
        FeatureCacheBuilder,
        FeatureCacheStore,
        SignatureMismatchError,
        apply_homogeneous_2d,
        assert_cache_matches,
        context_to_local_from_boxes,
    )


class MockFrozenExtractor(torch.nn.Module):
    """Deterministic stand-in for the real frozen DINOv3 local/context extractor."""

    def __init__(self):
        super().__init__()
        self.register_buffer("gain", torch.tensor(1.25))

    @staticmethod
    def _crop_resize(image: torch.Tensor, box, out_hw=(32, 32)) -> torch.Tensor:
        x0, y0, x1, y1 = [int(v) for v in box]
        crop = image[:, y0:y1, x0:x1].unsqueeze(0)
        return F.interpolate(
            crop,
            size=out_hw,
            mode="bilinear",
            align_corners=False,
        )

    def _features(self, x: torch.Tensor, prefix: str):
        # These are NOT DINO features; they only make a deterministic test oracle.
        scales = {
            "b4": (8, 8, 1.0),
            "b8": (4, 4, 2.0),
            "b12": (2, 2, 3.0),
        }
        out = {}
        for block, (h, w, factor) in scales.items():
            z = F.adaptive_avg_pool2d(x, (h, w)) * self.gain * factor
            # Expand channels to imitate a feature map without changing determinism.
            z = torch.cat([z, z.mean(dim=1, keepdim=True)], dim=1)
            out[f"{prefix}_{block}"] = z
        return out

    def forward(self, sample):
        image = sample["image"]
        local_box = sample["local_box"]
        context_box = sample["context_box"]

        local = self._crop_resize(image, local_box, (32, 32))
        context = self._crop_resize(image, context_box, (48, 48))

        geometry = {
            "local_box": local_box,
            "context_box": context_box,
            "local_hw": [32, 32],
            "context_hw": [48, 48],
            "context_to_local": context_to_local_from_boxes(
                local_box,
                context_box,
                local_hw=(32, 32),
                context_hw=(48, 48),
            ),
        }

        return {
            **self._features(local, "local"),
            **self._features(context, "context"),
            "geometry": geometry,
        }


@pytest.fixture
def sample():
    g = torch.Generator().manual_seed(20260901)
    image = torch.rand((3, 96, 128), generator=g)
    return {
        "image_id": "fabric/train/good/000.png",
        "category": "fabric",
        "image": image,
        "local_box": [32, 24, 80, 72],
        "context_box": [16, 8, 112, 88],
    }


@pytest.fixture
def signature():
    return {
        "backbone": "mock-dinov3-vits16",
        "checkpoint_sha256": "unit-test-only",
        "logical_layers_1based": [4, 8, 12],
        "internal_indices_0based": [3, 7, 11],
        "preprocess": "deterministic-local-context-v1",
    }


def test_feature_cache_roundtrip_matches_online(tmp_path, sample, signature):
    extractor = MockFrozenExtractor()

    builder = FeatureCacheBuilder(
        extractor=extractor,
        cache_dir=tmp_path,
        signature=signature,
        cache_dtype=torch.float32,
        verify_after_write=True,
        atol=1e-5,
        rtol=1e-5,
    )

    cached = builder.build_one(sample)
    online = extractor(sample)

    # Project hard gate: all six sources must reproduce online output.
    assert_cache_matches(online, cached, atol=1e-5, rtol=1e-5)

    for key in FEATURE_KEYS:
        assert online[key].shape == cached[key].shape
        assert torch.allclose(
            online[key].detach().cpu(),
            cached[key],
            atol=1e-5,
            rtol=1e-5,
        )

    # Explicitly retain the acceptance example from the task.
    assert torch.allclose(
        online["local_b8"].detach().cpu(),
        cached["local_b8"],
        atol=1e-5,
    )

    assert cached["image_id"] == sample["image_id"]
    assert cached["category"] == sample["category"]
    assert cached["geometry"]["local_box"] == [32.0, 24.0, 80.0, 72.0]
    assert cached["geometry"]["context_box"] == [16.0, 8.0, 112.0, 88.0]


def test_context_to_local_alignment_is_reconstructable():
    local_box = [40.0, 30.0, 80.0, 70.0]
    context_box = [20.0, 10.0, 100.0, 90.0]
    local_hw = (64, 64)
    context_hw = (128, 128)

    m = context_to_local_from_boxes(
        local_box,
        context_box,
        local_hw=local_hw,
        context_hw=context_hw,
    )

    # Where local top-left lies inside the resized context crop:
    uc0 = (local_box[0] - context_box[0]) / (context_box[2] - context_box[0]) * context_hw[1]
    vc0 = (local_box[1] - context_box[1]) / (context_box[3] - context_box[1]) * context_hw[0]

    # Where local bottom-right lies inside the resized context crop:
    uc1 = (local_box[2] - context_box[0]) / (context_box[2] - context_box[0]) * context_hw[1]
    vc1 = (local_box[3] - context_box[1]) / (context_box[3] - context_box[1]) * context_hw[0]

    x0, y0 = apply_homogeneous_2d(m, (uc0, vc0))
    x1, y1 = apply_homogeneous_2d(m, (uc1, vc1))

    assert x0 == pytest.approx(0.0, abs=1e-7)
    assert y0 == pytest.approx(0.0, abs=1e-7)
    assert x1 == pytest.approx(float(local_hw[1]), abs=1e-7)
    assert y1 == pytest.approx(float(local_hw[0]), abs=1e-7)


def test_signature_change_invalidates_cache(tmp_path, sample, signature):
    extractor = MockFrozenExtractor()
    builder = FeatureCacheBuilder(
        extractor=extractor,
        cache_dir=tmp_path,
        signature=signature,
    )
    builder.build_one(sample)

    wrong_signature = dict(signature)
    wrong_signature["preprocess"] = "changed-preprocess-v2"

    store = FeatureCacheStore(tmp_path)
    with pytest.raises(SignatureMismatchError):
        store.load(
            image_id=sample["image_id"],
            category=sample["category"],
            expected_signature=wrong_signature,
        )


def test_existing_cache_is_reused_only_with_same_signature(tmp_path, sample, signature):
    extractor = MockFrozenExtractor()
    builder = FeatureCacheBuilder(
        extractor=extractor,
        cache_dir=tmp_path,
        signature=signature,
    )
    first = builder.build_one(sample)
    second = builder.build_one(sample, overwrite=False)

    for key in FEATURE_KEYS:
        assert torch.equal(first[key], second[key])
