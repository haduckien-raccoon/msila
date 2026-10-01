from __future__ import annotations

import torch

from src.data.synthetic_anomaly import (
    SyntheticAnomalyConfig,
    SyntheticAnomalyGenerator,
    summarize_synthetic_metadata,
)


def make_image(h: int = 128, w: int = 160) -> torch.Tensor:
    # Smooth non-constant RGB image in [0,1].
    yy = torch.linspace(0, 1, h).view(1, h, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, w)
    r = xx.expand(1, h, w)
    g = yy.expand(1, h, w)
    b = (0.35 * r + 0.65 * g).clamp(0, 1)
    return torch.cat([r, g, b], dim=0).float()


def test_fixed_seed_is_exactly_reproducible():
    gen = SyntheticAnomalyGenerator()
    image = make_image()

    a = gen(image, seed=123, force_anomaly=True)
    b = gen(image, seed=123, force_anomaly=True)

    assert torch.equal(a.image, b.image)
    assert torch.equal(a.mask, b.mask)
    assert a.metadata == b.metadata


def test_mask_is_binary_and_exactly_synchronized_with_changes():
    cfg = SyntheticAnomalyConfig(
        anomaly_types=("intensity",),
        shape_types=("polygon",),
        boundary_probability=0.0,
    )
    gen = SyntheticAnomalyGenerator(cfg)
    image = make_image()
    sample = gen(image, seed=7, force_anomaly=True)

    assert sample.image.shape == image.shape
    assert sample.mask.shape == (1, image.shape[1], image.shape[2])
    assert set(torch.unique(sample.mask).tolist()).issubset({0.0, 1.0})
    assert sample.mask.sum() > 0

    delta = (sample.image - image).abs()
    outside = ~sample.mask.bool().expand_as(delta)
    inside = sample.mask.bool().expand_as(delta)

    assert torch.count_nonzero(delta[outside]) == 0
    assert float(delta[inside].mean()) > 0.0
    assert sample.metadata["max_abs_change_outside"] == 0.0


def test_force_normal_returns_true_normal_and_zero_mask():
    gen = SyntheticAnomalyGenerator()
    image = make_image()
    sample = gen(image, seed=99, force_anomaly=False)

    assert torch.equal(sample.image, image)
    assert torch.count_nonzero(sample.mask) == 0
    assert sample.metadata["is_anomaly"] is False
    assert sample.metadata["anomaly_type"] == "none"
    assert sample.metadata["area_ratio"] == 0.0


def test_metadata_area_and_bbox_match_binary_mask():
    gen = SyntheticAnomalyGenerator(
        SyntheticAnomalyConfig(
            anomaly_types=("noise",),
            shape_types=("ellipse",),
        )
    )
    image = make_image(96, 112)
    sample = gen(image, seed=314, force_anomaly=True)

    area_px = int(sample.mask.sum().item())
    assert sample.metadata["area_px"] == area_px
    assert abs(
        sample.metadata["area_ratio"]
        - area_px / (image.shape[1] * image.shape[2])
    ) < 1e-12

    ys, xs = torch.where(sample.mask[0] > 0.5)
    expected_bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    assert sample.metadata["bbox_xyxy"] == expected_bbox


def test_all_anomaly_types_preserve_shape_range_and_alignment():
    image = make_image(80, 96)
    for i, anomaly_type in enumerate(("intensity", "color", "noise", "cutpaste")):
        gen = SyntheticAnomalyGenerator(
            SyntheticAnomalyConfig(
                anomaly_types=(anomaly_type,),
                shape_types=("rectangle",),
                boundary_probability=0.0,
            )
        )
        sample = gen(image, seed=100 + i, force_anomaly=True)

        assert sample.image.shape == image.shape
        assert sample.mask.shape == (1, 80, 96)
        assert torch.isfinite(sample.image).all()
        assert 0.0 <= float(sample.image.min()) <= 1.0
        assert 0.0 <= float(sample.image.max()) <= 1.0
        assert float(sample.metadata["mean_abs_change_inside"]) > 0.0
        assert sample.metadata["max_abs_change_outside"] == 0.0


def test_summary_reports_type_size_location_without_realism_claim():
    gen = SyntheticAnomalyGenerator()
    image = make_image()

    records = []
    for i in range(10):
        records.append(
            gen(
                image,
                seed=1000 + i,
                force_anomaly=(i < 8),
            ).metadata
        )

    summary = summarize_synthetic_metadata(records)
    assert summary["n_total"] == 10
    assert summary["n_anomaly"] == 8
    assert summary["anomaly_fraction"] == 0.8
    assert summary["area_ratio"]["min"] > 0.0
    assert 0.0 <= summary["boundary_fraction"] <= 1.0
