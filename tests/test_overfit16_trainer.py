from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.train.overfit16 import (
    Overfit16Dataset,
    Overfit16DatasetError,
    Overfit16Trainer,
    OverfitTrainerError,
    make_overfit16_loader,
    seed_everything,
)

from src.losses.anomaly_loss import AnomalySegmentationLoss


def _write_task7_like_dataset(root: Path, *, size: int = 16) -> Path:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    records = []
    h = w = 16

    for i in range(size):
        is_anomaly = i >= 8

        mask = np.zeros((h, w), dtype=np.uint8)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Make the segmentation problem deliberately learnable:
        # anomaly pixels are bright in channel 0, background is dark.
        if is_anomaly:
            x0 = 2 + ((i - 8) % 4) * 2
            y0 = 2 + ((i - 8) // 4) * 4
            mask[y0:y0+4, x0:x0+4] = 255
            rgb[mask > 0, 0] = 255
        rgb[..., 1] = 20
        rgb[..., 2] = 10

        sample_id = f"ov16_{i:02d}_{'anomaly' if is_anomaly else 'normal'}"
        image_rel = f"images/{sample_id}.png"
        mask_rel = f"masks/{sample_id}.png"

        Image.fromarray(rgb, mode="RGB").save(root / image_rel)
        Image.fromarray(mask, mode="L").save(root / mask_rel)

        records.append(
            {
                "index": i,
                "sample_id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "is_anomaly": is_anomaly,
            }
        )

    manifest = {
        "schema_name": "msila_overfit16",
        "schema_version": 1,
        "purpose": "architecture_qa_only",
        "samples": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class TinySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.net(x)


def test_dataset_reads_exact_task7_contract(tmp_path):
    root = _write_task7_like_dataset(tmp_path / "ov16")
    ds = Overfit16Dataset(root)

    assert len(ds) == 16
    item0 = ds[0]
    item8 = ds[8]

    assert item0["image"].shape == (3, 16, 16)
    assert item0["mask"].shape == (1, 16, 16)
    assert float(item0["mask"].sum()) == 0.0
    assert float(item8["mask"].sum()) > 0.0
    assert item0["is_anomaly"] is False
    assert item8["is_anomaly"] is True


def test_dataset_rejects_wrong_sample_count(tmp_path):
    root = _write_task7_like_dataset(tmp_path / "bad", size=15)
    with pytest.raises(Overfit16DatasetError, match="exactly 16"):
        Overfit16Dataset(root)


def test_loader_is_deterministic_for_same_seed(tmp_path):
    root = _write_task7_like_dataset(tmp_path / "ov16")
    ds = Overfit16Dataset(root)

    loader_a = make_overfit16_loader(ds, batch_size=4, seed=77)
    loader_b = make_overfit16_loader(ds, batch_size=4, seed=77)

    order_a = [x for b in loader_a for x in b["sample_id"]]
    order_b = [x for b in loader_b for x in b["sample_id"]]
    assert order_a == order_b


def test_train_step_changes_model_and_returns_finite_diagnostics(tmp_path):
    seed_everything(3)
    root = _write_task7_like_dataset(tmp_path / "ov16")
    ds = Overfit16Dataset(root)
    loader = make_overfit16_loader(ds, batch_size=4, shuffle=False)

    model = TinySeg()
    criterion = AnomalySegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    trainer = Overfit16Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    before = [p.detach().clone() for p in model.parameters()]
    batch = next(iter(loader))
    log = trainer.train_step(batch, step=1, epoch=1)

    assert np.isfinite(log.loss)
    assert np.isfinite(log.bce)
    assert np.isfinite(log.dice_loss)
    assert np.isfinite(log.grad_norm)
    assert log.grad_norm > 0.0
    assert any(
        not torch.equal(a, b.detach())
        for a, b in zip(before, model.parameters())
    )


def test_fit_memorizes_controlled_16_sample_problem(tmp_path):
    seed_everything(11)
    root = _write_task7_like_dataset(tmp_path / "ov16")
    ds = Overfit16Dataset(root)
    loader = make_overfit16_loader(
        ds,
        batch_size=16,
        shuffle=True,
        seed=11,
    )

    model = TinySeg()
    criterion = AnomalySegmentationLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

    trainer = Overfit16Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        threshold=0.5,
    )

    result = trainer.fit(loader, epochs=80, log_every=10)

    assert result.final.loss < result.initial.loss
    assert result.loss_ratio < 0.20
    assert result.final.pixel_dice > 0.95
    assert result.final.iou > 0.90
    assert result.final.normal_fpr < 0.01
    assert result.steps == 80
    assert len(result.history) == 8


def test_custom_model_forward_supports_non_default_batch_contract(tmp_path):
    root = _write_task7_like_dataset(tmp_path / "ov16")
    ds = Overfit16Dataset(root)
    loader = make_overfit16_loader(ds, batch_size=4, shuffle=False)

    model = TinySeg()
    criterion = AnomalySegmentationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def forward_fn(model, batch):
        # Placeholder for the real project path:
        # image -> Local/Context/cache -> model.
        return model(batch["image"])

    trainer = Overfit16Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        model_forward=forward_fn,
    )

    log = trainer.train_step(next(iter(loader)), step=1, epoch=1)
    assert np.isfinite(log.loss)


def test_frozen_module_is_kept_eval_during_training(tmp_path):
    root = _write_task7_like_dataset(tmp_path / "ov16")
    ds = Overfit16Dataset(root)
    loader = make_overfit16_loader(ds, batch_size=4, shuffle=False)

    class ModelWithFrozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.Sequential(nn.BatchNorm2d(3))
            self.frozen.requires_grad_(False)
            self.head = nn.Conv2d(3, 1, 1)

        def forward(self, x):
            x = self.frozen(x)
            return self.head(x)

    model = ModelWithFrozen()
    criterion = AnomalySegmentationLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)

    trainer = Overfit16Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        frozen_modules={"dino_like": model.frozen},
    )

    trainer.train_step(next(iter(loader)), step=1, epoch=1)
    assert model.frozen.training is False
    assert all(p.grad is None for p in model.frozen.parameters())


def test_trainer_rejects_trainable_frozen_module():
    model = TinySeg()
    criterion = AnomalySegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    bad_frozen = nn.Linear(3, 3)

    with pytest.raises(OverfitTrainerError, match="trainable parameters"):
        Overfit16Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            frozen_modules={"dino": bad_frozen},
        )
