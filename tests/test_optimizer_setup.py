from __future__ import annotations

from collections import OrderedDict
import copy

import pytest
import torch
from torch import nn

from src.train.optimizer import (
    OptimizerContractError,
    assert_optimizer_contract,
    build_day3_msila_optimizer,
    build_optimizer,
    optimizer_parameter_ids,
)


class TinyFrozenDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(5, 5)


def make_modules():
    adapters = nn.ModuleDict(
        {
            "b4": nn.Sequential(nn.Conv2d(8, 4, 1), nn.GELU(), nn.Conv2d(4, 8, 1)),
            "b8": nn.Sequential(nn.Conv2d(8, 4, 1), nn.GELU(), nn.Conv2d(4, 8, 1)),
            "b12": nn.Sequential(nn.Conv2d(8, 4, 1), nn.GELU(), nn.Conv2d(4, 8, 1)),
        }
    )
    projection = nn.Conv2d(8, 6, kernel_size=1)
    fusion = nn.Sequential(nn.Conv2d(6, 6, kernel_size=1), nn.GELU())
    decoder = nn.Sequential(nn.Conv2d(6, 3, 3, padding=1), nn.GELU(), nn.Conv2d(3, 1, 1))

    dino = TinyFrozenDINO()
    dino.requires_grad_(False)

    return adapters, projection, fusion, decoder, dino


def test_day3_optimizer_contains_exact_target_modules_and_excludes_dino():
    adapters, projection, fusion, decoder, dino = make_modules()

    optimizer, report = build_day3_msila_optimizer(
        adapters=adapters,
        projection=projection,
        fusion=fusion,
        decoder=decoder,
        dino=dino,
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    target_ids = {
        id(p)
        for module in (adapters, projection, fusion, decoder)
        for p in module.parameters()
    }
    dino_ids = {id(p) for p in dino.parameters()}
    actual_ids = optimizer_parameter_ids(optimizer)

    assert actual_ids == target_ids
    assert actual_ids.isdisjoint(dino_ids)

    assert report.optimizer_name == "adamw"
    assert report.trainable_parameter_elements == sum(
        p.numel()
        for module in (adapters, projection, fusion, decoder)
        for p in module.parameters()
    )
    assert report.frozen_parameter_elements == sum(p.numel() for p in dino.parameters())
    assert [g.name for g in report.trainable_groups] == [
        "adapter",
        "projection",
        "fusion",
        "decoder",
    ]
    assert [g.name for g in report.frozen_groups] == ["dino"]


def test_builder_rejects_dino_if_any_parameter_is_trainable():
    adapters, projection, fusion, decoder, dino = make_modules()
    next(dino.parameters()).requires_grad_(True)

    with pytest.raises(OptimizerContractError, match="trainable parameters"):
        build_day3_msila_optimizer(
            adapters=adapters,
            projection=projection,
            fusion=fusion,
            decoder=decoder,
            dino=dino,
        )


def test_builder_rejects_accidentally_frozen_target_parameter():
    adapters, projection, fusion, decoder, dino = make_modules()
    next(projection.parameters()).requires_grad_(False)

    with pytest.raises(OptimizerContractError, match="requires_grad=False"):
        build_day3_msila_optimizer(
            adapters=adapters,
            projection=projection,
            fusion=fusion,
            decoder=decoder,
            dino=dino,
        )


def test_builder_rejects_parameter_overlap_between_logical_groups():
    shared = nn.Linear(4, 4)
    trainable = OrderedDict(
        [
            ("group_a", shared),
            ("group_b", shared),
        ]
    )

    with pytest.raises(OptimizerContractError, match="overlaps"):
        build_optimizer(trainable)


def test_contract_detects_manual_optimizer_contamination_with_dino():
    adapters, projection, fusion, decoder, dino = make_modules()

    trainable = OrderedDict(
        [
            ("adapter", adapters),
            ("projection", projection),
            ("fusion", fusion),
            ("decoder", decoder),
        ]
    )

    optimizer = torch.optim.AdamW(
        [
            *[p for m in trainable.values() for p in m.parameters()],
            *list(dino.parameters()),
        ],
        lr=1e-3,
    )

    with pytest.raises(OptimizerContractError):
        assert_optimizer_contract(
            optimizer,
            trainable_modules=trainable,
            frozen_modules={"dino": dino},
        )


def test_optimizer_step_changes_targets_but_not_frozen_dino():
    torch.manual_seed(7)
    adapters, projection, fusion, decoder, dino = make_modules()

    optimizer, _ = build_day3_msila_optimizer(
        adapters=adapters,
        projection=projection,
        fusion=fusion,
        decoder=decoder,
        dino=dino,
        optimizer_name="adamw",
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    before_target = {
        id(p): p.detach().clone()
        for module in (adapters, projection, fusion, decoder)
        for p in module.parameters()
    }
    before_dino = [p.detach().clone() for p in dino.parameters()]

    optimizer.zero_grad(set_to_none=True)
    loss = sum(
        p.square().mean()
        for module in (adapters, projection, fusion, decoder)
        for p in module.parameters()
    )
    loss.backward()
    optimizer.step()

    changed = 0
    for module in (adapters, projection, fusion, decoder):
        for p in module.parameters():
            if not torch.equal(before_target[id(p)], p.detach()):
                changed += 1

    assert changed > 0

    for before, after in zip(before_dino, dino.parameters()):
        assert torch.equal(before, after.detach())
        assert after.grad is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"weight_decay": -1e-4}, "weight_decay"),
        ({"eps": 0.0}, "eps"),
        ({"betas": (1.0, 0.999)}, "betas"),
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs, message):
    module = nn.Linear(3, 2)
    with pytest.raises(ValueError, match=message):
        build_optimizer({"module": module}, **kwargs)


@pytest.mark.parametrize("name", ["adamw", "adam", "sgd"])
def test_supported_optimizers_build_and_preserve_contract(name):
    adapters, projection, fusion, decoder, dino = make_modules()

    optimizer, _ = build_day3_msila_optimizer(
        adapters=adapters,
        projection=projection,
        fusion=fusion,
        decoder=decoder,
        dino=dino,
        optimizer_name=name,
    )

    assert len(optimizer.param_groups) == 4
    assert_optimizer_contract(
        optimizer,
        trainable_modules={
            "adapter": adapters,
            "projection": projection,
            "fusion": fusion,
            "decoder": decoder,
        },
        frozen_modules={"dino": dino},
    )
