from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.utils.checkpoint import (
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    build_checkpoint_payload,
    capture_rng_state,
    save_training_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.frozen(x))


def make_trained_state():
    torch.manual_seed(5)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

    x = torch.randn(3, 4)
    loss = model(x).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return model, optimizer


def load_raw(path: Path):
    # Tests inspect the serialized Task-12 payload only.
    # Restoring it into live objects belongs to Task 13.
    return torch.load(path, map_location="cpu", weights_only=False)


def test_checkpoint_contains_required_state(tmp_path):
    model, optimizer = make_trained_state()

    path = tmp_path / "checkpoint.pt"
    result = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=7,
        global_step=123,
        config={
            "seed": 2026,
            "lr": 1e-3,
            "dataset": Path("/data/overfit16"),
        },
        metadata={"purpose": "architecture_qa_only"},
    )

    assert path.is_file()
    assert result.size_bytes > 0

    payload = load_raw(path)
    assert payload["schema_name"] == CHECKPOINT_SCHEMA_NAME
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["training_state"] == {"epoch": 7, "global_step": 123}
    assert payload["config"]["seed"] == 2026
    assert payload["config"]["dataset"] == "/data/overfit16"
    assert payload["metadata"]["purpose"] == "architecture_qa_only"

    assert set(payload["model_state"]) == set(model.state_dict())
    assert "state" in payload["optimizer_state"]
    assert "param_groups" in payload["optimizer_state"]

    contract = payload["model_contract"]
    assert "head.weight" in contract["trainable_parameter_names"]
    assert "frozen.weight" in contract["frozen_parameter_names"]


def test_rng_capture_does_not_advance_rng():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    _ = capture_rng_state()

    assert random.getstate() == py_before

    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]

    assert torch.equal(torch.get_rng_state(), torch_before)


def test_named_generator_state_is_saved_exactly(tmp_path):
    model, optimizer = make_trained_state()
    generator = torch.Generator().manual_seed(2026)
    expected = generator.get_state().clone()

    path = tmp_path / "with_generator.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=4,
        config={},
        generators={"dataloader": generator},
    )

    payload = load_raw(path)
    saved = payload["rng_state"]["generators"]["dataloader"]
    assert torch.equal(saved, expected)


def test_optional_scheduler_state_is_saved(tmp_path):
    model, optimizer = make_trained_state()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5,
    )
    # Respect PyTorch scheduler order in the test: optimizer.step() first.
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    scheduler.step()

    path = tmp_path / "scheduler.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=8,
        config={},
        stateful_objects={"scheduler": scheduler},
    )

    payload = load_raw(path)
    assert "scheduler" in payload["stateful_states"]
    assert payload["stateful_states"]["scheduler"]["last_epoch"] == scheduler.last_epoch


def test_sha256_sidecar_matches_checkpoint(tmp_path):
    model, optimizer = make_trained_state()
    path = tmp_path / "integrity.pt"

    result = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config={},
        write_sha256=True,
    )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.sha256 == digest

    sidecar = path.with_suffix(path.suffix + ".sha256")
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {path.name}\n"


def test_atomic_overwrite_leaves_no_temp_files(tmp_path):
    model, optimizer = make_trained_state()
    path = tmp_path / "latest.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=10,
        config={"tag": "first"},
    )
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=20,
        config={"tag": "second"},
    )

    payload = load_raw(path)
    assert payload["training_state"]["epoch"] == 2
    assert payload["config"]["tag"] == "second"

    tmp_files = [
        p for p in tmp_path.iterdir()
        if p.name.startswith(".latest.pt.") and p.name.endswith(".tmp")
    ]
    assert tmp_files == []


@pytest.mark.parametrize(
    ("epoch", "global_step", "message"),
    [
        (-1, 0, "epoch"),
        (0, -1, "global_step"),
    ],
)
def test_negative_training_position_is_rejected(epoch, global_step, message):
    model, optimizer = make_trained_state()

    with pytest.raises(ValueError, match=message):
        build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config={},
        )


def test_nonserializable_config_is_rejected():
    model, optimizer = make_trained_state()

    with pytest.raises(CheckpointError, match="unsupported config value"):
        build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=0,
            global_step=0,
            config={"bad": object()},
        )


def test_save_does_not_change_model_or_optimizer_state(tmp_path):
    model, optimizer = make_trained_state()

    model_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    opt_before_groups = [
        dict(group)
        for group in optimizer.state_dict()["param_groups"]
    ]

    save_training_checkpoint(
        tmp_path / "pure_save.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=12,
        config={},
    )

    for name, value in model.state_dict().items():
        assert torch.equal(value, model_before[name])

    assert optimizer.state_dict()["param_groups"] == opt_before_groups
