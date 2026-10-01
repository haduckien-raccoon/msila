from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.utils.checkpoint import save_training_checkpoint
from src.utils.resume import (
    ResumeError,
    load_checkpoint_payload,
    resume_training_checkpoint,
    verify_checkpoint_sha256,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.frozen(x))


def make_model_optimizer():
    model = TinyModel()
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )
    return model, optimizer


def one_step(model, optimizer):
    x = torch.randn(5, 4)
    y = torch.randn(5, 2)

    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def clone_model_state(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def assert_model_state_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert torch.equal(a, b)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a) == set(b)
        for key in a:
            assert_nested_equal(a[key], b[key])
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, type(a))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
        return
    assert a == b


def test_resume_restores_model_optimizer_position_and_metadata(tmp_path):
    torch.manual_seed(5)
    model, optimizer = make_model_optimizer()
    one_step(model, optimizer)

    config = {
        "seed": 2026,
        "batch_size": 4,
        "learning_rate": 1e-2,
    }

    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=7,
        global_step=123,
        config=config,
        metadata={"purpose": "architecture_qa_only"},
    )

    expected_model = clone_model_state(model)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    # New live objects start from unrelated state.
    torch.manual_seed(999)
    resumed_model, resumed_optimizer = make_model_optimizer()

    result = resume_training_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_config=config,
    )

    assert result.epoch == 7
    assert result.global_step == 123
    assert result.config == config
    assert result.metadata["purpose"] == "architecture_qa_only"
    assert result.sha256_verified is True
    assert result.rng_restored is True

    assert_model_state_equal(
        clone_model_state(resumed_model),
        expected_model,
    )
    assert_nested_equal(
        resumed_optimizer.state_dict(),
        expected_optimizer,
    )


def test_resume_restores_python_numpy_torch_and_named_generator_rng(tmp_path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    generator = torch.Generator().manual_seed(2026)
    model, optimizer = make_model_optimizer()

    path = tmp_path / "rng.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=2,
        config={},
        generators={"dataloader": generator},
    )

    # Expected future sequence from exactly the saved state.
    expected_py = [random.random() for _ in range(3)]
    expected_np = np.random.rand(3)
    expected_torch = torch.rand(3)
    expected_gen = torch.rand(3, generator=generator)

    # Disturb every RNG and create a fresh generator.
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    resumed_generator = torch.Generator().manual_seed(777)
    resumed_model, resumed_optimizer = make_model_optimizer()

    resume_training_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        generators={"dataloader": resumed_generator},
    )

    assert [random.random() for _ in range(3)] == expected_py
    assert np.array_equal(np.random.rand(3), expected_np)
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(
        torch.rand(3, generator=resumed_generator),
        expected_gen,
    )


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)

    model, optimizer = make_model_optimizer()

    # Prefix shared by both trajectories.
    for _ in range(3):
        one_step(model, optimizer)

    path = tmp_path / "mid.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=3,
        config={"experiment": "deterministic_cpu_resume_test"},
    )

    # Uninterrupted reference continuation.
    reference_losses = [one_step(model, optimizer) for _ in range(4)]
    reference_model = clone_model_state(model)
    reference_optimizer = copy.deepcopy(optimizer.state_dict())

    # Fresh process-equivalent objects with unrelated state.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_model, resumed_optimizer = make_model_optimizer()

    result = resume_training_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_config={"experiment": "deterministic_cpu_resume_test"},
    )
    assert result.global_step == 3

    resumed_losses = [
        one_step(resumed_model, resumed_optimizer)
        for _ in range(4)
    ]

    # CPU deterministic test: resumed trajectory must exactly match.
    assert resumed_losses == reference_losses
    assert_model_state_equal(
        clone_model_state(resumed_model),
        reference_model,
    )
    assert_nested_equal(
        resumed_optimizer.state_dict(),
        reference_optimizer,
    )


def test_scheduler_state_is_restored(tmp_path):
    model, optimizer = make_model_optimizer()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5,
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    scheduler.step()

    path = tmp_path / "scheduler.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=5,
        config={},
        stateful_objects={"scheduler": scheduler},
    )

    new_model, new_optimizer = make_model_optimizer()
    new_scheduler = torch.optim.lr_scheduler.StepLR(
        new_optimizer,
        step_size=2,
        gamma=0.5,
    )

    result = resume_training_checkpoint(
        path,
        model=new_model,
        optimizer=new_optimizer,
        stateful_objects={"scheduler": new_scheduler},
    )

    assert result.restored_stateful_objects == ("scheduler",)
    assert new_scheduler.state_dict() == scheduler.state_dict()


def test_sha256_corruption_is_rejected_before_deserialization(tmp_path):
    model, optimizer = make_model_optimizer()
    path = tmp_path / "corrupt.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config={},
    )

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(data)

    with pytest.raises(ResumeError, match="SHA-256 mismatch"):
        verify_checkpoint_sha256(path)


def test_config_mismatch_is_rejected(tmp_path):
    model, optimizer = make_model_optimizer()
    path = tmp_path / "config.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=1,
        config={"lr": 1e-2, "fusion_dim": 64},
    )

    new_model, new_optimizer = make_model_optimizer()

    with pytest.raises(ResumeError, match="config mismatch"):
        resume_training_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
            expected_config={"lr": 1e-3, "fusion_dim": 64},
        )


def test_trainability_contract_mismatch_is_rejected(tmp_path):
    model, optimizer = make_model_optimizer()
    path = tmp_path / "contract.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=1,
        config={},
    )

    new_model, new_optimizer = make_model_optimizer()
    # This changes the frozen/trainable contract without changing parameter names.
    new_model.frozen.weight.requires_grad_(True)

    with pytest.raises(ResumeError, match="Trainable parameter-name contract"):
        resume_training_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
        )


def test_named_generator_mismatch_is_rejected(tmp_path):
    model, optimizer = make_model_optimizer()
    generator = torch.Generator().manual_seed(1)
    path = tmp_path / "gen.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config={},
        generators={"dataloader": generator},
    )

    new_model, new_optimizer = make_model_optimizer()

    with pytest.raises(ResumeError, match="torch.Generator name mismatch"):
        resume_training_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
            generators={},
        )


def test_payload_schema_mismatch_is_rejected(tmp_path):
    model, optimizer = make_model_optimizer()
    path = tmp_path / "schema.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config={},
        write_sha256=False,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 999
    torch.save(payload, path)

    with pytest.raises(ResumeError, match="schema_version mismatch"):
        load_checkpoint_payload(
            path,
            verify_sha256=False,
        )


def test_missing_sha_sidecar_is_rejected_by_default(tmp_path):
    model, optimizer = make_model_optimizer()
    path = tmp_path / "no_sidecar.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config={},
        write_sha256=False,
    )

    with pytest.raises(ResumeError, match="sidecar not found"):
        resume_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
        )
