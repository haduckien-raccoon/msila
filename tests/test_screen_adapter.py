from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml


def load_runner():
    path = Path(__file__).resolve().parents[1] / "train" / "screen_adapter.py"
    name = "screen_adapter_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def m():
    return load_runner()


def write_yaml(path: Path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def minimal_protocol(tmp_path: Path):
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    train.write_text("[]", encoding="utf-8")
    val.write_text("[]", encoding="utf-8")
    return {
        "version": 1,
        "hooks_module": "fake_hooks",
        "output_root": str(tmp_path / "out"),
        "frozen_backbone": {
            "checkpoint": "dinov3-fixed",
            "frozen": True,
            "blocks": [4, 8, 12],
            "local_context": "fixed",
            "alignment": "fixed",
        },
        "data": {
            "cache_dir": str(tmp_path / "cache"),
            "train_records": str(train),
            "val_records": str(val),
            "num_workers": 0,
        },
        "model": {
            "fusion": {"kind": "fixed"},
            "decoder": {"kind": "fixed"},
            "loss": {"kind": "fixed"},
            "augmentation": None,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-3,
                "kwargs": {},
            },
            "scheduler": None,
            "gradient_clip_norm": None,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "amp": {"enabled": False, "dtype": "bfloat16"},
        },
        "checkpoint": {"monitor": "val_loss", "mode": "min"},
    }


def test_grid_accepts_only_declared_candidate(m, tmp_path):
    grid = {
        "expected_num_candidates": 2,
        "adapter_defaults": {
            "in_dim": 8,
            "kernel_size": 3,
            "gamma_init": 0.0,
            "bias": True,
        },
        "candidates": [
            {
                "run_name": "adapter_r2_d4",
                "bottleneck_dim": 2,
                "projection_dim": 4,
            },
            {
                "run_name": "adapter_r4_d8",
                "bottleneck_dim": 4,
                "projection_dim": 8,
            },
        ],
    }
    path = tmp_path / "grid.yaml"
    write_yaml(path, grid)

    candidate, defaults = m.load_candidate_from_grid(path, r=2, d=4)
    assert candidate.run_name == "adapter_r2_d4"
    assert defaults["in_dim"] == 8

    with pytest.raises(m.ScreenConfigError, match="not in the locked grid"):
        m.load_candidate_from_grid(path, r=3, d=4)


def test_protocol_requires_locked_values_instead_of_guessing(m, tmp_path):
    cfg = minimal_protocol(tmp_path)
    cfg["training"]["epochs"] = None

    with pytest.raises(m.ScreenConfigError, match="training.epochs"):
        m.resolve_protocol(cfg, category="fabric")


def test_train_val_overlap_is_rejected(m):
    train = [{"category": "fabric", "image_id": "a"}]
    val = [{"category": "fabric", "image_id": "a"}]

    with pytest.raises(m.ScreenConfigError, match="Train/val leakage"):
        m.audit_split_disjoint(train, val)


def test_same_lock_passes_but_seed_or_protocol_drift_fails(m, tmp_path):
    payload = {
        "category": "fabric",
        "seed": 42,
        "protocol": {"lr": 1e-4},
    }
    out = tmp_path / "out"

    first = m.enforce_protocol_lock(out, category="fabric", payload=payload)
    second = m.enforce_protocol_lock(out, category="fabric", payload=payload)
    assert first == second

    changed = {
        "category": "fabric",
        "seed": 43,
        "protocol": {"lr": 1e-4},
    }
    with pytest.raises(m.ScreenConfigError, match="protocol drift"):
        m.enforce_protocol_lock(out, category="fabric", payload=changed)


class TinyAdapter(nn.Module):
    def __init__(self, r: int, d: int):
        super().__init__()
        self.down = nn.Linear(8, r)
        self.mid = nn.Linear(r, d)
        self.up = nn.Linear(d, 8)

    def forward(self, x):
        return x + self.up(torch.relu(self.mid(torch.relu(self.down(x)))))


class TinySystem(nn.Module):
    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.fusion = nn.Linear(8, 8)
        self.decoder = nn.Linear(8, 1)

    def forward(self, x):
        return self.decoder(torch.relu(self.fusion(self.adapter(x))))


def test_non_adapter_initialization_is_same_after_seed_reset(m):
    seed = 42

    torch.manual_seed(seed)
    adapter_a = TinyAdapter(2, 4)
    model_a = m.build_project_model(
        types.SimpleNamespace(
            build_model=lambda *, adapter, config: TinySystem(adapter)
        ),
        adapter=adapter_a,
        protocol={},
        seed=seed,
        deterministic=True,
        warn_only=False,
    )
    fp_a = m.non_adapter_model_fingerprint(model_a, adapter_a)

    torch.manual_seed(seed)
    adapter_b = TinyAdapter(4, 8)
    model_b = m.build_project_model(
        types.SimpleNamespace(
            build_model=lambda *, adapter, config: TinySystem(adapter)
        ),
        adapter=adapter_b,
        protocol={},
        seed=seed,
        deterministic=True,
        warn_only=False,
    )
    fp_b = m.non_adapter_model_fingerprint(model_b, adapter_b)

    assert fp_a["structure_sha256"] == fp_b["structure_sha256"]
    assert fp_a["initial_state_sha256"] == fp_b["initial_state_sha256"]


def test_fingerprint_detects_fixed_fusion_change(m):
    adapter_a = TinyAdapter(2, 4)
    model_a = TinySystem(adapter_a)
    fp_a = m.non_adapter_model_fingerprint(model_a, adapter_a)

    class ChangedSystem(nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.adapter = adapter
            self.fusion = nn.Linear(8, 16)
            self.decoder = nn.Linear(16, 1)

    adapter_b = TinyAdapter(4, 8)
    model_b = ChangedSystem(adapter_b)
    fp_b = m.non_adapter_model_fingerprint(model_b, adapter_b)

    assert fp_a["structure_sha256"] != fp_b["structure_sha256"]


def test_runner_refuses_hook_that_rebuilds_adapter(m):
    supplied = TinyAdapter(2, 4)

    bad_hooks = types.SimpleNamespace(
        build_model=lambda *, adapter, config: TinySystem(TinyAdapter(2, 4))
    )

    with pytest.raises(m.ScreenConfigError, match="exact Adapter instance"):
        m.build_project_model(
            bad_hooks,
            adapter=supplied,
            protocol={},
            seed=42,
            deterministic=True,
            warn_only=False,
        )


def test_step_contract_requires_scalar_loss_and_prediction(m):
    with pytest.raises(m.ScreenConfigError):
        m.normalize_step_output({"loss": torch.ones(2)}, stage="train")

    good = m.normalize_step_output(
        {"loss": torch.tensor(1.0), "metrics": {"x": 2.0}},
        stage="val",
    )
    assert "loss" in good

    with pytest.raises(m.ScreenConfigError, match="prediction"):
        m.normalize_step_output({}, stage="predict")
