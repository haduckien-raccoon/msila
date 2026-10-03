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
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "train"
        / "screen_adapter.py"
    )
    name = "screen_adapter_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def m():
    return load_runner()


def write_yaml(path: Path, data):
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


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
            "fusion": {
                "type": "attention_fusion_v0",
                "dim": 8,
                "share_projection_across_views": True,
            },
            "decoder": {
                "type": "basic_decoder",
                "output_size": [32, 32],
            },
            "loss": {
                "type": "bce_plus_dice",
                "bce_weight": 1.0,
                "dice_weight": 1.0,
            },
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
            "amp": {
                "enabled": False,
                "dtype": "bfloat16",
            },
        },
        "checkpoint": {
            "monitor": "val_loss",
            "mode": "min",
        },
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

    candidate, defaults = m.load_candidate_from_grid(
        path,
        r=2,
        d=4,
    )

    assert candidate.run_name == "adapter_r2_d4"
    assert candidate.r == 2
    assert candidate.d == 4
    assert defaults["in_dim"] == 8

    with pytest.raises(
        m.ScreenConfigError,
        match="not in the locked grid",
    ):
        m.load_candidate_from_grid(
            path,
            r=3,
            d=4,
        )


def test_protocol_requires_locked_values_instead_of_guessing(
    m,
    tmp_path,
):
    cfg = minimal_protocol(tmp_path)
    cfg["training"]["epochs"] = None

    with pytest.raises(
        m.ScreenConfigError,
        match="training.epochs",
    ):
        m.resolve_protocol(
            cfg,
            category="fabric",
        )


def test_train_val_overlap_is_rejected(m):
    train = [
        {
            "category": "fabric",
            "image_id": "a",
        }
    ]
    val = [
        {
            "category": "fabric",
            "image_id": "a",
        }
    ]

    with pytest.raises(
        m.ScreenConfigError,
        match="Train/val leakage",
    ):
        m.audit_split_disjoint(
            train,
            val,
        )


def test_same_lock_passes_but_seed_or_protocol_drift_fails(
    m,
    tmp_path,
):
    payload = {
        "category": "fabric",
        "seed": 42,
        "protocol": {
            "lr": 1e-4,
        },
    }

    out = tmp_path / "out"

    first = m.enforce_protocol_lock(
        out,
        category="fabric",
        payload=payload,
    )

    second = m.enforce_protocol_lock(
        out,
        category="fabric",
        payload=payload,
    )

    assert first == second

    changed = {
        "category": "fabric",
        "seed": 43,
        "protocol": {
            "lr": 1e-4,
        },
    }

    with pytest.raises(
        m.ScreenConfigError,
        match="protocol drift",
    ):
        m.enforce_protocol_lock(
            out,
            category="fabric",
            payload=changed,
        )


class TinyAdapter(nn.Module):
    """Small test double for one block-specific Adapter."""

    def __init__(
        self,
        r: int,
        d: int,
    ):
        super().__init__()

        self.r = int(r)
        self.d = int(d)

        self.down = nn.Linear(
            8,
            self.r,
        )
        self.mid = nn.Linear(
            self.r,
            self.d,
        )
        self.up = nn.Linear(
            self.d,
            8,
        )

    def forward(self, x):
        return x + self.up(
            torch.relu(
                self.mid(
                    torch.relu(
                        self.down(x)
                    )
                )
            )
        )


def make_tiny_adapters(
    r: int,
    d: int,
) -> nn.ModuleDict:
    """Three independent b4/b8/b12 modules with the same (r,d)."""
    return nn.ModuleDict(
        {
            "b4": TinyAdapter(r, d),
            "b8": TinyAdapter(r, d),
            "b12": TinyAdapter(r, d),
        }
    )


class TinySystem(nn.Module):
    """Fixed system around the exact supplied Adapter ModuleDict."""

    def __init__(
        self,
        adapters: nn.ModuleDict,
    ):
        super().__init__()

        # IMPORTANT: register the exact object supplied by the runner.
        self.adapters = adapters

        # Fixed non-screened components.
        self.fusion = nn.Linear(8, 8)
        self.decoder = nn.Linear(8, 1)

    def forward(self, x):
        # Exercise all three block-specific Adapters without sharing them.
        adapted = [
            self.adapters[key](x)
            for key in (
                "b4",
                "b8",
                "b12",
            )
        ]

        fused_input = torch.stack(
            adapted,
            dim=0,
        ).mean(dim=0)

        return self.decoder(
            torch.relu(
                self.fusion(
                    fused_input
                )
            )
        )


class FakeFactoryConfig:
    """Minimal AdapterFactoryConfig test double for build_adapters()."""

    @classmethod
    def from_mapping(cls, config):
        return dict(config)


class FakeBuild:
    def __init__(
        self,
        *,
        model: nn.Module,
        r: int,
        d: int,
    ):
        self.model = model
        self.run_name = (
            f"adapter_r{r}_d{d}"
        )
        self._r = int(r)
        self._d = int(d)

    def record(self):
        return {
            "run_name":
                self.run_name,

            "bottleneck_dim":
                self._r,

            "projection_dim":
                self._d,

            "trainable_params":
                sum(
                    p.numel()
                    for p in self.model.parameters()
                    if p.requires_grad
                ),
        }


class FakeResidualAdapterFactory:
    """Minimal ResidualAdapterFactory test double for build_adapters()."""

    def __init__(self, fixed):
        self.fixed = fixed

    def build_rd(
        self,
        *,
        r: int,
        d: int,
    ):
        return FakeBuild(
            model=TinyAdapter(
                r,
                d,
            ),
            r=r,
            d=d,
        )


def fake_project_deps(m):
    return m.ProjectDeps(
        CachedFeatureDataset=object,
        load_training_records=lambda _: [],
        make_cached_dataloader=lambda *args, **kwargs: None,
        AdapterFactoryConfig=FakeFactoryConfig,
        ResidualAdapterFactory=FakeResidualAdapterFactory,
    )


def _parameter_id_sets(
    adapters: nn.ModuleDict,
):
    return {
        key: {
            id(parameter)
            for parameter in module.parameters()
        }
        for key, module in adapters.items()
    }


def _state_clone(
    module: nn.Module,
):
    return {
        name: tensor.detach().clone()
        for name, tensor
        in module.state_dict().items()
    }


def test_build_adapters_creates_three_independent_block_modules(m):
    """Core regression gate for the new 3-Adapter Day-04 contract."""

    deps = fake_project_deps(m)

    candidate = m.Candidate(
        r=2,
        d=4,
        run_name="adapter_r2_d4",
    )

    adapters, records = m.build_adapters(
        deps,
        {
            "in_dim": 8,
            "kernel_size": 3,
            "gamma_init": 0.0,
            "bias": True,
        },
        candidate,
        seed=42,
    )

    assert isinstance(
        adapters,
        nn.ModuleDict,
    )

    assert set(
        adapters.keys()
    ) == {
        "b4",
        "b8",
        "b12",
    }

    assert set(
        records.keys()
    ) == {
        "b4",
        "b8",
        "b12",
    }

    # Same candidate architecture in all three blocks.
    for key in (
        "b4",
        "b8",
        "b12",
    ):
        assert adapters[key].r == 2
        assert adapters[key].d == 4

        assert (
            records[key][
                "bottleneck_dim"
            ]
            == 2
        )
        assert (
            records[key][
                "projection_dim"
            ]
            == 4
        )

    # Different module objects.
    assert (
        adapters["b4"]
        is not adapters["b8"]
    )
    assert (
        adapters["b4"]
        is not adapters["b12"]
    )
    assert (
        adapters["b8"]
        is not adapters["b12"]
    )

    # Different Parameter objects.
    ids = _parameter_id_sets(
        adapters
    )

    assert ids["b4"].isdisjoint(
        ids["b8"]
    )
    assert ids["b4"].isdisjoint(
        ids["b12"]
    )
    assert ids["b8"].isdisjoint(
        ids["b12"]
    )


def test_build_adapters_is_deterministic_per_block(m):
    """Same seed reproduces b4/b8/b12 exactly; blocks are still independent."""

    deps = fake_project_deps(m)

    candidate = m.Candidate(
        r=2,
        d=4,
        run_name="adapter_r2_d4",
    )

    kwargs = {
        "deps":
            deps,

        "adapter_defaults": {
            "in_dim": 8,
            "kernel_size": 3,
            "gamma_init": 0.0,
            "bias": True,
        },

        "candidate":
            candidate,

        "seed":
            42,
    }

    first, _ = m.build_adapters(
        **kwargs
    )
    second, _ = m.build_adapters(
        **kwargs
    )

    for key in (
        "b4",
        "b8",
        "b12",
    ):
        state_a = _state_clone(
            first[key]
        )
        state_b = _state_clone(
            second[key]
        )

        assert state_a.keys() == state_b.keys()

        for name in state_a:
            assert torch.equal(
                state_a[name],
                state_b[name],
            )

    # Block-specific seeds should not collapse the three modules to the same
    # initialized weights.
    assert not torch.equal(
        first["b4"].down.weight,
        first["b8"].down.weight,
    )
    assert not torch.equal(
        first["b4"].down.weight,
        first["b12"].down.weight,
    )


def test_non_adapter_initialization_is_same_after_seed_reset(m):
    """Changing r,d must not change fixed Fusion/Decoder initialization."""

    seed = 42

    torch.manual_seed(100)
    adapters_a = make_tiny_adapters(
        2,
        4,
    )

    model_a = m.build_project_model(
        types.SimpleNamespace(
            build_model=(
                lambda *,
                adapters,
                config:
                TinySystem(adapters)
            )
        ),
        adapters=adapters_a,
        protocol={},
        seed=seed,
        deterministic=True,
        warn_only=False,
    )

    fp_a = (
        m.non_adapter_model_fingerprint(
            model_a,
            adapters_a,
        )
    )

    # Deliberately construct a differently-sized candidate under a different
    # prior RNG state. build_project_model() must reset fixed-module RNG.
    torch.manual_seed(999)
    adapters_b = make_tiny_adapters(
        4,
        8,
    )

    model_b = m.build_project_model(
        types.SimpleNamespace(
            build_model=(
                lambda *,
                adapters,
                config:
                TinySystem(adapters)
            )
        ),
        adapters=adapters_b,
        protocol={},
        seed=seed,
        deterministic=True,
        warn_only=False,
    )

    fp_b = (
        m.non_adapter_model_fingerprint(
            model_b,
            adapters_b,
        )
    )

    assert (
        fp_a["structure_sha256"]
        == fp_b["structure_sha256"]
    )
    assert (
        fp_a["initial_state_sha256"]
        == fp_b["initial_state_sha256"]
    )


def test_fingerprint_detects_fixed_fusion_change(m):
    adapters_a = make_tiny_adapters(
        2,
        4,
    )

    model_a = TinySystem(
        adapters_a
    )

    fp_a = (
        m.non_adapter_model_fingerprint(
            model_a,
            adapters_a,
        )
    )

    class ChangedSystem(nn.Module):
        def __init__(
            self,
            adapters,
        ):
            super().__init__()

            self.adapters = adapters

            # Intentional scientific drift:
            self.fusion = nn.Linear(
                8,
                16,
            )
            self.decoder = nn.Linear(
                16,
                1,
            )

    adapters_b = make_tiny_adapters(
        4,
        8,
    )

    model_b = ChangedSystem(
        adapters_b
    )

    fp_b = (
        m.non_adapter_model_fingerprint(
            model_b,
            adapters_b,
        )
    )

    assert (
        fp_a["structure_sha256"]
        != fp_b["structure_sha256"]
    )


def test_runner_registers_exact_supplied_adapter_moduledict(m):
    """Good hook must register every supplied Adapter Parameter exactly."""

    supplied = make_tiny_adapters(
        2,
        4,
    )

    hooks = types.SimpleNamespace(
        build_model=(
            lambda *,
            adapters,
            config:
            TinySystem(adapters)
        )
    )

    model = m.build_project_model(
        hooks,
        adapters=supplied,
        protocol={},
        seed=42,
        deterministic=True,
        warn_only=False,
    )

    assert (
        model.adapters
        is supplied
    )

    supplied_ids = {
        id(p)
        for p in supplied.parameters()
    }
    model_ids = {
        id(p)
        for p in model.parameters()
    }

    assert supplied_ids.issubset(
        model_ids
    )


def test_runner_refuses_hook_that_rebuilds_adapters(m):
    """A hook may not silently replace the supplied candidate Adapters."""

    supplied = make_tiny_adapters(
        2,
        4,
    )

    bad_hooks = types.SimpleNamespace(
        build_model=(
            lambda *,
            adapters,
            config:
            TinySystem(
                make_tiny_adapters(
                    2,
                    4,
                )
            )
        )
    )

    with pytest.raises(
        m.ScreenConfigError,
        match="exact b4/b8/b12 Adapter",
    ):
        m.build_project_model(
            bad_hooks,
            adapters=supplied,
            protocol={},
            seed=42,
            deterministic=True,
            warn_only=False,
        )


def test_build_project_model_rejects_wrong_adapter_keys(m):
    supplied = nn.ModuleDict(
        {
            "b4":
                TinyAdapter(2, 4),

            "b8":
                TinyAdapter(2, 4),
        }
    )

    hooks = types.SimpleNamespace(
        build_model=(
            lambda *,
            adapters,
            config:
            TinySystem(adapters)
        )
    )

    with pytest.raises(
        m.ScreenConfigError,
        match="b4/b8/b12",
    ):
        m.build_project_model(
            hooks,
            adapters=supplied,
            protocol={},
            seed=42,
            deterministic=True,
            warn_only=False,
        )


def test_step_contract_requires_scalar_loss_and_prediction(m):
    with pytest.raises(
        m.ScreenConfigError
    ):
        m.normalize_step_output(
            {
                "loss":
                    torch.ones(2),
            },
            stage="train",
        )

    good = m.normalize_step_output(
        {
            "loss":
                torch.tensor(1.0),

            "metrics": {
                "x":
                    2.0,
            },
        },
        stage="val",
    )

    assert "loss" in good

    with pytest.raises(
        m.ScreenConfigError,
        match="prediction",
    ):
        m.normalize_step_output(
            {},
            stage="predict",
        )
