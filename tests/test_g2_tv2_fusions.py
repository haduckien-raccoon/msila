"""CPU numerical tests for the independent D8-TV2 fusion handoff."""

import io
import math

import pytest
import torch

from src.models.contracts import (
    MULTIVIEW_FEATURE_KEYS,
    ContractError,
    validate_g2_fused_feature,
)
from src.models.fusions import ConcatFusion, GatedFusion, WeightedSumFusion
from src.models.mean_fusion import MeanFusion


FUSIONS = (WeightedSumFusion, ConcatFusion, GatedFusion)


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(previous_threads)


def scalar_features(values=(1, 2, 3, 4, 5, 6), *, dim=2, shape=(2, 3, 5), dtype=torch.float32):
    """Reversed insertion order catches accidental source-order dependence."""
    batch, height, width = shape
    return {
        key: torch.full((batch, dim, height, width), float(values[i]), dtype=dtype)
        for i, key in reversed(list(enumerate(MULTIVIEW_FEATURE_KEYS)))
    }


def assert_probability_weights(weights, *, axis):
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    torch.testing.assert_close(weights.sum(dim=axis), torch.ones_like(weights.sum(dim=axis)))


@pytest.mark.parametrize("fusion_type", FUSIONS)
@pytest.mark.parametrize("dim", (1, 7, 64, 384, 768))
def test_width_shape_and_shared_output_contract(fusion_type, dim):
    features = scalar_features(dim=dim, shape=(1, 2, 3))
    fused = fusion_type(dim=dim)(features)
    assert isinstance(fused, torch.Tensor)
    validate_g2_fused_feature(fused, features, expected_channels=dim)


@pytest.mark.parametrize("fusion_type", FUSIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
def test_preserves_feature_dtype(fusion_type, dtype):
    features = scalar_features(dtype=dtype)
    fused = fusion_type(dim=2).to(dtype=dtype)(features)
    validate_g2_fused_feature(fused, features, expected_channels=2)


@pytest.mark.parametrize("fusion_type", FUSIONS)
def test_cpu_autocast_preserves_shared_contract(fusion_type):
    features = scalar_features()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fused = fusion_type(dim=2)(features)
    validate_g2_fused_feature(fused, features, expected_channels=2)


@pytest.mark.parametrize("fusion_type", FUSIONS)
@pytest.mark.parametrize("dim", (0, -1, 1.5, True, "64"))
def test_invalid_width_is_rejected(fusion_type, dim):
    with pytest.raises(ValueError, match="positive integer"):
        fusion_type(dim=dim)


@pytest.mark.parametrize("fusion_type", FUSIONS)
@pytest.mark.parametrize("violation", (
    "missing", "extra", "channels", "shape", "empty", "integer", "dtype", "nan", "inf", "not_tensor",
))
def test_reuses_six_feature_input_validation(fusion_type, violation):
    features = scalar_features()
    key = "context_b12"
    if violation == "missing":
        features.pop(key)
    elif violation == "extra":
        features["unknown_source"] = features[key]
    elif violation == "channels":
        features = scalar_features(dim=3)
    elif violation == "shape":
        features[key] = features[key][:, :, :2, :]
    elif violation == "empty":
        features[key] = features[key][:, :, :, :0]
    elif violation == "integer":
        features[key] = features[key].to(torch.int64)
    elif violation == "dtype":
        features[key] = features[key].double()
    elif violation in ("nan", "inf"):
        features[key][0, 0, 0, 0] = float(violation)
    else:
        features[key] = None
    with pytest.raises(ContractError):
        fusion_type(dim=2)(features)


@pytest.mark.parametrize("fusion_type", FUSIONS)
def test_gradients_reach_all_sources_and_learned_parameters(fusion_type):
    features = scalar_features()
    for feature in features.values():
        feature.requires_grad_()
    model = fusion_type(dim=2)
    model(features).square().mean().backward()
    for feature in features.values():
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad) > 0
    for parameter in model.parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize("fusion_type", FUSIONS)
@pytest.mark.parametrize("dim", (1, 7, 64))
def test_actual_trainable_parameter_counts(fusion_type, dim):
    expected = {
        WeightedSumFusion: 6,
        ConcatFusion: 6 * dim * dim + dim,
        GatedFusion: 6 * (dim + 1),
    }[fusion_type]
    model = fusion_type(dim=dim)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected
    assert all(p.requires_grad for p in model.parameters())
    if fusion_type is ConcatFusion:
        assert model.projection.in_channels == 6 * dim
        assert model.projection.out_channels == dim
        assert model.projection.kernel_size == (1, 1)
    if fusion_type is GatedFusion:
        assert len(model.gates) == 6
        assert len({id(gate.weight) for gate in model.gates}) == 6
        assert all(gate.kernel_size == (1, 1) for gate in model.gates)


@pytest.mark.parametrize("fusion_type", FUSIONS)
def test_checkpoint_roundtrip(fusion_type):
    features = scalar_features()
    original = fusion_type(dim=2)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.add_(torch.linspace(-0.2, 0.3, parameter.numel()).reshape_as(parameter))
    expected = original(features)
    checkpoint = io.BytesIO()
    torch.save(original.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = fusion_type(dim=2)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    torch.testing.assert_close(restored(features), expected, rtol=0, atol=0)


def test_weighted_uniform_initialization_matches_existing_mean():
    features = scalar_features()
    model = WeightedSumFusion(dim=2)
    assert torch.count_nonzero(model.source_logits) == 0
    weights = model.normalized_weights()
    torch.testing.assert_close(weights, torch.full((6,), 1 / 6))
    assert_probability_weights(weights, axis=0)
    torch.testing.assert_close(model(features), MeanFusion()(features))


def test_weighted_known_global_weights_and_canonical_order():
    model = WeightedSumFusion(dim=2).double()
    with torch.no_grad():
        model.source_logits.copy_(torch.tensor([math.log(i) for i in range(1, 7)]))
    weights = model.normalized_weights()
    torch.testing.assert_close(weights, torch.arange(1, 7, dtype=torch.float64) / 21)
    assert_probability_weights(weights, axis=0)
    # sum(i * i/21), with every source constant i.
    features = scalar_features(dtype=torch.float64)
    torch.testing.assert_close(model(features), torch.full_like(features["local_b4"], 91 / 21))


def test_weighted_zero_and_suppressed_source():
    model = WeightedSumFusion(dim=2)
    zero_source = scalar_features(values=(0, 2, 3, 4, 5, 6))
    torch.testing.assert_close(model(zero_source), torch.full_like(zero_source["local_b4"], 20 / 6))
    with torch.no_grad():
        model.source_logits[0] = -1000
    features = scalar_features(values=(1000000, 2, 3, 4, 5, 6))
    assert model.normalized_weights()[0] == 0
    assert_probability_weights(model.normalized_weights(), axis=0)
    torch.testing.assert_close(model(features), torch.full_like(features["local_b4"], 4))


@pytest.mark.parametrize("selected_source", range(6))
def test_concat_projects_known_source_in_canonical_channel_order(selected_source):
    model = ConcatFusion(dim=2)
    features = scalar_features()
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        for channel in range(2):
            model.projection.weight[channel, selected_source * 2 + channel, 0, 0] = 1
    expected = features[MULTIVIEW_FEATURE_KEYS[selected_source]]
    torch.testing.assert_close(model(features), expected, rtol=0, atol=0)
    # Ignored sources cannot leak into output. Selected source may be zero.
    features[MULTIVIEW_FEATURE_KEYS[selected_source]] = torch.zeros_like(expected)
    torch.testing.assert_close(model(features), torch.zeros_like(expected), rtol=0, atol=0)


def test_gated_uniform_initialization_matches_existing_mean():
    features = scalar_features()
    model = GatedFusion(dim=2)
    assert all(torch.count_nonzero(p) == 0 for p in model.parameters())
    weights = model.normalized_weights(features)
    assert weights.shape == (2, 6, 3, 5)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 6))
    assert_probability_weights(weights, axis=1)
    torch.testing.assert_close(model(features), MeanFusion()(features))


def test_gated_spatial_sigmoid_weights_have_numerical_ground_truth():
    model = GatedFusion(dim=1).double()
    # Two positions, independent source scorers: logits 0 at left, log(3)
    # at right for source 0. Its sigmoid gate is 1/2 then 3/4; others 1/2.
    features = scalar_features(dim=1, shape=(1, 1, 2), dtype=torch.float64)
    features["local_b4"] = torch.tensor([[[[0.0, math.log(3)]]]], dtype=torch.float64)
    with torch.no_grad():
        model.gates[0].weight.fill_(1)
    weights = model.normalized_weights(features)
    expected = torch.full_like(weights, 1 / 6)
    expected[:, 0, :, 1] = 3 / 13
    expected[:, 1:, :, 1] = 2 / 13
    torch.testing.assert_close(weights, expected)
    assert_probability_weights(weights, axis=1)
    torch.testing.assert_close(
        model(features),
        torch.tensor([[[[20 / 6, (3 * math.log(3) + 40) / 13]]]], dtype=torch.float64),
    )
    # Softmax(raw logits) would assign 3/8 at right, so this must differ.
    assert not torch.isclose(weights[0, 0, 0, 1], torch.tensor(3 / 8, dtype=torch.float64))


@pytest.mark.parametrize("bias", (-1000.0, 1000.0))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_gated_saturated_all_gates_remain_normalized_and_finite(bias, dtype):
    features = scalar_features(dtype=dtype)
    model = GatedFusion(dim=2).to(dtype=dtype)
    with torch.no_grad():
        for gate in model.gates:
            gate.bias.fill_(bias)
    weights = model.normalized_weights(features)
    assert_probability_weights(weights, axis=1)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 6))
    fused = model(features)
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused, torch.full_like(fused, 3.5))
    fused.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_gated_zero_source_and_gate_suppression():
    model = GatedFusion(dim=2)
    features = scalar_features(values=(0, 2, 3, 4, 5, 6))
    torch.testing.assert_close(model(features), torch.full_like(features["local_b4"], 20 / 6))
    with torch.no_grad():
        model.gates[0].bias.fill_(-1000)
    features["local_b4"].fill_(1000000)
    weights = model.normalized_weights(features)
    assert torch.count_nonzero(weights[:, 0]) == 0
    assert_probability_weights(weights, axis=1)
    torch.testing.assert_close(model(features), torch.full_like(features["local_b4"], 4))


def test_gated_each_scorer_sees_only_its_own_source():
    model = GatedFusion(dim=1).double()
    with torch.no_grad():
        for gate in model.gates:
            gate.weight.fill_(0.2)
    features = scalar_features(dim=1, dtype=torch.float64)
    before = model.normalized_weights(features)
    features["context_b12"].add_(3)
    after = model.normalized_weights(features)
    # The normalization changes all weights, but odds between unaffected
    # sources must stay fixed when only the sixth gate sees different input.
    torch.testing.assert_close(before[:, :5] / before[:, :1], after[:, :5] / after[:, :1])
    assert torch.all(after[:, 5] > before[:, 5])


@pytest.mark.parametrize("fusion_type", FUSIONS)
def test_disabling_validation_keeps_canonical_order(fusion_type):
    model = fusion_type(dim=2, validate_input=False)
    features = scalar_features()
    torch.testing.assert_close(model(features), model(dict(reversed(list(features.items())))), rtol=0, atol=0)
