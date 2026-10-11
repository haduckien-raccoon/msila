"""CUDA reduction promotion regression; CPU cases are behavioral fixtures."""
import logging

import pytest
import torch
from torch import nn

from src.models.contracts import ContractError, MULTIVIEW_FEATURE_KEYS
from src.models.fusion_factory import FUSION_METHODS, build_fusion
from tests.test_g2_tv1_runner import setup


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sources(device, dtype):
    generator = torch.Generator().manual_seed(42)
    return {key: torch.randn(2, 4, 3, 5, generator=generator).to(device=device, dtype=dtype).requires_grad_()
            for key in MULTIVIEW_FEATURE_KEYS}


@pytest.mark.parametrize('method', FUSION_METHODS)
@pytest.mark.parametrize('trace', [False, True])
def test_cpu_fixture_fp32_reduction_promotion_preserves_contract_and_gradients(method, trace, monkeypatch, caplog):
    """Emulate the CUDA sum policy explicitly, without claiming a GPU run."""
    model = build_fusion(method, 4)
    source = sources('cpu', torch.bfloat16)
    original = model.module.forward
    state_keys = set(model.state_dict())
    promoted = []

    def fp32_reduction(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, dict):
            result['feature'] = result['feature'].float()
            promoted.append(result['feature'])
        else:
            result = result.float()
            promoted.append(result)
        return result

    monkeypatch.setattr(model.module, 'forward', fp32_reduction)
    with caplog.at_level(logging.INFO, logger='src.models.fusion_factory'):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = model(source, return_trace=trace)
            model(source, return_trace=trace)
    feature = result['feature'] if trace else result
    assert feature.dtype == torch.bfloat16 and feature.device == source['local_b4'].device
    assert feature.shape == (2, 4, 3, 5) and torch.isfinite(feature).all()
    torch.testing.assert_close(feature, promoted[0].to(torch.bfloat16), rtol=0, atol=0)
    feature.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in source.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(model.state_dict()) == state_keys  # Checkpoint layout unchanged.
    assert sum('Fusion dtype under AMP' in record.message for record in caplog.records) == 1
    if trace and method == 'attention':
        assert result['attention'].shape == result['logits'].shape == (2, 6, 3, 5)


def test_wrong_dtype_outside_amp_still_fails(monkeypatch):
    model = build_fusion('attention', 4)
    source = sources('cpu', torch.float32)
    monkeypatch.setattr(model.module, 'forward', lambda *a, **k: source['local_b4'].double())
    with pytest.raises(ContractError, match=r'got cpu/torch.float64, expected cpu/torch.float32'):
        model(source)


def test_wrong_device_is_never_silently_moved_under_amp(monkeypatch):
    model = build_fusion('attention', 4)
    source = sources('cpu', torch.bfloat16)
    monkeypatch.setattr(model.module, 'forward', lambda *a, **k: torch.empty(2, 4, 3, 5, device='meta'))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises((ContractError, RuntimeError)):
            model(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT RUN: this host has CPU-only PyTorch')
@pytest.mark.parametrize('method', FUSION_METHODS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('trace', [False, True])
def test_real_cuda_autocast_all_fusions_forward_backward(method, dtype, trace):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('GPU does not support BF16')
    model = build_fusion(method, 4).cuda()
    source = sources('cuda', dtype)
    with torch.autocast('cuda', dtype=dtype):
        result = model(source, return_trace=trace)
    feature = result['feature'] if trace else result
    assert feature.shape == (2, 4, 3, 5) and feature.dtype == dtype and feature.device.type == 'cuda'
    assert torch.isfinite(feature).all()
    feature.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in source.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if trace and method == 'attention':
        assert result['attention'].shape == (2, 6, 3, 5)
        torch.testing.assert_close(result['attention'].float().sum(1), torch.ones(2, 3, 5, device='cuda'), atol=.004, rtol=.004)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT RUN: this host has CPU-only PyTorch')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cuda_paired_e5_grad_scaler_train_step_fixture(setup, dtype):
    """Real GPU operations with a fixture backbone; never experimental metrics."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('GPU does not support BF16')
    from tests.test_g2_fixed_attention import fixed_study
    from src.data.loader import G1NativeDataset, G1TileDataset
    from src.losses.anomaly_loss import AnomalySegmentationLoss
    from src.models.msila import build_g2_model
    from src.train import g2_fixed
    from src.train.g2_context import PairedG2Tiles, paired_tile_collate, paired_model_forward
    from src.train.overfit16 import Overfit16Trainer

    study, _ = fixed_study(setup)
    context, pools = g2_fixed.make_context(study, 'rice')
    cfg = context['config']
    native = G1NativeDataset(pools['train'], cfg['synthetic_protocol'], seed=42, role='train')
    paired = PairedG2Tiles(G1TileDataset(native))
    batch = paired_tile_collate([paired[0], paired[1]])
    model = build_g2_model(cfg, 'E5').cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    trainer = Overfit16Trainer(model=model, criterion=AnomalySegmentationLoss(), optimizer=optimizer,
                               device='cuda', frozen_modules={'backbone': model.extractor},
                               model_forward=paired_model_forward, amp_dtype=dtype)
    for step in (1, 2):
        entry = trainer.train_step(batch, step=step, epoch=0)
        assert torch.isfinite(torch.tensor(entry.loss))
    assert not model.extractor.training and all(p.grad is None for p in model.extractor.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert model.num_sources == 6 and (model.adapter_r, model.adapter_d) == (128, 512)
