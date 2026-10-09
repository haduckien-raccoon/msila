"""Tiny CPU training-code fixtures; never real DINO/cache/training evidence."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.train import screen_representation as training
from src.train.day05_contract import SOURCES
from src.geometry.view_meta import build_view_meta
from scripts import run_day05_pipeline as pipeline

ROOT = Path(__file__).parents[1]


@pytest.fixture
def small_head():
    config = training.read_yaml(ROOT / 'configs/day05_representation.yaml')
    protocol = training.read_yaml(ROOT / 'configs/day05_day04_full_v3_protocol.yaml')
    training.seed_everything(42, protocol)
    model = training.Day05RepresentationModel(day05_config=config, candidate='R1',
                                              in_channels=8, adapter_r=32, adapter_d=384)
    # This independent software fixture supplies detached tiny features and masks.
    # No scientific protocol fields/files are edited to obtain these tensors.
    batch = {key: torch.randn(1, 8, 4, 4) for key in SOURCES['R1']}
    batch['mask'] = torch.zeros(1, 1, 16, 16)
    batch['mask'][:, :, 4:10, 5:12] = 1
    return model, batch, config, protocol


def test_preflight_rejects_omitted_active_parameter(small_head):
    model, batch, config, protocol = small_head
    optimizer = training.build_optimizer(model, protocol)
    # Other adapters/projectors/decoder still update, so coarse prefix checks
    # can pass while this active gamma never receives an optimizer update.
    optimizer.param_groups[0]['params'] = [
        p for p in optimizer.param_groups[0]['params'] if p is not model.adapters['b4'].gamma]
    with pytest.raises(training.FullTrainError, match='optimizer.*missing.*adapters.b4.gamma'):
        training.run_preflight(model, [batch], training.make_criterion(config), optimizer,
                               torch.device('cpu'), 2)


def test_preflight_rejects_foreign_optimizer_parameter(small_head):
    model, batch, config, protocol = small_head
    optimizer = training.build_optimizer(model, protocol)
    optimizer.param_groups[0]['params'].append(torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(training.FullTrainError, match='optimizer.*foreign'):
        training.run_preflight(model, [batch], training.make_criterion(config), optimizer,
                               torch.device('cpu'), 2)


def test_pipeline_invalid_yaml_is_a_structured_failure(tmp_path, capsys):
    broken = tmp_path / 'broken.yaml'
    broken.write_text('locked: [\n')
    output = tmp_path / 'output'
    assert pipeline.main(['--stage', 'audit', '--day05-config', str(broken), '--output-root', str(output)]) == 2
    report = json.loads((output / 'last_gate_report.json').read_text())
    assert report['status'] == 'FAIL' and report['stage'] == 'audit'
    assert 'Traceback' not in capsys.readouterr().err


def test_training_cli_missing_selection_is_a_structured_failure(tmp_path, capsys):
    output = tmp_path / 'output'
    assert training.main(['--candidate', 'R0', '--r', '32', '--d', '384', '--device', 'cpu',
                          '--training-protocol', str(ROOT / 'configs/day05_day04_full_v3_protocol.yaml'),
                          '--cache-dir', str(tmp_path / 'cache'), '--train-records', str(tmp_path / 'train.json'),
                          '--val-records', str(tmp_path / 'dev.json'), '--output-root', str(output),
                          '--preflight-only']) == 2
    report = json.loads((output / 'last_gate_report.json').read_text())
    assert report['status'] == 'FAIL' and 'Day-04 selection' in report['reason']
    assert 'Traceback' not in capsys.readouterr().err


@pytest.mark.parametrize('candidate', SOURCES)
def test_cpu_preflight_owned_parameters_and_checkpoint_roundtrip(tmp_path, candidate):
    config = training.read_yaml(ROOT / 'configs/day05_representation.yaml')
    protocol = training.read_yaml(ROOT / 'configs/day05_day04_full_v3_protocol.yaml')
    training.seed_everything(42, protocol)
    model = training.Day05RepresentationModel(day05_config=config, candidate=candidate,
                                              in_channels=8, adapter_r=32, adapter_d=384)
    geometry = build_view_meta(source_hw=(768, 768), local_box_xyxy=(128, 128, 640, 640),
                               context_box_xyxy=(0, 0, 768, 768))
    batch = {key: torch.randn(1, 8, 4, 4) for key in SOURCES[candidate]}
    batch['meta'] = [{'geometry': dict(local_box=[128, 128, 640, 640], context_box=[0, 0, 768, 768],
                                      local_to_context=geometry.local_to_context.tolist(),
                                      context_to_local=geometry.context_to_local.tolist(),
                                      local_input_hw=[512, 512], context_input_hw=[512, 512])}]
    batch['mask'] = torch.zeros(1, 1, 16, 16)
    batch['mask'][:, :, 4:10, 5:12] = 1
    optimizer = training.build_optimizer(model, protocol)
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    observations = []
    hook = model.register_forward_hook(lambda module, args, out: observations.append((tuple(out[0].shape), out[0].requires_grad)))
    report = training.run_preflight(model, [batch], training.make_criterion(config, protocol), optimizer,
                                    torch.device('cpu'), 2, protocol)
    hook.remove()
    assert report['status'] == 'PASS' and observations == [((1, 1, 16, 16), True)] * 2
    assert report['optimizer_parameters']['optimizer'] == 'AdamW'
    assert set(report['source_usage'][-1]) == set(SOURCES[candidate])
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    grouped = [n for g in report['optimizer_parameters']['groups'] for n in g['parameter_names']]
    assert len(grouped) == len(trainable) and set(grouped) == set(trainable)
    assert report['optimizer_parameters']['trainable_parameters'] == sum(p.numel() for p in trainable.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    for name, old in frozen_before.items():
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is None and torch.equal(parameter, old)
    assert all(not x.requires_grad and x.grad is None for key, x in batch.items()
               if key.startswith(('local_b', 'context_b')))

    checkpoint = tmp_path / 'fixture_head.pt'
    resolved_hash = training.sha256_json(dict(fixture_only=True, candidate=candidate, source_keys=SOURCES[candidate]))
    training.save_checkpoint(checkpoint, model, optimizer, 0, 2, None, resolved_hash,
                             progress=dict(fixture_only=True, preflight_steps=2))
    restored = training.Day05RepresentationModel(day05_config=config, candidate=candidate,
                                                 in_channels=8, adapter_r=32, adapter_d=384)
    restored_optimizer = training.build_optimizer(restored, protocol)
    assert training.load_checkpoint(checkpoint, restored, restored_optimizer, resolved_hash,
                                     torch.device('cpu')) == (0, 2, None)
    assert checkpoint.with_suffix('.pt.sha256').read_text().strip() == training.sha256_file(checkpoint)
    assert all(torch.equal(v, restored.state_dict()[n]) for n, v in model.state_dict().items())
    assert len(restored_optimizer.state) == len(trainable)
    with torch.no_grad():
        torch.testing.assert_close(model(batch)[0], restored(batch)[0], rtol=0, atol=0)
    training.save_json(dict(evidence_scope='CPU_FIXTURE_ONLY', logits_shape=[1, 1, 16, 16],
                            input_feature_shape=[1, 8, 4, 4], checkpoint_sha256=training.sha256_file(checkpoint),
                            resolved_config_sha256=resolved_hash, preflight=report), tmp_path / 'fixture_preflight.json')


@pytest.mark.parametrize('violation', ['frozen', 'duplicate'])
def test_optimizer_ownership_rejects_frozen_and_duplicate_tensors(small_head, violation):
    model, batch, config, protocol = small_head
    optimizer = training.build_optimizer(model, protocol)
    parameter = model.adapters['b4'].gamma
    if violation == 'frozen':
        parameter.requires_grad_(False)
    else:
        optimizer.param_groups[0]['params'].append(parameter)
    with pytest.raises(training.FullTrainError, match=f'optimizer.*{violation}'):
        training.run_preflight(model, [batch], training.make_criterion(config), optimizer,
                               torch.device('cpu'), 2)
