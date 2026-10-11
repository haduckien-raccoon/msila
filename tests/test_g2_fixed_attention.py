"""Quantitative CPU fixtures; these tests provide no pretrained GPU metrics."""
from copy import deepcopy
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from scripts import run_g2_fixed
from scripts.eval_g2 import good_member_path
from src.eval.g1_test_public import evaluate_public
from src.train import g2_e2, g2_fixed
from src.train.g2_fixed_data import IlluminatedNativeDataset, NativeTileSampler
from src.data.loader import G1NativeDataset, G1TileDataset
from src.train.g2_context import PairedG2Tiles, prepare_pair, paired_record
from src.utils.resume import load_checkpoint_payload
from tests.test_g2_tv1_runner import setup, cpu_determinism, assert_nested_equal
from tests.test_g1_test_public import public_fixture
from tests.test_g2_tv1_model import BackboneFixture
from src.models.msila import build_g2_model
from src.losses.anomaly_loss import AnomalySegmentationLoss


def fixed_study(setup):
    settings = yaml.safe_load((g2_fixed.ROOT / 'configs/g2_fixed_attention.yaml').read_text())
    settings.update(model_config=setup['runner']['model_config'], epochs=2, batch_size=2,
                    num_workers=0, precision='fp32', checkpoint_interval_steps=1)
    p = setup['path'].parent / 'fixed.yaml'
    p.write_text(yaml.safe_dump(settings))
    args = SimpleNamespace(config=str(p), repo_dir=setup['cfg']['backbone']['repo_dir'],
                           weights=setup['cfg']['backbone']['checkpoints']['dinov3_vitb16'],
                           data_root=setup['cfg']['data']['root'], device='cpu',
                           output_root=str(setup['root'] / 'fixed'))
    return g2_fixed.prepare_study(args), args


def test_fixed_protocol_geometry_no_selection_lock_and_hashes(setup):
    study, _ = fixed_study(setup)
    context, pools = g2_fixed.make_context(study, 'rice')
    cfg = context['config']
    assert cfg['mode'] == 'fixed' and cfg['selection_status'] == 'user_fixed_no_D10'
    assert cfg['selection_sha256'] is None and context['expected_steps'] == 4
    assert (cfg['adapter']['r'], cfg['adapter']['d']) == (128, 512)
    assert cfg['backbone']['channels'] == 768 and cfg['backbone']['feature_blocks'] == [4, 8, 12]
    assert cfg['fusion']['dim'] == 64 and cfg['fusion']['method'] == 'attention'
    assert cfg['paired_views']['context_size'] == 768
    assert cfg['data']['tile_size'] == 512 and cfg['training']['seed'] == 42
    assert not list(study['root'].rglob('*selection_lock.json'))
    altered = deepcopy(context); altered['config']['adapter']['r'] = 64
    with pytest.raises(ValueError, match='mismatch'):
        g2_fixed.require_fixed_context(altered)
    assert all(r.split == 'train' for r in pools['train'])


def test_illumination_is_shared_native_deterministic_and_mask_preserving(setup):
    study, _ = fixed_study(setup)
    context, pools = g2_fixed.make_context(study, 'rice')
    cfg = context['config']
    options = dict(cfg['illumination'], probability=1.)
    kwargs = dict(seed=42, role='train', variants=2, fixed=False)
    plain = G1NativeDataset(pools['train'], cfg['synthetic_protocol'], **kwargs)
    augmented = IlluminatedNativeDataset(pools['train'], cfg['synthetic_protocol'], illumination=options, **kwargs)
    a = augmented[1]; b = augmented[1]
    assert torch.equal(a['image'], b['image']) and torch.equal(a['mask'], plain[1]['mask'])
    assert a['mask'].sum() > 0 and not torch.equal(a['image'], plain[1]['image'])
    tiles = G1TileDataset(augmented)
    paired = PairedG2Tiles(tiles)
    i = next(i for i, (index, _) in enumerate(tiles.records) if index == 1)
    sample = paired[i]
    local, ctx, meta = prepare_pair(a['image'], paired_record(tiles.records[i][1]), sample_id=a['meta']['sample_id'])
    assert torch.equal(sample['image'], local) and torch.equal(sample['context'], ctx)
    assert sample['view_meta']['local_sample_id'] == sample['view_meta']['context_sample_id']
    assert torch.equal(sample['mask'][..., :96, :128], a['mask'])
    augmented.set_epoch(1)
    changed = augmented[1]
    assert changed['meta']['illumination']['seed'] != a['meta']['illumination']['seed']
    augmented.set_epoch(0)
    assert torch.equal(augmented[1]['image'], a['image'])
    with pytest.raises(ValueError, match='TRAIN'):
        IlluminatedNativeDataset(pools['dev'], cfg['synthetic_protocol'], illumination=options,
                                 seed=17017, role='dev', fixed=True)


def test_native_sampler_exact_coverage_and_epoch_resume_order():
    tiles = SimpleNamespace(records=[(i // 4, None) for i in range(24)])
    class TileFixture:
        records = tiles.records
        def __len__(self): return len(self.records)
    t = TileFixture()
    first = list(NativeTileSampler(t, 42))
    assert sorted(first) == list(range(24)) and first == list(NativeTileSampler(t, 42))
    assert first != list(NativeTileSampler(t, 43))
    for start in range(0, 24, 4):
        assert len({i // 4 for i in first[start:start+4]}) == 1


def test_shared_full_training_checkpoint_resume_and_integrity(setup, monkeypatch):
    study, _ = fixed_study(setup)
    context, pools = g2_fixed.make_context(study, 'rice')
    full, interrupted = study['root'] / 'full/rice', study['root'] / 'interrupted/rice'
    result = g2_e2.train_fixed(context, pools, full, device='cpu')
    assert result['status'] == 'PASS' and result['global_step'] == 4
    assert result['verification_scope'] == 'fixture' and not result['scientific_evidence']
    assert result['frozen_backbone_unchanged'] and result['feature_sources'] == 6
    assert result['adapter_updated'] and result['projection_updated'] and result['fusion_updated']
    assert all(result['adapters_updated'].values()) and result['context_aligned']
    assert result['source_blocks'] == [4, 8, 12] and result['feature_width'] == 64
    assert g2_fixed.completed_training(full, context)
    original = g2_e2.Overfit16Trainer.train_step
    def crash(self, batch, *, step, epoch):
        if step == 2: raise RuntimeError('interruption fixture')
        return original(self, batch, step=step, epoch=epoch)
    monkeypatch.setattr(g2_e2.Overfit16Trainer, 'train_step', crash)
    with pytest.raises(RuntimeError, match='interruption'):
        g2_e2.train_fixed(context, pools, interrupted, device='cpu')
    monkeypatch.setattr(g2_e2.Overfit16Trainer, 'train_step', original)
    resumed = g2_e2.train_fixed(context, pools, interrupted, device='cpu', resume=True)
    assert resumed['global_step'] == 4
    expected, _ = load_checkpoint_payload(full / 'last.pt')
    actual, _ = load_checkpoint_payload(interrupted / 'last.pt')
    assert 'grad_scaler' in actual['metadata']
    assert_nested_equal(expected['model_state'], actual['model_state'])
    assert_nested_equal(expected['optimizer_state'], actual['optimizer_state'])
    with pytest.raises(ValueError, match='real pretrained GPU'):
        g2_fixed.public_evaluation(full, context, device='cpu', model_factory=lambda cfg: None)
    map_path = next((full / 'native_dev/maps').glob('*/score.npy'))
    map_path.write_bytes(b'corrupt fixture')
    assert g2_fixed.completed_training(full, context) is None


class PairedPixelFixture(nn.Module):
    def forward(self, batch):
        assert batch['image'].shape[-2:] == batch['context'].shape[-2:] == (512, 512)
        return 50 * batch['image'][:, :1]


def test_public_reuses_metric_native_hann_exact_orientation_and_aupro(tmp_path):
    cfg = public_fixture(tmp_path)
    cfg['data'].update(tile_size=512, overlap=128)
    cfg.update(paired_views=dict(context_size=768), training=dict(precision='fp32'))
    result = evaluate_public(PairedPixelFixture(), cfg, 'cpu', tmp_path / 'public',
                             predict_fn=g2_fixed.predict_fixed_native, example_limit=2)
    assert result['test_public_aupro_0_05'] == pytest.approx(1.)
    assert result['n_images'] == 2 and result['aupro_max_fpr'] == .05
    assert result['model_selection'].startswith('best.pt selected on synthetic DEV')
    score = np.load(sorted((tmp_path / 'public/examples').glob('*/predicted_map.npy'))[0])
    assert score.shape == (37, 53) and score[-1, -1] > .99 and score[0, 0] < .01


def test_amp_feature_dtype_geometry_six_sources_and_backward_cpu_fixture(setup, monkeypatch):
    """CPU BF16 emulates the contract; it is NOT a GPU throughput/accuracy test."""
    study, _ = fixed_study(setup)
    context, pools = g2_fixed.make_context(study, 'rice')
    class HalfFeatureFixture(BackboneFixture):
        def get_intermediate_layers(self, *args, **kwargs):
            return tuple(t.to(torch.bfloat16) for t in super().get_intermediate_layers(*args, **kwargs))
    monkeypatch.setattr(torch.hub, 'load', lambda **kw: HalfFeatureFixture(kw['model']))
    cfg = context['config']
    native = G1NativeDataset(pools['train'], cfg['synthetic_protocol'], seed=42, role='train')
    batch = g2_e2.paired_tile_collate([PairedG2Tiles(G1TileDataset(native))[1]])
    model = build_g2_model(cfg, 'E5')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits, trace = model(batch, return_trace=True)
        grid = model.aligner.build_sampling_grid(trace['adapted']['context_b4'], trace['geometry'])
    assert grid.dtype == torch.float32  # Geometry never quantized to BF16.
    assert logits.shape == (1, 1, 512, 512)
    assert len(trace['projected']) == 6 and all(t.dtype == torch.bfloat16 for t in trace['projected'].values())
    weights = trace['fusion_trace']['attention']
    assert weights.shape == (1, 6, 32, 32)
    assert torch.allclose(weights.float().sum(1), torch.ones(1, 32, 32), atol=.004)
    loss = AnomalySegmentationLoss()(logits.float(), batch['mask'])['loss']
    loss.backward()
    for module in g2_e2.trainable_modules(model, 'E5').values():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(p.grad is None for p in model.extractor.parameters())


def test_amp_probabilities_are_fp32_before_native_hann(setup):
    study, _ = fixed_study(setup)
    cfg = study['protocol']
    class HalfLogitFixture(nn.Module):
        def forward(self, batch):
            return torch.full((len(batch['image']), 1, 512, 512), .00123, dtype=torch.bfloat16)
    image = torch.zeros(3, 37, 53)
    score = g2_fixed.predict_fixed_native(HalfLogitFixture(), image, cfg, 'cpu')
    expected = torch.tensor(.00123, dtype=torch.bfloat16).float().sigmoid()
    assert score.dtype == torch.float32 and score.shape == (37, 53)
    assert torch.allclose(score, expected.expand_as(score), atol=1e-7)


def test_missing_coverage_never_produces_macro_or_selection(setup):
    study, _ = fixed_study(setup)
    summary = g2_fixed.write_summary(study)
    assert len(summary['categories']) == 8 and summary['status'] == 'INCOMPLETE'
    assert summary['macro_test_public_aupro_005'] is None and summary['selection_lock'] is None
    assert summary['training_pass'] == summary['test_public_pass'] == 0


def test_cpu_cli_preflight_and_failure_are_truthful(setup):
    study, args = fixed_study(setup)
    cfg = public_fixture(setup['path'].parent / 'public_fixture')
    import shutil
    source = Path(cfg['data']['root'])/'rice/test_public'
    shutil.copytree(source, Path(args.data_root)/'rice/TEST_PUBLIC')
    argv = ['--config',args.config,'--data-root',args.data_root,'--repo-dir',args.repo_dir,
            '--weights',args.weights,'--output-root',args.output_root,'--categories','rice',
            '--device','cpu','--preflight']
    assert run_g2_fixed.main(argv) == 0
    summary = json.loads((study['root']/'full/summary.json').read_text())
    assert summary['training_pass'] == summary['test_public_pass'] == 0
    assert not list(study['root'].rglob('*.pt'))
    argv[argv.index(args.weights)] = '/missing/checkpoint.pth'
    assert run_g2_fixed.main(argv) == 1
    failure = json.loads((Path(args.output_root)/'failure.json').read_text())
    assert failure['status'] == 'BLOCKED' and failure['training'] == 'NOT RUN'


@pytest.mark.parametrize('name,relative', [
    ('rice/TEST_PUBLIC/ground_truth/bad/0_mask.png', 'rice/TEST_PUBLIC/ground_truth/bad/0_mask.png'),
    ('rice/ground_truth/TEST_PUBLIC/bad/0_mask.png', 'rice/ground_truth/TEST_PUBLIC/bad/0_mask.png'),
    ('dataset/rice/TEST_PUBLIC/bad/0.png', 'rice/TEST_PUBLIC/bad/0.png'),
])
def test_archive_public_opt_in_keeps_real_gt_layout(name, relative):
    member = tarfile.TarInfo(name)
    assert good_member_path(member, 'rice') is None
    assert str(good_member_path(member, 'rice', include_test_public=True)) == relative
    assert good_member_path(tarfile.TarInfo('rice/TEST_PRIVATE/bad/0.png'), 'rice', include_test_public=True) is None


def test_cli_notebook_contract_and_cpu_cannot_claim_gpu():
    args = ['--data-root','/missing','--repo-dir','/missing','--weights','/missing']
    with pytest.raises(SystemExit): run_g2_fixed.parse_args([*args,'--device','cpu'])
    with pytest.raises(SystemExit): run_g2_fixed.parse_args([*args,'--smoke','--test-public'])
    assert run_g2_fixed.parse_args([*args,'--device','cpu','--preflight']).preflight
    notebook = json.loads((g2_fixed.ROOT/'notebooks/G2_05_E5_Fixed_Attention.ipynb').read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{index}', 'exec')
            assert cell['execution_count'] is None and cell['outputs'] == []
    text = ''.join(''.join(c['source']) for c in notebook['cells'])
    assert 'include_test_public=True' in text and 'g2/integration-D11' in text
    assert "SEED, DEV_SEED = 42, 17017" in text and 'scripts/run_g2_fixed.py' in text
    assert 'joint_selection_lock.json' not in text and 'loss.backward' not in text
