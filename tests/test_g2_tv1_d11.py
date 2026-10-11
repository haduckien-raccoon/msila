"""D11 technical tests on CPU fixtures; no experiment/selection evidence."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import nbformat
import pytest
import torch
import yaml

from scripts import run_g2
from src.eval import g2_joint as joint, g2_joint_selection as selection
from src.train import g2_e2, g2_main as main
from src.train.screen_representation import sha256_file, sha256_json
from src.utils.resume import load_checkpoint_payload
from tests.test_g2_tv1_runner import setup, cpu_determinism, assert_nested_equal
from tests.test_g2_tv1_d10 import settings_path, study_for
from tests.test_g2_tv2_joint_search import accepted_format_row

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(settings_path, tmp_path):
    """Format-only lock. Inject an audit ONLY at TV2's unit publication boundary."""
    study = study_for(settings_path)
    spec, proto = study['spec'], study['protocol']
    rows = [accepted_format_row(r, .5) for r in joint.skeleton(spec)]
    for row in rows:
        row['git_sha'] = spec['integration_sha']
        row['checkpoint'] = f"/unit/train/{row['run_id']}/best.pt"
        row['evaluation_json'] = f"/unit/evaluation/{row['run_id']}/metrics.json"
    rule = selection.declare_rule(spec, joint.skeleton(spec), tmp_path / 'rule.json', settings_path)
    lock_path = selection.publish_lock(rows, spec, rule, tmp_path / 'selection', audit=lambda row: None)
    return dict(lock=json.loads(lock_path.read_text()), spec=spec, protocol=proto,
                path=str(lock_path), file_sha256=sha256_file(lock_path), audited_cells=360, protocol_sha256=sha256_json(proto),
                owner_manifest_sha256={})


@pytest.fixture
def study(bundle, setup, tmp_path):
    settings = yaml.safe_load((ROOT / 'configs/g2_main.yaml').read_text())
    args = run_g2.parse_args(['--stage', 'adapter_screen', '--config', str(setup['path']),
                             '--device', 'cpu', '--backbone', 'dinov3_vitb16'])
    cfg, runner, _ = run_g2.load_config(args)
    settings['owner'] = 'tv1'
    return main.create_study(cfg, runner, settings, tmp_path / 'D11', bundle, device='cpu')


def context(study, experiment='E5', fusion='attention', **kwargs):
    return main.make_context(study, 'can', experiment, fusion if experiment == 'E5' else None, **kwargs)


def test_missing_lock_blocks_36_cells_without_metrics_or_checkpoints(tmp_path):
    args = run_g2.parse_args(['--stage', 'main', '--prepare', '--device', 'cpu', '--joint-lock', str(tmp_path / 'absent.json'),
                             '--output-root', str(tmp_path)])
    assert main.run_from_cli(args) == 2
    root = tmp_path / 'main_v1' / 'prepared' / 'tv1'
    rows = list(csv.DictReader((root / 'main_training_36.csv').open()))
    assert len(rows) == 36 and {r['status'] for r in rows} == {'BLOCKED'}
    assert {r['category'] for r in rows} == {'can', 'fabric', 'fruit_jelly', 'rice'}
    assert all(not r[k] for r in rows for k in ('aupro_005', 'synthetic_dice', 'checkpoint', 'r', 'd'))
    coverage = json.loads((root / 'main_coverage.json').read_text())
    assert coverage['real_pass'] == 0 and coverage['experimental_status'] == coverage['gpu_status'] == 'NOT RUN'
    assert not list(root.rglob('*.pt')) and not list(root.rglob('*selection_lock.json'))


@pytest.mark.parametrize('change', ['sha', 'selected', 'ranking', 'cells', 'metric', 'budget', 'main_missing', 'same_budget'])
def test_tampered_incomplete_or_unlocked_budget_blocks(bundle, change):
    b = deepcopy(bundle)
    if change == 'sha':
        b['lock']['sha256'] = '0' * 64
    elif change == 'selected':
        b['lock']['selected']['r'] = 256
    elif change == 'ranking':
        b['lock']['ranking'].reverse()
    elif change == 'cells':
        b['lock']['accounted_cells'] = 359
    elif change == 'metric':
        b['lock']['evidence'][0]['aupro_005'] = .9
    elif change == 'budget':
        b['lock']['screening_budget']['epochs'] += 1
    elif change == 'main_missing':
        b['protocol'].pop('main_budget')
    else:
        b['protocol']['main_budget'] = b['lock']['screening_budget']
    if change != 'sha':
        b['lock']['sha256'] = sha256_json({k:v for k,v in b['lock'].items() if k != 'sha256'})
    with pytest.raises((ValueError, KeyError)):
        main.validate_lock(b['lock'], b['spec'], bundle['lock']['evidence'], b['protocol'])


def test_all_36_logical_cells_have_same_locked_pair_and_e4_alias(study):
    rows = main.skeleton('tv1', study['selection_bundle']['lock'])
    assert len(rows) == 36 and len({r['physical_run_id'] for r in rows}) == 32
    assert len({r['run_id'] for r in rows}) == 36
    assert {r['fusion'] for r in rows if r['experiment'] == 'E5'} == set(joint.FUSIONS)
    for row in rows:
        ctx, _ = main.make_context(study, row['category'], row['experiment'], row['fusion'])
        cfg = ctx['config']
        assert cfg['main_budget'] == study['selection_bundle']['protocol']['main_budget']
        assert cfg['training']['batch_size'] == cfg['main_budget']['batch_size']
        if row['experiment'] == 'E1':
            assert row['r'] is row['d'] is None
        else:
            assert (row['r'], row['d']) == (64, 256)
    a, _ = context(study, 'E4')
    b, _ = context(study, 'E5', 'mean')
    assert a['config'] == b['config'] and a['config_sha256'] == b['config_sha256']


@pytest.mark.parametrize('experiment,fusion', [('E1',None), ('E2',None), ('E3',None), ('E4',None),
                                              *[('E5', f) for f in joint.FUSIONS if f != 'mean']])
def test_existing_loop_trains_each_architecture_and_exports_native(study, experiment, fusion, tmp_path):
    ctx, pools = context(study, experiment, fusion)
    epochs = []
    out = tmp_path / 'train'
    result = g2_e2.train_main(ctx, pools, out, device='cpu', on_epoch=lambda directory, epoch: epochs.append(epoch))
    assert result['training_complete'] and result['global_step'] == ctx['expected_steps'] == 4
    assert result['scientific_evidence'] is False and result['verification_scope'] == 'fixture'
    assert result['feature_sources'] == (1 if experiment in ('E1','E2') else 3 if experiment == 'E3' else 6)
    assert result['frozen_backbone_unchanged'] and result['decoder_updated']
    assert main.valid_training(out, ctx) == result
    assert epochs == [1, 2]  # Sync callback fires once per epoch, never per update.
    maps = json.loads((out / 'native_dev' / 'native_maps.json').read_text())
    assert len(maps) == result['best_synthetic_dev']['n_samples'] == 7
    assert all(m['original_hw'] == [96, 128] for m in maps)
    if experiment == 'E1':
        assert result['adapter'] is None and set(result['parameter_report']['groups']) == {'decoder'}
    if experiment == 'E2':
        assert set(result['parameter_report']['groups']) == {'adapter', 'decoder'}
    row = main.skeleton('tv1', study['selection_bundle']['lock'])[0]
    main.fill_row(row, result, out, ctx)
    assert row['status'] == 'CPU_PASS' and row['aupro_005'] is row['synthetic_dice'] is None


@pytest.mark.parametrize('field', ['r', 'budget', 'seed', 'loss', 'order', 'source'])
def test_main_context_rejects_scientific_overrides(study, field):
    ctx, _ = context(study)
    if field == 'r': ctx['config']['adapter']['r'] = 128
    elif field == 'budget': ctx['config']['training']['batch_size'] *= 2
    elif field == 'seed': ctx['config']['training']['seed'] += 1
    elif field == 'loss': ctx['config']['loss']['dice_weight'] += 1
    elif field == 'order': ctx['config']['fusion']['feature_order'].reverse()
    else: ctx['config']['sources']['dev'][0]['sha256'] = 'f' * 64
    ctx['config_sha256'] = sha256_json(ctx['config'])
    with pytest.raises(ValueError, match='BLOCKED'):
        main.require_main_context(ctx)


def test_e4_e5_average_models_state_logits_and_gradient_are_identical(study):
    ctx, pools = context(study, 'E4')
    cfg = ctx['config']
    f = g2_e2.FrozenE2Factory(cfg, device='cpu', experiment='E4')
    a = f(cfg)
    f.experiment = 'E5'
    b = f(cfg)
    assert a.extractor is b.extractor
    assert_nested_equal(g2_e2.trainable_modules(a, 'E4').state_dict(), g2_e2.trainable_modules(b, 'E5').state_dict())
    from src.data.loader import G1NativeDataset, G1TileDataset
    from src.train.g2_context import PairedG2Tiles, paired_tile_collate, paired_model_forward
    native = G1NativeDataset(pools['train'], cfg['synthetic_protocol'], seed=2026, role='train', variants=2, fixed=False)
    tiles = PairedG2Tiles(G1TileDataset(native, tile_size=512, overlap=128), context_size=768)
    batch = paired_tile_collate([tiles[0]])
    x, y = paired_model_forward(a, batch), paired_model_forward(b, batch)
    assert torch.equal(x, y)
    x.mean().backward(); y.mean().backward()
    assert all(torch.equal(pa.grad, pb.grad) for pa,pb in zip(a.decoder.parameters(), b.decoder.parameters()))


@pytest.mark.parametrize('experiment,fusion', [('E1',None), ('E3',None), ('E5','gated')])
def test_resume_after_epoch_boundary_matches_uninterrupted(study, experiment, fusion, tmp_path):
    ctx, pools = context(study, experiment, fusion)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    g2_e2.train_main(ctx, pools, full, device='cpu')
    def stop(directory, epoch):
        if epoch == 1: raise RuntimeError('Colab interrupted after epoch sync')
    with pytest.raises(RuntimeError, match='interrupted'):
        g2_e2.train_main(ctx, pools, resumed, device='cpu', on_epoch=stop)
    g2_e2.train_main(ctx, pools, resumed, device='cpu', resume=True)
    a,_ = load_checkpoint_payload(full / 'last.pt', require_sha256=True)
    b,_ = load_checkpoint_payload(resumed / 'last.pt', require_sha256=True)
    for k in ('model_state', 'optimizer_state', 'training_state'):
        assert_nested_equal(a[k], b[k])
    assert torch.equal(a['rng_state']['torch_cpu'], b['rng_state']['torch_cpu'])
    assert main.valid_training(resumed, ctx)


def test_corrupt_map_config_or_checkpoint_is_never_skipped(study, tmp_path):
    ctx, pools = context(study, 'E2')
    out = tmp_path / 'run'
    g2_e2.train_main(ctx, pools, out, device='cpu')
    path = out / 'resolved_config.yaml'; original = path.read_bytes()
    path.write_text('invalid: config')
    assert main.valid_training(out, ctx) is None
    path.write_bytes(original)
    maps = json.loads((out / 'native_dev' / 'native_maps.json').read_text())
    path = out / 'native_dev' / maps[0]['score_path']; original = path.read_bytes()
    path.write_bytes(b'corrupt')
    assert main.valid_training(out, ctx) is None
    path.write_bytes(original)
    path = out / 'best.pt'; original = path.read_bytes()
    path.write_bytes(b'corrupt')
    assert main.valid_training(out, ctx) is None
    path.write_bytes(original)
    assert main.valid_training(out, ctx)


def test_main_smoke_separate_budget_and_no_coverage_credit(study, tmp_path):
    ctx, pools = context(study, 'E5', 'attention', smoke=True)
    out = tmp_path / 'smoke'
    report = g2_e2.train_main(ctx, pools, out, device='cpu')
    assert report['global_step'] == 2 and not report['scientific_evidence']
    assert main.valid_training(out, ctx)
    rows = main.skeleton('tv1', study['selection_bundle']['lock'])
    main.fill_row(rows[-1], report, out, ctx)
    coverage = main.persist(tmp_path / 'manifest', rows, {})
    assert coverage['real_pass'] == coverage['completed_physical_runs'] == 0


def test_g1_incompatible_checkpoint_is_rejected(study, tmp_path):
    ctx, pools = context(study, 'E1')
    out = tmp_path / 'bad_g1'; out.mkdir()
    torch.save({'not_trained': True}, out / 'best.pt')
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        main.valid_g1_checkpoint(out / 'best.pt', ctx, pools)


def test_main_cli_cannot_override_locked_pair():
    with pytest.raises(SystemExit):
        run_g2.parse_args(['--stage','main','--r','64','--d','256'])


def test_colab_notebook_valid_and_main_is_opt_in():
    book = nbformat.read(ROOT / 'notebooks/G2_03_Training.ipynb', as_version=4)
    nbformat.validate(book)
    code = '\n'.join(c.source for c in book.cells if c.cell_type == 'code')
    assert 'RUN_MAIN_TRAINING = False' in code and "'--stage', 'main'" in code
    assert all(c.execution_count is None and not c.outputs for c in book.cells if c.cell_type == 'code')


def write_owner_evidence(bundle, tmp_path):
    paths = []
    for owner, cats in main.OWNERS.items():
        directory = tmp_path / owner; directory.mkdir()
        (directory / 'joint_search_spec.json').write_text(json.dumps(bundle['spec']))
        (directory / 'training_protocol.json').write_text(json.dumps(bundle['protocol']))
        rows = [r for r in bundle['lock']['evidence'] if r['category'] in cats]
        path = directory / f'{owner}_joint_search_180.csv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=joint.FIELDS)
            writer.writeheader()
            writer.writerows({**r,'budget':json.dumps(r['budget'])} for r in rows)
        paths.append(path)
    return paths


def test_format_lock_without_actual_360_checkpoint_exports_cannot_authorize_main(bundle, tmp_path):
    paths = write_owner_evidence(bundle, tmp_path)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        main.load_selection(bundle['path'], *paths)


def test_lock_intake_calls_unchanged_tv2_auditor_for_every_cell(bundle, tmp_path, monkeypatch):
    paths = write_owner_evidence(bundle, tmp_path)
    calls = []
    # Publication-format unit only; do not run training with this I/O injection.
    monkeypatch.setattr(selection, 'evidence_auditor', lambda spec, protocols: lambda row: calls.append(row['run_id']))
    loaded = main.load_selection(bundle['path'], *paths)
    assert len(calls) == len(set(calls)) == loaded['audited_cells'] == 360
    assert loaded['protocol_sha256'] == sha256_json(bundle['protocol'])


def main_args(setup, tmp_path, *extra):
    settings = yaml.safe_load((ROOT / 'configs/g2_main.yaml').read_text())
    settings.update(base_config=str(setup['path']), output_root=str(tmp_path / 'runner_main'))
    path = tmp_path / 'main.yaml'; path.write_text(yaml.safe_dump(settings))
    return run_g2.parse_args(['--stage','main','--config',str(path),'--backbone','dinov3_vitb16','--device','cpu',*extra])


def test_valid_lock_but_missing_colab_gpu_blocks_and_preserves_protocol(bundle, setup, tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'load_selection', lambda *args: bundle)
    args = main_args(setup, tmp_path)
    assert main.run_from_cli(args) == 2
    paths = list((tmp_path / 'runner_main').glob('main_v1/*/tv1/main_coverage.json'))
    assert len(paths) == 1
    report = json.loads(paths[0].read_text())
    assert report['real_pass'] == 0 and report['gpu_status'] == 'NOT RUN'
    assert report['status_counts'] == {'BLOCKED': 36}
    assert (paths[0].parent / 'training_protocol.json').is_file()
    assert not list(paths[0].parent.rglob('*.pt'))


def test_factory_oom_and_e4_average_alias_failures_survive_report_only(bundle, setup, tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'load_selection', lambda *args: bundle)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'MOCK OOM; no GPU executed')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    class OOMFactory:
        def __init__(self, *args, **kwargs):
            raise torch.cuda.OutOfMemoryError('unit injected OOM before backbone allocation')
    monkeypatch.setattr(g2_e2, 'FrozenE2Factory', OOMFactory)
    args = main_args(setup, tmp_path, '--categories','can','--experiments','E4','--max-jobs','1')
    args.device = 'cuda'
    assert main.run_from_cli(args) == 2
    args.report_only = True
    assert main.run_from_cli(args) == 2
    paths = list((tmp_path / 'runner_main').glob('main_v1/*/tv1/main_training_36.csv'))
    rows = list(csv.DictReader(paths[0].open()))
    failed = [r for r in rows if r['status'] == 'FAIL/OOM']
    assert len(failed) == 2 and {r['experiment'] for r in failed} == {'E4','E5'}
    assert all(not r['aupro_005'] and not r['checkpoint'] for r in failed)
    assert len(list(paths[0].parent.rglob('failure.json'))) == 1


def test_g1_same_scientific_protocol_is_reused_without_optimizer_updates(study, tmp_path, monkeypatch):
    from src.train import g1_e1
    ctx, pools = context(study, 'E1')
    cfg = deepcopy(ctx['config'])
    cfg['version'] = 'g1_e1_v1'; cfg['training']['mode'] = 'train'
    cfg['decoder'] = {k: cfg['decoder'][k] for k in ('hidden_channels','deterministic_resize')}
    old_dir = tmp_path / 'G1'; old_dir.mkdir()
    g1_e1.run(cfg, device='cpu', output_dir=old_dir)
    originals = {p.name: sha256_file(p) for p in old_dir.iterdir() if p.is_file()}
    assert main.valid_g1_checkpoint(old_dir / 'best.pt', ctx, pools)
    def forbidden_update(*args, **kwargs):
        raise AssertionError('Compatible G1 must not train again')
    monkeypatch.setattr(g2_e2.Overfit16Trainer, 'train_step', forbidden_update)
    out = tmp_path / 'D11_reuse'
    result = g2_e2.train_main(ctx, pools, out, device='cpu', reuse_checkpoint=old_dir / 'best.pt')
    assert result['reuse_g1_sha256'] == originals['best.pt'] and not result['scientific_evidence']
    assert main.valid_training(out, ctx)
    for p in old_dir.iterdir():
        if p.is_file(): assert sha256_file(p) == originals[p.name]
    best,_ = load_checkpoint_payload(out / 'best.pt', require_sha256=True)
    last,_ = load_checkpoint_payload(out / 'last.pt', require_sha256=True)
    old_best,_ = load_checkpoint_payload(old_dir / 'best.pt', require_sha256=True)
    old_last,_ = load_checkpoint_payload(old_dir / 'last.pt', require_sha256=True)
    assert_nested_equal(best['model_state']['decoder.head.0.weight'], old_best['model_state']['head.0.weight'])
    assert best['training_state']['global_step'] == old_best['training_state']['global_step']
    assert last['training_state']['global_step'] == old_last['training_state']['global_step']
    assert torch.equal(best['rng_state']['torch_cpu'], old_best['rng_state']['torch_cpu'])
    assert torch.equal(last['rng_state']['torch_cpu'], old_last['rng_state']['torch_cpu'])


def test_drive_namespace_matches_local_layout_for_restore(bundle, setup, tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'load_selection', lambda *args: bundle)
    args = main_args(setup, tmp_path, '--preflight', '--drive-output', str(tmp_path / 'drive'))
    assert main.run_from_cli(args) == 0
    local = next((tmp_path / 'runner_main').glob('main_v1/*/tv1/main_coverage.json'))
    mirror = tmp_path / 'drive' / local.relative_to(tmp_path / 'runner_main')
    assert mirror.read_bytes() == local.read_bytes()
    assert len(list(mirror.parent.glob('plan/**/resolved_config.yaml'))) == 32


def test_coverage_counts_g1_reuse_and_average_alias_without_inflating_new_training(tmp_path):
    # Counter-format unit only: no experiment metrics/checkpoints are fabricated.
    rows = main.skeleton('tv1')
    rows[0].update(status='PASS', reuse_g1_checkpoint='/format-only/G1/best.pt')
    rows[3]['status'] = 'PASS'  # E4
    rows[4]['status'] = 'PASS'  # E5 mean aliases that same physical E4
    result = main.persist(tmp_path / 'counter_format', rows, {})
    assert result['real_pass'] == 3 and result['completed_physical_runs'] == 2
    assert result['g1_reused_physical_runs'] == result['newly_trained_physical_runs'] == 1
