"""Quantitative CPU evidence tests. No fabricated scientific GPU results."""
from copy import deepcopy
import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts import eval_g2 as cli
from src.eval import g2_final as final
from src.train import g2_main as main, g2_e2
from src.train.screen_representation import sha256_file, sha256_json
from tests.test_g2_tv1_d11 import bundle, study, context
from tests.test_g2_tv1_d10 import settings_path
from tests.test_g2_tv1_runner import setup, cpu_determinism

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return yaml.safe_load((ROOT / 'configs/full_scale_synthetic.yaml').read_text())


@pytest.fixture
def maps(tmp_path, protocol):
    rows = []
    # Asymmetric native image, tiny single pixel, all-normal control.
    for index in range(2):
        mask = np.zeros((7, 13), dtype=np.uint8)
        if index:
            mask[1, 9] = 1
        score = np.where(mask, .75, .25).astype(np.float32)
        np.save(tmp_path / f'{index}_score.npy', score)
        np.save(tmp_path / f'{index}_mask.npy', mask)
        rows.append(dict(sample_id=str(index), original_hw=[7, 13], source=dict(path='can/validation/good/a.png', sha256='a'*64),
                         image_sha256=str(index)*64, synthetic=dict(is_anomaly=bool(index),
                         size_bin=protocol['dev_tiny_bins'][0] if index else None),
                         score_path=f'{index}_score.npy', mask_path=f'{index}_mask.npy',
                         score_sha256=sha256_file(tmp_path / f'{index}_score.npy'), mask_sha256=sha256_file(tmp_path / f'{index}_mask.npy')))
    return tmp_path, rows


def test_exact_ground_truth_metrics_and_orientation(maps, protocol):
    directory, rows = maps
    result = final.metrics_from_exports(directory, rows, protocol, 'can')
    assert result == pytest.approx(dict(aupro_005=1., tiny=1., mixed=1., dice=1., p99=.25))
    for row in rows:
        path = directory / row['score_path']
        np.save(path, 1 - np.load(path))
        row['score_sha256'] = sha256_file(path)
    inverse = final.metrics_from_exports(directory, rows, protocol, 'can')
    assert inverse['aupro_005'] == inverse['tiny'] == inverse['mixed'] == 0
    assert inverse['dice'] == 0 and inverse['p99'] == .75


@pytest.mark.parametrize('change', ['transpose', 'nan', 'unbounded', 'checksum', 'duplicate', 'mask', 'unsafe'])
def test_native_evidence_rejected(maps, change):
    directory, rows = maps
    row = rows[0]
    path = directory / row['score_path']
    if change == 'duplicate':
        rows.append(deepcopy(row))
    elif change == 'unsafe':
        row['score_path'] = '../outside.npy'
    elif change == 'checksum':
        row['score_sha256'] = '0' * 64
    elif change == 'mask':
        row['synthetic']['is_anomaly'] = True
    else:
        score = np.load(path)
        if change == 'transpose':
            score = score.T
        else:
            score[0, 0] = np.nan if change == 'nan' else 2.
        np.save(path, score)
        row['score_sha256'] = sha256_file(path)
    with pytest.raises(ValueError):
        final.audit_native_maps(directory, rows)


def test_no_tiny_or_normal_metric_is_not_full_coverage(maps, protocol):
    with pytest.raises(ValueError):
        final.metrics_from_exports(maps[0], maps[1][1:], protocol, 'can')
    with pytest.raises(ValueError):
        final.metrics_from_exports(maps[0], maps[1][:1], protocol, 'can')
    p = deepcopy(protocol); p['prediction_threshold'] = .6
    with pytest.raises(ValueError, match='threshold'):
        final.metrics_from_exports(*maps, p, 'can')


def format_pass(row, value):
    # Reporting-format fixture only, never a checkpoint/experimental artifact.
    return dict(row, status='PASS', reason=None, **{k: value for k in final.METRICS}, params=10,
                checkpoint_sha256='fixture', evidence_sha256='fixture', dev_hash='same_fixture')


def test_80_cells_40_fusions_full8_macros_and_no_partial_mean(tmp_path):
    rows = final.skeleton()
    rows = [format_pass(r, .2 if r['experiment'] == 'E1' else .5) for r in rows]
    target = next(r for r in rows if r['experiment'] == 'E2' and r['category'] == 'walnuts')
    target.update(status='BLOCKED', reason='Fixture missing category')
    summary = final.write_reports(tmp_path, rows)
    assert summary['real_pass'] == 79 and summary['status'] == 'BLOCKED'
    assert len(list(csv.DictReader((tmp_path / 'fusion_comparison.csv').open()))) == 40
    macro = {r['experiment']: r for r in csv.DictReader((tmp_path / 'metrics_g2_macro.csv').open())}
    assert macro['E2']['aupro_005'] == '' and macro['E2']['valid_categories'] == '7'
    assert float(macro['E1']['aupro_005']) == pytest.approx(.2)
    ablation = list(csv.DictReader((tmp_path / 'ablation_g2.csv').open()))
    assert len(ablation) == 8 * 9
    assert next(r for r in ablation if r['contrast'] == 'E2 - E1' and r['category'] == 'macro_8')['aupro_005'] == ''
    assert next(r for r in ablation if r['contrast'] == 'E2 - E1' and r['category'] == 'can')['aupro_005'] == '0.3'
    assert target['aupro_005'] is None


def test_report_rejects_duplicate_missing_and_unverified_pass(tmp_path):
    rows = final.skeleton()
    with pytest.raises(ValueError):
        final.write_reports(tmp_path, rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        final.write_reports(tmp_path, rows)
    rows = final.skeleton(); rows[0]['status'] = 'PASS'
    with pytest.raises(ValueError, match='Unverified'):
        final.write_reports(tmp_path, rows)


def test_missing_lock_accounts_all_cells_without_random_metrics(tmp_path):
    assert cli.main(['--final', '--device', 'cpu', '--experiments', 'E1', 'E2', 'E3', 'E4', 'E5',
                     '--joint-lock', str(tmp_path / 'absent.json'), '--output-root', str(tmp_path / 'out')]) == 2
    out = tmp_path / 'out'
    rows = list(csv.DictReader((out / 'metrics_g2.csv').open()))
    assert len(rows) == 80 and {r['status'] for r in rows} == {'BLOCKED'}
    assert all(not r[k] for r in rows for k in final.METRICS)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['real_pass'] == 0 and summary['experimental_status'] == 'NOT RUN'
    assert summary['selection_performed'] is False and summary['selected_D10'] is None
    assert not list(out.rglob('*.pt')) and not list(out.rglob('*selection_lock.json'))
    assert not list(out.rglob('*.png'))


def test_targeted_final_retains_other_categories_not_run(tmp_path):
    cli.main(['--final', '--device', 'cpu', '--categories', 'vial', '--experiments', 'E2',
              '--output-root', str(tmp_path / 'out')])
    rows = list(csv.DictReader((tmp_path / 'out/metrics_g2.csv').open()))
    assert sum(r['status'] == 'BLOCKED' for r in rows) == 2  # E0 plus E2.
    assert sum(r['status'] == 'NOT RUN' for r in rows) == 78


@pytest.mark.parametrize('flags', [['--final', '--smoke'], ['--final', '--init-joint'],
                                  ['--joint-lock', 'x'], ['--final', '--r', '64', '--d', '256'],
                                  ['--final', '--report-only', '--benchmark']])
def test_invalid_cli_modes(flags):
    with pytest.raises(SystemExit):
        cli.parse_args(flags)


def test_cached_metric_skip_and_hash_repair(maps, protocol, monkeypatch):
    directory, rows = maps
    out = directory / 'final'
    evidence = dict(evidence_sha256='fixture', config=dict(category='can', synthetic_protocol=protocol),
                    native=directory, maps=rows)
    metrics, signature, skipped = final.cached_metrics(out, evidence, {}, resume=True)
    assert not skipped
    out.mkdir()
    path = out / 'metrics.json'
    path.write_text(json.dumps(dict(metrics, status='PASS', evaluation_sha256=signature)))
    (out / 'metrics.json.sha256').write_text(sha256_file(path))
    def trap(*args):
        pytest.fail('Valid cache must avoid metric recomputation')
    original = final.metrics_from_exports
    monkeypatch.setattr(final, 'metrics_from_exports', trap)
    assert final.cached_metrics(out, evidence, {}, resume=True)[2]
    monkeypatch.setattr(final, 'metrics_from_exports', original)
    path.write_text('{}')
    assert not final.cached_metrics(out, evidence, {}, resume=True)[2]


def test_owner_schema_and_duplicate_detection(study, bundle, tmp_path):
    root = tmp_path / 'owner'
    rows = main.skeleton('tv1', bundle['lock'])
    manifest = dict(owner='tv1', selection_sha256=bundle['lock']['sha256'], main_protocol_sha256=study['sha256'])
    main.persist(root, rows, manifest)
    (root / 'training_protocol.json').write_text(json.dumps(study['protocol']))
    parsed, proto = final.read_owner(root / 'main_training_36.csv', 'tv1', bundle)
    assert len(parsed) == 36 and proto == study['protocol']
    rows[-1] = deepcopy(rows[0])
    main.persist(root, rows, manifest)
    with pytest.raises(ValueError, match='Duplicate/missing'):
        final.read_owner(root / 'main_training_36.csv', 'tv1', bundle)


def test_real_cpu_checkpoint_maps_validate_but_cannot_be_experiment(study, bundle, tmp_path, cpu_determinism):
    ctx, pools = context(study, 'E4')
    root = tmp_path / 'owner'
    directory = root / 'main' / ctx['config']['run_id']
    report = g2_e2.train_main(ctx, pools, directory, device='cpu')
    assert main.valid_training(directory, ctx) is not None
    rows = json.loads((directory / 'native_dev/native_maps.json').read_text())
    computed = final.metrics_from_exports(directory / 'native_dev', rows, ctx['config']['synthetic_protocol'], 'can')
    assert computed['aupro_005'] == pytest.approx(report['best_synthetic_dev']['synthetic_dev_aupro_0_05'])
    assert computed['dice'] == report['best_synthetic_dev']['synthetic_dice']
    row = next(r for r in main.skeleton('tv1', bundle['lock']) if r['experiment'] == 'E4' and r['category'] == 'can')
    main.fill_row(row, report, directory, ctx)
    assert row['status'] == 'CPU_PASS' and row['aupro_005'] is None
    with pytest.raises(final.FinalBlocked, match='CPU_PASS'):
        final.audit_main(row, root / 'main_training_36.csv', study['protocol'], bundle)
    # Changing only a CSV status cannot turn mock backbone training into science.
    row['status'] = 'PASS'
    with pytest.raises(ValueError, match='CPU fixture'):
        final.audit_main(row, root / 'main_training_36.csv', study['protocol'], bundle)


def test_backbone_cache_reuses_only_extractor_and_invalidates_mode(monkeypatch, tmp_path):
    calls = []
    def build(cfg, *, experiment, root, extractor=None):
        calls.append(extractor)
        return SimpleNamespace(extractor=extractor or torch.nn.Linear(1, 1).requires_grad_(False),
                               decoder=torch.nn.Linear(1, 1))
    monkeypatch.setattr(cli, 'build_g2_model', build)
    cfg = dict(backbone=dict(name='fixture', repo_dir=str(tmp_path), weights=str(tmp_path / 'weights'),
                            checkpoint_sha256='fixture', source_sha256={}, norm=True))
    cache = cli.EvaluationBackboneCache()
    a, b = cache.build(cfg, 'E1', 'cpu'), cache.build(cfg, 'E2', 'cpu')
    assert a.extractor is b.extractor and a.decoder is not b.decoder
    c, d = cache.build(cfg, 'E3', 'cpu'), cache.build(cfg, 'E5', 'cpu')
    assert c.extractor is d.extractor and c.extractor is not a.extractor
    assert calls[0] is calls[2] is None
    cfg['backbone']['checkpoint_sha256'] = 'changed'
    assert cache.build(cfg, 'E5', 'cpu').extractor is not d.extractor


def test_benchmark_on_cpu_never_reports_latency_or_t4(tmp_path):
    args = cli.parse_args(['--final', '--device', 'cpu', '--benchmark', '--output-root', str(tmp_path)])
    for schema in ('msila.g2.e0.v1', None):
        result = final.benchmark_evidence(dict(config={}, report=dict(schema=schema)), args, cli.EvaluationBackboneCache())
        assert result['status'] == 'BLOCKED' and result['latency'] is result['peak_vram'] is None
        assert 'GPU' in result['reason'] and not final.valid_benchmark(result)


def test_80_metrics_without_cuda_measurements_is_not_g2_exp_pass(tmp_path):
    rows = [format_pass(r, .5) for r in final.skeleton()]
    result = final.write_reports(tmp_path, rows)
    assert result['real_pass'] == 80 and result['benchmark_pass'] == 0
    assert result['status'] == 'BLOCKED'


def test_e1_main_uses_wrapped_state_and_rejects_fixture(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = torch.nn.Module()
            self.extractor.backbone = torch.nn.Linear(1, 1).requires_grad_(False)
            self.decoder = torch.nn.Linear(2, 1)
    saved, built = Model(), Model()
    with torch.no_grad():
        saved.decoder.weight.fill_(.7)
        saved.decoder.bias.fill_(.8)
    payload = dict(config=dict(mode='main'), model_state=g2_e2.trainable_modules(saved, 'E1').state_dict())
    monkeypatch.setattr(cli, 'build_g2_model', lambda *a, **k: built)
    with pytest.raises(cli.Blocked, match='fixture'):
        cli.restore_model(payload, {}, 'E1', 'cpu')
    assert torch.equal(built.decoder.weight, saved.decoder.weight)
    assert torch.equal(built.decoder.bias, saved.decoder.bias)


def test_runbook_cli_flags_match_existing_tv1_and_tv2_interfaces():
    from scripts import run_g2
    text = (ROOT / 'docs/G2_SOLO_RUNBOOK.md').read_text()
    blocks = [part.split('```', 1)[0] for part in text.split('```python\n')[1:]]
    namespace = {'Path': Path}
    names = {'REPO', 'DATA', 'DINO', 'WEIGHTS', 'D10', 'LOCK', 'TV1_D10', 'TV2_D10',
             'MAIN', 'DRIVE_MAIN', 'COMMON', 'E0', 'FINAL', 'DRIVE_FINAL', 'EVAL'}
    namespace['STUDY'] = Path('/content/G2_D11/main_v1/protocol')
    for block in blocks:
        tree = ast.parse(block)
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), 'runbook', 'exec'), namespace)
    args = run_g2.parse_args([*namespace['COMMON'], '--owner', 'tv2', '--resume'])
    assert args.stage == 'main' and args.resume and args.owner == 'tv2'
    evaluated = cli.parse_args([*namespace['EVAL'], '--resume', '--benchmark'])
    assert evaluated.final and evaluated.benchmark and evaluated.experiments == ['E1','E2','E3','E4','E5']


def test_missing_evidence_and_failed_evidence_have_distinct_statuses():
    assert final.error_status(FileNotFoundError('absent')) == 'BLOCKED'
    assert final.error_status(final.FinalBlocked('not trained')) == 'BLOCKED'
    assert final.error_status(ValueError('Native map checksum drift')) == 'FAIL'
