"""D11 evidence intake and reports for eval_g2; no training or selection.

Main training already exports the selected checkpoint's native DEV maps.
Audit those maps and reuse the existing metric implementations instead of
performing feature extraction again. CPU fixtures never become GPU evidence.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.eval.evaluator import MVTEC_AD2_CATEGORIES as CATEGORIES, build_anomaly_map_qa_report, write_metrics_json
from src.eval.full_scale import DEVMetricAccumulator
from src.metrics.segf1 import aggregate_seg_f1
from src.train.screen_representation import sha256_file, sha256_json
from src.utils.resume import load_checkpoint_payload

FUSIONS = ('mean', 'weighted_sum', 'concat', 'gated', 'attention')
EXPERIMENTS = ('E0', 'E1', 'E2', 'E3', 'E4', *(f'E5/{f}' for f in FUSIONS))
METRICS = ('aupro_005', 'tiny', 'mixed', 'dice', 'p99')
FIELDS = ('experiment', 'category', 'status', 'reason', *METRICS, 'params', 'latency', 'peak_vram',
          'benchmark_status', 'checkpoint', 'checkpoint_sha256', 'config_hash', 'selection_sha256',
          'dev_hash', 'feature_sources', 'fusion_sources', 'reuse_of', 'evidence_sha256')


class FinalBlocked(RuntimeError):
    """Required scientific evidence is not available yet."""


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def error_status(exc):
    reason = str(exc)
    return 'BLOCKED' if isinstance(exc, (FinalBlocked, OSError)) or reason.startswith('BLOCKED:') else 'FAIL'


def skeleton():
    return [dict.fromkeys(FIELDS) | dict(experiment=e, category=c, status='NOT RUN', reason='Not requested',
                                        benchmark_status='NOT RUN') for e in EXPERIMENTS for c in CATEGORIES]


def safe_path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe evidence path')
    path = (root / relative).resolve()
    require(root in path.parents, 'Evidence symlink escapes its directory')
    return path


def audit_native_maps(directory, rows):
    require(bool(rows), 'Missing native maps')
    ids = [r['sample_id'] for r in rows]
    require(len(set(ids)) == len(ids), 'Duplicate native sample ID')
    for row in rows:
        score, mask = [safe_path(directory, row[k + '_path']) for k in ('score', 'mask')]
        for kind, path in (('score', score), ('mask', mask)):
            require(sha256_file(path) == row[kind + '_sha256'], 'Native map checksum drift')
        s, m = np.load(score, mmap_mode='r'), np.load(mask, mmap_mode='r')
        require(list(s.shape) == list(m.shape) == row['original_hw'] and s.ndim == 2,
                'Native H/W or orientation mismatch')
        require(np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1,
                'Invalid native score range or values')
        require(np.isin(m, (0, 1)).all(), 'Native mask must be binary')
        require(bool(m.any()) == row['synthetic']['is_anomaly'], 'Synthetic metadata/mask differs')
    return sha256_json([{k: r[k] for k in ('sample_id', 'source', 'original_hw', 'synthetic',
                                         'mask_sha256', 'image_sha256')} for r in rows])


def metrics_from_exports(directory, rows, protocol, category):
    """Exact existing AU-PRO and pooled F1, with normal-only pixel P99.

    Only score/mask evidence is read. No synthetic image or metric is invented.
    Disk-backed accumulators and quantile workspace bound native-image RAM.
    """
    require(protocol['prediction_threshold'] == .5, 'Dice threshold is locked at 0.5')
    audit_native_maps(directory, rows)
    mixed, tiny = [DEVMetricAccumulator(protocol, per_region=False, disk_backed=True) for _ in range(2)]
    normals, samples = [], []
    try:
        for row in rows:
            score = np.load(safe_path(directory, row['score_path']), mmap_mode='r')
            mask = np.load(safe_path(directory, row['mask_path']), mmap_mode='r')
            qa = build_anomaly_map_qa_report([dict(anomaly_map=score, gt_mask=mask, meta=dict(
                image_id=row['sample_id'], category=category, split='dev_synthetic',
                original_hw=row['original_hw'], anomaly_map_space='original_image', gt_mask_space='original_image'))])
            require(qa['summary']['status'] == 'PASS', 'Native map QA failed')
            synthetic = row['synthetic']
            mixed.add(score, mask, split='dev_mixed', image_id=row['sample_id'])
            if not synthetic['is_anomaly'] or synthetic['size_bin'] in protocol['dev_tiny_bins']:
                tiny.add(score, mask, split='dev_tiny', image_id=row['sample_id'])
            if not synthetic['is_anomaly']:
                normals.append(score)
            samples.append(dict(category=category, anomaly_map=score, gt_mask=mask))
        gm, gt = mixed.result()['groups']['all'], tiny.result()['groups']['all']
        require(gm['regions'] > 0 and gt['regions'] > 0 and normals, 'Missing tiny/mixed regions or normal images')
        # Same exact np.quantile rule as g1_e1.evaluate_dev; no pixel subsampling.
        with tempfile.TemporaryDirectory(prefix='g2_final_p99_') as scratch:
            count = sum(a.size for a in normals)
            pixels = np.memmap(Path(scratch) / 'normal.bin', dtype=np.float32, mode='w+', shape=(count,))
            offset = 0
            for array in normals:
                pixels[offset:offset + array.size] = array.ravel()
                offset += array.size
            p99 = float(np.quantile(pixels, .99, overwrite_input=True))
            del pixels
        dice = aggregate_seg_f1(samples, .5)['per_category'][category]['f1']
        result = dict(aupro_005=gm['aupro_0_05'], tiny=gt['aupro_0_05'], mixed=gm['aupro_0_05'], dice=dice, p99=p99)
        require(all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in result.values()),
                'Unavailable or nonfinite final metric')
        return result
    finally:
        for accumulator in (mixed, tiny):
            accumulator._scratch.cleanup()


def read_owner(path, owner, bundle):
    """Strict 36 logical rows; a copied Drive tree changes locations only."""
    from src.train import g2_main as main
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == main.FIELDS, 'Unexpected main manifest schema')
        rows = list(reader)
    expected = main.skeleton(owner, bundle['lock'])
    ids = [r['run_id'] for r in rows]
    require(len(ids) == len(set(ids)) == 36 and set(ids) == {r['run_id'] for r in expected},
            'Duplicate/missing main run IDs or wrong category/seed')
    manifest = json.loads((path.parent / 'run_manifest.json').read_text())
    proto = json.loads((path.parent / 'training_protocol.json').read_text())
    require(manifest['owner'] == owner and manifest['selection_sha256'] == bundle['lock']['sha256']
            and manifest['main_protocol_sha256'] == sha256_json(proto)
            and proto['version'] == 'g2_d11_study_v1'
            and proto['joint_selection_sha256'] == bundle['lock']['sha256'], 'Main protocol/owner/selection drift')
    lookup = {r['run_id']: r for r in expected}
    for row in rows:
        ref = lookup[row['run_id']]
        for key in ('category', 'experiment', 'fusion', 'r', 'd', 'seed', 'selection_sha256'):
            require(str(row[key]) == ('' if ref[key] is None else str(ref[key])), f'Main row {key} differs')
        alternate = main.run_id(row['category'], row['experiment'], row['fusion'] or None, ref['seed'])
        allowed = {alternate}
        if row['experiment'] == 'E5' and row['fusion'] == 'mean':
            allowed.add(main.run_id(row['category'], 'E4', None, ref['seed']))
        require(row['physical_run_id'] in allowed, 'Invalid physical alias')
        if row['physical_run_id'] != alternate:
            require(row['reuse_of'] == row['physical_run_id'], 'Alias lacks reuse provenance')
        else:
            require(not row['reuse_of'], 'Independent run must not claim reuse')
    return rows, proto


def audit_main(row, manifest_path, protocol, bundle):
    from src.train import g2_main as main
    if row['status'] != 'PASS':
        reason = row.get('reason') or f"Main checkpoint status {row['status']}"
        raise (ValueError(reason) if row['status'] in {'FAIL', 'OOM'} else FinalBlocked(reason))
    root = Path(manifest_path).parent
    physical = row['physical_run_id']
    directory = safe_path(root, 'main/' + physical)
    checkpoint = directory / 'best.pt'
    require(Path(row['checkpoint']).as_posix().endswith('/main/' + physical + '/best.pt'),
            'Checkpoint path differs from physical run ID')
    payload, _ = load_checkpoint_payload(checkpoint, require_sha256=True)
    cfg = payload['config']
    context = dict(config=cfg, config_sha256=sha256_json(cfg), expected_steps=cfg['expected_steps'], selection_bundle=bundle)
    main.require_main_context(context)
    require(cfg['mode'] == 'main' and cfg['category'] == row['category'] and cfg['run_id'] == physical
            and cfg['study_sha256'] == sha256_json(protocol) and cfg['git_sha'] == protocol['git_sha'],
            'Checkpoint/main study identity differs')
    require(cfg['stage'] == row['experiment'] or (row['experiment'] == 'E5' and row['fusion'] == 'mean'
            and cfg['stage'] == 'E4' and cfg.get('equivalent_experiments') == ['E4', 'E5/mean']),
            'Only an explicitly equivalent E4 can represent E5 Average')
    if cfg['stage'] == 'E5':
        require(cfg['fusion']['method'] == row['fusion'], 'Fusion/checkpoint mismatch')
    require(row['config_sha256'] == sha256_json(cfg) and row['checkpoint_sha256'] == sha256_file(checkpoint),
            'Manifest config/checkpoint SHA differs')
    report = main.valid_training(directory, context)
    require(report is not None, 'Checkpoint/budget/optimizer/native export audit failed')
    require(report.get('scientific_evidence') is True and report['verification_scope'] == 'real_pretrained'
            and torch.device(report['device']).type == 'cuda', 'CPU fixture/smoke is not experimental evidence')
    for key, metric in (('aupro_005', 'synthetic_dev_aupro_0_05'), ('synthetic_dice', 'synthetic_dice'),
                        ('normal_p99', 'normal_score_p99')):
        require(float(row[key]) == report['best_synthetic_dev'][metric], 'Manifest metric/report differs')
    require(float(row['tiny_aupro_005']) == report['best_synthetic_dev']['dev_tiny']['aupro_0_05']
            and float(row['mixed_aupro_005']) == report['best_synthetic_dev']['dev_mixed']['aupro_0_05']
            and int(row['params']) == report['parameter_report']['trainable_total'], 'Subset/params report differs')
    native = directory / 'native_dev'
    rows = json.loads((native / 'native_maps.json').read_text())
    dev_hash = audit_native_maps(native, rows)
    require(len(rows) == len(cfg['sources']['dev']) * (cfg['data']['dev_variants_per_image'] + 1),
            'Incomplete full synthetic DEV sample count')
    return dict(config=cfg, report=report, checkpoint=checkpoint, native=native, maps=rows, dev_hash=dev_hash,
                evidence_sha256=sha256_json(dict(config=row['config_sha256'], checkpoint=row['checkpoint_sha256'],
                    report=sha256_file(directory / 'metrics.json'), last=report['last_checkpoint_sha256'],
                    exports=sha256_file(native / 'native_maps.json'), qa=sha256_file(native / 'qa_report.json'),
                    selection=bundle['lock']['sha256'], study=sha256_json(protocol))))


def audit_e0(root, category, bundle):
    from src.eval import e0
    from scripts.eval_g2 import source_identity
    directory = Path(root) / category
    cfg = json.loads((directory / 'resolved_config.json').read_text())
    report = json.loads((directory / 'metrics.json').read_text())
    require(report['status'] == 'PASS' and report['mode'] == 'full' and report['schema'] == e0.SCHEMA
            and report['category'] == category and report['verification_scope'] == 'pretrained_dinov3'
            and torch.device(report['device']).type == 'cuda' and report['frozen_backbone_unchanged'] is True,
            'E0 must be a real frozen pretrained CUDA baseline, not a fixture/smoke')
    proto, signature = bundle['protocol'], report['signature']
    settings = deepcopy(cfg)
    for key in ('output_root', 'smoke'):
        settings.pop(key, None)
    settings['backbone'].pop('checkpoints', None)
    require(signature['config_hash'] == e0.object_sha256(settings)
            and signature['source_hash'] == e0.object_sha256(report['source_manifest'])
            and signature['weights_sha256'] == proto['backbone']['checkpoint_sha256']
            and signature['producer_hash'] == e0.object_sha256(proto['backbone']['source_sha256'])
            and cfg['backbone']['name'] == proto['backbone']['name']
            and cfg['backbone']['norm'] == proto['backbone']['norm'], 'E0 backbone/config/source provenance differs')
    require(cfg['synthetic_protocol'] == proto['synthetic_protocol']
            and cfg['evaluation']['dev_seed'] == proto['training']['dev_seed']
            and all(cfg['data'][k] == proto['data'][k] for k in
                    ('tile_size', 'overlap', 'max_dev_sources', 'dev_variants_per_image'))
            and cfg['memory']['sampling_seed'] == proto['training']['seed']
            and sha256_json(source_identity(report['source_manifest'])) == proto['dataset_source_sha256'][category],
            'E0/main TRAIN/DEV sources, seeds or synthetic protocol differ')
    require(report['score_transform'] == e0.SCORE_TRANSFORM and cfg['evaluation']['max_fpr'] == .05
            and report['score_orientation'] == 'higher_is_more_anomalous', 'E0 scoring protocol drift')
    bank = directory / 'normal_memory.pt'
    _, checksum = e0.load_memory(bank, signature, cfg, category)
    require(checksum == report['bank_sha256'], 'E0 bank/report SHA differs')
    for name in ('qa_report.json', 'dev_manifest.json', 'native_maps.json'):
        require(report['evaluation_artifacts'][name] == sha256_file(directory / name), 'E0 export SHA differs; rerun with --save-maps')
    rows = json.loads((directory / 'native_maps.json').read_text())
    require(len(rows) == report['n_samples'] == len(report['source_manifest']['dev']) * (cfg['data']['dev_variants_per_image'] + 1),
            'Incomplete E0 DEV coverage')
    return dict(config=cfg, report=report, checkpoint=bank, native=directory, maps=rows,
                dev_hash=audit_native_maps(directory, rows), evidence_sha256=sha256_json(dict(
                    report=sha256_file(directory / 'metrics.json'), config=sha256_file(directory / 'resolved_config.json'),
                    bank=checksum, exports=sha256_file(directory / 'native_maps.json'), selection=bundle['lock']['sha256'])))


def cached_metrics(output, evidence, implementation, *, resume):
    signature = sha256_json(dict(evidence=evidence['evidence_sha256'], implementation=implementation))
    path = Path(output) / 'metrics.json'
    if resume:
        try:
            saved = json.loads(path.read_text())
            require((path.parent / 'metrics.json.sha256').read_text().strip() == sha256_file(path)
                    and saved['evaluation_sha256'] == signature and saved['status'] == 'PASS', 'Stale final cache')
            require(all(type(saved[k]) in (int, float) and np.isfinite(saved[k]) and 0 <= saved[k] <= 1
                        for k in METRICS), 'Invalid cached metric')
            evidence['saved_result'] = saved
            return {k: saved[k] for k in METRICS}, signature, True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    cfg = evidence['config']
    return metrics_from_exports(evidence['native'], evidence['maps'], cfg['synthetic_protocol'], cfg['category']
                                if 'category' in cfg else evidence['report']['category']), signature, False


def benchmark_evidence(evidence, args, cache):
    """Restore genuine main heads or E0 bank; measure only on real CUDA."""
    from scripts import eval_g2 as cli
    from src.eval.g2_benchmark import BenchmarkUnavailable, benchmark_native
    cfg = evidence['config']
    try:
        require(torch.device(args.device).type == 'cuda' and torch.cuda.is_available(), 'CUDA GPU unavailable')
        if evidence['report'].get('schema') == 'msila.g2.e0.v1':
            from src.eval import e0
            from src.models.dinov3_extractor import DINOv3FeatureExtractor
            relocated = deepcopy(cfg)
            for flag, section, key in (('data_root', 'data', 'root'), ('repo_dir', 'backbone', 'repo_dir'),
                                       ('weights', 'backbone', 'weights')):
                if getattr(args, flag):
                    relocated[section][key] = getattr(args, flag)
            reasons = e0.asset_reasons(relocated, args.device)
            require(not reasons, '; '.join(reasons))
            identity = e0.experiment_identity(relocated)
            require(identity['weights_sha256'] == evidence['report']['signature']['weights_sha256']
                    and identity['producer_hash'] == evidence['report']['signature']['producer_hash'], 'E0 relocated assets differ')
            pools, sources = e0.discover_sources(relocated, evidence['report']['category'])
            require(cli.source_identity(sources) == cli.source_identity(evidence['report']['source_manifest']), 'E0 relocated sources differ')
            key = ('E0', relocated['backbone']['name'], relocated['backbone']['repo_dir'], relocated['backbone']['weights'],
                   identity['weights_sha256'], identity['producer_hash'], relocated['backbone']['norm'], args.device)
            if cache.key != key:
                cache.extractor, cache.key = None, None
                b = relocated['backbone']
                cache.extractor = DINOv3FeatureExtractor(b['repo_dir'], b['weights'], b['name'],
                                     norm=b['norm'], feature_mode='deepest', check_finite=True).to(args.device).eval()
                cache.key = key
            payload, _ = e0.load_memory(evidence['checkpoint'], evidence['report']['signature'], cfg, evidence['report']['category'])
            m = cfg['memory']
            bank = e0.NormalFeatureMemoryBank(payload['vectors'], distance=m['distance'], normalize=m['normalize'],
                            query_chunk_size=m['query_chunk_size'], bank_chunk_size=m['bank_chunk_size'], device=m['device'])
            model = e0.E0MemoryModel(cache.extractor, bank, score_scale=cfg['evaluation']['score_scale'])
            native = cli.G1NativeDataset(pools['dev'], cfg['synthetic_protocol'], seed=cfg['evaluation']['dev_seed'],
                                        role='dev', variants=cfg['data']['dev_variants_per_image'], fixed=True)
            predict = e0.predict_native_e0
        else:
            relocated, pools = cli.relocate_config(cfg, args)
            payload, _ = load_checkpoint_payload(evidence['checkpoint'], require_sha256=True)
            model = cli.restore_model(payload, relocated, cfg['stage'], args.device, cache=cache)
            native = cli.G1NativeDataset(pools['dev'], cfg['synthetic_protocol'], seed=cfg['training']['dev_seed'],
                                        role='dev', variants=cfg['data']['dev_variants_per_image'], fixed=True)
            predict = cli.predict_native_e4 if cfg['stage'] in {'E4', 'E5'} else cli.predict_native
        result = benchmark_native(model, native[0]['image'], relocated, args.device, predict_fn=predict,
                                  warmup=args.benchmark_warmup, repeats=args.benchmark_repeats)
        result.update(checkpoint_sha256=sha256_file(evidence['checkpoint']), config_hash=sha256_json(cfg),
                      score_transform=evidence['report'].get('score_transform', 'sigmoid'))
        return result
    except (BenchmarkUnavailable, OSError, ValueError, KeyError, RuntimeError) as exc:
        return dict(status='BLOCKED', reason=str(exc), latency=None, peak_vram=None)


def valid_benchmark(result):
    return (result.get('status') == 'PASS' and result.get('cpu_fixture') is False
            and result.get('gpu') and result.get('cuda') and str(result.get('device', '')).startswith('cuda')
            and type(result.get('latency')) in (int, float) and np.isfinite(result['latency']) and result['latency'] > 0
            and type(result.get('peak_vram')) is int and result['peak_vram'] > 0
            and result.get('native_hw') and result.get('tile_batch_size')
            and type(result.get('repeats')) is int and result['repeats'] >= 1
            and len(result.get('samples_ms', [])) == result['repeats'])


def write_reports(root, rows, *, selected=None, error=None, provenance=None, render=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(len(rows) == 80 and len({(r['experiment'], r['category']) for r in rows}) == 80
            and {(r['experiment'], r['category']) for r in rows} == {(e, c) for e in EXPERIMENTS for c in CATEGORIES},
            'Final report must account for all 80 cells without duplicates')
    for row in rows:
        if row['status'] == 'PASS':
            require(all(type(row[k]) in (float, int) and np.isfinite(row[k]) and 0 <= row[k] <= 1 for k in METRICS)
                    and row.get('evidence_sha256') and row.get('dev_hash'), 'Unverified PASS row')
        else:
            row.update(dict.fromkeys((*METRICS, 'params', 'latency', 'peak_vram')))
    def csv_file(name, records, fields):
        with (root / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader(); writer.writerows(records)
    csv_file('metrics_g2.csv', rows, FIELDS)
    csv_file('fusion_comparison.csv', [r for r in rows if r['experiment'].startswith('E5/')], FIELDS)
    lookup = {(r['experiment'], r['category']): r for r in rows}
    macro = []
    for exp in EXPERIMENTS:
        group = [lookup[exp, c] for c in CATEGORIES]
        n = sum(r['status'] == 'PASS' for r in group)
        macro.append(dict(experiment=exp, category='macro_8', valid_categories=n, expected_categories=8,
                          status='PASS' if n == 8 else 'INCOMPLETE',
                          **{k: float(np.mean([r[k] for r in group])) if n == 8 else None for k in METRICS}))
    csv_file('metrics_g2_macro.csv', macro, ('experiment', 'category', 'status', 'valid_categories', 'expected_categories', *METRICS))
    contrasts = [('E2', 'E1'), ('E3', 'E2'), ('E4', 'E3'), *((f'E5/{f}', 'E5/mean') for f in FUSIONS)]
    ablation = []
    for later, before in contrasts:
        deltas = []
        for category in CATEGORIES:
            a, b = lookup[later, category], lookup[before, category]
            ok = a['status'] == b['status'] == 'PASS' and a['dev_hash'] == b['dev_hash']
            row = dict(contrast=f'{later} - {before}', category=category, status='PASS' if ok else 'BLOCKED',
                       valid_categories=int(ok), expected_categories=1,
                       **{k: a[k] - b[k] if ok else None for k in METRICS})
            ablation.append(row); deltas.append(row)
        n = sum(r['status'] == 'PASS' for r in deltas)
        ablation.append(dict(contrast=f'{later} - {before}', category='macro_8', status='PASS' if n == 8 else 'INCOMPLETE',
                            valid_categories=n, expected_categories=8,
                            **{k: float(np.mean([r[k] for r in deltas])) if n == 8 else None for k in METRICS}))
    csv_file('ablation_g2.csv', ablation, ('contrast', 'category', 'status', 'valid_categories', 'expected_categories', *METRICS))
    passed = sum(r['status'] == 'PASS' for r in rows)
    physical = {r['checkpoint_sha256'] for r in rows if r['status'] == 'PASS'}
    measured = sum(r['benchmark_status'] == 'PASS' for r in rows)
    summary = dict(schema='g2_final_v1', status='G2 EXP PASS' if passed == measured == 80 else 'BLOCKED',
                   experimental_status='NOT RUN' if not passed else 'INCOMPLETE' if passed < 80 else 'PASS',
                   real_pass=passed, expected_cells=80, fusion_real_pass=sum(r['status'] == 'PASS' for r in rows if r['experiment'].startswith('E5/')),
                   fusion_expected_cells=40, status_counts=dict(Counter(r['status'] for r in rows)),
                   per_experiment={m['experiment']: {'pass': m['valid_categories'], 'expected': 8} for m in macro},
                   completed_physical_artifacts=len(physical), selected_D10=selected, selection_performed=False,
                   main_validation='conditional verification of the unchanged D10 selection; no reselection',
                   selected_main_macro=next((m for m in macro if selected and m['experiment'] == 'E5/' + selected['fusion']), None),
                   split='synthetic DEV', heldout_test_status='NOT RUN', error=error,
                   benchmark_pass=measured, benchmark_expected_cells=80,
                   limits=['Synthetic DEV also used for checkpoint and D10 selection; not held-out generalization evidence.',
                           'Short screening budget and one seed may rank candidates differently from full training.',
                           'Macro means require eight valid categories; missing results remain null.'],
                   provenance=provenance or {}, records=rows)
    write_metrics_json(summary, root / 'g2_final_coverage.json')
    write_metrics_json(summary, root / 'summary.json')
    lines = ['# G2 final report', '', f"Status: **{summary['status']}**. Actual results: **{passed}/80**; Fusion: **{summary['fusion_real_pass']}/40**.",
             '', f'D10 selection (unchanged): `{json.dumps(selected)}`.', '',
             '| Experiment | Valid categories | Macro synthetic DEV AU-PRO@0.05 |', '|---|---:|---:|']
    for m in macro:
        value = '—' if m['aupro_005'] is None else f"{m['aupro_005']:.6f}"
        lines.append(f"| {m['experiment']} | {m['valid_categories']}/8 | {value} |")
    lines += ['', 'E2 uses one Local source; E3 Average uses three Local sources; E4/E5 Average uses six aligned Local/Context sources.',
              'E4 reuse for E5 Average requires an identical physical checkpoint/config and explicit equivalent_experiments.', '',
              'GPU benchmark numbers are published only with the measured GPU, native H/W, batch size and synchronized timing conditions.', '',
              '## Limits', '', *[f'- {s}' for s in summary['limits']], '', 'Held-out TEST/real defects: **NOT RUN**. No tuning on TEST.', '',
              'See CSV rows and per-category attempt_status.json for missing/failed evidence. All artifacts are synthetic DEV only.']
    (root / 'G2_FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    # Draw only genuine paired deltas. A skeleton produces no numeric chart.
    plot_rows = [r for r in ablation if r['category'] == 'macro_8' and r['status'] == 'PASS']
    chart = root / 'figures' / 'ablation_macro.png'
    if render and plot_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        chart.parent.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([r['contrast'] for r in plot_rows], [r['aupro_005'] for r in plot_rows])
        ax.axvline(0, color='black', linewidth=.5); ax.set_xlabel('Macro synthetic DEV ΔAU-PRO@0.05 (8/8)')
        fig.tight_layout(); fig.savefig(chart, dpi=150); plt.close(fig)
    elif render and chart.exists():
        chart.unlink()
    return summary


def run_final(args):
    """Read-only evidence audit. Missing lock blocks, never invokes training."""
    from scripts import eval_g2 as cli
    from src.train import g2_main as main
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows, bundle, global_error = skeleton(), None, None
    provenance = dict(arguments=vars(args), git_sha=cli.git_commit(), implementation=cli.implementation_identity(),
                      torch=str(torch.__version__), cuda_available=torch.cuda.is_available(), training_performed=False,
                      git_dirty=bool(cli.subprocess.check_output(['git', 'status', '--porcelain'], cwd=cli.ROOT, text=True)),
                      tv1_source_sha='2605739571b55caab4418dc8ac3bb648243e728a')
    write_metrics_json(provenance, root / 'run_manifest.json')
    owners, protocols, owner_errors = {}, {}, {}
    try:
        require(args.joint_lock, 'Missing immutable D10 joint selection lock')
        bundle = main.load_selection(args.joint_lock, args.tv1_manifest, args.tv2_manifest)
        # Portable metadata copies; the full checkpoint/map trees stay in owner storage.
        artifacts = root / 'evidence'
        artifacts.mkdir(exist_ok=True)
        cli.copy_atomic(bundle['path'], artifacts / 'locked_D10_selection.json')
        for owner in ('tv1', 'tv2'):
            source = Path(getattr(args, f'{owner}_manifest'))
            for name in (source.name, 'training_protocol.json', 'joint_search_spec.json'):
                cli.copy_atomic(source.parent / name, artifacts / 'D10' / owner / name)
        for owner in ('tv1', 'tv2'):
            path = getattr(args, f'main_{owner}_manifest')
            try:
                if not path:
                    raise FinalBlocked(f'Missing {owner} main_training_36.csv')
                owner_rows, proto = read_owner(path, owner, bundle)
                owners[owner] = {r['run_id']: r for r in owner_rows}
                protocols[owner] = proto
                for name in ('main_training_36.csv', 'run_manifest.json', 'training_protocol.json', 'selection_audit.json'):
                    if (Path(path).parent / name).is_file():
                        cli.copy_atomic(Path(path).parent / name, artifacts / 'D11' / owner / name)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                owners[owner] = str(exc)
                owner_errors[owner] = error_status(exc)
        if len(protocols) == 2:
            require(protocols['tv1'] == protocols['tv2'], 'TV1/TV2 main protocol/code/config parity differs')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        global_error = str(exc)
    physical_cache, category_dev, backbone_cache = {}, {}, cli.EvaluationBackboneCache()
    for row in rows:
        exp, category = row['experiment'], row['category']
        stage = exp.split('/')[0]
        # Final always accounts for E0. Nonrequested main cells are NOT RUN.
        requested = category in args.categories and (stage == 'E0' or stage in args.experiments)
        output = root / exp / category
        if not requested:
            continue
        try:
            if global_error is not None:
                raise FinalBlocked(global_error)
            if stage == 'E0':
                evidence = audit_e0(args.e0_root, category, bundle)
                reuse = None
            else:
                owner = 'tv1' if category in main.OWNERS['tv1'] else 'tv2'
                if not isinstance(owners[owner], dict):
                    raise (FinalBlocked(owners[owner]) if owner_errors[owner] == 'BLOCKED' else ValueError(owners[owner]))
                fusion = exp.split('/')[1] if stage == 'E5' else None
                rid = main.run_id(category, stage, fusion, bundle['lock']['seed'])
                manifest_row = owners[owner][rid]
                cache_key = (owner, manifest_row['physical_run_id'])
                audited = audit_main(manifest_row, getattr(args, f'main_{owner}_manifest'), protocols[owner], bundle)
                if cache_key not in physical_cache:
                    physical_cache[cache_key] = audited
                else:
                    require(physical_cache[cache_key]['evidence_sha256'] == audited['evidence_sha256'], 'Average reuse evidence differs')
                evidence = physical_cache[cache_key]
                reuse = 'E4/' + category if manifest_row['reuse_of'] else None
            require(category not in category_dev or category_dev[category] == evidence['dev_hash'],
                    'Native synthetic DEV input/mask/source parity differs across experiments')
            category_dev[category] = evidence['dev_hash']
            if 'computed_metrics' not in evidence:
                metrics, signature, skipped = cached_metrics(output, evidence, provenance['implementation'],
                                                            resume=args.resume or args.report_only)
                evidence['computed_metrics'] = metrics
            else:
                metrics = evidence['computed_metrics']
                signature = sha256_json(dict(evidence=evidence['evidence_sha256'], implementation=provenance['implementation']))
                skipped = True
            report, cfg = evidence['report'], evidence['config']
            recorded = report if stage == 'E0' else report['best_synthetic_dev']
            for name, val in (('aupro_005', recorded['synthetic_dev_aupro_0_05']),
                              ('tiny', recorded['dev_tiny']['aupro_0_05']), ('mixed', recorded['dev_mixed']['aupro_0_05'])):
                require(np.isclose(metrics[name], val, rtol=0, atol=1e-12), 'Export AU-PRO/report differs')
            if stage != 'E0':
                require(np.isclose(metrics['dice'], recorded['synthetic_dice'], rtol=0, atol=1e-12)
                        and np.isclose(metrics['p99'], recorded['normal_score_p99'], rtol=0, atol=1e-12), 'Export Dice/P99 differs')
            benchmark = evidence.get('benchmark', evidence.get('saved_result', {}).get('benchmark',
                                     dict(status='NOT RUN', reason='Not requested', latency=None, peak_vram=None)))
            if benchmark.get('status') == 'PASS':
                require(valid_benchmark(benchmark) and benchmark.get('checkpoint_sha256') == sha256_file(evidence['checkpoint'])
                        and benchmark.get('config_hash') == sha256_json(cfg), 'Unbound/non-CUDA benchmark evidence')
            if args.benchmark and 'benchmark' not in evidence:
                benchmark = benchmark_evidence(evidence, args, backbone_cache)
                evidence['benchmark'] = benchmark
            row.update(status='PASS', reason=None, **metrics, params=0 if stage == 'E0' else report['parameter_report']['trainable_total'],
                       checkpoint=str(evidence['checkpoint'].resolve()), checkpoint_sha256=sha256_file(evidence['checkpoint']),
                       config_hash=sha256_json(cfg), selection_sha256=bundle['lock']['sha256'], dev_hash=evidence['dev_hash'],
                       feature_sources=1 if stage in {'E0', 'E1', 'E2'} else 3 if stage == 'E3' else 6,
                       fusion_sources=0 if stage in {'E0', 'E1', 'E2'} else 3 if stage == 'E3' else 6, reuse_of=reuse,
                       evidence_sha256=evidence['evidence_sha256'], benchmark_status=benchmark['status'],
                       latency=benchmark.get('latency'), peak_vram=benchmark.get('peak_vram'))
            output.mkdir(parents=True, exist_ok=True)
            write_metrics_json(dict(row, evaluation_sha256=signature, resumed=skipped, native_exports=str(evidence['native']),
                                    benchmark=benchmark, score_transform=report.get('score_transform', 'sigmoid'),
                                    params_scope='trainable head; frozen DINOv3 excluded; E0=0',
                                    resolved_config=cfg,
                                    dice_threshold=.5, split='dev_synthetic', au_pro_fpr=.05), output / 'metrics.json')
            (output / 'metrics.json.sha256').write_text(sha256_file(output / 'metrics.json') + '\n')
            # Reuse real input/mask/score examples; never reconstruct images from scores.
            examples = evidence['native'] / 'examples'
            if examples.exists():
                cli.sync_tree(examples, output / 'examples')
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            row.update(status=error_status(exc), reason=str(exc))
            write_metrics_json(row, output / 'attempt_status.json')
        # Persist coverage after every cell, including interruptions.
        write_reports(root, rows, selected=bundle['lock']['selected'] if bundle else None,
                      error=global_error, provenance=provenance, render=False)
    summary = write_reports(root, rows, selected=bundle['lock']['selected'] if bundle else None,
                            error=global_error, provenance=provenance)
    (root / 'eval.log').write_text(json.dumps({k: v for k, v in summary.items() if k not in ('records', 'provenance')}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('records', 'provenance')}, indent=2))
    return 0 if summary['status'] == 'G2 EXP PASS' else 2
