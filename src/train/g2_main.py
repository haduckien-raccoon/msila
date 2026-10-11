"""D11 TV1/TV2 orchestration of the existing E1--E5 trainer.

Only a re-audited TV2 D10 lock authorizes main training. No lock publication,
TEST selection, new metric, or GPU execution during local preparation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import gc
import json
import logging
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from src.eval import g2_joint as joint, g2_joint_selection as selection
from src.train.g2_joint_search import absolute, require, PAIRS, ROOT, OWNERS
from src.train.screen_representation import read_yaml, save_json, save_yaml, sha256_file, sha256_json
from src.utils.resume import load_checkpoint_payload

EXPERIMENTS = ('E1', 'E2', 'E3', 'E4', 'E5')
FIELDS = ('category', 'experiment', 'fusion', 'r', 'd', 'seed', 'run_id', 'physical_run_id',
          'status', 'reason', 'config_sha256', 'selection_sha256', 'checkpoint', 'checkpoint_sha256',
          'aupro_005', 'tiny_aupro_005', 'mixed_aupro_005', 'synthetic_dice', 'normal_p99',
          'params', 'reuse_of', 'reuse_g1_checkpoint', 'reuse_g1_sha256', 'git_sha')


def validate_lock(payload, spec, rows, protocol):
    """Read-only verification against the exact immutable TV2 publication format."""
    body = deepcopy(payload)
    checksum = body.pop('sha256', None)
    require(checksum == sha256_json(body), 'BLOCKED: joint lock checksum mismatch')
    require(body.get('version') == 'g2_joint_selection_lock_v1', 'BLOCKED: unsupported joint lock version')
    decision = selection.decision(rows, spec, body.get('rule'))
    require(decision['status'] == 'READY', 'BLOCKED: joint search is incomplete or rule is missing')
    evidence = sorted(deepcopy(rows), key=lambda row: row['run_id'])
    expected = dict(spec_sha256=sha256_json(spec), rule_sha256=sha256_json(body['rule']),
                    evidence_sha256=sha256_json(evidence), training_git_sha=spec['integration_sha'],
                    seed=spec['seed'], screening_budget=spec['budget'], expected_cells=360,
                    accounted_cells=360, valid_metric_cells=360, eligible_candidates=45,
                    excluded_infeasible_candidates=[], scope='all_45_candidates', evidence=evidence,
                    ranking=decision['ranking'], limitations=list(selection.LIMITATIONS),
                    selected={k: decision['selected'][k] for k in ('r', 'd', 'fusion', 'aupro_005', 'params')})
    require(all(body.get(k) == v for k, v in expected.items()), 'BLOCKED: lock differs from D10 evidence/decision')
    require(protocol.get('version') == 'g2_d10_study_v1' and protocol.get('git_sha') == spec['integration_sha']
            and protocol.get('screening_budget') == spec['budget'], 'BLOCKED: missing original D10 training protocol')
    for cat in joint.CATEGORIES:
        group = [r for r in rows if r['category'] == cat]
        require(len({(r['protocol_hash'], r['dev_hash']) for r in group}) == 1,
                'BLOCKED: D10 synthetic DEV/protocol parity differs')
    main_budget = protocol.get('main_budget')
    joint.validate_budget(main_budget)
    require(main_budget != spec['budget'] and main_budget['max_minutes'] is None,
            'BLOCKED: locked main budget must differ from screening and cannot truncate by wall time')
    require((body['selected']['r'], body['selected']['d']) in PAIRS, 'BLOCKED: locked pair outside verified ViT-B grid')
    require(protocol['backbone']['name'] == 'dinov3_vitb16' and protocol['backbone']['channels'] == 768,
            'BLOCKED: main protocol expects the verified ViT-B/16 backbone')
    return payload


def load_selection(lock_path, tv1_manifest, tv2_manifest):
    """Audit every checkpoint/native export using unchanged TV2 code; no GPU."""
    lock_path = absolute(lock_path)
    require(lock_path.is_file(), f'BLOCKED: missing joint_selection_lock.json: {lock_path}')
    require(tv1_manifest and tv2_manifest, 'BLOCKED: original TV1 and TV2 D10 manifests/evidence are required')
    paths = dict(tv1=absolute(tv1_manifest), tv2=absolute(tv2_manifest))
    rows, spec, owners, protocols = selection.merge_owners(paths['tv1'], paths['tv2'])
    require(set(protocols) == {'tv1', 'tv2'}, 'BLOCKED: both original training protocols are required')
    lock = json.loads(lock_path.read_text())
    protocol = protocols['tv1']
    validate_lock(lock, spec, rows, protocol)
    audit = selection.evidence_auditor(spec, protocols)
    # Copying evidence from Drive changes only its location, never signed content.
    for owner, path in paths.items():
        part = [r for r in rows if r['category'] in OWNERS[owner]]
        for row in selection.relocate(part, path.parent):
            audit(row)
    return dict(lock=lock, spec=spec, protocol=protocol, path=str(lock_path),
                file_sha256=sha256_file(lock_path), audited_cells=360,
                protocol_sha256=sha256_json(protocol),
                owner_manifest_sha256={k: v['manifest_sha256'] for k, v in owners.items()})


def require_main_context(context):
    cfg = context['config']
    bundle = context.get('selection_bundle', {})
    require(bundle.get('audited_cells') == 360, 'BLOCKED: D11 requires an audited D10 selection bundle')
    lock, protocol = bundle['lock'], bundle['protocol']
    require(sha256_json(protocol) == bundle['protocol_sha256'], 'BLOCKED: audited D10 protocol changed')
    require(sha256_file(bundle['path']) == bundle['file_sha256'], 'BLOCKED: joint lock changed after preflight')
    validate_lock(lock, bundle['spec'], lock['evidence'], protocol)
    require(context['config_sha256'] == sha256_json(cfg) and cfg['expected_steps'] == context['expected_steps'],
            'BLOCKED: main config/step hash drift')
    require(cfg.get('stage') in EXPERIMENTS and cfg.get('mode') in {'main', 'main_smoke'}
            and cfg.get('selection_status') == 'locked_joint_D10'
            and cfg.get('selection_sha256') == lock['sha256']
            and cfg.get('adapter_pair_role') == ('not_applicable' if cfg['stage'] == 'E1' else 'locked_joint_pair'),
            'BLOCKED: unbound main experiment/selection')
    require(cfg['main_budget'] == protocol['main_budget'] and cfg['screening_budget'] == lock['screening_budget'],
            'BLOCKED: main/screening budget changed after selection')
    if cfg['stage'] != 'E1':
        require((cfg['adapter']['r'], cfg['adapter']['d']) == (lock['selected']['r'], lock['selected']['d']),
                'BLOCKED: r/d must equal the D10 locked pair')
    budget = {k: cfg['training'][k] for k in joint.BUDGET_FIELDS}
    if cfg['mode'] == 'main':
        require(budget == cfg['main_budget'], 'BLOCKED: main training budget differs from locked protocol')
    else:
        require(budget['epochs'] == 1 and budget['max_steps'] in (1, 2) and context['expected_steps'] <= 2
                and budget['batch_size'] == cfg['main_budget']['batch_size'] and budget['max_minutes'] is None,
                'BLOCKED: main smoke is bounded and uses the locked batch')
    # All scientific training variables stay equal to D10; only the budget differs.
    reference = protocol['training']
    for key in reference.keys() - set(joint.BUDGET_FIELDS):
        require(cfg['training'].get(key) == reference[key], f'BLOCKED: locked training field changed: {key}')
    for key in ('loss', 'synthetic_protocol', 'decoder', 'checkpoint', 'paired_views'):
        require(cfg[key] == protocol[key], f'BLOCKED: locked {key} changed')
    b = cfg['backbone']
    require(all(b[k] == protocol['backbone'][k] for k in ('name', 'norm', 'channels', 'patch_size', 'deepest_block'))
            and b['checkpoint_sha256'] == protocol['backbone']['checkpoint_sha256']
            and b['source_sha256'] == protocol['backbone']['source_sha256'] and b['frozen'] is True,
            'BLOCKED: pretrained backbone differs from D10')
    if cfg['stage'] in {'E3', 'E4'}:
        require(cfg['fusion']['method'] == 'mean', 'BLOCKED: E3/E4 are Average ablation baselines')
    if cfg['stage'] == 'E5':
        require(cfg['fusion']['method'] in joint.FUSIONS, 'BLOCKED: invalid E5 Fusion')
    require({k: v for k, v in cfg['fusion'].items() if k != 'method'} ==
            {k: v for k, v in protocol['fusion'].items() if k != 'method'}, 'BLOCKED: shared Fusion dimensions/order changed')
    from scripts.eval_g2 import source_identity
    if cfg['mode'] == 'main':
        require({k: v for k, v in cfg['data'].items() if k != 'root'} ==
                {k: v for k, v in protocol['data'].items() if k != 'root'}, 'BLOCKED: locked dataset configuration changed')
        require(sha256_json(source_identity(cfg['sources'])) == protocol['dataset_source_sha256'][cfg['category']],
                'BLOCKED: main TRAIN/DEV source split differs from D10')


def valid_g1_checkpoint(path, context, pools):
    """A best-only, overfit, different-budget or different-split G1 is retrained."""
    from scripts.eval_g2 import load_trained_checkpoint, scientific_identity, validate_e1_budget
    cfg = context['config']
    payload = load_trained_checkpoint(path, 'E1', cfg['category'])
    require(scientific_identity(payload['config']) == scientific_identity(cfg), 'G1/main scientific protocols differ')
    require(payload['config']['backbone'].get('source_sha256') == cfg['backbone']['source_sha256'],
            'G1 DINO implementation provenance differs')
    require(validate_e1_budget(path, payload, cfg, pools), 'G1 full training budget cannot be verified')
    payload['_last_payload'], _ = load_checkpoint_payload(Path(path).parent / 'last.pt', require_sha256=True)
    return payload


def load_settings(args):
    from scripts import run_g2 as legacy
    path = absolute(args.config)
    if path == ROOT / 'configs/g2_runner.yaml':
        path = ROOT / 'configs/g2_main.yaml'
    settings = read_yaml(path)
    require(settings.get('version') == 'g2_d11_runner_v1', 'Expected g2_d11_runner_v1')
    require(settings.get('drive_sync') == 'epoch_and_job_end' and settings.get('oom_policy') == 'record_failure_no_automatic_batch_change',
            'D11 requires epoch/job sync and a fixed locked batch')
    base_args = deepcopy(args)
    base_args.config, base_args.categories = str(absolute(settings['base_config'])), ['all']
    cfg, runner, _ = legacy.load_config(base_args)
    require(cfg['backbone']['name'] == 'dinov3_vitb16', 'D11 uses the verified ViT-B/16 search')
    for flag, key in (('joint_lock', 'joint_lock'), ('tv1_manifest', 'tv1_manifest'), ('tv2_manifest', 'tv2_manifest')):
        settings[key] = getattr(args, flag) or settings.get(key)
    name = args.run_id or settings['run_id']
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) is not None, 'Invalid run_id')
    owned = OWNERS[args.owner]
    requested = [c for group in args.categories for c in group.split(',')]
    if requested == ['all']:
        requested = list(owned)
    require(requested and len(set(requested)) == len(requested) and set(requested).issubset(owned), 'Categories outside owner assignment')
    args.categories = [c for c in owned if c in requested]
    from src.models.fusion_factory import canonical_fusion_method
    args.fusions = [canonical_fusion_method(f) for f in args.fusions] if args.fusions else list(joint.FUSIONS)
    require(len(set(args.fusions)) == len(args.fusions), 'Duplicate Fusion filter')
    args.experiments = args.experiments or list(EXPERIMENTS)
    return cfg, runner, settings, absolute(args.output_root or settings['output_root']) / name


def create_study(cfg, runner, settings, root, bundle, *, device):
    from scripts import run_g2 as legacy
    from scripts.eval_g2 import source_identity
    prototype = deepcopy(bundle['protocol'])
    # The D10 protocol is authoritative, including dataset caps, seeds and main budget.
    prototype['backbone']['repo_dir'] = cfg['backbone']['repo_dir']
    prototype['backbone']['weights'] = cfg['backbone']['weights']
    prototype['data']['root'] = cfg['data']['root']
    require(sha256_file(prototype['backbone']['weights']) == prototype['backbone']['checkpoint_sha256'],
            'BLOCKED: DINO checkpoint checksum differs from lock')
    dino_root = Path(prototype['backbone']['repo_dir'])
    hashes = {str(p.relative_to(dino_root)): sha256_file(p) for p in sorted(dino_root.rglob('*.py'))}
    require(hashes == prototype['backbone']['source_sha256'], 'BLOCKED: DINO source differs from lock')
    from src.train.g1_e1 import check_assets
    check_assets(prototype, device)
    prototype['training'].update(prototype['main_budget'])
    prototype.update(version='g2_d11_study_v1', joint_selection_sha256=bundle['lock']['sha256'],
                     screening_git_sha=bundle['lock']['training_git_sha'],
                     git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                     runtime={'torch': str(torch.__version__), 'cuda': torch.version.cuda},
                     average_reuse=bool(settings['reuse_e4_average']))
    source_files = set(prototype['source_code_sha256']) | {'src/train/g2_main.py'}
    prototype['source_code_sha256'] = {p: sha256_file(ROOT / p) for p in sorted(source_files)}
    sha = sha256_json(prototype)
    study = dict(protocol=prototype, sha256=sha, root=root / sha / settings['owner'], contexts={},
                 selection_bundle=bundle, settings=settings, commit=prototype['git_sha'])
    # Verify all eight source hashes, so TV2 can use the identical main protocol.
    for category in joint.CATEGORIES:
        context, _ = legacy.make_context(study, category, PAIRS[0], 'adapter_screen')
        require(sha256_json(source_identity(context['config']['sources'])) == prototype['dataset_source_sha256'][category],
                f'BLOCKED: dataset/source split changed: {category}')
    return study


def make_context(study, category, experiment, fusion=None, *, smoke=False):
    from scripts import run_g2 as legacy
    lock = study['selection_bundle']['lock']
    pair = (lock['selected']['r'], lock['selected']['d'])
    physical = 'E4' if experiment == 'E5' and fusion == 'mean' and study['protocol']['average_reuse'] else experiment
    context, pools = legacy.make_context(study, category, pair, 'adapter_screen', smoke=smoke)
    cfg = context['config']
    # Labels are canonicalized before hashing: E4 and its E5-Average alias use exactly one resolved config.
    cfg.update(stage=physical, mode='main_smoke' if smoke else 'main', selection_status='locked_joint_D10',
               selection_sha256=lock['sha256'], adapter_pair_role='not_applicable' if physical == 'E1' else 'locked_joint_pair',
               run_id=run_id(category, physical, fusion if physical == 'E5' else None, cfg['training']['seed']),
               equivalent_experiments=['E4', 'E5/mean'] if physical == 'E4' and cfg['average_reuse'] else [physical])
    cfg['fusion']['method'] = fusion if physical == 'E5' else 'mean'
    context.update(config_sha256=sha256_json(cfg), selection_bundle=study['selection_bundle'],
                   fairness_root=study['root'] / ('smoke' if smoke else 'main') / '_fairness' / physical)
    require_main_context(context)
    return context, pools


def run_id(category, experiment, fusion, seed):
    return f'{experiment}/{fusion}/{category}/seed{seed}' if experiment == 'E5' else f'{experiment}/{category}/seed{seed}'


def skeleton(owner, lock=None, *, reuse=True):
    rows = []
    for category in OWNERS[owner]:
        for experiment in EXPERIMENTS:
            for fusion in joint.FUSIONS if experiment == 'E5' else (None,):
                physical = 'E4' if reuse and experiment == 'E5' and fusion == 'mean' else experiment
                seed = lock['seed'] if lock else None
                rid = run_id(category, experiment, fusion, seed)
                pid = run_id(category, physical, fusion if physical == 'E5' else None, seed)
                row = dict.fromkeys(FIELDS)
                row.update(category=category, experiment=experiment, fusion=fusion, seed=seed, run_id=rid,
                           physical_run_id=pid, status='NOT RUN',
                           r=lock['selected']['r'] if lock and experiment != 'E1' else None,
                           d=lock['selected']['d'] if lock and experiment != 'E1' else None,
                           selection_sha256=lock['sha256'] if lock else None,
                           reuse_of=pid if pid != rid else None)
                rows.append(row)
    return rows


def valid_training(directory, context):
    """Strict completion, optimizer/cursor/hash and native-map checks before skip."""
    try:
        cfg = context['config']
        report_path = directory / 'metrics.json'
        require((directory / 'metrics.json.sha256').read_text().strip() == sha256_file(report_path), 'Training report checksum differs')
        report = json.loads(report_path.read_text())
        expected = dict(status='PASS', training_complete=True, stage=cfg['stage'], mode=cfg['mode'],
                        category=cfg['category'], run_id=cfg['run_id'], config_sha256=context['config_sha256'],
                        expected_steps=context['expected_steps'], global_step=context['expected_steps'],
                        git_sha=cfg['git_sha'], selection_sha256=cfg['selection_sha256'], main_budget=cfg['main_budget'],
                        frozen_backbone_unchanged=True, decoder_updated=True,
                        adapter=None if cfg['stage'] == 'E1' else {k: cfg['adapter'][k] for k in ('r', 'd')})
        require(all(report.get(k) == v for k, v in expected.items()), 'Training completion/protocol differs')
        require(read_yaml(directory / 'resolved_config.yaml') == cfg
                and json.loads((directory / 'config_hash.json').read_text())['sha256'] == context['config_sha256'], 'Resolved config differs')
        metric = report['best_synthetic_dev']['synthetic_dev_aupro_0_05']
        require(joint.unit_metric(metric) and report['best_synthetic_dev']['qa_status'] == 'PASS', 'Invalid best DEV metric')
        from scripts.eval_g2 import validate_metrics
        dev = report['best_synthetic_dev']
        validate_metrics({**dev, 'tiny_aupro_0_05': dev['dev_tiny']['aupro_0_05'],
                          'mixed_aupro_0_05': dev['dev_mixed']['aupro_0_05']})
        history = json.loads((directory / 'loss_history.json').read_text())
        require(len(history) == context['expected_steps'] and all(
                all(type(e[k]) in (int, float) and np.isfinite(e[k]) for k in ('loss', 'bce', 'dice_loss')) for e in history), 'Missing/nonfinite loss history')
        keys = {'E1': ('decoder',), 'E2': ('adapter', 'decoder'), 'E3': ('adapters', 'projection', 'decoder'),
                'E4': ('adapters', 'projection', 'decoder'), 'E5': ('adapters', 'projection', 'decoder', 'fusion')}[cfg['stage']]
        for name in ('best', 'last'):
            path = directory / f'{name}.pt'
            payload, _ = load_checkpoint_payload(path, require_sha256=True)
            meta, state = payload['metadata'], payload['training_state']
            require(payload['config'] == cfg and meta['config_sha256'] == context['config_sha256']
                    and meta['selection_sha256'] == cfg['selection_sha256'] and meta['git_sha'] == cfg['git_sha']
                    and sha256_file(path) == report[f'{name}_checkpoint_sha256'], 'Checkpoint provenance differs')
            require(0 < state['global_step'] <= context['expected_steps'] and meta['history']
                    and meta['best_metric'] == metric and meta['best_epoch'] == report['best_epoch'], 'Untrained or mismatched best checkpoint')
            require(payload['optimizer_state']['state'] and payload['rng_state']
                    and type(meta['next_epoch']) is int and type(meta['next_batch']) is int, 'Missing optimizer/RNG/cursor')
            require(all(bool(torch.isfinite(v).all()) for v in payload['model_state'].values())
                    and all(bool(torch.isfinite(v).all()) for s in payload['optimizer_state']['state'].values()
                            for v in s.values() if isinstance(v, torch.Tensor)), 'Nonfinite checkpoint')
            groups = {k: sum(v.numel() for n, v in payload['model_state'].items() if n.startswith(k + '.')) for k in keys}
            require(report['parameter_report']['groups'] == groups
                    and report['parameter_report']['trainable_total'] == sum(groups.values()), 'Parameter count differs')
            if name == 'last':
                require(state['global_step'] == context['expected_steps'] and meta['dev_step'] == state['global_step']
                        and len(meta['history']) == state['global_step'], 'Incomplete final budget/DEV')
        if cfg['stage'] != 'E1':
            require(report['adapter_updated'] is True, 'Adapter not trained')
        if cfg['stage'] in ('E3', 'E4', 'E5'):
            require(report['projection_updated'] is True and all(report['adapters_updated'].values())
                    and len(report['adapters_updated']) == 3 and report['feature_sources'] == (3 if cfg['stage'] == 'E3' else 6), 'Multilayer contracts differ')
        if cfg['stage'] in ('E4', 'E5'):
            require(report['context_aligned'] is True, 'Context not aligned')
        if cfg['stage'] == 'E5' and cfg['fusion']['method'] != 'mean':
            require(report['fusion_updated'] is True, 'Fusion not trained')
        native = directory / 'native_dev'
        export = report['native_exports']
        require(sha256_file(native / 'native_maps.json') == export['manifest_sha256']
                and sha256_file(native / 'qa_report.json') == export['qa_sha256'], 'Native export manifest/QA differs')
        rows = json.loads((native / 'native_maps.json').read_text())
        require(len(rows) == export['count'] == report['best_synthetic_dev']['n_samples'] > 0, 'Incomplete native coverage')
        for row in rows:
            for kind in ('score', 'mask'):
                relative = Path(row[kind + '_path'])
                require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid native path')
                path = native / relative
                require(sha256_file(path) == row[kind + '_sha256'] and list(np.load(path, mmap_mode='r').shape) == row['original_hw'], 'Native map checksum/shape drift')
        return report
    except (OSError, KeyError, ValueError, TypeError, RuntimeError):
        return None


def persist(root, rows, manifest):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'main_training_36.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    counts = dict(Counter(r['status'] for r in rows))
    real = sum(r['status'] == 'PASS' for r in rows)
    physical = {r['physical_run_id'] for r in rows if r['status'] == 'PASS'}
    reused_g1 = {r['physical_run_id'] for r in rows if r['status'] == 'PASS' and r.get('reuse_g1_checkpoint')}
    coverage = dict(status='PASS' if real == 36 else ('BLOCKED' if counts.get('BLOCKED') else 'INCOMPLETE'),
                    experimental_status='NOT RUN' if real == 0 else 'INCOMPLETE' if real < 36 else 'PASS',
                    real_pass=real, expected_cells=36, completed_physical_runs=len(physical),
                    g1_reused_physical_runs=len(reused_g1), newly_trained_physical_runs=len(physical - reused_g1),
                    planned_physical_runs=len({r['physical_run_id'] for r in rows}), status_counts=counts,
                    gpu_status=manifest.get('gpu_status', 'NOT RUN'), winner_claim=False)
    save_json(coverage, root / 'main_coverage.json')
    save_json(manifest, root / 'run_manifest.json')
    return coverage


def mirror(root, target, directory=None):
    if not target:
        return
    from scripts.eval_g2 import copy_atomic, sync_tree
    target = Path(target)
    if directory:
        sync_tree(directory, target / Path(directory).relative_to(root))
    for name in ('run_manifest.json', 'main_training_36.csv', 'main_coverage.json', 'training_protocol.json', 'selection_audit.json'):
        if (root / name).is_file():
            copy_atomic(root / name, target / name)


def fill_row(row, report, directory, context):
    cfg, dev = context['config'], report['best_synthetic_dev']
    science = (cfg['mode'] == 'main' and report.get('scientific_evidence') is True
               and report['verification_scope'] == 'real_pretrained' and torch.device(report['device']).type == 'cuda')
    row.update(status='PASS' if science else 'SMOKE_PASS' if cfg['mode'] == 'main_smoke' else 'CPU_PASS',
               config_sha256=context['config_sha256'], checkpoint=str(directory / 'best.pt'),
               checkpoint_sha256=report['best_checkpoint_sha256'], params=report['parameter_report']['trainable_total'],
               reuse_g1_checkpoint=report.get('reuse_g1_checkpoint'), reuse_g1_sha256=report.get('reuse_g1_sha256'),
               git_sha=cfg['git_sha'], reason=None)
    # Fixture/smoke metrics remain in isolated diagnostic artifacts, never scientific tables.
    if science:
        row.update(aupro_005=dev['synthetic_dev_aupro_0_05'], tiny_aupro_005=dev['dev_tiny']['aupro_0_05'],
                   mixed_aupro_005=dev['dev_mixed']['aupro_0_05'], synthetic_dice=dev['synthetic_dice'],
                   normal_p99=dev['normal_score_p99'])


def run_from_cli(args):
    cfg, runner, settings, namespace = load_settings(args)
    settings['owner'] = args.owner
    blocked_root = namespace / 'prepared' / args.owner
    drive_namespace = absolute(args.drive_output) / namespace.name if args.drive_output else None
    blocked_target = drive_namespace / 'prepared' / args.owner if drive_namespace else None
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    try:
        bundle = load_selection(settings['joint_lock'], settings['tv1_manifest'], settings['tv2_manifest'])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        rows = skeleton(args.owner, reuse=settings['reuse_e4_average'])
        for row in rows:
            row.update(status='BLOCKED', reason=str(exc))
        manifest = dict(task='D11', owner=args.owner, categories=list(OWNERS[args.owner]), git_sha=commit,
                        status='BLOCKED', reason=str(exc), gpu_status='NOT RUN', selection_performed=False,
                        main_budget_source='original D10 training_protocol.json; unresolved', settings=settings)
        persist(blocked_root, rows, manifest)
        save_yaml(settings, blocked_root / 'runner_config.yaml')
        mirror(blocked_root, blocked_target)
        print(json.dumps(dict(status='BLOCKED', reason=str(exc), real_pass=0, expected_cells=36, output=str(blocked_root))))
        return 2
    try:
        # Asset checks happen on CPU even when CUDA is absent; model weights are never loaded here.
        study = create_study(cfg, runner, settings, namespace, bundle, device='cpu')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        rows = skeleton(args.owner, bundle['lock'], reuse=settings['reuse_e4_average'])
        for row in rows:
            row.update(status='BLOCKED', reason=str(exc))
        persist(blocked_root, rows, dict(task='D11', owner=args.owner, git_sha=commit, reason=str(exc), gpu_status='NOT RUN'))
        mirror(blocked_root, blocked_target)
        print(json.dumps(dict(status='BLOCKED', reason=str(exc))))
        return 2
    root = study['root']
    target = drive_namespace / study['sha256'] / args.owner if drive_namespace else None
    rows = skeleton(args.owner, bundle['lock'], reuse=settings['reuse_e4_average'])
    manifest = dict(task='D11', owner=args.owner, categories=list(OWNERS[args.owner]), git_sha=commit,
                    screening_git_sha=bundle['lock']['training_git_sha'], main_protocol_sha256=study['sha256'],
                    selection_sha256=bundle['lock']['sha256'], selected=bundle['lock']['selected'],
                    main_budget=study['protocol']['main_budget'], screening_budget=study['protocol']['screening_budget'],
                    seed=bundle['lock']['seed'], gpu_status='NOT RUN', selection_performed=False,
                    runner_settings=settings, native_maps_split='dev_synthetic',
                    interpretation='Conditional architecture ablation; synthetic DEV reused for checkpoint and D10 selection',
                    drive_sync='epoch_and_job_end', oom_policy=settings['oom_policy'])
    old_manifest_path = root / 'run_manifest.json'
    if old_manifest_path.is_file():
        previous = json.loads(old_manifest_path.read_text())
        require(previous.get('main_protocol_sha256') == study['sha256'], 'BLOCKED: main namespace protocol drift')
        for key in ('gpu_status', 'gpu'):
            if key in previous:
                manifest[key] = previous[key]
    save_json(study['protocol'], root / 'training_protocol.json')
    save_json({k: v for k, v in bundle.items() if k not in ('lock', 'protocol', 'spec')}, root / 'selection_audit.json')
    contexts = {}
    for row in rows:
        ctx, pools = make_context(study, row['category'], row['experiment'], row['fusion'])
        directory = root / 'main' / row['physical_run_id']
        plan = root / 'plan' / row['physical_run_id']
        save_yaml(ctx['config'], plan / 'resolved_config.yaml')
        save_json({'sha256': ctx['config_sha256']}, plan / 'config_hash.json')
        row.update(config_sha256=ctx['config_sha256'], git_sha=commit)
        contexts[row['run_id']] = ctx, pools, directory
        report = valid_training(directory, ctx)
        if report:
            fill_row(row, report, directory, ctx)
        elif (directory / 'failure.json').is_file():
            failed = json.loads((directory / 'failure.json').read_text())
            if failed.get('config_sha256') == ctx['config_sha256']:
                row.update(status=failed['status'], reason=failed['reason'])
        elif (directory / 'metrics.json').is_file():
            row.update(status='FAIL', reason='Existing result failed checksum/config/native export audit; use resume to repair')
        elif (directory / 'last.pt').is_file():
            row.update(status='INCOMPLETE', reason='Checkpoint exists; resume required')
    persist(root, rows, manifest); mirror(root, target)
    if target:
        from scripts.eval_g2 import sync_tree
        sync_tree(root / 'plan', target / 'plan')
    if args.prepare or args.preflight or args.report_only:
        print(json.dumps(dict(status='PREPARED', output=str(root), real_pass=sum(r['status'] == 'PASS' for r in rows), expected_cells=36)))
        return 0 if not args.report_only or all(r['status'] == 'PASS' for r in rows) else 2
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        for row in rows:
            if row['status'] == 'NOT RUN':
                row.update(status='BLOCKED', reason='BLOCKED: main training requires Colab CUDA GPU; local GPU is prohibited')
        persist(root, rows, manifest); mirror(root, target)
        print(json.dumps(dict(status='BLOCKED', reason='Colab GPU missing', real_pass=sum(r['status'] == 'PASS' for r in rows), expected_cells=36)))
        return 2
    from src.train.g2_e2 import FrozenE2Factory, train_main
    attempted, finished_ids, factory = 0, set(), None
    try:
        for row in sorted(rows, key=lambda r: (EXPERIMENTS.index(r['physical_run_id'].split('/')[0]), r['category'], r['fusion'] or '')):
            if row['category'] not in args.categories or row['experiment'] not in args.experiments:
                continue
            if row['experiment'] == 'E5' and row['fusion'] not in args.fusions:
                continue
            if (row['status'] == 'PASS' and not args.smoke) or row['physical_run_id'] in finished_ids:
                continue
            if attempted >= (1 if args.smoke else args.max_jobs or (32 if settings['reuse_e4_average'] else 36)):
                break
            attempted += 1
            finished_ids.add(row['physical_run_id'])
            manifest['gpu_status'], manifest['gpu'] = 'ATTEMPTED', torch.cuda.get_device_name(torch.device(args.device))
            ctx, pools, directory = contexts[row['run_id']]
            if args.smoke:
                ctx, pools = make_context(study, row['category'], row['experiment'], row['fusion'], smoke=True)
                directory = root / 'smoke' / row['physical_run_id']
            directory.mkdir(parents=True, exist_ok=True)
            save_yaml(ctx['config'], directory / 'resolved_config.yaml')
            save_json({'sha256': ctx['config_sha256']}, directory / 'config_hash.json')
            handler = logging.FileHandler(directory / 'train.log')
            logging.getLogger().addHandler(handler)
            try:
                # At most one cached frozen backbone is retained. Fresh heads/optimizer for every run.
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
                experiment = ctx['config']['stage']
                mode = 'deepest' if experiment in ('E1', 'E2') else 'multilayer'
                if factory is None or factory.extractor.feature_mode != mode:
                    factory = None
                    gc.collect(); torch.cuda.empty_cache()
                    factory = FrozenE2Factory(ctx['config'], device=args.device, experiment=experiment)
                else:
                    factory.experiment = experiment
                    factory.extractor.to(args.device)
                reuse = None
                candidate = settings.get('g1_checkpoints', {}).get(row['category']) if experiment == 'E1' and not args.smoke else None
                if candidate and not (directory / 'last.pt').exists():
                    try:
                        valid_g1_checkpoint(absolute(candidate), ctx, pools)
                        reuse = absolute(candidate)
                        save_json({'decision': 'REUSE', 'checkpoint': str(reuse), 'sha256': sha256_file(reuse)}, directory / 'g1_reuse_decision.json')
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        save_json({'decision': 'RETRAIN', 'reason': str(exc)}, directory / 'g1_reuse_decision.json')
                def on_epoch(job, epoch):
                    save_json({'epoch': epoch, 'config_sha256': ctx['config_sha256']}, job / 'epoch_progress.json')
                    mirror(root, target, job)
                train_main(ctx, pools, directory, device=args.device, resume=args.resume, model_factory=factory,
                           on_epoch=on_epoch, reuse_checkpoint=reuse)
                save_json(dict(gpu=manifest['gpu'], device=args.device, batch_size=ctx['config']['training']['batch_size'],
                               pair_strategy=ctx['config']['paired_views']['pair_strategy'],
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(torch.device(args.device))), directory / 'gpu_memory.json')
                report = valid_training(directory, ctx)
                require(report is not None, 'Completed output failed checkpoint/native export audit')
                if args.smoke:
                    save_json(dict(status='SMOKE_PASS', scientific_evidence=False, config_sha256=ctx['config_sha256'],
                                   category=row['category'], experiment=experiment, gpu=manifest['gpu']), root / 'smoke_result.json')
                else:
                    for alias in rows:
                        if alias['physical_run_id'] == row['physical_run_id']:
                            alias_ctx = contexts[alias['run_id']][0]
                            require(alias_ctx['config'] == ctx['config'], 'E4/E5-Average configs differ: cannot reuse')
                            fill_row(alias, report, directory, alias_ctx)
                    finished_ids.add(row['physical_run_id'])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()
                failure = dict(status='FAIL/OOM' if oom else 'FAIL', reason=str(exc), config_sha256=ctx['config_sha256'],
                               selection_sha256=bundle['lock']['sha256'], git_sha=commit, gpu=manifest['gpu'])
                save_json(failure, directory / 'failure.json')
                if args.smoke:
                    save_json({**failure, 'scientific_evidence': False}, root / 'smoke_result.json')
                else:
                    for alias in rows:
                        if alias['physical_run_id'] == row['physical_run_id']:
                            alias.update(status=failure['status'], reason=str(exc))
                factory = None
                gc.collect(); torch.cuda.empty_cache()
            finally:
                logging.getLogger().removeHandler(handler); handler.close()
                manifest['attempted_jobs_this_invocation'] = attempted
                persist(root, rows, manifest); mirror(root, target, directory)
        coverage = persist(root, rows, manifest)
        mirror(root, target)
        print(json.dumps({**coverage, 'output': str(root)}))
        if args.smoke:
            smoke_path = root / 'smoke_result.json'
            return 0 if attempted and smoke_path.exists() and json.loads(smoke_path.read_text())['status'] == 'SMOKE_PASS' else 2
        return 0 if coverage['real_pass'] == 36 else 2
    finally:
        factory = None
        gc.collect(); torch.cuda.empty_cache()
