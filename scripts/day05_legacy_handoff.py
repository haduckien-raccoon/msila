"""Read the original TV1-C + TV1-D artifacts without changing their config/hash.

This is an inference compatibility reader, not a training metadata migration.
Legacy artifacts did not have a producer-time artifact checksum index. TV1-D's
checkpoint/config/epoch-log evidence is verified; other hashes are receipt-time
snapshots and are explicitly labelled that way in the handoff.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

from src.train.day05_contract import SOURCES, digest, file_hash, validate_day05

LEGACY_LOCK_KEYS = (
    'category', 'seed', 'adapter', 'backbone', 'input', 'projection', 'fusion',
    'decoder', 'loss', 'training', 'data_fingerprints', 'cache_producer_sha256',
    'git_commit', 'runner_sha256',
)
MODES = dict(R0='deep_only', R1='multi_local', R2='multi_local_context')


def model_config(cfg):
    """Return a separate model-construction view; preserve cfg's original hash."""
    if 'day05_config' in cfg:
        return cfg['day05_config']
    candidate = cfg['candidate']
    if cfg['representation']['mode'] != MODES[candidate]:
        raise ValueError('Legacy representation mode differs from the source contract')
    locked = {k: copy.deepcopy(cfg[k]) for k in
              ('backbone', 'input', 'projection', 'fusion', 'decoder', 'loss')}
    locked['adapter'] = {k: cfg['adapter'][k] for k in ('kernel_size', 'gamma_init')}
    locked['training'] = {'seed': cfg['seed']}
    result = dict(locked=locked, representations={
        c: dict(mode=MODES[c], sources=sources) for c, sources in SOURCES.items()})
    validate_day05(result)
    return result


def verify_legacy(path, cfg, run):
    """Verify the exact legacy structure recorded by the attached TV1 notebook."""
    import torch

    path = Path(path)
    if 'artifact_sha256' in run or cfg.get('schema') != 'msila.day05.full_train.resolved.v1':
        raise ValueError('Unsupported/incomplete legacy TV1 artifact format')
    if cfg['checkpoint_rule'] != {'monitor': 'val_total_loss', 'mode': 'min'}:
        raise ValueError('Legacy TV1 checkpoint rule drift')
    lock_sha = digest({k: cfg[k] for k in LEGACY_LOCK_KEYS})
    lock_file = path.parents[2] / 'protocol_lock.json'
    control = json.loads(lock_file.read_text())
    if (cfg['protocol_lock_sha256'] != lock_sha or run['protocol_lock_sha256'] != lock_sha
            or control['sha256'] != lock_sha or digest(control['payload']) != lock_sha):
        raise ValueError('Legacy controlled protocol hash mismatch')

    selection = json.loads((path / 'selection_record.json').read_text())
    if (selection.get('schema') != 'msila.day05.checkpoint_selection_record.v1'
            or selection.get('status') != 'PASS'
            or selection['candidate'] != cfg['candidate'] or selection['category'] != cfg['category']
            or selection['seed'] != cfg['seed'] or selection['dev_split'] != 'dev_synthetic'):
        raise ValueError('Legacy TV1-D selection evidence missing or mismatched')
    evidence, rule = selection['evidence'], selection['rule']
    for name, key in [('epoch_log.csv', 'epoch_log_sha256'),
                      ('resolved_config.yaml', 'resolved_config_sha256_file')]:
        if file_hash(path / name) != evidence[key]:
            raise ValueError(f'Legacy TV1-D evidence changed: {name}')
    if (evidence['checkpoint_resolved_config_sha256'] != digest(cfg)
            or evidence['protocol_lock_sha256'] != lock_sha or evidence['git_commit'] != cfg['git_commit']):
        raise ValueError('Legacy TV1-D config identity mismatch')
    rule_file = path.parents[2] / 'checkpoint_rule_lock.json'
    locked_rule = json.loads(rule_file.read_text())
    identity = {k: v for k, v in locked_rule.items() if k not in ('rule_lock_sha256', 'locked_at_unix')}
    if (locked_rule.get('schema') != 'msila.day05.checkpoint_rule_lock.v1'
            or locked_rule.get('status') != 'LOCKED_BEFORE_TV1C_RESULTS'
            or locked_rule.get('candidate_scope') != list(SOURCES)
            or digest(identity) != locked_rule['rule_lock_sha256']
            or locked_rule['rule_lock_sha256'] != rule['rule_lock_sha256']
            or locked_rule['locked_at_unix'] > run['started_at_unix']):
        raise ValueError('Legacy checkpoint rule lock changed or was created after training')
    for key, expected in dict(day05_log_monitor='val_total_loss', mode='min', dev_split='dev_synthetic',
                              comparison='strict_improvement', tie_policy='keep_earliest_epoch',
                              category=cfg['category'], seed=cfg['seed']).items():
        if locked_rule[key] != expected:
            raise ValueError(f'Legacy checkpoint rule {key} mismatch')
    for key in ('day04_monitor', 'day05_log_monitor', 'mode', 'comparison', 'tie_policy',
                'rule_source_sha256', 'selector_file_sha256'):
        if rule[key] != locked_rule[key]:
            raise ValueError(f'Legacy TV1-D rule evidence {key} mismatch')

    with (path / 'epoch_log.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    epochs = cfg['training']['epochs']
    if [int(r['epoch']) for r in rows] != list(range(1, epochs + 1)):
        raise ValueError('Legacy epoch log must contain every epoch exactly once')
    if any(not math.isfinite(float(r['val_total_loss'])) for r in rows):
        raise ValueError('Legacy checkpoint metric must be finite')
    selected = min(rows, key=lambda r: float(r['val_total_loss']))  # First tied epoch wins.
    selected_epoch, selected_value = int(selected['epoch']), float(selected['val_total_loss'])
    budget, per_epoch = cfg['training']['total_update_budget'], cfg['training']['updates_per_epoch']
    if ([int(r['global_step']) for r in rows] != [i * per_epoch for i in range(1, epochs + 1)]
            or budget != epochs * per_epoch):
        raise ValueError('Legacy epoch/update budget mismatch')
    count = 0
    with (path / 'training_log.csv').open(newline='') as stream:
        for count, row in enumerate(csv.DictReader(stream), start=1):
            if (int(row['global_step']) != count or int(row['epoch']) != (count - 1) // per_epoch + 1
                    or int(row['step_in_epoch']) != (count - 1) % per_epoch + 1):
                raise ValueError('Legacy optimizer-step log incomplete or duplicated')
    if count != budget:
        raise ValueError('Legacy optimizer-step count differs from the training budget')

    for name in ('best.pt', 'last.pt'):
        checkpoint = torch.load(path / name, map_location='cpu', weights_only=False)
        if (checkpoint.get('schema') != 'msila.day05.full_train.checkpoint.v1'
                or checkpoint.get('resolved_config_sha256') != digest(cfg)):
            raise ValueError(f'Legacy checkpoint config mismatch: {name}')
        expected_epoch = selected_epoch if name == 'best.pt' else epochs
        if (checkpoint['epoch'] != expected_epoch or checkpoint['global_step'] != expected_epoch * per_epoch
                or not math.isclose(float(checkpoint['best_val']), selected_value, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'Legacy selected checkpoint/last update mismatch: {name}')
    decision = selection['selection']
    if (decision['selected_epoch'] != selected_epoch or decision['metric_name'] != 'val_total_loss'
            or not math.isclose(decision['selected_metric'], selected_value, rel_tol=0, abs_tol=1e-12)
            or decision['checkpoint_sha256'] != file_hash(path / 'best.pt')):
        raise ValueError('Legacy TV1-D selected checkpoint evidence mismatch')
    snapshots = {name: file_hash(path / name) for name in
                 ('best.pt', 'last.pt', 'resolved_config.yaml', 'run_manifest.json', 'preflight_report.json',
                  'selection_record.json', 'training_log.csv', 'epoch_log.csv', 'sample_anomaly_map.png')}
    snapshots['protocol_lock.json'] = file_hash(lock_file)
    snapshots['checkpoint_rule_lock.json'] = file_hash(rule_file)
    return dict(artifact_format='legacy_tv1c_with_tv1d_selection',
                checksum_scope='TV1-D verifies best/config/epoch log; remaining hashes recorded at TV2 receipt',
                artifact_sha256_at_receipt=snapshots,
                selected_epoch=selected_epoch, selected_val_total_loss=selected_value)
