#!/usr/bin/env python3
"""Build the TV1 checkpoint handoff (distinct from the pixel metric manifest)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from src.train.day05_contract import SOURCES, digest, file_hash, validate_day05, validate_signature
from src.train.screen_representation import lock_payload


def read_artifact(path, candidate):
    path = Path(path).absolute()
    cfg = yaml.safe_load((path / 'resolved_config.yaml').read_text(encoding='utf-8'))
    run = json.loads((path / 'run_manifest.json').read_text(encoding='utf-8'))
    if cfg['candidate'] != candidate or run['candidate'] != candidate:
        raise ValueError(f'{candidate}: artifact candidate mismatch')
    if run['status'] != 'COMPLETE' or cfg['seed'] != 42 or run['seed'] != 42:
        raise ValueError(f'{candidate}: need COMPLETE seed42 run')
    validate_day05(cfg['day05_config'])
    validate_signature(cfg['cache']['provenance_check']['producer_signature'])
    if cfg['cache']['in_channels'] != 384:
        raise ValueError('ViT-S/16 feature channels must be 384')
    if (cfg['adapter']['r'], cfg['adapter']['d']) != (32, 384):
        raise ValueError('Day-04 adapter lock mismatch')
    if cfg['representation']['sources'] != SOURCES[candidate] or run['sources'] != SOURCES[candidate]:
        raise ValueError(f'{candidate}: source mismatch')
    if digest(cfg) != run['resolved_config_sha256']:
        raise ValueError(f'{candidate}: resolved config hash mismatch')
    if cfg['protocol_lock_sha256'] != digest(lock_payload(cfg)):
        raise ValueError(f'{candidate}: controlled protocol hash mismatch')
    for name in ('best.pt', 'last.pt', 'resolved_config.yaml', 'preflight_report.json',
                 'selection_record.json', 'training_log.csv', 'epoch_log.csv', 'sample_anomaly_map.png'):
        if run.get('artifact_sha256', {}).get(name) != file_hash(path / name):
            raise ValueError(f'{candidate}: missing or changed {name}')
    if run.get('completed_epochs') != cfg['training']['epochs'] or run.get('global_step') != cfg['training']['total_update_budget']:
        raise ValueError(f'{candidate}: incomplete epoch/update budget')
    if json.loads((path / 'preflight_report.json').read_text())['status'] != 'PASS':
        raise ValueError(f'{candidate}: preflight did not PASS')
    return cfg, dict(candidate=candidate, artifact_path=str(path), category=cfg['category'],
                     seed=cfg['seed'], checkpoint_path=str(path / 'best.pt'),
                     checkpoint_sha256=file_hash(path / 'best.pt'),
                     resolved_config_sha256=digest(cfg), protocol_lock_sha256=cfg['protocol_lock_sha256'])


def verify_and_build_manifest(r0, r1, r2, output_path):
    artifacts = [read_artifact(p, c) for p, c in zip((r0, r1, r2), SOURCES)]
    configs, runs = zip(*artifacts)
    if len({r['protocol_lock_sha256'] for r in runs}) != 1:
        raise ValueError('R0/R1/R2 controlled protocol differs')
    manifest = dict(schema='msila.day05.handoff.v1', runs=list(runs),
                    category=configs[0]['category'], seed=42,
                    protocol_lock_sha256=runs[0]['protocol_lock_sha256'])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    if output_path.exists() and output_path.read_text() != content:
        raise ValueError('Existing handoff differs; choose a new output path')
    output_path.write_text(content, encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('r0', 'r1', 'r2', 'output'): p.add_argument(k, type=Path)
    a = p.parse_args()
    verify_and_build_manifest(a.r0, a.r1, a.r2, a.output)
