#!/usr/bin/env python3
"""Fixed E5 r128/d512/attention: shared G2 training, DEV best, real TEST_PUBLIC."""
import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.train.g2_e2 import FrozenE5Factory, train_fixed
from src.train.g2_fixed import (CATEGORIES, prepare_study, make_context, completed_training,
                               public_evaluation, write_summary)
from src.train.screen_representation import save_json
from scripts.eval_g2 import sync_tree


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/g2_fixed_attention.yaml')
    for flag in ('data-root', 'repo-dir', 'weights'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--output-root', default='outputs/G2/E5_FIXED')
    parser.add_argument('--drive-output')
    parser.add_argument('--categories', nargs='+', default=['all'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--test-public', action='store_true')
    args = parser.parse_args(argv)
    names = [c for group in args.categories for c in group.split(',')]
    if names == ['all']:
        names = list(CATEGORIES)
    if len(set(names)) != len(names) or set(names) - set(CATEGORIES):
        parser.error('categories must be all or distinct canonical category names')
    args.categories = [c for c in CATEGORIES if c in names]
    if args.smoke and (args.test_public or args.eval_only):
        parser.error('Smoke is separate from TEST_PUBLIC evaluation')
    if args.eval_only:
        args.test_public = True
    if not args.preflight and torch.device(args.device).type != 'cuda':
        parser.error('Actual full/smoke runs require CUDA; CPU is supported for preflight/unit fixtures')
    return args


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(root / 'run.log'), logging.StreamHandler()], force=True)
    study = None
    try:
        study = prepare_study(args)
        mode = 'smoke' if args.smoke else 'full'
        jobs = [(c, *make_context(study, c, smoke=args.smoke)) for c in args.categories]
        if args.test_public or args.preflight:
            from src.eval.g1_test_public import test_records
            for _, context, _ in jobs:
                test_records(context['config'])

        def mirror(*unused):
            if args.drive_output:
                sync_tree(root, Path(args.drive_output))

        summary = write_summary(study, smoke=args.smoke)
        mirror()
        if args.preflight:
            save_json(dict(status='PREPARED', selection_status='user_fixed_no_D10', GPU='NOT RUN',
                           categories=args.categories, expected_steps={c: ctx['expected_steps'] for c, ctx, _ in jobs}),
                      study['root'] / 'preflight.json')
            mirror()
            return 0
        factory = FrozenE5Factory(jobs[0][1]['config'], device=args.device)
        for category, context, pools in jobs:
            directory = study['root'] / mode / category
            try:
                result = completed_training(directory, context)
                if result is None and not args.eval_only:
                    result = train_fixed(context, pools, directory, device=args.device, resume=args.resume,
                                         model_factory=factory, on_epoch=mirror, on_checkpoint=mirror)
                elif result is not None:
                    logging.info('SKIP valid completed training: %s', category)
                if args.test_public and result is not None and result['training_complete']:
                    public_evaluation(directory, context, device=args.device, model_factory=factory)
                elif args.test_public:
                    save_json(dict(status='BLOCKED', reason='Missing complete real best.pt; training incomplete or eval-only',
                                   category=category), directory / 'test_public/metrics.json')
            except Exception as exc:
                # Preserve resumable checkpoints, record the failure and continue categories.
                save_json(dict(status='FAIL', reason=str(exc), category=category), directory / 'failure.json')
                logging.exception('Category %s failed', category)
                if args.test_public:
                    save_json(dict(status='FAIL', reason=str(exc), category=category), directory / 'test_public/metrics.json')
            summary = write_summary(study, smoke=args.smoke)
            mirror()
        print(summary)
        return 0 if summary['status'] == 'PASS' or (args.smoke and all(
            (study['root'] / mode / c / 'metrics.json').is_file() and completed_training(study['root'] / mode / c, ctx)
            for c, ctx, _ in jobs)) else 2
    except Exception as exc:
        save_json(dict(status='BLOCKED' if isinstance(exc, FileNotFoundError) else 'FAIL', reason=str(exc),
                       training='NOT RUN', test_public='NOT RUN'), root / 'failure.json')
        logging.exception('Fixed E5 preparation failed')
        return 1
    finally:
        if study is not None:
            write_summary(study, smoke=args.smoke)
        if args.drive_output:
            sync_tree(root, Path(args.drive_output))


if __name__ == '__main__':
    raise SystemExit(main())
