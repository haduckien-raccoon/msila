"""User-fixed E5 orchestration, reusing G2's training/inference/evaluation loop.

This independent namespace never manufactures a D10 selection lock. The chosen
configuration and illumination are frozen before DEV/TEST results are observed.
"""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import subprocess

from PIL import Image
import torch

from src.eval.evaluator import MVTEC_AD2_CATEGORIES
from src.data.synthetic_anomaly import validate_native_protocol
from src.data.tiling import generate_tile_records
from src.eval.g1_test_public import evaluate_public, test_records
from src.models.msila import resolve_g2_config
from src.train.g1_e1 import check_assets, discover_sources
from src.train.g2_context import predict_native_e4
from src.train.g2_fixed_data import validate_illumination
from src.train.screen_representation import read_yaml, save_json, save_yaml, sha256_file as raw_sha256_file, sha256_json
from src.utils.resume import load_checkpoint_payload

ROOT = Path(__file__).resolve().parents[2]
SELECTION = 'user_fixed_no_D10'
CATEGORIES = tuple(MVTEC_AD2_CATEGORIES)
_FILE_HASHES = {}


def sha256_file(path):
    """Reuse verified hashes within this process while file identity is unchanged.

    Repeated coverage writes must not reread gigabytes of native exports. Atomic
    checkpoint replacements and modified files invalidate inode/ctime/mtime.
    Nothing is trusted from a previous process or from a Drive cache file.
    """
    path = Path(path).resolve()
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns, stat.st_mtime_ns)
    cached = _FILE_HASHES.get(path)
    if cached is None or cached[0] != identity:
        cached = identity, raw_sha256_file(path)
        _FILE_HASHES[path] = cached
    return cached[1]


def absolute(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def precision_dtype(value):
    return {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[value]


def predict_fixed_native(model, image, cfg, device):
    dtype = precision_dtype(cfg['training']['precision'])
    with torch.autocast(torch.device(device).type, dtype=dtype, enabled=dtype is not None):
        return predict_native_e4(model, image, cfg, device)


def prepare_study(args):
    settings = read_yaml(absolute(args.config))
    if settings.get('version') != 'g2_fixed_e5_v1':
        raise ValueError('Expected g2_fixed_e5_v1 settings')
    if (settings['r'], settings['d'], settings['fusion']) != (128, 512, 'attention'):
        raise ValueError('This runner is the requested fixed r128/d512/attention E5 experiment')
    if Path(settings['run_id']).name != settings['run_id'] or settings['run_id'] in {'', '.', '..'}:
        raise ValueError('run_id must be one directory name')
    cfg = read_yaml(absolute(settings['model_config']))
    cfg['backbone'].update(name='dinov3_vitb16', repo_dir=str(absolute(args.repo_dir)),
                           weights=str(absolute(args.weights)))
    cfg['adapter'].update(r=128, d=512)
    cfg['fusion'].update(method='attention', dim=64)
    cfg['data'].update(root=str(absolute(args.data_root)), max_train_sources=None,
                       max_dev_sources=settings['max_dev_sources'], num_workers=settings['num_workers'])
    cfg['decoder']['deterministic_resize'] = settings['deterministic']
    cfg['paired_views'].update(context_size=768, pair_strategy='concat',
                               deterministic_sampling=settings['deterministic'])
    cfg['evaluation'].update(tile_batch_size=settings['tile_batch_size'], example_limit=3, probability_fp32=True)
    cfg = resolve_g2_config(cfg, root=ROOT)
    check_assets(cfg, args.device)
    cfg['synthetic_protocol'] = validate_native_protocol(read_yaml(absolute(cfg['synthetic_protocol'])))
    cfg['loss'] = {k: cfg['loss'][k] for k in ('bce_weight', 'dice_weight', 'dice_eps')}
    precision = settings['precision']
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if torch.device(args.device).type == 'cuda' else 'fp32'
    precision_dtype(precision)
    if precision != 'fp32' and (torch.device(args.device).type != 'cuda' or
                               (precision == 'bf16' and not torch.cuda.is_bf16_supported())):
        raise ValueError('Requested mixed precision is unavailable on this device')
    t = cfg['training']
    t.update(seed=settings['seed'], dev_seed=settings['dev_seed'], epochs=settings['epochs'],
             batch_size=settings['batch_size'], checkpoint_interval_steps=settings['checkpoint_interval_steps'],
             learning_rate=t['optimizer']['lr'], weight_decay=t['optimizer']['weight_decay'],
             precision=precision, amp=precision != 'fp32', deterministic_algorithms=settings['deterministic'],
             max_steps=None, max_minutes=settings['max_minutes'])
    for key in ('epochs', 'batch_size', 'checkpoint_interval_steps'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if any(type(t[k]) is not int or t[k] < 0 for k in ('seed', 'dev_seed')) or t['seed'] == t['dev_seed']:
        raise ValueError('TRAIN and DEV need independent nonnegative integer seeds')
    if (type(cfg['data']['num_workers']) is not int or cfg['data']['num_workers'] < 0 or
            type(cfg['evaluation']['tile_batch_size']) is not int or cfg['evaluation']['tile_batch_size'] < 1):
        raise ValueError('Invalid worker count or inference batch size')
    if settings['max_dev_sources'] is not None and (type(settings['max_dev_sources']) is not int or settings['max_dev_sources'] < 1):
        raise ValueError('max_dev_sources must be positive or null')
    if t['max_minutes'] is not None and (not math.isfinite(t['max_minutes']) or t['max_minutes'] <= 0):
        raise ValueError('max_minutes must be positive or null')
    if type(settings['deterministic']) is not bool:
        raise ValueError('deterministic must be boolean')
    if (cfg['data']['tile_size'] != 512 or cfg['data']['local_source_fov'] != [512, 512]
            or cfg['data']['context_source_fov'] != [768, 768]
            or cfg['data']['context_input_size'] != [512, 512] or not 0 <= cfg['data']['overlap'] < 512
            or cfg['data']['train_split'] != 'TRAIN/good' or cfg['data']['dev_split'] != 'VALIDATION/good'
            or cfg['evaluation']['max_fpr'] != .05 or cfg['evaluation']['test_for_selection'] is not False
            or cfg['evaluation']['score_transform'] != 'sigmoid' or cfg['evaluation']['stitching'] != 'hann_native_coordinates'
            or cfg['checkpoint'] != dict(monitor='synthetic_dev_aupro_0_05', mode='max',
                                        tie_break='earliest_epoch', selection_split='VALIDATION/good')):
        raise ValueError('Fixed E5 still requires the native G2 geometry, loss and DEV selection protocol')
    validate_illumination(settings['illumination'])
    cfg.update(version='g2_user_fixed_e5_v1', stage='E5', run_id=settings['run_id'],
               selection_status=SELECTION, selection_sha256=None, adapter_pair_role='user_fixed',
               illumination=deepcopy(settings['illumination']), tile_sampler='native_grouped_v1')
    cfg['backbone']['checkpoint_sha256'] = sha256_file(cfg['backbone']['weights'])
    repo = Path(cfg['backbone']['repo_dir'])
    cfg['backbone']['source_sha256'] = {str(p.relative_to(repo)): sha256_file(p) for p in sorted(repo.rglob('*.py'))}
    cfg['git_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cfg['source_code_sha256'] = {str(p.relative_to(ROOT)): sha256_file(p)
                                for p in sorted((ROOT / 'src').rglob('*.py'))}
    cfg['source_code_sha256']['scripts/run_g2_fixed.py'] = sha256_file(ROOT / 'scripts/run_g2_fixed.py')
    cfg['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda)
    digest = sha256_json(cfg)
    root = absolute(args.output_root) / settings['run_id'] / digest
    save_yaml(cfg, root / 'study.yaml')
    save_json(dict(device=str(args.device), torch=str(torch.__version__), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.device(args.device).type == 'cuda' else None,
                   precision=precision, deterministic=settings['deterministic']), root / 'environment.json')
    save_json(dict(study_sha256=digest, selection_status=SELECTION, selection_lock=None,
                   purpose='Declared settings only; actual status is in full/summary.json'), root / 'plan.json')
    return dict(protocol=cfg, sha256=digest, root=root, contexts={})


def make_context(study, category, *, smoke=False):
    cfg = deepcopy(study['protocol'])
    cfg.update(category=category, mode='fixed_smoke' if smoke else 'fixed', study_sha256=study['sha256'])
    if smoke:
        cfg['training'].update(epochs=1, max_steps=2, max_minutes=None)
        cfg['data'].update(max_train_sources=1, max_dev_sources=1, dev_variants_per_image=3)
    pools, manifest = discover_sources(cfg)
    count = 0
    for record in pools['train']:
        with Image.open(record.image_path) as image:
            count += len(generate_tile_records(image.height, image.width, 512, cfg['data']['overlap'], context_size=768))
    expected = math.ceil(count * (cfg['data']['train_variants_per_image'] + 1) / cfg['training']['batch_size']) * cfg['training']['epochs']
    if cfg['training']['max_steps'] is not None:
        expected = min(expected, cfg['training']['max_steps'])
    cfg.update(sources=manifest, expected_steps=expected,
               fixed_budget={k: cfg['training'][k] for k in ('epochs', 'batch_size', 'max_steps', 'max_minutes')})
    context = dict(config=cfg, config_sha256=sha256_json(cfg), expected_steps=expected,
                   fairness_root=str(study['root'] / ('smoke' if smoke else 'full')))
    require_fixed_context(context)
    return context, pools


def require_fixed_context(context):
    cfg = context['config']
    if (context['config_sha256'] != sha256_json(cfg) or context['expected_steps'] != cfg['expected_steps']
            or cfg['stage'] != 'E5' or cfg['mode'] not in {'fixed', 'fixed_smoke'}
            or cfg['selection_status'] != SELECTION or cfg['selection_sha256'] is not None
            or cfg['adapter_pair_role'] != 'user_fixed'
            or (cfg['adapter']['r'], cfg['adapter']['d'], cfg['fusion']['method']) != (128, 512, 'attention')
            or cfg['backbone']['name'] != 'dinov3_vitb16' or cfg['backbone']['feature_blocks'] != [4, 8, 12]
            or cfg['paired_views']['context_size'] != 768 or cfg['fusion']['dim'] != 64):
        raise ValueError('Fixed E5 context/config mismatch')
    if cfg['mode'] == 'fixed_smoke' and (cfg['training']['epochs'] != 1 or not 1 <= cfg['expected_steps'] <= 2):
        raise ValueError('Smoke is bounded and never experimental acceptance')
    validate_illumination(cfg['illumination'])


def completed_training(directory, context):
    """Skip only integrity-checked complete best/last checkpoints with DEV exports."""
    directory = Path(directory)
    try:
        result = json.loads((directory / 'metrics.json').read_text())
        if (result['status'] != 'PASS' or not result['training_complete'] or
                result['config_sha256'] != context['config_sha256'] or
                result['global_step'] != context['expected_steps'] or
                result['feature_sources'] != 6 or result['selection_status'] != SELECTION):
            return None
        if (directory / 'metrics.json.sha256').read_text().strip() != sha256_file(directory / 'metrics.json'):
            return None
        for name in ('best', 'last'):
            path = directory / f'{name}.pt'
            payload, _ = load_checkpoint_payload(path, require_sha256=True)
            if payload['config'] != context['config'] or result[f'{name}_checkpoint_sha256'] != sha256_file(path):
                return None
            if name == 'last' and payload['training_state']['global_step'] != context['expected_steps']:
                return None
        dev = result['best_synthetic_dev']['synthetic_dev_aupro_0_05']
        if not isinstance(dev, (int, float)) or not math.isfinite(dev) or not 0 <= dev <= 1:
            return None
        for name, key in (('native_maps.json', 'manifest_sha256'), ('qa_report.json', 'qa_sha256')):
            if sha256_file(directory / 'native_dev' / name) != result['native_exports'][key]:
                return None
        rows = json.loads((directory / 'native_dev/native_maps.json').read_text())
        if len(rows) != result['native_exports']['count']:
            return None
        for row in rows:
            for key in ('score', 'mask'):
                if sha256_file(directory / 'native_dev' / row[f'{key}_path']) != row[f'{key}_sha256']:
                    return None
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def public_evaluation(directory, context, *, device, model_factory):
    from src.train.g2_e2 import trainable_modules
    directory = Path(directory)
    result = completed_training(directory, context)
    if result is None or context['config']['mode'] != 'fixed':
        raise ValueError('TEST_PUBLIC requires a complete fixed full-training run, never smoke')
    if torch.device(device).type != 'cuda' or result['verification_scope'] != 'real_pretrained':
        raise ValueError('Experimental TEST_PUBLIC requires a real pretrained GPU training checkpoint')
    cfg, path = context['config'], directory / 'best.pt'
    # Validate real GT and source checksums before expensive model inference.
    records = test_records(cfg)
    sources = [dict(image=r.image_path, image_sha256=sha256_file(r.image_path), mask=r.mask_path,
                    mask_sha256=sha256_file(r.mask_path) if r.mask_path else None) for r in records]
    signature = sha256_json(dict(config=context['config_sha256'], checkpoint=sha256_file(path), sources=sources))
    out = directory / 'test_public'
    try:
        saved = json.loads((out / 'metrics.json').read_text())
        value = saved.get('test_public_aupro_0_05')
        if (saved.get('signature') == signature and saved.get('status') == 'PASS'
                and saved.get('native_resolution') is True and saved.get('n_images') == len(records)
                and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                and saved.get('artifact_sha256') and all(sha256_file(out / p) == digest
                    for p, digest in saved['artifact_sha256'].items())):
            return saved
    except (OSError, ValueError, TypeError, KeyError):
        pass  # Missing/corrupt exports are recomputed, never silently skipped.
    payload, _ = load_checkpoint_payload(path, require_sha256=True)
    model = model_factory(cfg).to(device).eval()
    trainable_modules(model, 'E5').load_state_dict(payload['model_state'], strict=True)
    if not model.extractor.backbone.__class__.__module__.startswith('dinov3.'):
        raise ValueError('Fixture backbone cannot produce experimental TEST_PUBLIC metrics')
    public = evaluate_public(model, cfg, device, out, predict_fn=predict_fixed_native, example_limit=6)
    public.update(signature=signature, selection_status=SELECTION, checkpoint=str(path),
                  checkpoint_sha256=sha256_file(path), config_sha256=context['config_sha256'],
                  adapter=dict(r=128, d=512), fusion='attention', seed=cfg['training']['seed'],
                  git_sha=cfg['git_sha'], scientific_evidence=public['status'] == 'PASS',
                  artifact_sha256={str(p.relative_to(out)): sha256_file(p)
                                   for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'metrics.json'})
    save_json(public, out / 'metrics.json')
    return public


def write_summary(study, *, smoke=False):
    root = study['root'] / ('smoke' if smoke else 'full')
    rows = []
    for category in CATEGORIES:
        folder = root / category
        row = dict(category=category, training_status='NOT RUN', test_public_status='NOT RUN',
                   synthetic_dev_aupro_005=None, test_public_aupro_005=None, checkpoint=None,
                   selection_status=SELECTION)
        for name, prefix in (('metrics.json', 'train'), ('test_public/metrics.json', 'test')):
            p = folder / name
            if p.is_file():
                result = json.loads(p.read_text())
                row['training_status' if prefix == 'train' else 'test_public_status'] = result['status']
                if prefix == 'train':
                    row['synthetic_dev_aupro_005'] = (result.get('best_synthetic_dev') or {}).get('synthetic_dev_aupro_0_05')
                    row['checkpoint'] = str(folder / 'best.pt') if (folder / 'best.pt').is_file() else None
                else:
                    row['test_public_aupro_005'] = result.get('test_public_aupro_0_05')
        failure = folder / 'failure.json'
        if failure.is_file() and row['training_status'] == 'NOT RUN':
            row['training_status'] = json.loads(failure.read_text())['status']
        if row['training_status'] == 'PASS':
            try:
                cfg = read_yaml(folder / 'resolved_config.yaml')
                ctx = dict(config=cfg, config_sha256=sha256_json(cfg), expected_steps=cfg['expected_steps'])
                actual = completed_training(folder, ctx)
                if actual is None:
                    row['training_status'] = 'INCOMPLETE'
                elif not smoke and not actual['scientific_evidence']:
                    row['training_status'] = 'FIXTURE PASS'
            except (OSError, ValueError, KeyError):
                row['training_status'] = 'INCOMPLETE'
        if row['test_public_status'] == 'PASS':
            try:
                public = json.loads((folder / 'test_public/metrics.json').read_text())
                value = public['test_public_aupro_0_05']
                if (not public.get('scientific_evidence') or public['category'] != category
                        or public['config_sha256'] != sha256_json(read_yaml(folder / 'resolved_config.yaml'))
                        or public['checkpoint_sha256'] != sha256_file(folder / 'best.pt')
                        or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                        or not public.get('artifact_sha256') or any(
                            sha256_file(folder / 'test_public' / p) != digest
                            for p, digest in public['artifact_sha256'].items())):
                    row['test_public_status'] = 'INCOMPLETE'
            except (OSError, ValueError, KeyError, TypeError):
                row['test_public_status'] = 'INCOMPLETE'
        if smoke:
            row['test_public_status'] = 'NOT RUN'
        rows.append(row)
    full = all(r['training_status'] == r['test_public_status'] == 'PASS' and
               isinstance(r['test_public_aupro_005'], (float, int)) and math.isfinite(r['test_public_aupro_005']) for r in rows) and not smoke
    summary = dict(status='PASS' if full else 'INCOMPLETE', scope='fixed_E5_only_no_D10_selection',
                   selection_lock=None, training_pass=sum(r['training_status'] == 'PASS' for r in rows),
                   test_public_pass=sum(r['test_public_status'] == 'PASS' for r in rows), expected_categories=8,
                   macro_test_public_aupro_005=sum(r['test_public_aupro_005'] for r in rows)/8 if full else None,
                   smoke=smoke, categories=rows,
                   limitations='User-fixed configuration; no D10 optimum or E0–E5 comparison claim. Illumination is train-only. Best uses synthetic DEV; TEST_PUBLIC is evaluation only.')
    save_json(summary, root / 'summary.json')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'metrics_e5_fixed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return summary
