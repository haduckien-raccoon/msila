"""Final G1 E1 evaluation on real TEST_PUBLIC; model selection stays on DEV."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data.loader import load_rgb_native, load_mask_native, scan_mvtec_ad2
from src.eval.evaluator import build_anomaly_map_qa_report, write_metrics_json
from src.eval.full_scale import DEVMetricAccumulator
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.models.msila import E1
from src.train.g1_e1 import (check_assets, discover_sources, file_sha256, parse_args as train_args,
                            predict_native, resolve_config, resume_identity)
from src.utils.resume import load_checkpoint_payload


def test_records(cfg):
    records = scan_mvtec_ad2(cfg['data']['root'], split='test_public',
                             categories=[cfg['category']], require_pixel_gt=True)
    if not records:
        raise FileNotFoundError(f"No {cfg['category']}/TEST_PUBLIC images under {cfg['data']['root']}")
    return records


def save_public_example(image, mask, score, record, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    rgb = image.permute(1, 2, 0).numpy()
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(directory / 'original.png')
    Image.fromarray((mask * 255).astype(np.uint8)).save(directory / 'mask.png')
    np.save(directory / 'predicted_map.npy', score)
    plt.imsave(directory / 'predicted_map.png', score, cmap='inferno', vmin=0, vmax=1)
    overlay = .65 * rgb + .35 * plt.get_cmap('inferno')(score)[..., :3]
    plt.imsave(directory / 'overlay.png', np.clip(overlay, 0, 1))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, value, title in zip(axes, (rgb, mask, score, overlay),
                                ('Real TEST_PUBLIC', 'GT', 'Probability', 'Prediction overlay')):
        ax.imshow(value, **({'cmap': 'inferno', 'vmin': 0, 'vmax': 1} if value.ndim == 2 else {}))
        ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(directory / 'comparison.png', dpi=140); plt.close(fig)
    write_metrics_json(dict(image=record.image_path, mask=record.mask_path,
                            defect_type=record.defect_type, synthetic=False,
                            native_hw=list(mask.shape)), directory / 'metadata.json')


@torch.no_grad()
def evaluate_public(model, cfg, device, output, *, example_limit=8):
    """All public images, exact native geometry, existing disk-backed AU-PRO."""
    records = test_records(cfg)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    accumulator = DEVMetricAccumulator(cfg['synthetic_protocol'], per_region=False, disk_backed=True)
    qa_rows, manifest = [], []
    normal_count = abnormal_count = examples = 0
    model.eval()
    for index, record in enumerate(records):
        image = load_rgb_native(record.image_path)
        hw = tuple(image.shape[-2:])
        mask = np.zeros(hw, dtype=np.uint8) if record.is_normal else load_mask_native(record.mask_path).numpy()
        if mask.shape != hw:
            raise ValueError(f'Native TEST_PUBLIC image/mask shape mismatch: {record.image_path}: {hw} != {mask.shape}')
        if not record.is_normal and not mask.any():
            raise ValueError(f'Abnormal TEST_PUBLIC mask is empty: {record.mask_path}')
        score = predict_native(model, image, cfg, device).numpy()
        sample_id = f'{index:05d}_{Path(record.image_path).stem}'
        qa = build_anomaly_map_qa_report([dict(anomaly_map=score, gt_mask=mask, meta=dict(
            image_id=sample_id, category=cfg['category'], split='test_public', original_hw=list(hw),
            anomaly_map_space='original_image', gt_mask_space='original_image'))])
        if qa['summary']['status'] != 'PASS':
            raise ValueError(f'Native TEST_PUBLIC map QA failed: {qa}')
        qa_rows.extend(qa['per_sample'])
        accumulator.add(score, mask, split='test_public', image_id=sample_id)
        normal_count += int(record.is_normal); abnormal_count += int(not record.is_normal)
        manifest.append(dict(image=record.image_path, image_sha256=file_sha256(record.image_path),
                             mask=record.mask_path, mask_sha256=file_sha256(record.mask_path) if record.mask_path else None,
                             native_hw=list(hw), is_normal=record.is_normal))
        if examples < example_limit:
            save_public_example(image, mask, score, record, output / 'examples' / sample_id)
            examples += 1
        logging.info('TEST_PUBLIC category=%s image=%d/%d native=%s', cfg['category'], index+1, len(records), hw)
    metric = accumulator.result()['groups']['all']
    result = dict(status='PASS' if metric['aupro_0_05'] is not None else 'UNDEFINED',
                  category=cfg['category'], split='test_public', synthetic=False, full_split=True,
                  model_selection='best.pt selected on synthetic DEV only; no TEST tuning',
                  n_images=len(records), n_normal=normal_count, n_abnormal=abnormal_count,
                  native_resolution=True, qa_status='PASS', score_orientation='higher_is_more_anomalous',
                  score_normalization='sigmoid_no_rescaling', aupro_max_fpr=.05,
                  test_public_aupro_0_05=metric['aupro_0_05'], metric_details=metric)
    write_metrics_json(dict(status='PASS', per_sample=qa_rows), output / 'qa_report.json')
    write_metrics_json(dict(split='test_public', images=manifest), output / 'source_manifest.json')
    write_metrics_json(result, output / 'metrics.json')
    return result


def run_public(cfg, checkpoint, device, output, *, example_limit=8):
    check_assets(cfg, device)
    _, sources = discover_sources(cfg)
    cfg['sources'] = sources
    cfg['backbone']['checkpoint_sha256'] = file_sha256(cfg['backbone']['weights'])
    payload, _ = load_checkpoint_payload(checkpoint, require_sha256=True)
    if payload['config']['training']['mode'] != 'train':
        raise ValueError('TEST_PUBLIC requires the full train checkpoint, not smoke/Overfit-16')
    if resume_identity(payload['config']) != resume_identity(cfg):
        raise ValueError('TEST_PUBLIC config/source/pretrained checkpoint provenance mismatch')
    test_records(cfg)  # Fail on missing/ambiguous public GT before loading the model.
    extractor = DINOv3FeatureExtractor(cfg['backbone']['repo_dir'], cfg['backbone']['weights'],
                                     model_name=cfg['backbone']['name'], feature_mode='deepest', check_finite=True)
    model = E1(extractor, **cfg['decoder']).to(device)
    model.decoder.load_state_dict(payload['model_state'], strict=True)
    result = evaluate_public(model, cfg, device, output, example_limit=example_limit)
    result.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_sha256(checkpoint))
    write_metrics_json(result, Path(output) / 'metrics.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--example-limit', type=int, default=8)
    args = parser.parse_args(argv)
    if args.example_limit < 0:
        parser.error('--example-limit must be >=0')
    cfg = resolve_config(train_args(['--config', args.config]))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(output / 'test_public.log'), logging.StreamHandler()], force=True)
    try:
        result = run_public(cfg, args.checkpoint, args.device, output, example_limit=args.example_limit)
        logging.info('TEST_PUBLIC result: %s', result)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        write_metrics_json(dict(status='NOT RUN' if isinstance(exc, FileNotFoundError) else 'FAIL',
                                reason=str(exc), split='test_public'), output / 'failure.json')
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
