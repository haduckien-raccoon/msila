#!/usr/bin/env python3
"""Strict TV1 handoff -> real Local/Context tiles -> native float32 maps."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as TF
from src.data.loader import MVTecAD2HighResDataset, load_rgb_native, normalize_dinov3
from src.data.tiling import generate_tile_records, crop_with_padding, stitch_tiles_hann
from src.geometry.view_meta import build_view_meta
from src.models.dinov3_extractor import build_online_extractor
from src.train.day05_contract import SOURCES, digest, file_hash, validate_signature, validate_day05
from src.train.screen_representation import Day05RepresentationModel
from scripts.build_evaluation_manifest import read_artifact, verify_and_build_manifest
from scripts.day05_legacy_handoff import model_config
from scripts.eval_day05_representation import evaluator_hashes

PREPROCESSING = dict(rgb_decode='PIL.convert_RGB_uint8_div255', local_size=512, context_size=768, input_size=512, overlap=128,
    context_resize='bicubic_antialias', normalization='ImageNet mean/std',
    padding='reflect_with_small_image_replicate_fallback', blocks=[4, 8, 12])
STITCHING = 'sigmoid_per_tile; hann2d(periodic=False,clamp_min=0.001); weighted_mean; native_HW'


def load_tv1_model(run_dir, candidate, device='cpu'):
    cfg, row = read_artifact(run_dir, candidate)
    construction_config = model_config(cfg)
    validate_day05(construction_config)
    model = Day05RepresentationModel(day05_config=construction_config, candidate=candidate,
        in_channels=cfg['cache']['in_channels'], adapter_r=cfg['adapter']['r'], adapter_d=cfg['adapter']['d'])
    state = torch.load(row['checkpoint_path'], map_location='cpu', weights_only=False)
    if (state.get('schema') != 'msila.day05.full_train.checkpoint.v1'
            or state.get('resolved_config_sha256') != digest(cfg)):
        raise ValueError('TV1 checkpoint schema/config mismatch')
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), cfg, row


def make_tile_batch(image, record, extractor, device, context=True):
    local = crop_with_padding(image, record.local_xyxy, pad_mode='reflect')
    x_local = normalize_dinov3(local).unsqueeze(0).to(device)
    if context:
        context_crop = crop_with_padding(image, record.context_xyxy, pad_mode='reflect')
        context_input = TF.resize(context_crop, [512, 512],
                                  interpolation=InterpolationMode.BICUBIC, antialias=True)
        features = extractor.extract_online_cache_features(x_local,
            normalize_dinov3(context_input).unsqueeze(0).to(device), strategy='sequential')
    else:
        single = extractor(x_local)
        physical = getattr(extractor, 'blocks', (4,8,12))
        features = {f'local_b{slot}': single[f'b{block}']
                    for slot, block in zip((4,8,12),physical)}
    # Validate in the padded Context frame; translations cancel in L->C.
    lx0, ly0, lx1, ly1 = record.local_xyxy
    cx0, cy0, cx1, cy1 = record.context_xyxy
    g = build_view_meta(source_hw=(cy1-cy0, cx1-cx0),
                        local_box_xyxy=(lx0-cx0, ly0-cy0, lx1-cx0, ly1-cy0),
                        context_box_xyxy=(0, 0, cx1-cx0, cy1-cy0), local_input_hw=(512, 512),
                        context_input_hw=(512, 512), validate=True)
    features.update(meta=[{'geometry': dict(local_to_context=g.local_to_context.tolist(),
                    context_to_local=g.context_to_local.tolist(), local_box=list(record.local_xyxy),
                    context_box=list(record.context_xyxy), local_input_hw=[512, 512], context_input_hw=[512, 512])}],
                    output_hw=(512, 512))
    return features


@torch.inference_mode()
def infer_image(image, model, extractor, device):
    records = generate_tile_records(*image.shape[-2:], local_size=512, overlap=128, context_size=768)
    maps = []
    for record in records:
        batch = make_tile_batch(image, record, extractor, device, context=model.candidate == 'R2')
        logits, _ = model(batch)
        score = logits[0, 0].float().sigmoid().cpu()
        if score.shape != (512, 512) or not torch.isfinite(score).all():
            raise ValueError('Invalid tile anomaly map')
        maps.append(score)
    result = stitch_tiles_hann(maps, records, tuple(image.shape[-2:])).numpy().astype(np.float32)
    if result.shape != tuple(image.shape[-2:]) or not np.isfinite(result).all():
        raise ValueError('Invalid stitched native anomaly map')
    return result


def load_input_image(path):
    if Path(path).suffix.lower() != '.npy':
        with Image.open(path) as im: arr = np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2,0,1).float().div_(255).contiguous()
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.float32 or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError('Synthetic native RGB must be HWC float32 [0,1]')
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()


def load_inputs(args, cfg, validate_only=False):
    if args.split == 'dev_synthetic':
        if not args.input_manifest:
            raise ValueError('DEV-synthetic is not an official loader split: supply --input-manifest with real synthetic images/GT')
        path = Path(args.input_manifest).absolute()
        obj = json.loads(path.read_text())
        if obj.get('schema') != 'msila.day05.inference_inputs.v1' or obj.get('split') != args.split:
            raise ValueError('Invalid DEV input manifest schema/split')
        samples = [dict(s) for s in obj['samples'] if s['category'] == cfg['category']]
        from src.data.cached_dataset import load_training_records
        records = {(r['category'],r['image_id']):r for r in load_training_records(args.val_records)
                   if r['category']==cfg['category']}
        expected = set(records)
        if {(s['category'], s['image_id']) for s in samples} != expected:
            raise ValueError('DEV map IDs must equal the locked validation IDs 1:1')
        if file_hash(args.val_records) != cfg['data_fingerprints']['val_source_sha256']:
            raise ValueError('Validation records changed since TV1 training')
        for s in samples:
            record=records[s['category'],s['image_id']]
            if (s.get('source_image_id')!=record['meta'].get('source_identity')
                    or digest(s.get('synthetic'))!=digest(record['meta'].get('synthetic'))):
                raise ValueError('DEV source/synthetic provenance differs from locked validation records')
            s['is_anomaly']=record['is_anomaly']
            if not s.get('synthetic') or not s.get('source_image_id'):
                raise ValueError('DEV image lacks synthetic/source provenance')
            for field in ('image', 'gt_mask'):
                p = Path(s[field]); p = p if p.is_absolute() else path.parent / p
                s[field] = str(p.absolute())
                if file_hash(p) != s.get(field + '_sha256'):
                    raise ValueError(f'{s["image_id"]}: {field} hash mismatch')
    else:
        if not args.representation_lock:
            raise ValueError('Final TEST inference requires the DEV representation lock')
        lock = yaml.safe_load(Path(args.representation_lock).read_text())
        if lock.get('schema') != 'msila.representation_lock.v1' or lock['evaluation_scope']['split'] != 'dev_synthetic':
            raise ValueError('Invalid representation lock')
        # Tie final TEST use to the DEV evidence and these exact TV1 artifacts.
        provenance=lock['provenance']
        for p,sha in provenance['input_files_sha256'].items():
            if file_hash(p)!=sha: raise ValueError('Representation lock evidence changed')
        dev_manifest=Path(args.r0).absolute().parents[2]/'maps/dev_synthetic/evaluation_manifest.json'
        if file_hash(dev_manifest)!=provenance['evaluation_manifest_sha256']:
            raise ValueError('Representation lock belongs to a different DEV evaluation')
        dev=json.loads(dev_manifest.read_text())
        hp=Path(dev['handoff_manifest'])
        if file_hash(hp)!=dev['handoff_sha256']: raise ValueError('DEV checkpoint handoff changed')
        handoff=json.loads(hp.read_text())
        for r,p in zip(handoff['runs'],(args.r0,args.r1,args.r2)):
            if r['checkpoint_sha256']!=file_hash(Path(p)/'best.pt'):
                raise ValueError('Representation lock checkpoint mismatch')
        ds = MVTecAD2HighResDataset(args.data_root, split=args.split, categories=[cfg['category']])
        samples = []
        for r in ds.records:
            # Official loader has image_norm, but only native image is cropped here.
            iid=Path(r.image_path).relative_to(args.data_root).as_posix()
            with Image.open(r.image_path) as im: hw = [im.height, im.width]
            normal = r.is_normal
            if r.mask_path is None and not normal:
                raise ValueError(f'Missing abnormal GT: {r.image_path}')
            mp=Path(r.mask_path).absolute() if r.mask_path else Path(args.output_dir).absolute()/'ground_truth'/(digest(iid)+'.png')
            if normal and not validate_only:
                mp.parent.mkdir(parents=True,exist_ok=True)
                if mp.exists() and (np.array(Image.open(mp)).any() or list(np.array(Image.open(mp)).shape)!=hw):
                    raise ValueError('Stored normal GT differs from the official good label')
                if not mp.exists():Image.fromarray(np.zeros(hw,dtype=np.uint8)).save(mp)
            samples.append(dict(image_id=iid,category=r.category,image=str(Path(r.image_path).absolute()),gt_mask=str(mp),
                original_hw=hw,image_sha256=file_hash(r.image_path),
                gt_mask_sha256=None if normal and validate_only else file_hash(mp),
                gt_provenance='official_good_label_zero_mask' if normal else 'official_abnormal_mask',
                known_normal=normal))
    ids = [s['image_id'] for s in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate inference IDs')
    positive = False
    for s in samples:
        hw = list(load_input_image(s['image']).shape[-2:])
        if validate_only and s.get('known_normal'): gt = np.zeros(hw,dtype=np.uint8)
        else:
            with Image.open(s['gt_mask']) as im: gt = np.array(im)
        if hw != s['original_hw'] or gt.shape != tuple(hw) or not set(np.unique(gt)).issubset({0, 1, 255}):
            raise ValueError(f'{s["image_id"]}: native image/GT shape or binary-mask mismatch')
        if 'is_anomaly' in s and bool(gt.any())!=bool(s['is_anomaly']):
            raise ValueError('Native synthetic GT disagrees with its anomaly label')
        positive |= bool(gt.any())
    if not positive:
        raise ValueError('AU-PRO requires at least one anomalous GT region')
    return samples


def run_inference(args):
    root = Path(args.output_dir).absolute()
    runs = [Path(args.r0), Path(args.r1), Path(args.r2)]
    # Validate all three artifacts before extracting any image.
    inspected = [read_artifact(p, c) for p, c in zip(runs, SOURCES)]
    if len({row['protocol_lock_sha256'] for _, row in inspected}) != 1:
        raise ValueError('Representation protocol mismatch')
    samples = load_inputs(args, inspected[0][0])
    bb = inspected[0][0]['backbone']['name']
    blocks = inspected[0][0]['backbone']['feature_blocks']
    if getattr(args,'backbone',None) and args.backbone != bb:
        raise ValueError('CLI/inference run backbone mismatch')
    for cfg, _ in inspected:
        if cfg['backbone']['name'] != bb or cfg['backbone']['feature_blocks'] != blocks:
            raise ValueError('Inference runs have different backbone/block identities')
        validate_signature(cfg['cache']['provenance_check']['producer_signature'], args.dino_checkpoint,
                           expected_backbone=bb)
    device = torch.device(args.device)
    extractor = build_online_extractor(repo_dir=args.dinov3_repo, weights=args.dino_checkpoint,
                                      model_name=bb, blocks=blocks, device=device)
    if extractor.out_channels != inspected[0][0]['cache']['in_channels'] or not extractor.backbone_is_frozen():
        raise ValueError('Real backbone does not match the frozen training backbone')
    root.mkdir(parents=True, exist_ok=True)
    verify_and_build_manifest(*runs, root / 'handoff_manifest.json')
    for s in samples:
        s['maps'] = {}; s['map_provenance'] = {}; s['image_path'] = s['image']
    for run_dir, c in zip(runs, SOURCES):
        model, cfg, row = load_tv1_model(run_dir, c, device)
        out = root / c; out.mkdir(exist_ok=True)
        for s in samples:
            name = digest([s['category'], s['image_id']])
            map_path, prov_path = out / f'{name}.npy', out / f'{name}.json'
            expected = dict(candidate=c, checkpoint_sha256=row['checkpoint_sha256'],
                image_id=s['image_id'], source_image_id=s.get('source_image_id', s['image_id']),
                image_sha256=file_hash(s['image']), gt_mask_sha256=file_hash(s['gt_mask']), original_hw=s['original_hw'], split=args.split,
                preprocessing=PREPROCESSING, stitching=STITCHING, coordinate_space='original_image')
            if map_path.exists() or prov_path.exists():
                old = json.loads(prov_path.read_text()) if prov_path.exists() else {}
                if any(old.get(k) != v for k, v in expected.items()) or not map_path.exists() or old.get('map_sha256') != file_hash(map_path):
                    raise ValueError('Existing map/provenance mismatch; use a new output directory')
                score = np.load(map_path, allow_pickle=False)
            else:
                image = load_input_image(s['image'])
                start = time.perf_counter()
                score = infer_image(image, model, extractor, device)
                np.save(map_path, score)
                prov_path.write_text(json.dumps(expected | dict(map_sha256=file_hash(map_path),
                    elapsed_seconds=time.perf_counter()-start), indent=2) + '\n')
            if score.dtype != np.float32 or score.shape != tuple(s['original_hw']) or not np.isfinite(score).all():
                raise ValueError('Stored map dtype/shape/finite gate failed')
            s['maps'][c] = str(map_path); s['map_provenance'][c] = str(prov_path)
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
    manifest = dict(schema='msila.day05.metric_manifest.v1', split=args.split,
        categories=[inspected[0][0]['category']], evaluator_sha256=evaluator_hashes(),
        normalization_by_candidate={c: 'sigmoid_logits_without_map_normalization' for c in SOURCES},
        seg_f1_threshold=args.seg_f1_threshold, samples=samples,
        handoff_manifest=str(root / 'handoff_manifest.json'),
        handoff_sha256=file_hash(root / 'handoff_manifest.json'))
    path = root / 'evaluation_manifest.json'
    content = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != content: raise ValueError('Existing metric manifest differs')
    path.write_text(content, encoding='utf-8')
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for c in SOURCES: p.add_argument('--' + c.lower(), type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--split', choices=['dev_synthetic', 'test_public'], default='dev_synthetic')
    p.add_argument('--input-manifest', type=Path)
    p.add_argument('--val-records', type=Path)
    p.add_argument('--representation-lock', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dinov3-repo', type=Path, required=True)
    p.add_argument('--dino-checkpoint', type=Path, required=True)
    p.add_argument('--backbone', default=None)
    p.add_argument('--seg-f1-threshold', type=float, required=True)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args(argv)


if __name__ == '__main__': run_inference(parse_args())
