#!/usr/bin/env python3
"""Day-04 FULL-v3 source plan and synthetic generator, with native DEV images.

One deterministic crop per raw TRAIN image, hash80/20 v3, alternating anomaly,
TRAIN intensity/color/noise, DEV cutpaste. Reuses the sharded cache writer.
--export-dev-only replays Day-04 records and verifies masks AND all six features
before publishing native DEV inputs. It never relabels cache metadata.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as TF
from src.data.loader import scan_mvtec_ad2, normalize_dinov3
from src.data.feature_cache import FeatureCacheWriter, FeatureCacheReader, FEATURE_KEYS, sample_key, validate_cache
from src.data.cached_dataset import load_training_records
from src.data.synthetic_anomaly import SyntheticAnomalyConfig, SyntheticAnomalyGenerator
from src.geometry.view_meta import build_view_meta
from src.models.dinov3_extractor import build_online_extractor
from src.train.day05_contract import file_hash, validate_signature, digest
from src.train.screen_representation import save_json
import hashlib

SPLIT_VERSION = 'day04_full_hash80_20_v3'


def stable_hash(s): return hashlib.sha256(s.encode()).hexdigest()


def source_plan(data_root, categories):
    rows = []
    scanned = scan_mvtec_ad2(data_root)
    for cat in categories:
        roots = set()
        for r in scanned:
            if r.category != cat or r.split != 'train': continue
            p=Path(r.image_path); parts=p.relative_to(data_root).parts
            for j in range(len(parts)-1):
                if parts[j:j+2] == ('train','good'): roots.add(Path(data_root).joinpath(*parts[:j+2]))
        if len(roots)!=1:
            raise FileNotFoundError(f'BLOCKED_MISSING_DATA: require one TRAIN/good root for {cat}, found {len(roots)}')
        root=roots.pop()
        paths=sorted(Path(r.image_path) for r in scanned if r.category==cat and r.split=='train'
                     and Path(r.image_path).is_relative_to(root))
        if not paths: raise FileNotFoundError(f'BLOCKED_MISSING_DATA: {root}')
        parts = {'train_core': [], 'dev_synthetic': []}
        for p in paths:
            sid = f'{cat}/{p.relative_to(root).as_posix()}'
            split = 'train_core' if int(stable_hash(f'{SPLIT_VERSION}|{sid}')[:8], 16) % 10000 < 8000 else 'dev_synthetic'
            parts[split].append((sid, p))
        if min(map(len, parts.values())) < 4:
            raise ValueError(f'{cat}: FULL-v3 requires at least four sources per split')
        for split, items in parts.items():
            for i, (sid, p) in enumerate(sorted(items, key=lambda x: stable_hash(x[0]))):
                anomaly = bool(i % 2)
                iid = f'{cat}_{split}_{i:05d}_{stable_hash(sid)[:12]}_{"anom" if anomaly else "normal"}'
                rows.append(dict(category=cat, split=split, image_id=iid, source_identity=sid,
                                 source_path=str(p), make_anomaly=anomaly))
    return rows


def prepare_sample(entry):
    with Image.open(entry['source_path']) as im: arr=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
    raw=torch.from_numpy(arr).permute(2,0,1).float().div_(255).contiguous()
    _, h0, w0 = raw.shape
    ph, pw = max(0, 768-h0), max(0, 768-w0)
    top, left = ph//2, pw//2
    padding = (left, pw-left, top, ph-top)
    padded = raw
    if ph or pw:
        mode = 'reflect' if h0 > max(top, ph-top) and w0 > max(left, pw-left) else 'replicate'
        padded = F.pad(raw, padding, mode=mode)
    h, w = padded.shape[-2:]
    sh = stable_hash(entry['source_identity'])
    y, x = int(sh[:16], 16) % (h-768+1), int(sh[16:32], 16) % (w-768+1)
    crop = padded[:, y:y+768, x:x+768].contiguous()
    types = ('intensity', 'color', 'noise') if entry['split'] == 'train_core' else ('cutpaste',)
    generator = SyntheticAnomalyGenerator(SyntheticAnomalyConfig(anomaly_probability=1., anomaly_types=types))
    seed = int(stable_hash(f'{SPLIT_VERSION}|{entry["split"]}|{entry["source_identity"]}')[:15], 16)
    for attempt in range(128):
        synth = generator(crop, seed=seed+attempt, force_anomaly=entry['make_anomaly'])
        local_mask = synth.mask[:, 128:640, 128:640].contiguous()
        if not entry['make_anomaly'] or bool(local_mask.any()): break
    else: raise ValueError('No synthetic anomaly in the Local crop after locked 128 attempts')
    native = padded.clone(); native[:, y:y+768, x:x+768] = synth.image
    native_mask = torch.zeros((1,h,w)); native_mask[:,y:y+768,x:x+768] = synth.mask
    native = native[:,top:top+h0,left:left+w0].contiguous()
    native_mask = native_mask[:,top:top+h0,left:left+w0].contiguous()
    meta = dict(source_original_hw=[h0,w0], padding_ltrb=[left,top,pw-left,ph-top],
                context_crop_xy_on_padded=[x,y], synthetic=dict(synth.metadata),
                split=entry['split'], original_hw=[512,512], evaluation_unit='derived_local_crop_512',
                smoke_only=False, full_source_coverage=True, source_identity=entry['source_identity'],
                source_relpath=entry['source_identity'])
    return synth.image[:,128:640,128:640].contiguous(), synth.image, local_mask, native, native_mask, meta


def write_array(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), arr): raise ValueError(f'Existing image differs: {path}')
        return
    tmp = path.with_suffix('.tmp.npy'); np.save(tmp, arr); os.replace(tmp, path)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask[0].numpy() > .5).astype(np.uint8)*255
    if path.exists():
        if not np.array_equal(np.array(Image.open(path)), arr): raise ValueError(f'Existing mask differs: {path}')
        return
    tmp = path.with_suffix('.tmp.png'); Image.fromarray(arr).save(tmp); os.replace(tmp, path)


def build(args):
    for path in (args.dino_checkpoint, args.dinov3_repo/'hubconf.py'):
        if not path.is_file(): raise FileNotFoundError(f'BLOCKED_MISSING_DATA: {path}')
    if args.export_dev_only and any(x is None for x in (args.cache_dir,args.train_records,args.val_records,args.mask_root)):
        raise ValueError('--export-dev-only requires cache/train/dev/mask paths from Day-04')
    plan = source_plan(args.data_root, args.categories.split(','))
    if args.dry_run:
        return dict(status='ASSETS_VALID', samples=len(plan), extraction_executed=False)
    root = Path(args.output_root).absolute()
    sig = dict(schema='msila.day05.full_cache.v1', dataset='MVTec_AD_2', categories=args.categories.split(','),
        smoke_only=False, full_source_coverage=True, source_split_version=SPLIT_VERSION, train_fraction=.8,
        backbone='dinov3_vits16', checkpoint_sha256=file_hash(args.dino_checkpoint),
        logical_layers_1based=[4,8,12], internal_indices_0based=[3,7,11], local_source_size=[512,512],
        context_source_size=[768,768], model_input_size=[512,512], normalization='ImageNet mean/std',
        train_anomaly_types=['intensity','color','noise'], dev_anomaly_types=['cutpaste'],
        preprocessing='context_bicubic_antialias; local_identity; normalize_after_resize',
        config_version='day04_full_crop512_ctx768_all3_v3', evaluation_unit='derived_local_crop_512',
        synthetic_code_sha256=file_hash(Path(__file__).parents[1]/'src/data/synthetic_anomaly.py'),
        builder_sha256=file_hash(__file__),
        source_sha256={e['source_identity']:file_hash(e['source_path']) for e in plan})
    cache = Path(args.cache_dir) if args.cache_dir else root/'feature_cache'
    reader, existing = None, {}
    if args.export_dev_only:
        reader = FeatureCacheReader(cache)
        validate_signature(reader.manifest['producer_signature'], args.dino_checkpoint)
        for p in (args.train_records, args.val_records):
            for r in load_training_records(p): existing[(r['category'],r['image_id'])] = r
        required = {(e['category'],e['image_id']) for e in plan}
        if set(existing) != required: raise ValueError('Day-04 records differ from the FULL source plan; pass the same categories')
    else:
        reader = FeatureCacheReader(cache) if (cache/'manifest.json').exists() else None
        if reader and reader.manifest['producer_signature'] != sig: raise ValueError('Cache producer drift; use a new root')
    extractor = build_online_extractor(repo_dir=args.dinov3_repo, weights=args.dino_checkpoint,
                                      device=args.device, model_name='dinov3_vits16')
    if extractor.out_channels != 384 or extractor.patch_size != 16: raise ValueError('Not ViT-S/16')
    g = build_view_meta(source_hw=(768,768),local_box_xyxy=(128,128,640,640),context_box_xyxy=(0,0,768,768))
    geometry = dict(image_hw=[768,768],local_hw=[512,512],context_hw=[768,768],local_input_hw=[512,512],
        context_input_hw=[512,512],local_box=[128,128,640,640],context_box=[0,0,768,768],
        local_to_context=g.local_to_context.tolist(),context_to_local=g.context_to_local.tolist())
    records = {'train_core':[], 'dev_synthetic':[]}; inputs=[]
    writer = None if args.export_dev_only else FeatureCacheWriter(cache,producer_signature=sig,max_samples_per_shard=16)
    try:
        for e in plan:
            local, ctx, lm, native, nm, meta = prepare_sample(e)
            rel = Path(e['split'])/e['category']/(e['image_id']+'.png')
            record = dict(image_id=e['image_id'],category=e['category'],mask_path=str(rel),mask_hw=[512,512],
                          is_anomaly=e['make_anomaly'],meta=meta)
            if args.export_dev_only:
                old = existing[e['category'],e['image_id']]
                old_mask = np.array(Image.open(Path(args.mask_root)/old['mask_path'])) > 0
                if not np.array_equal(old_mask, lm[0].numpy()>0) or digest(old['meta']['synthetic']) != digest(meta['synthetic']):
                    raise ValueError('Synthetic replay differs from Day-04; do not reuse this cache')
            else: write_mask(root/'masks'/rel,lm)
            # Export mode checks both independent Context and Local against real cached DINO outputs.
            key = sample_key(e['image_id'],e['category'])
            if args.export_dev_only or key not in writer.manifest['index']:
                ctx_input=TF.resize(ctx,[512,512],interpolation=InterpolationMode.BICUBIC,antialias=True)
                with torch.inference_mode():
                    features=extractor.extract_online_cache_features(normalize_dinov3(local)[None].to(args.device),
                        normalize_dinov3(ctx_input)[None].to(args.device),strategy='sequential',to_cpu=True)
                if args.export_dev_only:
                    cached=reader.get(image_id=e['image_id'],category=e['category'])
                    for k in FEATURE_KEYS:
                        if not torch.allclose(features[k].float(),cached[k].float(),rtol=1e-4,atol=1e-5):
                            raise ValueError(f'Day-04 replay feature mismatch: {e["image_id"]}/{k}')
                else:
                    writer.add(dict(image_id=e['image_id'],category=e['category'],geometry=geometry,**features))
            records[e['split']].append(record)
            if e['split']=='dev_synthetic':
                ip=root/'images'/e['category']/(e['image_id']+'.npy')
                mp=root/'native_masks'/e['category']/(e['image_id']+'.png')
                write_array(ip,native.permute(1,2,0).numpy().astype(np.float32));write_mask(mp,nm)
                inputs.append(dict(image_id=e['image_id'],category=e['category'],image=str(ip),gt_mask=str(mp),
                    image_sha256=file_hash(ip),gt_mask_sha256=file_hash(mp),original_hw=list(native.shape[-2:]),
                    source_image_id=e['source_identity'],synthetic=meta['synthetic'],crop_provenance=meta))
    finally:
        if writer: writer.close()
    if not args.export_dev_only:
        for split, rows in records.items(): save_json(rows,root/'records'/f'{split}.json')
    save_json(dict(schema='msila.day05.inference_inputs.v1',split='dev_synthetic',
        evaluation_unit='native_source_image_with_locked_context_synthesis',samples=inputs),root/'dev_inputs.json')
    validate_cache(cache)
    result=dict(status='COMPLETE',samples=len(plan),cache=str(cache),export_dev_only=args.export_dev_only)
    save_json(result,root/'COMPLETE.json');return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','output-root','dinov3-repo','dino-checkpoint'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--categories',default='fabric')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--export-dev-only',action='store_true')
    for k in ('train-records','val-records','mask-root'):p.add_argument('--'+k,type=Path)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args(argv)


if __name__=='__main__': print(json.dumps(build(parse_args()),indent=2))
