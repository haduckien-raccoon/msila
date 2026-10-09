#!/usr/bin/env python3
"""DINOv3 full cache builder: native tiny/mixed DEV v2 and legacy Day-04 replay.

--synthetic-protocol selects native v2 synthesis, source-disjoint DEV groups,
all native Local/Context tiles and configurable S/S+/B/L/H+ frozen backbones.

Legacy mode:
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
    if getattr(args, 'backbone', 'dinov3_vits16') != 'dinov3_vits16' and not getattr(args, 'synthetic_protocol', None):
        args.synthetic_protocol = Path('configs/full_scale_synthetic.yaml')
    if getattr(args, 'synthetic_protocol', None):
        return build_native_cache(args)
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
        backbone=getattr(args, 'backbone', 'dinov3_vits16'), checkpoint_sha256=file_hash(args.dino_checkpoint),
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
                                      device=args.device, model_name=getattr(args, 'backbone', 'dinov3_vits16'),
                                      blocks=getattr(args, 'blocks', None))
    if getattr(extractor,'blocks',(4,8,12)) != (4,8,12):
        raise ValueError('Physical blocks differ from legacy cache; use --synthetic-protocol for full-scale cache')
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
    from src.models.backbone_registry import BACKBONES
    p.add_argument('--backbone', choices=list(BACKBONES), default='dinov3_vits16')
    p.add_argument('--blocks', type=int, nargs=3)
    p.add_argument('--seed', type=int, default=42, help='Data-generation seed, shared across training seeds')
    p.add_argument('--synthetic-protocol', type=Path)
    p.add_argument('--data-artifact-root', type=Path)
    return p.parse_args(argv)


NATIVE_SPLIT_VERSION = 'train_normal_hash80_10_10_tiny_v2'


def native_source_plan(data_root, categories, protocol, seed=42):
    """Only official TRAIN normals; disjoint sources across TRAIN/tiny/mixed."""
    from itertools import product
    from src.data.synthetic_anomaly import validate_native_protocol
    validate_native_protocol(protocol)
    scanned = scan_mvtec_ad2(data_root, split='train', categories=categories)
    rows = []
    for cat in categories:
        sources = sorted((r for r in scanned if r.category == cat and r.is_normal), key=lambda r: r.image_path)
        roles = dict(train_core=[], dev_tiny=[], dev_mixed=[])
        for r in sources:
            sid = Path(r.image_path).relative_to(data_root).as_posix()
            u = int(stable_hash(f'{NATIVE_SPLIT_VERSION}|{seed}|{sid}')[:8], 16) % 10000
            role = 'train_core' if u < 8000 else 'dev_tiny' if u < 9000 else 'dev_mixed'
            roles[role].append((sid, r.image_path))
        if any(not values for values in roles.values()):
            raise ValueError(f'{cat}: empty source-disjoint TRAIN/DEV partition; counts=' +
                             str({k: len(v) for k, v in roles.items()}))
        for role, items in roles.items():
            variants = list(product(protocol['defect_types'], protocol[role+'_bins'], protocol['placements']))
            for i, (sid, path) in enumerate(items):
                # All DEV strata for each source, plus an unchanged normal.
                # TRAIN uses a fixed, stratified defect per source plus normal;
                # every native tile is retained, including negative tiles.
                chosen = [variants[i % len(variants)]] if role == 'train_core' else variants
                for j, variant in enumerate([None] + chosen):
                    kind, size, placement = variant or variants[0]
                    iid = f'{cat}_{role}_{stable_hash(sid)[:16]}_v{j:03d}'
                    rows.append(dict(category=cat, split=role, image_id=iid, source_identity=sid,
                                     source_path=path, make_anomaly=variant is not None,
                                     defect_type=kind, size_bin=size, placement=placement,
                                     synthetic_seed=int(stable_hash(f'{seed}|{iid}')[:15],16)))
    return rows


def build_native_cache(args):
    import yaml
    import subprocess
    import time
    import importlib.metadata
    from collections import Counter
    from src.data.synthetic_anomaly import NativeTinyDefectGenerator
    from src.data.loader import load_rgb_native
    from src.data.tiling import generate_tile_records, crop_with_padding
    from src.eval.region_stats import component_geometry
    from src.models.backbone_registry import backbone_spec, validate_blocks
    from scripts.day05_full_inference import make_tile_batch

    started = time.perf_counter()
    for path in (args.dino_checkpoint, args.dinov3_repo/'hubconf.py'):
        if not path.is_file():
            raise FileNotFoundError(f'BLOCKED_MISSING_DATA: {path}')
    if getattr(args, 'export_dev_only', False):
        raise ValueError('Native v2 data cannot relabel legacy caches via --export-dev-only')
    protocol = yaml.safe_load(Path(args.synthetic_protocol).read_text())
    seed = getattr(args, 'seed', 42)
    plan = native_source_plan(args.data_root, args.categories.split(','), protocol, seed)
    spec = backbone_spec(args.backbone)
    blocks = validate_blocks(spec.blocks if getattr(args, 'blocks', None) is None else args.blocks, spec.depth)
    if args.dry_run:
        return dict(status='ASSETS_VALID', backbone=args.backbone, physical_blocks=list(blocks),
                    native_images=len(plan), extraction_executed=False, vram_validated=False)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('BLOCKED_MISSING_GPU: CUDA requested but unavailable')
    root = Path(args.output_root).absolute()
    data_root = Path(getattr(args, 'data_artifact_root', None) or root/'dataset').absolute()
    cache = Path(getattr(args, 'cache_dir', None) or root/'feature_cache').absolute()
    unique_sources = {e['source_identity']:e['source_path'] for e in plan}
    source_hashes = {sid:file_hash(path) for sid,path in unique_sources.items()}
    try:
        revision = subprocess.check_output(['git', '-C', str(args.dinov3_repo), 'rev-parse', 'HEAD'],
                                           text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = None
    sig = dict(schema='msila.full_scale.cache.v2', dataset='MVTec_AD_2', categories=args.categories.split(','),
        backbone=args.backbone, checkpoint_sha256=file_hash(args.dino_checkpoint),
        architecture=dict(channels=spec.channels, depth=spec.depth, patch_size=spec.patch_size),
        logical_layers_1based=list(blocks), internal_indices_0based=[b-1 for b in blocks],
        cache_slot_blocks=dict(zip(('b4','b8','b12'),blocks)),
        local_source_size=[512,512], context_source_size=[768,768], model_input_size=[512,512],
        normalization='ImageNet mean/std', preprocessing='native_synthesis;all_tiles512_overlap128;reflect_RGB;'
        'constant_zero_mask_padding;context768_bicubic_antialias_to512;normalize_after_resize',
        source_split_version=NATIVE_SPLIT_VERSION, data_seed=seed, source_sha256=source_hashes,
        synthetic_protocol=protocol, synthetic_protocol_sha256=digest(protocol),
        source_plan_sha256=digest(plan), dinov3_git_revision=revision,
        dinov3_source_sha256={str(p.relative_to(args.dinov3_repo)):file_hash(p)
                             for p in sorted(Path(args.dinov3_repo).rglob('*.py')) if '.git' not in p.parts},
        synthetic_code_sha256=file_hash(Path(__file__).parents[1]/'src/data/synthetic_anomaly.py'),
        builder_sha256=file_hash(__file__), smoke_only=False, full_source_coverage=True,
        feature_storage_dtype='float32', evaluation_unit='native_source_image',
        feature_norm=True, extraction_amp=False,
        producer_versions={name:importlib.metadata.version(name) for name in
                           ('torch','torchvision','numpy','Pillow')},
        training_sampling='every_tile_of_normal_and_one_stratified_defect_per_train_source')
    validate_signature(sig, args.dino_checkpoint, expected_backbone=args.backbone)
    if (cache/'manifest.json').is_file():
        saved=FeatureCacheReader(cache,mmap=True,shard_cache_size=1)
        if saved.manifest['producer_signature']!=sig:
            raise ValueError('Native cache producer changed; use a new cache root')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    extractor = None
    generator = NativeTinyDefectGenerator(protocol)
    records = dict(train_core=[], dev_tiny=[], dev_mixed=[])
    inputs, stats = [], []
    try:
        with FeatureCacheWriter(cache, producer_signature=sig, max_samples_per_shard=16) as writer:
            for e in plan:
                raw = load_rgb_native(e['source_path'])
                synth = generator(raw, seed=e['synthetic_seed'], defect_type=e['defect_type'],
                                  size_bin=e['size_bin'], placement=e['placement'], normal=not e['make_anomaly'])
                hw = list(raw.shape[-2:])
                geometry = component_geometry(synth.mask[0].numpy(), protocol['boundary_band_px'])
                if e['make_anomaly'] and (len(geometry) != 1 or geometry[0]['area'] != synth.metadata['area']):
                    raise ValueError('Native generator must produce one connected region of declared area')
                audit = dict(image_id=e['image_id'], category=e['category'], split=e['split'],
                             native_hw=hw, synthetic=synth.metadata, native_components=geometry, tiles=[])
                for tile in generate_tile_records(*hw):
                    iid = f'{e["image_id"]}_tile{tile.tile_id:05d}'
                    mask = crop_with_padding(synth.mask, tile.local_xyxy, pad_mode='constant')
                    rel = Path(e['split'])/e['category']/(iid+'.png')
                    write_mask(data_root/'masks'/rel, mask)
                    key = sample_key(iid,e['category'])
                    if key not in writer.manifest['index']:
                        if extractor is None:
                            extractor=build_online_extractor(repo_dir=args.dinov3_repo,weights=args.dino_checkpoint,
                                                            model_name=args.backbone,blocks=blocks,device=device)
                        features = make_tile_batch(synth.image, tile, extractor, device, context=True)
                        g = features.pop('meta')[0]['geometry']
                        features.pop('output_hw')
                        features={key:value.detach().float() for key,value in features.items()}
                        # The cache geometry describes both views in the Context frame.
                        lx0,ly0,lx1,ly1=tile.local_xyxy;cx0,cy0,cx1,cy1=tile.context_xyxy
                        g.update(image_hw=[768,768],local_hw=[512,512],context_hw=[768,768],
                                 local_box=[lx0-cx0,ly0-cy0,lx1-cx0,ly1-cy0],context_box=[0,0,768,768])
                        writer.add(dict(image_id=iid,category=e['category'],geometry=g,**features))
                    components = component_geometry(mask[0].numpy(), protocol['boundary_band_px'])
                    context_mask=crop_with_padding(synth.mask,tile.context_xyxy,pad_mode='constant')
                    resized_context=F.interpolate(context_mask[None],size=(512,512),mode='nearest')[0,0].numpy()
                    audit['tiles'].append(dict(tile_id=tile.tile_id, local_xyxy=list(tile.local_xyxy),
                                               local_hw=[512,512], components=components,
                                               context_resize_scale=512/768,
                                               context_components=component_geometry(context_mask[0].numpy()),
                                               context_input_components=component_geometry(resized_context),
                                               component_coordinate_frame='local_tile_or_context_input',
                                               boundary_definition='native_components refer to image edge; tile components refer to tile edge'))
                    meta = dict(split=e['split'],source_identity=e['source_identity'], synthetic=synth.metadata,
                                native_image_id=e['image_id'], native_hw=hw, tile_xyxy=list(tile.local_xyxy),
                                smoke_only=False,full_source_coverage=True)
                    records[e['split']].append(dict(image_id=iid,category=e['category'],mask_path=str(rel),
                        mask_hw=[512,512],is_anomaly=bool(mask.any()),meta=meta))
                stats.append(audit)
                if e['split'] != 'train_core':
                    ip=data_root/'images'/e['category']/(e['image_id']+'.npy')
                    mp=data_root/'native_masks'/e['category']/(e['image_id']+'.png')
                    write_array(ip,synth.image.permute(1,2,0).numpy().astype(np.float32));write_mask(mp,synth.mask)
                    inputs.append(dict(image_id=e['image_id'],category=e['category'],split=e['split'],
                        image=str(ip),gt_mask=str(mp),image_sha256=file_hash(ip),gt_mask_sha256=file_hash(mp),
                        original_hw=hw,source_image_id=e['source_identity'],synthetic=synth.metadata))
                print(f'[CACHE] {args.backbone}: {e["image_id"]}', flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        save_json(dict(status='FAILED_OOM', backbone=args.backbone, reason=str(exc),
                       configuration_preserved=True),root/'cache_failure.json')
        raise
    for role, rows in records.items():
        save_json(rows, data_root/'records'/f'{role}.json')
    save_json(records['dev_tiny']+records['dev_mixed'],data_root/'records/dev.json')
    save_json(dict(schema='msila.full_scale.native_inputs.v2',samples=inputs,
                   synthetic_protocol=protocol,synthetic_protocol_sha256=digest(protocol)),data_root/'dev_inputs.json')
    coverage = Counter((e['split'], e['size_bin'], e['placement'], e['defect_type'])
                       for e in plan if e['make_anomaly'])
    save_json(dict(schema='msila.full_scale.gt_stats.v2', images=stats,
                   coverage=[dict(split=k[0],size_bin=k[1],placement=k[2],defect_type=k[3],images=v)
                             for k,v in sorted(coverage.items())],
                   synthetic_proxy=True, real_defect_distribution_calibrated=False),data_root/'gt_statistics.json')
    validate_cache(cache)
    result = dict(status='COMPLETE', backbone=args.backbone, cache=str(cache), data_root=str(data_root),
                  native_images=len(plan), tile_records=sum(map(len,records.values())),
                  producer_sha256=digest(sig), elapsed_sec=time.perf_counter()-started,
                  peak_vram_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.,
                  dev_inputs_sha256=file_hash(data_root/'dev_inputs.json'))
    save_json(result,root/'COMPLETE.json')
    return result


if __name__=='__main__': print(json.dumps(build(parse_args()),indent=2))
