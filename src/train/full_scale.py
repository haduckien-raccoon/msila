"""Full grid orchestration over the existing cache, trainer and native inference.

No model/training loop is defined here. Every declared job calls
screen_representation.train_candidate and uses the same update protocol.
"""
from __future__ import annotations
import copy
import csv
import gc
import itertools
import json
import platform
import os
import importlib.metadata
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml

from src.models.backbone_registry import BACKBONES, backbone_spec, adapter_pairs, validate_blocks
from src.train.day05_contract import digest, file_hash, validate_signature
from src.train.screen_representation import (read_yaml, save_json, save_yaml, train_candidate,
                                              enforce_lock, verify_protocol)


def expand_grid(config):
    if config.get('schema') != 'msila.full_scale.grid.v2':
        raise ValueError('Expected a versioned full-scale grid')
    if set(config['backbones']) - BACKBONES.keys() or not config['backbones']:
        raise ValueError('Grid contains an unsupported or empty backbone list')
    reps, seeds, cats = config['representations'], config['seeds'], config['categories']
    if not reps or set(reps) - {'R0','R1','R2'}:
        raise ValueError('Representations must be R0/R1/R2')
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or not cats:
        raise ValueError('Declare nonnegative integer seeds and nonempty categories')
    for values in (reps,seeds,cats):
        if len(set(values)) != len(values):
            raise ValueError('Duplicate grid axis value')
    jobs = []
    for bb, entry in config['backbones'].items():
        spec = backbone_spec(bb)
        validate_blocks(entry.get('blocks',spec.blocks),spec.depth)
        pairs = adapter_pairs(spec.channels,entry.get('adapter',config['adapter']))
        for (r,d),rep,seed,cat in itertools.product(pairs,reps,seeds,cats):
            jobs.append(dict(backbone=bb,r=r,d=d,representation=rep,seed=seed,category=cat))
    return jobs


def job_id(job):
    return f'{job["backbone"]}/adapter_r{job["r"]}_d{job["d"]}/seed_{job["seed"]}/{job["category"]}/{job["representation"]}'


def resolve_jobs(args, config):
    if args.train_all:
        if any(getattr(args,k,None) is not None for k in ('backbone','adapter_r','adapter_d','representation')):
            raise ValueError('--train-all executes exactly the YAML grid; single-job selectors require single mode')
        return expand_grid(config)
    if any(getattr(args,k,None) is None for k in ('backbone','adapter_r','adapter_d','representation')):
        raise ValueError('Single mode requires --backbone --adapter-r --adapter-d --representation --seed')
    if args.backbone not in config['backbones'] or args.adapter_r <= 0 or args.adapter_d <= 0:
        raise ValueError('Invalid single backbone/adapter configuration')
    return [dict(backbone=args.backbone,r=args.adapter_r,d=args.adapter_d,
                 representation=args.representation,seed=args.seed,category=args.category)]


def validate_extractor_source(extractor,signature):
    if signature.get('dinov3_source_sha256') is None:
        raise ValueError('Native evaluation requires the cache producer DINOv3 source identity')
    root=Path(extractor.repo_dir)
    actual={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.py')) if '.git' not in p.parts}
    if actual!=signature['dinov3_source_sha256']:
        raise ValueError('DINOv3 source code differs from the cache producer')


def evaluate_native_dev(run_dir, data_root, extractor, device):
    from scripts.day05_full_inference import load_tv1_model, load_input_image, infer_image
    from src.eval.full_scale import DEVMetricAccumulator
    from PIL import Image
    run_dir, data_root = Path(run_dir), Path(data_root)
    cfg = read_yaml(run_dir/'resolved_config.yaml')
    model, cfg, row = load_tv1_model(run_dir,cfg['candidate'],device)
    sig = cfg['cache']['provenance_check']['producer_signature']
    if (extractor.model_name != cfg['backbone']['name'] or list(extractor.blocks) != cfg['backbone']['feature_blocks']
            or extractor.out_channels != cfg['cache']['in_channels'] or not extractor.backbone_is_frozen()):
        raise ValueError('Native extractor/run backbone, blocks or frozen state mismatch')
    validate_signature(sig, extractor.weights, expected_backbone=extractor.model_name)
    validate_extractor_source(extractor,sig)
    manifest = json.loads((data_root/'dev_inputs.json').read_text())
    from src.data.cached_dataset import load_training_records
    validation_path=data_root/'records/dev.json'
    if file_hash(validation_path)!=cfg['data_fingerprints']['val_source_sha256']:
        raise ValueError('Native validation record index changed since training')
    expected={r['meta']['native_image_id']:r['meta'] for r in load_training_records(validation_path)
              if r['category']==cfg['category']}
    selected=[s for s in manifest['samples'] if s['category']==cfg['category']]
    if len(selected)!=len({s['image_id'] for s in selected}) or {s['image_id'] for s in selected}!=set(expected):
        raise ValueError('Native DEV IDs must equal the locked validation native-image IDs')
    for sample in selected:
        meta=expected[sample['image_id']]
        if (sample['source_image_id']!=meta['source_identity'] or sample['split']!=meta['split']
                or sample['synthetic']!=meta['synthetic']):
            raise ValueError('Native DEV source/synthetic metadata differs from training validation records')
    if manifest['synthetic_protocol'] != cfg['evaluation']['synthetic_protocol']:
        raise ValueError('Native DEV/evaluation protocol mismatch')
    provenance = dict(checkpoint_sha256=row['checkpoint_sha256'], resolved_config_sha256=digest(cfg),
                      dev_inputs_sha256=file_hash(data_root/'dev_inputs.json'),
                      producer_sha256=cfg['cache_producer_sha256'])
    directory = run_dir/'native_dev'
    report_path, map_manifest_path = directory/'metrics.json', directory/'map_manifest.json'
    directory.mkdir(parents=True,exist_ok=True)
    for sample in selected:
        for field in ('image','gt_mask'):
            if file_hash(sample[field])!=sample[field+'_sha256']:
                raise ValueError(f'Native DEV artifact changed: {sample["image_id"]}/{field}')
    maps = {}
    previous_elapsed = 0.
    previous_peak = 0.
    if map_manifest_path.exists():
        old = json.loads(map_manifest_path.read_text())
        if old['provenance'] != provenance:
            raise ValueError('Native evaluation provenance drift; no reuse of mismatched maps')
        maps = old['maps']
        previous_elapsed = old.get('elapsed_sec',0.)
        previous_peak = old.get('peak_vram_mb',0.)
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['provenance'] != provenance:
            raise ValueError('Native metric provenance drift')
        for key, info in maps.items():
            if not (directory/info['path']).is_file() or file_hash(directory/info['path']) != info['sha256']:
                raise ValueError(f'Native map missing/changed: {key}')
        if old['map_manifest_sha256'] != file_hash(map_manifest_path):
            raise ValueError('Native map manifest changed')
        return old
    accumulator = DEVMetricAccumulator(manifest['synthetic_protocol'],disk_backed=True)
    started = time.perf_counter()
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rows = [s for s in manifest['samples'] if s['category'] == cfg['category']]
    if not rows:
        raise ValueError('Empty native DEV input manifest')
    for sample in rows:
        image = load_input_image(sample['image'])
        if list(image.shape[-2:]) != sample['original_hw']:
            raise ValueError('Native DEV image resolution changed')
        with Image.open(sample['gt_mask']) as im:
            gt = np.array(im)>0
        key = sample['image_id']
        out = directory/(key+'.npy')
        if key in maps:
            if file_hash(out) != maps[key]['sha256']:
                raise ValueError('Cached native anomaly map corrupted')
            score = np.load(out,allow_pickle=False)
        else:
            score = infer_image(image,model,extractor,device)
            tmp = out.with_suffix('.tmp.npy');np.save(tmp,score);tmp.replace(out)
            maps[key]=dict(path=out.name,sha256=file_hash(out),native_hw=sample['original_hw'])
        accumulator.add(score,gt,split=sample['split'],image_id=key)
        peak = torch.cuda.max_memory_allocated(device)/1024**2 if torch.device(device).type=='cuda' else 0.
        save_json(dict(provenance=provenance,maps=maps,elapsed_sec=previous_elapsed+time.perf_counter()-started,
                       peak_vram_mb=max(previous_peak,peak)),map_manifest_path)
    report = accumulator.result()
    if report['dev_aupro'] is None:
        raise ValueError('Native DEV tiny/mixed metric undefined; cannot rank this configuration')
    report.update(provenance=provenance,map_manifest_sha256=file_hash(map_manifest_path),
                  elapsed_sec=previous_elapsed+time.perf_counter()-started,
                  peak_vram_mb=max(previous_peak,peak), evaluation_unit='native_source_image',
                  checkpoint_selection_unit='cached_512_native_tiles')
    save_json(report,report_path)
    del model
    return report


def rank_models(root, jobs, completed):
    """Require every declared seed/job; rank families by seed-mean native DEV."""
    from collections import defaultdict
    if set(completed) != {job_id(j) for j in jobs}:
        raise ValueError('Grid incomplete; TEST_PUBLIC selection lock cannot be created')
    grouped = defaultdict(list)
    for job in jobs:
        jid = job_id(job)
        grouped[(job['category'],job['backbone'],job['r'],job['d'],job['representation'])].append((job,completed[jid]))
    ranking = []
    for key, rows in grouped.items():
        values = [row['dev_aupro'] for job,row in rows]
        ranking.append(dict(category=key[0],backbone=key[1],r=key[2],d=key[3],representation=key[4],
                            seeds=sorted(j['seed'] for j,r in rows),mean_dev_aupro=float(np.mean(values)),
                            std_dev_aupro=float(np.std(values)),
                            runs=[dict(job=j,**row) for j,row in rows]))
    ranking.sort(key=lambda r:(r['category'],-r['mean_dev_aupro'],r['backbone'],r['r'],r['d'],r['representation']))
    winners = {}
    for row in ranking:
        winners.setdefault(row['category'],row)
    save_json(dict(schema='msila.full_scale.ranking.v2',ranking=ranking),Path(root)/'ranking.json')
    with (Path(root)/'ranking.csv').open('w',newline='') as f:
        fields=['category','backbone','r','d','representation','seeds','mean_dev_aupro','std_dev_aupro']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in ranking)
    lock=dict(schema='msila.full_scale.selection.v2',status='COMPLETE',evaluation_split=['dev_tiny','dev_mixed'],
              rule='mean_across_declared_seeds(0.5*DEV_tiny_AU_PRO005+0.5*DEV_mixed_AU_PRO005)',
              tie_rule='lexicographic_backbone_r_d_representation',winners=winners,
              grid_lock_sha256=file_hash(Path(root)/'protocol_lock.json'),
              ranking_sha256=file_hash(Path(root)/'ranking.json'))
    lock['selection_sha256']=digest(lock)
    save_json(lock,Path(root)/'selection_lock.json')
    return lock


def run_full_scale(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from scripts.build_day05_cache import build
    from src.models.dinov3_extractor import build_online_extractor
    config=read_yaml(args.grid_config or 'configs/full_scale_grid.yaml')
    all_jobs=expand_grid(config)
    jobs=resolve_jobs(args,config)
    if args.data_root is None or args.dinov3_repo is None:
        raise ValueError('Full-scale requires --data-root and --dinov3-repo')
    synthetic=read_yaml(config['synthetic_protocol'])
    from src.data.synthetic_anomaly import validate_native_protocol
    validate_native_protocol(synthetic)
    protocol=read_yaml(config['training_protocol'])
    verify_protocol(protocol)
    if protocol['checkpoint'] != dict(monitor='dev_aupro',mode='max'):
        raise ValueError('Full-scale protocol must select checkpoints by DEV AU-PRO')
    protocol['evaluation']=dict(synthetic_protocol=synthetic,checkpoint_unit='cached_native_tiles',
                                ranking_unit='full_native_Hann_stitched_images',
                                sampling_backend='fixed_geometry_bilinear_gather_v1')
    categories=config['categories'] if args.train_all else [args.category]
    from scripts.build_day05_cache import native_source_plan
    native_source_plan(args.data_root,categories,synthetic,config['data_seed'])
    checkpoints={}
    for bb in dict.fromkeys(j['backbone'] for j in jobs):
        checkpoint=args.dino_checkpoint if not args.train_all and args.dino_checkpoint else Path(config['backbones'][bb]['checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError(f'BLOCKED_MISSING_WEIGHTS: {bb}: {checkpoint}')
        checkpoints[bb]=checkpoint
    if not (args.dinov3_repo/'hubconf.py').is_file():
        raise FileNotFoundError('BLOCKED_MISSING_SOURCE: DINOv3 hubconf.py')
    root=Path(args.output_root).absolute()
    evidence=dict(grid=config,jobs=jobs,synthetic_protocol=synthetic,training_protocol=protocol,
                  checkpoint_sha256={bb:file_hash(p) for bb,p in checkpoints.items()},
                  source_code_sha256={name:file_hash(Path(__file__).parents[2]/name) for name in
                  ('src/train/full_scale.py','src/train/screen_representation.py','src/data/synthetic_anomaly.py',
                   'src/models/dinov3_extractor.py','scripts/build_day05_cache.py','scripts/day05_full_inference.py',
                   'src/eval/full_scale.py','src/models/backbone_registry.py')})
    for name in ('src/models/residual_adapter.py','src/models/feature_projection.py',
                 'src/models/context_alignment.py','src/models/feature_selector.py',
                 'src/models/basic_decoder.py','src/models/bilinear_sampling.py',
                 'src/models/mean_fusion.py','src/losses/anomaly_loss.py',
                 'src/metrics/aupro.py','src/data/feature_cache.py','src/data/cached_dataset.py',
                 'src/data/loader.py','src/data/tiling.py','src/geometry/view_meta.py','src/train/day05_contract.py'):
        evidence['source_code_sha256'][name]=file_hash(Path(__file__).parents[2]/name)
    if args.dry_run or args.stage=='audit':
        report=dict(status='CONFIG_VALID',jobs=len(jobs),all_declared_jobs=len(all_jobs),
                    backbones=list(checkpoints),epochs=protocol['training']['epochs'],
                    batch_size=protocol['training']['batch_size'],training_executed=False,
                    vram_validated=False,config_sha256=digest(evidence))
        print(json.dumps(report,indent=2));return report
    if torch.device(args.device).type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('BLOCKED_MISSING_GPU: CUDA requested but unavailable')
    if (torch.device(args.device).type=='cuda' and protocol['training'].get('amp',{}).get('enabled')
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Configured bfloat16 is unsupported on this GPU; protocol preserved')
    enforce_lock(root,evidence)
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','torchvision','Pillow','PyYAML')},
        device=str(args.device),gpu_name=torch.cuda.get_device_name(args.device)
        if torch.device(args.device).type=='cuda' else None,recorded_at_unix=time.time())
    env_dir=root/'environments';env_dir.mkdir(exist_ok=True)
    save_json(environment,env_dir/f'{time.time_ns()}.json')
    if args.stage=='test-public':
        return evaluate_selected_test(args,root,checkpoints)
    resolved_dir=root/'configs';resolved_dir.mkdir(parents=True,exist_ok=True)
    protocol_path=resolved_dir/'training_protocol.yaml';save_yaml(protocol,protocol_path)
    state_path=root/'grid_manifest.json'
    state=dict(schema='msila.full_scale.grid_manifest.v2',status='RUNNING',declared_jobs=jobs,
               config_sha256=digest(evidence),completed={})
    if state_path.exists():
        old=json.loads(state_path.read_text())
        if old['config_sha256'] != state['config_sha256']:
            raise ValueError('Grid manifest config mismatch')
        state['completed']=old['completed']
    save_json(state,state_path)
    current_job=None
    try:
        for bb,checkpoint in checkpoints.items():
            current_job=None
            bb_jobs=[j for j in jobs if j['backbone']==bb]
            cache_root=root/'assets'/bb
            shared_data=root/'assets/dataset'
            blocks=config['backbones'][bb].get('blocks',backbone_spec(bb).blocks)
            built=build(SimpleNamespace(data_root=args.data_root,output_root=cache_root,
                dinov3_repo=args.dinov3_repo,dino_checkpoint=checkpoint,backbone=bb,blocks=blocks,
                categories=','.join(categories),seed=config['data_seed'],device=args.device,
                synthetic_protocol=Path(config['synthetic_protocol']),data_artifact_root=shared_data,
                cache_dir=None,export_dev_only=False,dry_run=False))
            # Training consumes only cache, so remove the frozen model before
            # each head fit. Native evaluation loads the exact backbone again.
            for job in bb_jobs:
                current_job=job
                family_root=root/'runs'/bb/f'adapter_r{job["r"]}_d{job["d"]}'
                representation=read_yaml('configs/day05_representation.yaml')
                representation['schema']='msila.full_scale.representation.v2'
                representation['experiment']['name']='full_scale_backbone_adapter_representation'
                representation['locked']['backbone']=dict(name=bb,frozen=True,feature_blocks=list(blocks))
                representation['locked']['adapter']['source']='predeclared_search_candidate'
                representation['locked']['training']['seed']=job['seed']
                representation['forbidden_changes']={k:True for k in
                    ('attention_fusion','illumination_loss','candidate_specific_decoder','candidate_specific_loss',
                     'candidate_specific_tile_or_context_size')}
                path=resolved_dir/f'{bb}_seed{job["seed"]}.yaml';save_yaml(representation,path)
                trainer_args=SimpleNamespace(day05_config=path,training_protocol=protocol_path,
                    r=job['r'],d=job['d'],seed=job['seed'],category=job['category'],candidate=job['representation'],
                    backbone=bb,dino_checkpoint=checkpoint,adapter_selection_report=None,
                    train_records=shared_data/'records/train_core.json',val_records=shared_data/'records/dev.json',
                    mask_root=shared_data/'masks',cache_dir=Path(built['cache']),output_root=family_root,
                    device=args.device,preflight_steps=args.preflight_steps,preflight_only=args.stage=='preflight',
                    resume=args.resume,allow_unverified_cache_provenance=False)
                run_dir=family_root/f'seed_{job["seed"]}'/job['category']/job['representation']
                if args.stage in ('train','all','preflight'):
                    train_candidate(trainer_args)
                if args.stage!='preflight':
                    gc.collect()
                    if torch.device(args.device).type=='cuda':torch.cuda.empty_cache()
                    extractor=build_online_extractor(repo_dir=args.dinov3_repo,weights=checkpoint,
                        model_name=bb,blocks=blocks,device=args.device)
                    result=evaluate_native_dev(run_dir,shared_data,extractor,args.device)
                    del extractor
                    manifest_path=run_dir/'run_manifest.json'
                    run_manifest=json.loads(manifest_path.read_text())
                    run_manifest['native_dev']=dict(dev_aupro=result['dev_aupro'],
                        metrics_path=str(run_dir/'native_dev/metrics.json'),
                        metrics_sha256=file_hash(run_dir/'native_dev/metrics.json'),
                        elapsed_sec=result['elapsed_sec'],peak_vram_mb=result['peak_vram_mb'])
                    run_manifest['total_training_and_native_eval_sec']=run_manifest['total_elapsed_sec']+result['elapsed_sec']
                    run_manifest['total_peak_vram_mb']=max(run_manifest['peak_vram_mb'],result['peak_vram_mb'],built['peak_vram_mb'])
                    save_json(run_manifest,manifest_path)
                    state['completed'][job_id(job)]=dict(run_dir=str(run_dir),dev_aupro=result['dev_aupro'],
                        checkpoint_sha256=result['provenance']['checkpoint_sha256'],
                        metrics_sha256=file_hash(run_dir/'native_dev/metrics.json'))
                    save_json(state,state_path)
            gc.collect()
            if torch.device(args.device).type=='cuda':torch.cuda.empty_cache()
        if args.stage=='preflight':
            state['status']='PREFLIGHT_PASS'
        else:
            rank_models(root,jobs,state['completed'])
            state['status']='COMPLETE'
        save_json(state,state_path)
        return state
    except BaseException as exc:
        state.update(status='FAILED_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else
                     'INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'FAILED',
                     current_job=current_job,reason=str(exc),configuration_preserved=True)
        save_json(state,state_path)
        if current_job:
            save_json(dict(status=state['status'],reason=str(exc),job=current_job,
                           configuration_preserved=True),root/'last_failure.json')
        raise


def evaluate_selected_test(args,root,checkpoints):
    """TEST_PUBLIC is reachable only through an intact, complete DEV lock."""
    from scripts.day05_full_inference import load_tv1_model, infer_image
    from src.models.dinov3_extractor import build_online_extractor
    from src.data.loader import scan_mvtec_ad2, load_rgb_native, load_mask_native
    from src.eval.full_scale import DEVMetricAccumulator
    lock=json.loads((root/'selection_lock.json').read_text())
    sha=lock.pop('selection_sha256')
    if (lock['status']!='COMPLETE' or lock['evaluation_split']!=['dev_tiny','dev_mixed'] or digest(lock)!=sha
            or file_hash(root/'ranking.json')!=lock['ranking_sha256']
            or file_hash(root/'protocol_lock.json')!=lock['grid_lock_sha256']):
        raise ValueError('Incomplete or modified DEV selection lock')
    reports=[]
    for cat,winner in lock['winners'].items():
        bb=winner['backbone']
        for selected in winner['runs']:
            run_dir=Path(selected['run_dir'])
            if (file_hash(run_dir/'best.pt')!=selected['checkpoint_sha256']
                    or file_hash(run_dir/'native_dev/metrics.json')!=selected['metrics_sha256']):
                raise ValueError('Selected model or DEV evidence changed')
            model,cfg,_=load_tv1_model(run_dir,winner['representation'],args.device)
            signature=cfg['cache']['provenance_check']['producer_signature']
            validate_signature(signature,checkpoints[bb],expected_backbone=bb)
            extractor=build_online_extractor(repo_dir=args.dinov3_repo,weights=checkpoints[bb],model_name=bb,
                                             blocks=cfg['backbone']['feature_blocks'],device=args.device)
            validate_extractor_source(extractor,signature)
            metric=DEVMetricAccumulator(cfg['evaluation']['synthetic_protocol'],disk_backed=True)
            records=scan_mvtec_ad2(args.data_root,split='test_public',categories=[cat],require_pixel_gt=True)
            if not records:raise ValueError(f'No TEST_PUBLIC samples for {cat}')
            output=run_dir/'test_public';output.mkdir(exist_ok=True)
            samples=[]
            for record in records:
                image=load_rgb_native(record.image_path)
                hw=tuple(image.shape[-2:])
                gt=np.zeros(hw,dtype=bool) if record.is_normal else load_mask_native(record.mask_path,hw).numpy()>0
                if not record.is_normal and not gt.any():raise ValueError('Abnormal TEST_PUBLIC GT is empty')
                score=infer_image(image,model,extractor,args.device)
                iid=Path(record.image_path).relative_to(args.data_root).as_posix()
                path=output/(digest(iid)+'.npy');np.save(path,score)
                metric.add(score,gt,split='test_public',image_id=iid)
                samples.append(dict(image_id=iid,image_sha256=file_hash(record.image_path),
                    gt_sha256=None if record.is_normal else file_hash(record.mask_path),
                    normal_zero_mask=record.is_normal,native_hw=list(hw),map_sha256=file_hash(path)))
            report=metric.result()
            report.update(category=cat,seed=cfg['seed'],selection_sha256=sha,
                          checkpoint_sha256=selected['checkpoint_sha256'],samples=samples,
                          purpose='final_evaluation_after_DEV_selection')
            save_json(report,output/'metrics.json');reports.append(report)
            del model,extractor
    save_json(dict(schema='msila.full_scale.test_public.v2',reports=reports),root/'test_public_results.json')
    return dict(status='TEST_PUBLIC_EVALUATED',runs=len(reports))
