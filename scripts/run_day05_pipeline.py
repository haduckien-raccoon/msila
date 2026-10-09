#!/usr/bin/env python3
"""Day-05 gates and full-scale backbone × adapter × representation × seed training."""
from __future__ import annotations
import argparse
import csv
import gc
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
import yaml

from src.train.day05_contract import SOURCES, file_hash, validate_day05, validate_selection, validate_signature, validate_record_sources


class Blocked(RuntimeError): pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['audit','preflight','train','inference','evaluate','all','test-public'], default=None)
    p.add_argument('--train-all', action='store_true', help='Run every declared full-scale Cartesian configuration')
    p.add_argument('--grid-config', type=Path)
    p.add_argument('--backbone', choices=['dinov3_vits16','dinov3_vits16plus','dinov3_vitb16','dinov3_vitl16','dinov3_vith16plus'])
    p.add_argument('--adapter-r', type=int)
    p.add_argument('--adapter-d', type=int)
    p.add_argument('--representation', choices=['R0','R1','R2'])
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--train-records',type=Path)
    p.add_argument('--val-records',type=Path)
    p.add_argument('--mask-root',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--dinov3-repo',type=Path)
    p.add_argument('--dino-checkpoint',type=Path)
    p.add_argument('--adapter-selection-report',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('outputs/day05'))
    p.add_argument('--day05-config',type=Path,default=Path('configs/day05_representation.yaml'))
    p.add_argument('--training-protocol',type=Path,default=Path('configs/day05_day04_full_v3_protocol.yaml'))
    p.add_argument('--category',default='fabric')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--preflight-steps',type=int,default=3)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--input-manifest',type=Path)
    p.add_argument('--split',choices=['dev_synthetic','test_public'],default='dev_synthetic')
    p.add_argument('--representation-lock',type=Path)
    p.add_argument('--tiny-protocol',type=Path)
    p.add_argument('--boundary-protocol',type=Path)
    p.add_argument('--seg-f1-threshold',type=float)
    p.add_argument('--efficiency-csv',type=Path,action='append')
    args=p.parse_args(argv)
    if args.stage is None:
        args.stage='all' if args.train_all or args.grid_config else 'audit'
    return args


def audit(a):
    missing_deps = [m for m in ('torch','torchvision','numpy','scipy','PIL','yaml','matplotlib') if importlib.util.find_spec(m) is None]
    if missing_deps: raise Blocked('BLOCKED_MISSING_DEPENDENCY: '+', '.join(missing_deps))
    from src.train.screen_representation import read_yaml, verify_protocol, verify_cache, filter_records, make_dataset
    from src.data.cached_dataset import load_training_records
    cfg=read_yaml(a.day05_config);protocol=read_yaml(a.training_protocol)
    validate_day05(cfg);verify_protocol(protocol)
    requested = {'backbone': (a.backbone, cfg['locked']['backbone']['name']),
                 'adapter-r': (a.adapter_r, 32), 'adapter-d': (a.adapter_d, 384),
                 'seed': (a.seed, cfg['locked']['training']['seed'])}
    for option, (value, locked) in requested.items():
        if value is not None and value != locked:
            raise ValueError(f'Day-05 locked --{option}={locked}; received {value}')
    assets = dict(cache_manifest=None if a.cache_dir is None else a.cache_dir/'manifest.json',
                  train_records=a.train_records,val_records=a.val_records,
                  dino_checkpoint=a.dino_checkpoint,adapter_selection_report=a.adapter_selection_report,
                  dino_source=None if a.dinov3_repo is None else a.dinov3_repo/'hubconf.py')
    if a.stage in ('inference','all') and a.split=='dev_synthetic': assets['dev_inputs']=a.input_manifest
    if a.stage in ('evaluate','all'):
        assets.update(tiny_protocol=a.tiny_protocol,boundary_protocol=a.boundary_protocol)
    missing = [f'{k}={p}' for k,p in assets.items() if p is None or not p.is_file()]
    if missing:
        status='BLOCKED_MISSING_PROTOCOL' if all(x.startswith(('tiny_protocol=','boundary_protocol=')) for x in missing) else 'BLOCKED_MISSING_DATA'
        raise Blocked(status+': '+ '; '.join(missing))
    validate_selection(a.adapter_selection_report)
    train=filter_records(load_training_records(a.train_records),a.category,'train')
    dev=filter_records(load_training_records(a.val_records),a.category,'val')
    validate_record_sources(train,dev)
    reader,provenance=verify_cache(a.cache_dir,train,dev,'dinov3_vits16')
    validate_signature(provenance['producer_signature'],a.dino_checkpoint)
    expected=protocol['data'].get('expected_producer_signature')
    if expected is not None and expected != provenance['producer_signature']:
        raise ValueError('Cache differs from the locked training protocol producer')
    for records in (train,dev):
        ds=make_dataset(a.cache_dir,records,protocol,a.mask_root)
        for i in range(len(ds)):
            item=ds[i]
            for key in SOURCES['R2']:
                if tuple(item[key].shape)!=(384,32,32): raise ValueError(f'{item["image_id"]}: invalid ViT-S/16 feature shape')
            if tuple(item['mask'].shape)!=(1,512,512): raise ValueError('Local mask geometry mismatch')
            if bool(item['mask'].any()) != bool(records[i]['is_anomaly']): raise ValueError('Synthetic anomaly label/mask mismatch')
    if a.stage in ('inference','evaluate','all'):
        if a.seg_f1_threshold is None: raise Blocked('BLOCKED_MISSING_PROTOCOL: supply the locked --seg-f1-threshold')
        if not 0 <= a.seg_f1_threshold <= 1: raise ValueError('Invalid score threshold')
    if a.stage in ('evaluate','all'):
        from src.eval.region_stats import validate_tiny_protocol
        from src.eval.boundary_analysis import validate_inputs
        tiny=json.loads(a.tiny_protocol.read_text());boundary=json.loads(a.boundary_protocol.read_text())
        if (tiny.get('tiny_area_px') is None or tiny.get('locked_before_candidate_results') is not True
                or boundary.get('tolerance_px') is None or boundary.get('locked_before_candidate_results') is not True
                or (boundary.get('boundary_mode')=='image_border_band' and boundary.get('band_width_px') is None)):
            raise Blocked('BLOCKED_MISSING_PROTOCOL: tiny/boundary thresholds have not been locked')
        validate_tiny_protocol(tiny)
        validate_inputs(dict(seg_f1_threshold=a.seg_f1_threshold, categories=[a.category],
            samples=[{'image_id':'asset_validation','category':a.category,'maps':dict.fromkeys(SOURCES)}],
            split=a.split,normalization_by_candidate={c:'same' for c in SOURCES}),boundary)
    if a.stage in ('inference','all'):
        from scripts.day05_full_inference import load_inputs
        load_inputs(inference_args(a),dict(category=a.category,data_fingerprints=dict(val_source_sha256=file_hash(a.val_records))),validate_only=True)
    if a.stage in ('inference','evaluate'):
        from scripts.build_evaluation_manifest import read_artifact
        for c,run in zip(SOURCES,run_dirs(a)): read_artifact(run,c)
    import torch
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise Blocked('BLOCKED_MISSING_GPU: requested CUDA runtime is unavailable')
    return dict(status='ASSETS_VALID',backbone='dinov3_vits16',train_samples=len(train),dev_samples=len(dev),
                protocol=str(a.training_protocol),real_forward_executed=False,full_training_complete=False)


def train_args(a,c,preflight=False):
    return SimpleNamespace(**(vars(a) | dict(candidate=c,r=32,d=384,
        allow_unverified_cache_provenance=False,preflight_only=preflight)))


def run_dirs(a): return [a.output_root/f'seed_{a.seed}'/a.category/c for c in SOURCES]


def inference_args(a):
    return SimpleNamespace(**(vars(a) | dict(zip(('r0','r1','r2'),run_dirs(a))) | dict(output_dir=a.output_root/'maps'/a.split)))


def csv_write(path, rows, fields=None):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def stage_once(root, inputs, action):
    """Resume verified finished stages; preserve interrupted attempts in quarantine."""
    root=Path(root); gate=root/'stage_gate.json'
    if gate.is_file():
        saved=json.loads(gate.read_text())
        if saved['inputs'] != inputs: raise ValueError(f'Stage inputs changed: {root}')
        for path, sha in saved['artifacts'].items():
            if file_hash(root/path)!=sha: raise ValueError(f'Stage artifact changed: {root/path}')
        return root
    if root.exists(): root.rename(root.with_name(root.name+'.incomplete.'+uuid.uuid4().hex[:8]))
    root.mkdir(parents=True)
    action(root)
    artifacts={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()}
    gate.write_text(json.dumps(dict(status='COMPLETE',inputs=inputs,artifacts=artifacts),indent=2)+'\n')
    return root


def measure_efficiency(a, output):
    import torch
    from src.eval.efficiency import parameter_report, make_benchmark_scope, benchmark_inference_efficiency
    from src.train.screen_representation import read_yaml, make_dataset, make_loader, move_to_device, filter_records
    from src.data.cached_dataset import load_training_records
    from scripts.day05_full_inference import load_tv1_model
    if not torch.cuda.is_available() or not a.device.startswith('cuda'):
        raise Blocked('BLOCKED_MISSING_GPU: E5/E6 require real CUDA measurements')
    device=torch.device(a.device);torch.cuda.set_device(device)
    props=torch.cuda.get_device_properties(device)
    protocol=read_yaml(a.training_protocol)
    protocol['training']['batch_size']=1  # Existing E8 measurement scope, independent of training batch64.
    records=filter_records(load_training_records(a.val_records),a.category,'val')
    ds=make_dataset(a.cache_dir,records,protocol,a.mask_root)
    batch=next(iter(make_loader(ds,protocol,False,a.seed)))
    benchmark=dict(latency_warmup=10,latency_iterations=50,latency_rounds=3,
        stability_cv_threshold=.10,vram_warmup=10,vram_iterations=1,use_inference_mode=True)
    hardware=dict(gpu=props.name,gpu_uuid=str(getattr(props,'uuid','unavailable')),device=str(device),
        total_memory_bytes=props.total_memory,torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version())
    scope=make_benchmark_scope(scope_name='cached_trainable_pipeline',device=device,batch_size=1,precision='float32',
        input_signature='six frozen ViT-S/16 BCHW features; geometry; Local mask 512',
        pipeline_stages=['adapter','alignment','projection','mean_fusion','decoder'],
        extra=dict(hardware=hardware,output_hw=[512,512],benchmark=benchmark,sample_ids=[batch['meta'][0]['image_id']],cache_manifest_sha256=file_hash(a.cache_dir/'manifest.json'),
            val_records_sha256=file_hash(a.val_records)))
    (output/'efficiency_protocol.json').write_text(json.dumps(scope,indent=2)+'\n')
    rows=[]
    # No live frozen backbone is counted toward this cached-head scope.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for c,run in zip(SOURCES,run_dirs(a)):
        gc.collect();torch.cuda.empty_cache()
        model,cfg,row=load_tv1_model(run,c,device)
        params=parameter_report(model,candidate_id=c,scope='cached_trainable_pipeline')
        gpu_batch=move_to_device(batch,device)
        def once():
            with torch.autocast(device_type='cuda',enabled=False): return model(gpu_batch)[0]
        measured=benchmark_inference_efficiency(once,candidate_id=c,scope=scope,**benchmark)
        if measured['status']!='PASS': raise ValueError(f'{c}: unstable latency/invalid VRAM gate')
        (output/f'{c}_efficiency.json').write_text(json.dumps(dict(parameters=params,efficiency=measured,
            checkpoint_sha256=row['checkpoint_sha256'],resolved_config_sha256=row['resolved_config_sha256']),indent=2)+'\n')
        rows.append(dict(candidate=c,category=a.category,seed=a.seed,trainable_params=params['totals']['trainable_parameters'],
            inference_ms_per_batch=measured['latency']['latency_ms']['median'],peak_vram_MiB=measured['peak_vram']['memory']['peak_allocated']['MiB'],
            gpu=props.name,tile_resolution='512x512',batch_size=1,precision='float32',scope_sha256=scope['fingerprint_sha256'],
            status='PASS',**{k:v for k,v in benchmark.items() if k not in ('stability_cv_threshold','use_inference_mode')}))
        del model,gpu_batch,once
    csv_write(output/'efficiency.csv',rows)
    torch.cuda.empty_cache()


def evaluate(a):
    from scripts.eval_day05_representation import evaluate_manifest, validate_map_provenance
    from src.eval import tiny_analysis, boundary_analysis, region_stats, compare_representation
    from scripts.aggregate_week06_multiscale import aggregate,write_outputs
    from scripts.create_representation_lock import build_lock,write_lock
    manifest=a.output_root/'maps'/a.split/'evaluation_manifest.json'
    obj=json.loads(manifest.read_text());validate_map_provenance(obj,manifest.parent)
    ids=dict(manifest=file_hash(manifest),tiny=file_hash(a.tiny_protocol),boundary=file_hash(a.boundary_protocol),
             runner=file_hash(__file__))
    base=a.output_root/'evaluation'/a.split
    primary=stage_once(base/'primary',ids,lambda d:(d/'metrics.json').write_text(
        json.dumps(evaluate_manifest(manifest),indent=2,allow_nan=False)+'\n'))
    tiny=stage_once(base/'tiny',ids,lambda d:csv_write(d/'tiny.csv',tiny_analysis.analyze(manifest,a.tiny_protocol)))
    def boundary_stage(d):
        rows=boundary_analysis.analyze(manifest,a.boundary_protocol,d/'visualizations')
        csv_write(d/'boundary.csv',rows)
    boundary=stage_once(base/'boundary',ids,boundary_stage)
    stage_once(base/'regions',ids,lambda d:csv_write(d/'per_region_stats.csv',
        region_stats.analyze(manifest,a.tiny_protocol,a.boundary_protocol),region_stats.FIELDS))
    stage_once(base/'comparison',ids,lambda d:compare_representation.compare(
        manifest,a.tiny_protocol,a.boundary_protocol,d/'visualizations'))
    efficiency=a.efficiency_csv
    if not efficiency:
        measured=stage_once(base/'efficiency',ids,lambda d:measure_efficiency(a,d))
        efficiency=[measured/'efficiency.csv']
    def aggregate_stage(d):
        result=aggregate(primary/'metrics.json',tiny/'tiny.csv',boundary/'boundary.csv',efficiency)
        csv=d/'week06_multiscale.csv';write_outputs(result,csv)
        if a.split=='dev_synthetic': write_lock(build_lock(result),d/'representation_lock.yaml',csv)
    # Adjacent E8 reports must belong to these exact TV1 checkpoints.
    from scripts.build_evaluation_manifest import read_artifact
    for path in efficiency:
        with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        for row in rows:
            c=row['candidate']
            if row['category']!=a.category or int(row['seed'])!=a.seed:
                raise ValueError('Efficiency category/seed does not match TV1')
            run=run_dirs(a)[list(SOURCES).index(c)]
            _,expected=read_artifact(run,c)
            saved=json.loads((path.parent/f'{c}_efficiency.json').read_text())
            for k in ('checkpoint_sha256','resolved_config_sha256'):
                if saved.get(k)!=expected[k]: raise ValueError(f'Efficiency {c}: TV1 {k} mismatch')
    summary=stage_once(base/'summary',ids|{str(p):file_hash(p) for p in efficiency},aggregate_stage)
    return summary


def main(argv=None):
    a=parse_args(argv)
    try:
        if a.train_all or a.grid_config:
            from src.train.full_scale import run_full_scale
            report=run_full_scale(a)
            print(json.dumps({k:v for k,v in report.items() if k not in ('declared_jobs','completed')},indent=2))
            return 0
        if a.stage=='test-public':
            raise ValueError('--stage test-public requires --grid-config/--train-all and an intact DEV selection lock')
        report=audit(a)
        if a.stage=='audit' or a.dry_run:
            print(json.dumps(report,indent=2));return 0
        from src.train.screen_representation import train_candidate
        if a.stage in ('preflight','all'):
            for c in SOURCES: train_candidate(train_args(a,c,True))
        if a.stage in ('train','all'):
            for c in SOURCES: train_candidate(train_args(a,c))
        if a.stage in ('inference','all'):
            from scripts.day05_full_inference import run_inference
            run_inference(inference_args(a))
        if a.stage in ('evaluate','all'): evaluate(a)
        print(json.dumps(dict(status='STAGE_COMPLETE',stage=a.stage,output_root=str(a.output_root)),indent=2));return 0
    except (Blocked,ValueError,RuntimeError,FileNotFoundError,KeyError,TypeError) as exc:
        report=dict(status=str(exc).split(':')[0] if str(exc).startswith('BLOCKED_') else 'FAIL',reason=str(exc),stage=a.stage)
        a.output_root.mkdir(parents=True,exist_ok=True)
        (a.output_root/'last_gate_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),file=sys.stderr);return 2


if __name__=='__main__': raise SystemExit(main())
