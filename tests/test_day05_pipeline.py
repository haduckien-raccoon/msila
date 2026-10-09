"""Software regression fixtures only: these are never REAL-DATA/FULL-TRAIN evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from src.data.feature_cache import FEATURE_KEYS, FeatureCacheWriter
from src.geometry.view_meta import build_view_meta
from src.train.day05_contract import SOURCES, digest, file_hash, validate_signature, validate_record_sources
from src.train.screen_representation import (Day05RepresentationModel, train_candidate, run_preflight,
    make_criterion, build_optimizer, load_checkpoint, save_checkpoint, seed_everything, verify_cache)
from scripts.build_evaluation_manifest import verify_and_build_manifest
from scripts.day05_full_inference import (load_tv1_model, make_tile_batch, infer_image, run_inference)
from scripts.eval_day05_representation import evaluate_manifest
from scripts.run_day05_pipeline import main, stage_once
from scripts.build_day05_cache import source_plan, prepare_sample, build
from scripts.day05_legacy_handoff import LEGACY_LOCK_KEYS

ROOT=Path(__file__).parents[1]


@pytest.fixture
def legacy_artifacts(artifacts, tmp_path):
    """Reproduce original TV1-C/TV1-D serialization; only synthetic QA data."""
    import csv
    import shutil
    root = tmp_path / 'legacy_runs'
    rule = dict(schema='msila.day05.checkpoint_rule_lock.v1', status='LOCKED_BEFORE_TV1C_RESULTS',
        rule_source='fixture_protocol', rule_source_sha256='a'*64, selector_file='fixture_selector',
        selector_file_sha256='b'*64, selector_source='src.train.screen_adapter.is_better',
        dev_split='dev_synthetic', day04_monitor='val_loss', day05_log_monitor='val_total_loss',
        mode='min', comparison='strict_improvement', tie_policy='keep_earliest_epoch',
        candidate_scope=list(SOURCES), category='fabric', seed=42)
    rule['rule_lock_sha256'] = digest(rule)
    rule['locked_at_unix'] = 1.0
    runs = []
    for c, original in zip(SOURCES, artifacts.runs):
        run = root/'seed_42/fabric'/c
        shutil.copytree(original, run)
        cfg = yaml.safe_load((run/'resolved_config.yaml').read_text())
        for key in ('day05_config','day05_config_sha256','data_loader'):
            del cfg[key]
        cfg['training'] = {key: cfg['training'][key] for key in
            ('epochs','batch_size','optimizer','scheduler','gradient_clip_norm','updates_per_epoch','total_update_budget')}
        cfg['data_fingerprints'] = {key: value for key,value in cfg['data_fingerprints'].items() if 'masks' not in key}
        cfg['checkpoint_rule'] = dict(monitor='val_total_loss',mode='min')
        payload = {key:cfg[key] for key in LEGACY_LOCK_KEYS}
        cfg['protocol_lock_sha256'] = digest(payload)
        (root/'protocol_lock.json').write_text(json.dumps(dict(sha256=digest(payload),payload=payload)))
        (run/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
        for name in ('best.pt','last.pt'):
            state = torch.load(run/name, map_location='cpu', weights_only=False)
            state['resolved_config_sha256'] = digest(cfg)
            torch.save(state,run/name)
        manifest = json.loads((run/'run_manifest.json').read_text())
        del manifest['artifact_sha256']
        manifest.update(resolved_config_sha256=digest(cfg), protocol_lock_sha256=cfg['protocol_lock_sha256'], started_at_unix=2.0)
        (run/'run_manifest.json').write_text(json.dumps(manifest))
        with (run/'epoch_log.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        selected = min(rows,key=lambda row:float(row['val_total_loss']))
        selection = dict(schema='msila.day05.checkpoint_selection_record.v1',status='PASS',
            candidate=c,category='fabric',seed=42,dev_split='dev_synthetic',
            rule={key:rule[key] for key in ('rule_lock_sha256','day04_monitor','day05_log_monitor','mode',
                                          'comparison','tie_policy','rule_source_sha256','selector_file_sha256')},
            selection=dict(selected_epoch=int(selected['epoch']),selected_metric=float(selected['val_total_loss']),
                metric_name='val_total_loss',checkpoint_sha256=file_hash(run/'best.pt')),
            evidence=dict(epoch_log_sha256=file_hash(run/'epoch_log.csv'),
                resolved_config_sha256_file=file_hash(run/'resolved_config.yaml'),
                checkpoint_resolved_config_sha256=digest(cfg),protocol_lock_sha256=cfg['protocol_lock_sha256'],
                git_commit=cfg['git_commit']))
        (run/'selection_record.json').write_text(json.dumps(selection))
        runs.append(run)
    (root/'checkpoint_rule_lock.json').write_text(json.dumps(rule))
    return SimpleNamespace(root=root,runs=runs)


def test_legacy_tv1d_handoff_preserves_original_config_and_model(artifacts, legacy_artifacts):
    from scripts.build_evaluation_manifest import read_artifact
    before = {str(p):file_hash(p) for p in legacy_artifacts.root.rglob('*') if p.is_file()}
    for c,original,legacy in zip(SOURCES,artifacts.runs,legacy_artifacts.runs):
        model,cfg,row = load_tv1_model(legacy,c)
        reference,_,_ = load_tv1_model(original,c)
        assert 'day05_config' not in cfg
        assert row['artifact_format']=='legacy_tv1c_with_tv1d_selection'
        assert row['resolved_config_sha256']==digest(cfg)
        batch={key:torch.randn(1,384,4,4) for key in FEATURE_KEYS}
        batch.update(meta=[{'geometry':artifacts.geometry}],output_hw=(17,19))
        with torch.no_grad():
            assert torch.equal(model(batch)[0],reference(batch)[0])
        assert read_artifact(legacy,c)[0]==cfg
    assert before=={str(p):file_hash(p) for p in legacy_artifacts.root.rglob('*') if p.is_file()}
    handoff=verify_and_build_manifest(*legacy_artifacts.runs,legacy_artifacts.root/'receipt.json')
    assert all('artifact_sha256_at_receipt' in row for row in handoff['runs'])


@pytest.mark.parametrize('changed',['epoch_log','training_log','last_checkpoint','rule_lock','selection','config','missing_selection'])
def test_legacy_handoff_rejects_incomplete_or_modified_evidence(legacy_artifacts,changed):
    from scripts.build_evaluation_manifest import read_artifact
    run=legacy_artifacts.runs[0]
    if changed in ('epoch_log','training_log'):
        path=run/(changed+'.csv')
        path.write_text(path.read_text().splitlines()[0]+'\n')
    elif changed=='last_checkpoint':
        path=run/'last.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        state['epoch']-=1;torch.save(state,path)
    elif changed=='rule_lock':
        path=legacy_artifacts.root/'checkpoint_rule_lock.json';obj=json.loads(path.read_text())
        obj['locked_at_unix']=3.0;path.write_text(json.dumps(obj))
    elif changed=='selection':
        path=run/'selection_record.json';obj=json.loads(path.read_text())
        obj['selection']['checkpoint_sha256']='0'*64;path.write_text(json.dumps(obj))
    elif changed=='config':
        path=run/'resolved_config.yaml';obj=yaml.safe_load(path.read_text())
        obj['training']['batch_size']+=1;path.write_text(yaml.safe_dump(obj))
    else: (run/'selection_record.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        read_artifact(run,'R0')


def test_tv2_notebook_receives_legacy_runs_and_never_retrains(artifacts,legacy_artifacts,tmp_path,monkeypatch):
    """Execute notebook orchestration on fixtures; DINO/CUDA metrics are not simulated PASS."""
    import ast
    import gc
    import matplotlib
    import subprocess
    import sys
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notebook_path=ROOT/'notebooks/Day05_TV2_Receive_Inference_Evaluation_ViTS16_Colab.ipynb'
    if not notebook_path.is_file():
        pytest.skip('TV2 notebook delivered separately from its code ZIP; install the notebook to run orchestration QA.')
    notebook=json.loads(notebook_path.read_text())
    code_cells=[cell for cell in notebook['cells'] if cell['cell_type']=='code']
    for cell in code_cells:
        tree=ast.parse(''.join(cell['source']))
        for call in (node for node in ast.walk(tree) if isinstance(node,ast.Call)):
            if isinstance(call.func,ast.Name) and call.func.id=='run':
                assert call.args[0].value in ('scripts.build_day05_cache','scripts.day05_full_inference')
    native=tmp_path/'native_dev';native.mkdir()
    samples=[]
    for record in artifacts.records['dev_synthetic']:
        gt=np.zeros((40,48),np.uint8)
        if record['is_anomaly']:gt[:2,:2]=255;gt[20:23,20:23]=255
        path=native/(record['image_id']+'.png');Image.fromarray(gt).save(path)
        samples.append(dict(image_id=record['image_id'],category='fabric',gt_mask=str(path),
                            original_hw=[40,48],gt_mask_sha256=file_hash(path)))
    inputs=native/'dev_inputs.json'
    inputs.write_text(json.dumps(dict(schema='msila.day05.inference_inputs.v1',split='dev_synthetic',samples=samples)))
    env=dict(__name__='__main__',Path=Path,json=json,gc=gc,yaml=yaml,torch=torch,sys=sys,subprocess=subprocess,
             PROJECT_DRIVE=tmp_path,PROJECT_ROOT=ROOT,DEVICE='cpu')
    commands=[]
    def run_mock(module,*args):commands.append((module,list(map(str,args))))
    for cell in code_cells:
        stage=cell['metadata']['day05_stage']
        if stage=='setup':continue
        if stage=='protocol_gate':gate=cell;break
        exec(compile(''.join(cell['source']),cell['id'],'exec'),env)
        if stage=='config':
            env.update(run=run_mock,TV1_ROOT=legacy_artifacts.root,OUT=legacy_artifacts.root,
                RUN_DIRS=legacy_artifacts.runs,TV2_RECEIPT_DIR=legacy_artifacts.root/'tv2_handoff',
                CACHE=artifacts.root/'feature_cache',TRAIN=artifacts.root/'train_core.json',
                DEV=artifacts.root/'dev_synthetic.json',DINO_CKPT=artifacts.checkpoint,
                MASKS=None,INPUTS=inputs,DEV_EXPORT=native,MAP_DIR=legacy_artifacts.root/'maps/dev_synthetic')
    assert len(commands)==1 and '--export-dev-only' in commands[0][1]
    with pytest.raises(RuntimeError,match='Chưa có protocol lock'):
        exec(''.join(gate['source']),env)
    lock=next(cell for cell in code_cells if cell['metadata']['day05_stage']=='gt_lock')
    source=''.join(lock['source']).replace('EXPORT_LOCKED_PROTOCOLS = False','EXPORT_LOCKED_PROTOCOLS = True')
    source=source.replace("TINY_LOCK_BASIS = ''","TINY_LOCK_BASIS = 'SYNTHETIC QA ONLY'")
    source=source.replace("BOUNDARY_LOCK_BASIS = ''","BOUNDARY_LOCK_BASIS = 'SYNTHETIC QA ONLY'")
    exec(source,env);exec(''.join(gate['source']),env)
    evaluation_calls=[]
    def evaluate_spy(args):
        evaluation_calls.append(args)
        # No fake metric/PASS: an empty QA-only display fixture in pytest tmp_path.
        path=args.output_root/'evaluation/dev_synthetic/primary';path.mkdir(parents=True)
        (path/'metrics.json').write_text(json.dumps(dict(schema='QA_DISPLAY_ONLY',candidate_category=[])))
        return path.parent/'summary'
    monkeypatch.setattr('scripts.run_day05_pipeline.evaluate',evaluate_spy)
    for cell in code_cells:
        if cell['metadata']['day05_stage'] in ('inference','evaluate','results'):
            exec(compile(''.join(cell['source']),cell['id'],'exec'),env)
    assert [module for module,args in commands]==['scripts.build_day05_cache','scripts.day05_full_inference']
    assert commands[-1][1][commands[-1][1].index('--r0')+1]==str(legacy_artifacts.runs[0])
    assert evaluation_calls[0].output_root==legacy_artifacts.root
    assert evaluation_calls[0].tiny_protocol==env['TINY']
    assert env['first_cfg']['training']['batch_size']==artifacts.protocol['training']['batch_size']
    assert 'msila.day05.tv2.efficiency_input.v1' in env['MEASUREMENT_INPUT_PATH'].read_text()
    plt.close('all')


def signature(checkpoint):
    return dict(schema='unit_fixture_only',categories=['fabric'],backbone='dinov3_vits16',checkpoint_sha256=file_hash(checkpoint),
        logical_layers_1based=[4,8,12],local_source_size=[512,512],context_source_size=[768,768],
        model_input_size=[512,512],normalization='ImageNet mean/std',smoke_only=False,full_source_coverage=True,
        source_split_version='day04_full_hash80_20_v3',train_fraction=.8,
        train_anomaly_types=['intensity','color','noise'],dev_anomaly_types=['cutpaste'])


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    torch.set_num_threads(2)
    root=tmp_path_factory.mktemp('day05_fixture')
    checkpoint=root/'fixture_dino.pth';checkpoint.write_bytes(b'fixture ONLY: no real DINO weights')
    cfg=yaml.safe_load((ROOT/'configs/day05_representation.yaml').read_text())
    protocol=yaml.safe_load((ROOT/'configs/day05_day04_full_v3_protocol.yaml').read_text())
    # A separate tiny update budget in an isolated software fixture, never edited scientific files.
    protocol['training'].update(epochs=2,batch_size=2,amp={'enabled':False,'dtype':'bfloat16'})
    protocol['data'].update(num_workers=0,persistent_workers=False,pin_memory=False)
    pp=root/'fixture_protocol.yaml';pp.write_text(yaml.safe_dump(protocol))
    report=root/'fixture_selection.json';report.write_text(json.dumps(dict(analysis_status='PASS',
        selection=dict(status='SELECTED',selected_candidate='adapter_r32_d384'))))
    g=build_view_meta(source_hw=(768,768),local_box_xyxy=(128,128,640,640),context_box_xyxy=(0,0,768,768))
    geometry=dict(local_box=[128,128,640,640],context_box=[0,0,768,768],local_to_context=g.local_to_context.tolist(),
                  context_to_local=g.context_to_local.tolist(),local_input_hw=[512,512],context_input_hw=[512,512])
    records={'train_core':[],'dev_synthetic':[]}
    torch.manual_seed(7)
    with FeatureCacheWriter(root/'feature_cache',producer_signature=signature(checkpoint),max_samples_per_shard=2) as w:
        for split in records:
            for i in range(2):
                iid=f'{split}_{i}';mask=np.zeros((512,512),np.uint8)
                if i: mask[200:232,211:259]=255
                mp=root/f'{iid}.png';Image.fromarray(mask).save(mp)
                records[split].append(dict(image_id=iid,category='fabric',mask_path=str(mp),mask_hw=[512,512],
                    is_anomaly=bool(i),meta=dict(split=split,source_identity=f'fixture/{iid}',
                    synthetic={'fixture':True},smoke_only=False,full_source_coverage=True)))
                w.add(dict(image_id=iid,category='fabric',geometry=geometry,
                           **{k:torch.randn(1,384,32,32) for k in FEATURE_KEYS}))
    for split,rows in records.items(): (root/f'{split}.json').write_text(json.dumps(rows))
    common=dict(category='fabric',r=32,d=384,seed=42,cache_dir=root/'feature_cache',
        train_records=root/'train_core.json',val_records=root/'dev_synthetic.json',mask_root=None,
        day05_config=ROOT/'configs/day05_representation.yaml',training_protocol=pp,
        adapter_selection_report=report,output_root=root/'runs',device='cpu',preflight_steps=2,
        allow_unverified_cache_provenance=False,resume=True,preflight_only=False)
    runs=[train_candidate(SimpleNamespace(**common,candidate=c)) for c in SOURCES]
    return SimpleNamespace(root=root,checkpoint=checkpoint,cfg=cfg,protocol=protocol,common=common,
                           records=records,runs=runs,geometry=geometry)


@pytest.mark.parametrize('change', [None,'missing','wrong','filename_only','geometry','checkpoint'])
def test_backbone_metadata_has_no_filename_or_substring_bypass(tmp_path,change):
    p=tmp_path/'dinov3_vits16.pth';p.write_bytes(b'fixture')
    s=signature(p)
    if change=='missing':del s['backbone']
    if change=='wrong':s['backbone']='dinov3_vitb16'
    if change=='filename_only':s['backbone']='unknown';s['checkpoint_path']=str(p)
    if change=='geometry':s['context_source_size']=[512,512]
    if change=='checkpoint':s['checkpoint_sha256']='0'*64
    if change is None:validate_signature(s,p)
    else:
        with pytest.raises(ValueError):validate_signature(s,p)


def test_cache_contract_and_forbidden_unverified_flag(artifacts):
    a=artifacts
    reader,prov=verify_cache(a.root/'feature_cache',a.records['train_core'],a.records['dev_synthetic'],'dinov3_vits16')
    assert prov['verified']
    assert set(FEATURE_KEYS)<=set(reader.get(image_id='train_core_0',category='fabric'))
    with pytest.raises(RuntimeError,match='forbidden'):
        verify_cache(a.root/'feature_cache',[],[],'dinov3_vits16',allow_unverified=True)


@pytest.mark.parametrize('c',SOURCES)
def test_real_tv1_class_checkpoint_strict_load_and_finite_forward(artifacts,c):
    run=artifacts.runs[list(SOURCES).index(c)]
    model,cfg,row=load_tv1_model(run,c)
    assert cfg['seed']==42 and cfg['projection']['fusion_dim']==64
    batch={k:torch.randn(1,384,4,4) for k in FEATURE_KEYS}
    batch.update(meta=[{'geometry':artifacts.geometry}],output_hw=(17,19))
    logits,trace=model(batch)
    assert logits.shape==(1,1,17,19) and torch.isfinite(logits).all()
    assert list(trace['projected'])==SOURCES[c]
    assert not list(model.aligner.parameters()) and not list(model.fusion.parameters())


def test_handoff_three_candidates_same_seed_and_shared_protocol(artifacts):
    p=artifacts.root/'handoff.json'
    m=verify_and_build_manifest(*artifacts.runs,p)
    assert m['schema']=='msila.day05.handoff.v1'
    assert [r['candidate'] for r in m['runs']]==list(SOURCES)
    assert {r['seed'] for r in m['runs']}=={42}
    assert len({r['protocol_lock_sha256'] for r in m['runs']})==1
    assert verify_and_build_manifest(*artifacts.runs,p)==m
    with pytest.raises(ValueError,match='candidate'):
        verify_and_build_manifest(artifacts.runs[1],artifacts.runs[0],artifacts.runs[2],p)


def test_complete_run_is_preserved_and_corruption_rejected(artifacts):
    run=artifacts.runs[0];before={p.name:file_hash(p) for p in run.iterdir() if p.is_file()}
    train_candidate(SimpleNamespace(**artifacts.common,candidate='R0'))
    assert before=={p.name:file_hash(p) for p in run.iterdir() if p.is_file()}
    p=run/'last.pt';content=p.read_bytes();p.write_bytes(content+b'corruption')
    try:
        with pytest.raises(RuntimeError,match='COMPLETE artifact'):train_candidate(SimpleNamespace(**artifacts.common,candidate='R0'))
    finally:p.write_bytes(content)


@pytest.mark.parametrize('c',SOURCES)
def test_preflight_updates_weights_and_every_selected_source(artifacts,c):
    seed_everything(42)
    model=Day05RepresentationModel(day05_config=artifacts.cfg,candidate=c,in_channels=8,adapter_r=32,adapter_d=384)
    batch={k:torch.randn(1,8,4,4) for k in FEATURE_KEYS}
    mask=torch.zeros(1,1,16,16);mask[:,:,4:10,5:12]=1
    batch.update(mask=mask,meta=[{'geometry':artifacts.geometry}])
    opt=build_optimizer(model,artifacts.protocol)
    report=run_preflight(model,[batch],make_criterion(artifacts.cfg),opt,torch.device('cpu'),2)
    assert report['status']=='PASS'
    assert set(report['source_usage'][-1])==set(SOURCES[c])
    for prefix in ('adapters.','projection.','decoder.'):
        assert any(n.startswith(prefix) for n in report['changed_parameters'])
    if c=='R2':
        model.eval()
        with torch.no_grad():
            first=model(batch)[0]
            different=dict(batch,**{k:batch[k]+5 for k in FEATURE_KEYS if k.startswith('context')})
            assert not torch.allclose(first,model(different)[0])


class FixtureExtractor:
    """Records true independent view inputs; software fixture, no backbone claim."""
    def __init__(self):self.pairs=[]
    def __call__(self,x):
        f=torch.nn.functional.adaptive_avg_pool2d(x.mean(1,keepdim=True),(4,4)).repeat(1,384,1,1)
        return {f'b{b}':f+b/100 for b in (4,8,12)}
    def extract_online_cache_features(self,local,context,**kw):
        self.pairs.append((local.clone(),context.clone()))
        result={f'local_{k}':v for k,v in self(local).items()}
        result.update({f'context_{k}':v for k,v in self(context).items()});return result


def test_r2_tile_has_independent_context_and_true_alignment():
    image=torch.rand(3,790,870)
    from src.data.tiling import generate_tile_records
    rec=generate_tile_records(790,870)[0];ex=FixtureExtractor()
    batch=make_tile_batch(image,rec,ex,'cpu')
    assert not torch.equal(ex.pairs[0][0],ex.pairs[0][1])
    assert not torch.equal(batch['local_b4'],batch['context_b4'])
    m=torch.tensor(batch['meta'][0]['geometry']['local_to_context'])
    assert torch.allclose(m,torch.tensor([[2/3,0,128*2/3],[0,2/3,128*2/3],[0,0,1.]]))


@pytest.mark.parametrize('hw',[(211,217),(790,870)])
def test_tiled_inference_preserves_native_shape_finite_float32(artifacts,hw):
    model,_,_=load_tv1_model(artifacts.runs[2],'R2')
    score=infer_image(torch.rand(3,*hw),model,FixtureExtractor(),'cpu')
    assert score.shape==hw and score.dtype==np.float32 and np.isfinite(score).all()


def test_stitching_preserves_pixel_coordinates():
    from src.data.tiling import generate_tile_records,crop_with_padding,stitch_tiles_hann
    h,w=713,991;yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    ramp=(xx+yy*w).float()/(h*w)
    records=generate_tile_records(h,w)
    tiles=[crop_with_padding(ramp[None],r.local_xyxy)[0] for r in records]
    assert torch.allclose(stitch_tiles_hann(tiles,records,(h,w)),ramp,atol=2e-7)


def test_native_map_provenance_evaluator_and_mismatch_rejection(artifacts,monkeypatch):
    a=artifacts; samples=[]
    for i in range(2):
        im=a.root/f'input_{i}.npy';np.save(im,np.random.default_rng(i).random((133,149,3),dtype=np.float32))
        gt=np.zeros((133,149),np.uint8)
        if i:gt[30:55,70:82]=255
        mp=a.root/f'input_{i}.png';Image.fromarray(gt).save(mp)
        samples.append(dict(image_id=f'dev_synthetic_{i}',category='fabric',image=str(im),gt_mask=str(mp),
            image_sha256=file_hash(im),gt_mask_sha256=file_hash(mp),original_hw=[133,149],
            source_image_id=f'fixture/dev_synthetic_{i}',synthetic={'fixture':True}))
    ip=a.root/'inputs.json';ip.write_text(json.dumps(dict(schema='msila.day05.inference_inputs.v1',split='dev_synthetic',samples=samples)))
    ex=FixtureExtractor();ex.out_channels=384;ex.patch_size=16;ex.backbone_is_frozen=lambda:True
    monkeypatch.setattr('scripts.day05_full_inference.build_online_extractor',lambda **kwargs:ex)
    args=SimpleNamespace(r0=a.runs[0],r1=a.runs[1],r2=a.runs[2],output_dir=Path(a.common['output_root'])/'maps'/'dev_synthetic',
        input_manifest=ip,split='dev_synthetic',val_records=a.root/'dev_synthetic.json',
        dino_checkpoint=a.checkpoint,dinov3_repo=a.root,device='cpu',seg_f1_threshold=.5)
    manifest=run_inference(args)
    result=evaluate_manifest(manifest)
    assert set(result['macro_by_candidate'])==set(SOURCES)
    assert result['evaluator_sha256']
    assert run_inference(args)==manifest
    # Exercise real diagnostic modules on fixtures, and stop E8 without inventing VRAM.
    from scripts.run_day05_pipeline import evaluate, Blocked
    tiny=a.root/'fixture_tiny.json';boundary=a.root/'fixture_boundary.json'
    tiny.write_text(json.dumps(dict(tiny_area_px=100,area_unit='original_image_pixels',connectivity=8,max_fpr=.05,
        locked_before_candidate_results=True,threshold_basis='unit fixture only')))
    boundary.write_text(json.dumps(dict(boundary_mode='image_border_band',band_width_px=2,connectivity=8,max_fpr=.05,
        prediction_threshold=.5,tolerance_px=1,locked_before_candidate_results=True,rule_basis='unit fixture only')))
    ev=SimpleNamespace(**(a.common|dict(tiny_protocol=tiny,boundary_protocol=boundary,
        seg_f1_threshold=.5,split='dev_synthetic',efficiency_csv=None)))
    with pytest.raises(Blocked,match='MISSING_GPU'):evaluate(ev)
    evroot=Path(ev.output_root)/'evaluation/dev_synthetic'
    assert (evroot/'primary/metrics.json').is_file() and (evroot/'regions/per_region_stats.csv').is_file()
    assert not (evroot/'summary/representation_lock.yaml').exists()
    # Resume validated outputs without overwriting completed metric/diagnostic stages.
    before=file_hash(evroot/'primary/metrics.json')
    with pytest.raises(Blocked,match='MISSING_GPU'):evaluate(ev)
    assert before==file_hash(evroot/'primary/metrics.json')
    obj=json.loads(manifest.read_text());prov=Path(obj['samples'][0]['map_provenance']['R2'])
    old=prov.read_text();d=json.loads(old);d['image_id']='wrong_GT_id';prov.write_text(json.dumps(d))
    try:
        with pytest.raises(ValueError,match='ID'):evaluate_manifest(manifest)
    finally:prov.write_text(old)
    bad=copy.deepcopy(samples);bad[0]['image_id']='missing_id';ip.write_text(json.dumps(dict(schema='msila.day05.inference_inputs.v1',split='dev_synthetic',samples=bad)))
    with pytest.raises(ValueError,match='IDs'):run_inference(args)


def test_source_leakage_is_rejected(artifacts):
    train=copy.deepcopy(artifacts.records['train_core']);dev=copy.deepcopy(artifacts.records['dev_synthetic'])
    dev[0]['meta']['source_identity']=train[0]['meta']['source_identity']
    with pytest.raises(ValueError,match='leakage'):validate_record_sources(train,dev)


def test_day04_source_plan_is_deterministic_and_excludes_test(tmp_path):
    train=tmp_path/'fabric/train/good';train.mkdir(parents=True)
    for i in range(64):(train/f'{i}.png').write_bytes(b'plan fixture')
    test=tmp_path/'fabric/test_public/bad';test.mkdir(parents=True);(test/'0.png').write_bytes(b'never used')
    rows=source_plan(tmp_path,['fabric'])
    assert rows==source_plan(tmp_path,['fabric']) and len(rows)==64
    assert all('/train/good/' in r['source_path'] for r in rows)
    tr={r['source_identity'] for r in rows if r['split']=='train_core'}
    dv={r['source_identity'] for r in rows if r['split']=='dev_synthetic'}
    assert not tr&dv


def test_synthesis_precedes_extraction_and_native_masks_match(tmp_path):
    p=tmp_path/'source.png';Image.fromarray(np.full((800,850,3),120,np.uint8)).save(p)
    e=dict(source_path=str(p),source_identity='fabric/fixture.png',split='dev_synthetic',make_anomaly=True)
    local,context,lm,native,nm,meta=prepare_sample(e)
    assert torch.equal(local,context[:,128:640,128:640])
    x,y=meta['context_crop_xy_on_padded']
    assert torch.equal(local,native[:,y+128:y+640,x+128:x+640])
    assert torch.equal(lm,nm[:,y+128:y+640,x+128:x+640])
    assert lm.any() and meta['synthetic']['anomaly_type']=='cutpaste'


def test_checkpoint_restores_rng_optimizer_and_sampler_order(tmp_path,artifacts):
    seed_everything(42)
    model=Day05RepresentationModel(day05_config=artifacts.cfg,candidate='R0',in_channels=8,adapter_r=32,adapter_d=384)
    opt=build_optimizer(model,artifacts.protocol);p=tmp_path/'last.pt'
    save_checkpoint(p,model,opt,2,4,.7,'fixture')
    expected=torch.rand(5);torch.rand(12)
    epoch,step,best=load_checkpoint(p,model,opt,'fixture',torch.device('cpu'))
    assert (epoch,step,best)==(2,4,.7) and torch.equal(expected,torch.rand(5))
    with pytest.raises(RuntimeError,match='mismatch'):load_checkpoint(p,model,opt,'wrong',torch.device('cpu'))


def test_dry_run_reports_missing_assets_without_training(tmp_path):
    assert main(['--stage','all','--dry-run','--output-root',str(tmp_path)])==2
    report=json.loads((tmp_path/'last_gate_report.json').read_text())
    assert report['status']=='BLOCKED_MISSING_DATA'
    assert not list(tmp_path.rglob('best.pt'))


def test_stage_resume_checks_hashes_and_preserves_incomplete_attempt(tmp_path):
    root=tmp_path/'stage';root.mkdir();(root/'partial.txt').write_text('interrupted')
    count=[]
    def work(d):count.append(1);(d/'report.json').write_text('{}')
    stage_once(root,{'asset':'hash'},work);stage_once(root,{'asset':'hash'},work)
    assert len(count)==1 and list(tmp_path.glob('stage.incomplete.*'))
    (root/'report.json').write_text('changed')
    with pytest.raises(ValueError,match='artifact'):stage_once(root,{'asset':'hash'},work)


def test_locked_bfloat16_amp_preflight_keeps_protocol(artifacts):
    from src.train.screen_representation import amp_context
    protocol=copy.deepcopy(artifacts.protocol)
    protocol['training']['amp']={'enabled':True,'dtype':'bfloat16'}
    before=copy.deepcopy(protocol)
    seed_everything(42,protocol)
    model=Day05RepresentationModel(day05_config=artifacts.cfg,candidate='R2',in_channels=8,adapter_r=32,adapter_d=384)
    batch={k:torch.randn(1,8,4,4) for k in FEATURE_KEYS}
    mask=torch.zeros(1,1,16,16);mask[:,:,4:10,5:12]=1
    batch.update(mask=mask,meta=[{'geometry':artifacts.geometry}])
    report=run_preflight(model,[batch],make_criterion(artifacts.cfg),build_optimizer(model,protocol),torch.device('cpu'),2,protocol)
    assert report['status']=='PASS' and protocol==before
    with amp_context(protocol,torch.device('cpu')): assert model(batch)[0].dtype==torch.bfloat16


def test_fixture_epoch_resume_matches_uninterrupted_updates(tmp_path,artifacts,monkeypatch):
    import src.train.screen_representation as training
    common=artifacts.common|dict(output_root=tmp_path/'resume')
    original=training.save_checkpoint
    def interrupt(path,*args,**kwargs):
        original(path,*args,**kwargs)
        if Path(path).name=='last.pt' and args[2]==1: raise RuntimeError('fixture simulated runtime interruption')
    monkeypatch.setattr(training,'save_checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='interruption'):train_candidate(SimpleNamespace(**common,candidate='R0'))
    monkeypatch.setattr(training,'save_checkpoint',original)
    run=train_candidate(SimpleNamespace(**common,candidate='R0'))
    expected=torch.load(artifacts.runs[0]/'last.pt',map_location='cpu',weights_only=False)
    actual=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert actual['global_step']==expected['global_step']==2
    assert all(torch.equal(actual['model'][k],v) for k,v in expected['model'].items())
    import csv
    with (run/'epoch_log.csv').open() as f: assert [r['epoch'] for r in csv.DictReader(f)]==['1','2']


def test_builder_reuses_shards_and_verifies_day04_replay(tmp_path,monkeypatch):
    import scripts.build_day05_cache as builder
    plan=[]
    for split in ('train_core','dev_synthetic'):
        for i in range(2):
            p=tmp_path/f'{split}_{i}.png';Image.fromarray(np.full((800,850,3),80+i*20,np.uint8)).save(p)
            plan.append(dict(category='fabric',split=split,image_id=f'{split}_{i}',source_identity=f'fabric/{split}_{i}.png',
                             source_path=str(p),make_anomaly=bool(i)))
    ex=FixtureExtractor();ex.out_channels=384;ex.patch_size=16
    monkeypatch.setattr(builder,'source_plan',lambda *args:plan)
    monkeypatch.setattr(builder,'build_online_extractor',lambda **kwargs:ex)
    ckpt=tmp_path/'fixture.pth';ckpt.write_bytes(b'fixture DINO')
    repo=tmp_path/'repo';repo.mkdir();(repo/'hubconf.py').write_text('# fixture')
    a=SimpleNamespace(data_root=tmp_path,output_root=tmp_path/'data',dinov3_repo=repo,dino_checkpoint=ckpt,
        categories='fabric',device='cpu',cache_dir=None,export_dev_only=False,dry_run=False)
    assert build(a)['status']=='COMPLETE' and len(ex.pairs)==4
    cache=a.output_root/'feature_cache';manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['num_samples']==4 and manifest['num_shards']==1
    assert manifest['producer_signature']['backbone']=='dinov3_vits16'
    assert build(a)['status']=='COMPLETE' and len(ex.pairs)==4
    export=SimpleNamespace(**(vars(a)|dict(output_root=tmp_path/'export',export_dev_only=True,cache_dir=cache,
        train_records=a.output_root/'records/train_core.json',val_records=a.output_root/'records/dev_synthetic.json',
        mask_root=a.output_root/'masks')))
    assert build(export)['status']=='COMPLETE' and len(ex.pairs)==8
    inputs=json.loads((export.output_root/'dev_inputs.json').read_text())
    assert len(inputs['samples'])==2 and inputs['samples'][0]['original_hw']==[800,850]
    mp=a.output_root/'masks/dev_synthetic/fabric/dev_synthetic_1.png'
    arr=np.array(Image.open(mp));arr[0,0]=255-arr[0,0];Image.fromarray(arr).save(mp)
    with pytest.raises(ValueError,match='replay'):build(export)


def test_sampler_resume_ignores_persistent_worker_base_seed_consumption(artifacts):
    from src.train.screen_representation import make_dataset, make_loader
    ds=make_dataset(artifacts.root/'feature_cache',artifacts.records['train_core'],artifacts.protocol,None)
    p=copy.deepcopy(artifacts.protocol);p['training']['batch_size']=1
    live=make_loader(ds,p,True,42)
    # Worker creation consumes loader RNG; a reused persistent worker does not.
    # Iterate the actual zero-worker loader to exercise its base-seed draw without
    # forking this multithreaded CPU-only test runner.
    list(live)
    live.sampler.generator.manual_seed(43)
    expected=list(live.sampler)
    resumed=make_loader(ds,p,True,42);resumed.sampler.generator.manual_seed(43)
    resumed_iterator=iter(resumed) # New iterator draws a base seed, without pulling a sample.
    actual=list(resumed.sampler)
    assert actual==expected
    live.sampler.generator.manual_seed(44);resumed.sampler.generator.manual_seed(44)
    assert list(live.sampler)==list(resumed.sampler)
    assert live.sampler.generator is not live.generator


def test_final_test_uses_official_normal_labels_and_rejects_missing_abnormal_gt(artifacts,tmp_path):
    from scripts.day05_full_inference import load_inputs
    for role,name in [('good','normal'),('bad','abnormal')]:
        p=tmp_path/'fabric/test_public'/role/f'{name}.png';p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.full((153,179,3),100,np.uint8)).save(p)
    mp=tmp_path/'fabric/test_public/ground_truth/bad/abnormal_mask.png';mp.parent.mkdir(parents=True)
    mask=np.zeros((153,179),np.uint8);mask[30:40,44:50]=255;Image.fromarray(mask).save(mp)
    # Independent checkpoint scope for this test; no dependence on another test.
    scope=tmp_path/'fixture_artifact_scope'
    run_aliases=[]
    for c,run in zip(SOURCES,artifacts.runs):
        alias=scope/f'seed_42/fabric/{c}';alias.mkdir(parents=True)
        (alias/'best.pt').symlink_to(run/'best.pt');run_aliases.append(alias)
    maps=scope/'maps/dev_synthetic';maps.mkdir(parents=True)
    hp=maps/'handoff_manifest.json';verify_and_build_manifest(*artifacts.runs,hp)
    dev_manifest=maps/'evaluation_manifest.json'
    dev_manifest.write_text(json.dumps(dict(fixture_only=True,handoff_manifest=str(hp),handoff_sha256=file_hash(hp))))
    metrics=scope/'fixture_evidence.json';metrics.write_text('{"fixture_only": true}')
    lp=tmp_path/'fixture_lock.yaml';lp.write_text(yaml.safe_dump(dict(schema='msila.representation_lock.v1',
        evaluation_scope={'split':'dev_synthetic'},provenance=dict(input_files_sha256={str(metrics):file_hash(metrics)},
        evaluation_manifest_sha256=file_hash(dev_manifest)))))
    a=SimpleNamespace(split='test_public',representation_lock=lp,data_root=tmp_path,
        r0=run_aliases[0],r1=run_aliases[1],r2=run_aliases[2],output_dir=tmp_path/'maps')
    samples=load_inputs(a,{'category':'fabric'},validate_only=True)
    assert len(samples)==2 and not (a.output_dir/'ground_truth').exists()
    samples=load_inputs(a,{'category':'fabric'})
    normal=next(s for s in samples if s['known_normal'])
    assert normal['gt_provenance']=='official_good_label_zero_mask'
    assert not np.array(Image.open(normal['gt_mask'])).any()
    mp.unlink()
    with pytest.raises(ValueError,match='abnormal GT'):load_inputs(a,{'category':'fabric'})


def test_source_plan_accepts_official_archive_wrapper(tmp_path):
    train=tmp_path/'fabric/fabric/train/good';train.mkdir(parents=True)
    for i in range(64):(train/f'{i}.png').write_bytes(b'plan fixture')
    rows=source_plan(tmp_path,['fabric'])
    assert len(rows)==64 and rows[0]['source_identity'].startswith('fabric/')
    assert 'fabric/fabric' not in rows[0]['source_identity']


def test_changed_masks_cannot_reuse_a_complete_training_run(artifacts):
    p=Path(artifacts.records['train_core'][1]['mask_path']);old=p.read_bytes()
    arr=np.array(Image.open(p));arr[0,0]=255-arr[0,0];Image.fromarray(arr).save(p)
    try:
        with pytest.raises(RuntimeError,match='PROTOCOL DRIFT'):
            train_candidate(SimpleNamespace(**artifacts.common,candidate='R0'))
    finally:p.write_bytes(old)


@pytest.mark.parametrize('override', [[], ['--backbone', 'dinov3_vitb16'], ['--adapter-r', '64'],
                                    ['--adapter-d', '128'], ['--seed', '17']])
def test_pipeline_audit_rejects_silently_ignored_day05_overrides(artifacts, tmp_path, override, capsys):
    repo = tmp_path / 'fixture_repo'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# Fixture only; no real DINO implementation or weights\n')
    output = tmp_path / 'audit'
    args = ['--stage', 'audit', '--device', 'cpu', '--output-root', str(output),
            '--cache-dir', str(artifacts.root / 'feature_cache'),
            '--train-records', str(artifacts.common['train_records']),
            '--val-records', str(artifacts.common['val_records']),
            '--training-protocol', str(artifacts.common['training_protocol']),
            '--adapter-selection-report', str(artifacts.common['adapter_selection_report']),
            '--dinov3-repo', str(repo), '--dino-checkpoint', str(artifacts.checkpoint)]
    assert main(args + override) == (2 if override else 0)
    if override:
        report = json.loads((output / 'last_gate_report.json').read_text())
        assert report['status'] == 'FAIL' and 'locked' in report['reason']
    else:
        report = json.loads(capsys.readouterr().out)
        assert report['status'] == 'ASSETS_VALID' and report['real_forward_executed'] is False


def test_pipeline_cpu_preflight_records_cache_hashes_and_optimizer_contract(artifacts, tmp_path):
    repo = tmp_path / 'fixture_repo'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# Fixture only; this cache CLI does not execute DINO\n')
    output = tmp_path / 'preflight'
    assert main(['--stage', 'preflight', '--device', 'cpu', '--preflight-steps', '2',
                 '--output-root', str(output), '--cache-dir', str(artifacts.root / 'feature_cache'),
                 '--train-records', str(artifacts.common['train_records']),
                 '--val-records', str(artifacts.common['val_records']),
                 '--training-protocol', str(artifacts.common['training_protocol']),
                 '--adapter-selection-report', str(artifacts.common['adapter_selection_report']),
                 '--dinov3-repo', str(repo), '--dino-checkpoint', str(artifacts.checkpoint)]) == 0
    from src.train.screen_representation import sha256_json, read_yaml
    for candidate in SOURCES:
        run = output / 'seed_42/fabric' / candidate
        manifest = json.loads((run / 'run_manifest.json').read_text())
        config = read_yaml(run / 'resolved_config.yaml')
        preflight = json.loads((run / 'preflight_report.json').read_text())
        assert manifest['status'] == 'PREFLIGHT_PASS' and manifest['sources'] == SOURCES[candidate]
        assert manifest['resolved_config_sha256'] == sha256_json(config)
        assert config['backbone']['frozen'] is True
        assert config['cache']['manifest_sha256'] == file_hash(artifacts.root / 'feature_cache/manifest.json')
        assert config['cache']['provenance_check']['producer_signature']['checkpoint_sha256'] == file_hash(artifacts.checkpoint)
        assert config['training']['amp'] == artifacts.protocol['training']['amp']
        assert preflight['status'] == preflight['optimizer_parameters']['status'] == 'PASS'
        assert preflight['steps'] == 2 and set(preflight['source_usage'][-1]) == set(SOURCES[candidate])
        assert not (run / 'best.pt').exists() and not (run / 'training_log.csv').exists()


@pytest.mark.parametrize('shape', [(1, 8, 32, 32), (1, 384, 4, 4), (2, 384, 32, 32)])
def test_day05_verify_cache_rejects_shape_in_any_source(tmp_path, shape):
    checkpoint = tmp_path / 'fixture.pth'
    checkpoint.write_bytes(b'fixture only, not pretrained DINO')
    cache = tmp_path / 'cache'
    record = dict(image_id='source0', category='fabric')
    geometry = dict(local_box=[0, 0, 512, 512], context_box=[-128, -128, 640, 640],
                    context_to_local=torch.eye(3).tolist())
    features = {key: torch.zeros(1, 384, 32, 32) for key in FEATURE_KEYS}
    features['context_b12'] = torch.zeros(shape)
    with FeatureCacheWriter(cache, producer_signature=signature(checkpoint)) as writer:
        writer.add(dict(record, geometry=geometry, **features))
    with pytest.raises(RuntimeError, match='source0/context_b12 shape'):
        verify_cache(cache, [record], [], 'dinov3_vits16')
