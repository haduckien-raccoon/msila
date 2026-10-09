"""Real CPU optimizer/resume tests on isolated synthetic feature fixtures.

These fixtures are software tests, not DINOv3 or MVTec experimental results.
The scientific 150-epoch config and the full grid are never changed here.
"""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from src.data.feature_cache import FeatureCacheWriter, FEATURE_KEYS, FeatureCacheReader, ProducerMismatchError
from src.geometry.view_meta import build_view_meta
from src.train.day05_contract import digest, file_hash
import src.train.screen_representation as trainer
from scripts.day05_full_inference import load_tv1_model

ROOT=Path(__file__).parents[1]


@pytest.fixture(scope='module')
def full_assets(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('full_scale_software_fixture')
    synthetic=yaml.safe_load((ROOT/'configs/full_scale_synthetic.yaml').read_text())
    config=yaml.safe_load((ROOT/'configs/day05_representation.yaml').read_text())
    config['schema']='msila.full_scale.representation.v2'
    config['locked']['training']['seed']=17
    cfg=root/'representation.yaml';cfg.write_text(yaml.safe_dump(config))
    p=yaml.safe_load((ROOT/'configs/full_scale_train.yaml').read_text())
    p['training'].update(epochs=2,batch_size=2,amp={'enabled':False,'dtype':'bfloat16'},checkpoint_interval_steps=1)
    p['data'].update(num_workers=0,persistent_workers=False,pin_memory=False)
    p['evaluation']=dict(synthetic_protocol=synthetic)
    pp=root/'fixture_training.yaml';pp.write_text(yaml.safe_dump(p))
    weights=root/'fixture_weights.pth';weights.write_bytes(b'SOFTWARE FIXTURE ONLY, NOT DINO WEIGHTS')
    sig=dict(schema='msila.full_scale.cache.v2',backbone='dinov3_vits16',checkpoint_sha256=file_hash(weights),
             architecture=dict(channels=384,depth=12,patch_size=16),logical_layers_1based=[4,8,12],
             internal_indices_0based=[3,7,11],cache_slot_blocks={'b4':4,'b8':8,'b12':12},
             smoke_only=False,full_source_coverage=True,source_split_version='train_normal_hash80_10_10_tiny_v2',
             local_source_size=[512,512],context_source_size=[768,768],model_input_size=[512,512],
             normalization='ImageNet mean/std',synthetic_protocol=synthetic,
             preprocessing='SOFTWARE_FIXTURE_ONLY',source_sha256={'fixture':'f'*64})
    g=build_view_meta(source_hw=(768,768),local_box_xyxy=(128,128,640,640),context_box_xyxy=(0,0,768,768))
    geometry=dict(local_box=[128,128,640,640],context_box=[0,0,768,768],
        local_to_context=g.local_to_context.tolist(),context_to_local=g.context_to_local.tolist(),
        local_input_hw=[512,512],context_input_hw=[512,512])
    rows={'train_core':[],'dev_tiny':[],'dev_mixed':[]}
    rng=torch.Generator().manual_seed(17)
    with FeatureCacheWriter(root/'cache',producer_signature=sig,max_samples_per_shard=4) as writer:
        for split in rows:
            for i in range(4 if split=='train_core' else 2):
                iid=f'{split}_{i}'
                mask=np.zeros((512,512),dtype=np.uint8)
                if i%2:mask[200:202,210:214]=255
                mp=root/(iid+'.png');Image.fromarray(mask).save(mp)
                meta=dict(split=split,source_identity=f'fixture/train/good/{iid}',synthetic={'fixture_only':True},
                          native_image_id=iid,smoke_only=False,full_source_coverage=True)
                rows[split].append(dict(image_id=iid,category='fabric',mask_path=str(mp),mask_hw=[512,512],
                                        is_anomaly=bool(i%2),meta=meta))
                writer.add(dict(image_id=iid,category='fabric',geometry=geometry,
                                **{k:torch.randn(1,384,32,32,generator=rng) for k in FEATURE_KEYS}))
    for split,values in rows.items():(root/(split+'.json')).write_text(json.dumps(values))
    (root/'dev.json').write_text(json.dumps(rows['dev_tiny']+rows['dev_mixed']))
    args=SimpleNamespace(day05_config=cfg,training_protocol=pp,backbone='dinov3_vits16',dino_checkpoint=weights,
        r=3,d=7,seed=17,category='fabric',candidate='R0',adapter_selection_report=None,
        train_records=root/'train_core.json',val_records=root/'dev.json',mask_root=None,cache_dir=root/'cache',
        output_root=root/'runs',device='cpu',preflight_steps=2,preflight_only=False,resume=True,
        allow_unverified_cache_provenance=False)
    return SimpleNamespace(root=root,args=args,protocol=p,config=config,signature=sig)


def test_actual_training_logs_DEV_metric_and_resumes_mid_epoch_exactly(full_assets,tmp_path,monkeypatch):
    baseline=copy.copy(full_assets.args);baseline.output_root=tmp_path/'baseline'
    expected_run=trainer.train_candidate(baseline)
    resumed=copy.copy(full_assets.args);resumed.output_root=tmp_path/'resumed'
    original=trainer.save_checkpoint
    def stop_after_committed_update(path,*args,**kwargs):
        original(path,*args,**kwargs)
        progress=kwargs.get('progress')
        if Path(path).name=='last.pt' and progress and progress.get('step_in_epoch')==1:
            raise RuntimeError('fixture interruption after a committed update')
    monkeypatch.setattr(trainer,'save_checkpoint',stop_after_committed_update)
    with pytest.raises(RuntimeError,match='fixture interruption'):
        trainer.train_candidate(resumed)
    monkeypatch.setattr(trainer,'save_checkpoint',original)
    actual_run=trainer.train_candidate(resumed)
    expected=torch.load(expected_run/'last.pt',map_location='cpu',weights_only=False)
    actual=torch.load(actual_run/'last.pt',map_location='cpu',weights_only=False)
    assert expected['global_step']==actual['global_step']==4
    assert all(torch.equal(actual['model'][k],v) for k,v in expected['model'].items())
    for path in (expected_run,actual_run):
        with (path/'training_log.csv').open() as f:
            assert [int(r['global_step']) for r in csv.DictReader(f)]==[1,2,3,4]
        with (path/'epoch_log.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==2 and all(float(r['dev_aupro'])>=0 for r in rows)
        manifest=json.loads((path/'run_manifest.json').read_text())
        assert manifest['status']=='COMPLETE' and manifest['total_update_budget']==4
        assert manifest['best_dev_aupro']>=0 and manifest['total_elapsed_sec']>0
        assert (path/'last.pt.sha256').read_text().strip()==file_hash(path/'last.pt')
        model,cfg,_=load_tv1_model(path,'R0')
        assert cfg['seed']==17 and cfg['adapter']['d']==7
        assert model.aligner.deterministic_sampling and model.decoder.deterministic_resize
    before=file_hash(actual_run/'last.pt')
    assert trainer.train_candidate(resumed)==actual_run
    assert file_hash(actual_run/'last.pt')==before
    changed=copy.copy(resumed);changed.d=8
    with pytest.raises(trainer.FullTrainError,match='(?i)drift'):
        trainer.train_candidate(changed)
    assert file_hash(actual_run/'last.pt')==before


def test_seed_protocol_locks_coexist(full_assets,tmp_path):
    runs=[]
    for seed in (17,42):
        cfg=copy.deepcopy(full_assets.config);cfg['locked']['training']['seed']=seed
        path=tmp_path/f'seed{seed}.yaml';path.write_text(yaml.safe_dump(cfg))
        args=copy.copy(full_assets.args);args.seed=seed;args.day05_config=path;args.output_root=tmp_path/'runs'
        runs.append(trainer.train_candidate(args))
    assert all((p/'best.pt').is_file() for p in runs)


def test_full_cache_content_checksum_is_enforced(full_assets,tmp_path):
    geometry=dict(local_box=[0,0,16,16],context_box=[0,0,16,16],context_to_local=np.eye(3).tolist())
    with FeatureCacheWriter(tmp_path/'cache',producer_signature=full_assets.signature,max_samples_per_shard=1) as writer:
        writer.add(dict(image_id='checksum',category='fabric',geometry=geometry,
                        **{k:torch.zeros(1,384,1,1) for k in FEATURE_KEYS}))
    reader=FeatureCacheReader(tmp_path/'cache')
    assert reader.get(image_id='checksum',category='fabric')['local_b4'].shape==(1,384,1,1)
    shard=next((tmp_path/'cache/shards').glob('*.pt'))
    state=torch.load(shard,weights_only=False)
    state['records'][next(iter(state['records']))]['local_b4'][0,0,0,0]=1
    torch.save(state,shard)
    with pytest.raises(ProducerMismatchError,match='checksum'):
        FeatureCacheReader(tmp_path/'cache').get(image_id='checksum',category='fabric')


def test_OOM_leaves_full_job_and_protocol_unchanged(tmp_path,monkeypatch):
    import src.train.full_scale as full
    import scripts.build_day05_cache as builder
    config=yaml.safe_load((ROOT/'configs/full_scale_grid.yaml').read_text())
    weights=tmp_path/'weights.pth';weights.write_bytes(b'fixture')
    config['backbones']={'dinov3_vith16plus':{'checkpoint':str(weights)}}
    config.update(adapter={'pairs':[[13,29]]},representations=['R2'],seeds=[17],categories=['fabric'])
    grid=tmp_path/'grid.yaml';grid.write_text(yaml.safe_dump(config));before=grid.read_bytes()
    repo=tmp_path/'repo';repo.mkdir();(repo/'hubconf.py').write_text('# fixture')
    monkeypatch.setattr(builder,'native_source_plan',lambda *args:[])
    monkeypatch.setattr(builder,'build',lambda *args:dict(cache=str(tmp_path/'cache'),peak_vram_mb=0.))
    def oom(*args):raise torch.cuda.OutOfMemoryError('fixture OOM')
    monkeypatch.setattr(full,'train_candidate',oom)
    args=SimpleNamespace(grid_config=grid,train_all=True,backbone=None,adapter_r=None,adapter_d=None,
        representation=None,data_root=tmp_path,dinov3_repo=repo,output_root=tmp_path/'outputs',
        dino_checkpoint=None,device='cpu',stage='all',dry_run=False,resume=True,preflight_steps=2)
    with pytest.raises(torch.cuda.OutOfMemoryError):full.run_full_scale(args)
    report=json.loads((args.output_root/'grid_manifest.json').read_text())
    assert report['status']=='FAILED_OOM' and report['configuration_preserved']
    assert report['current_job']==dict(backbone='dinov3_vith16plus',r=13,d=29,representation='R2',seed=17,category='fabric')
    assert grid.read_bytes()==before


def test_native_cache_train_inference_evaluation_and_map_resume(tmp_path,monkeypatch):
    import scripts.build_day05_cache as builder
    from src.models.dinov3_extractor import DINOv3FeatureExtractor
    from src.train.full_scale import evaluate_native_dev
    class MockDINO(torch.nn.Module):
        embed_dim=384
        patch_size=16
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList(torch.nn.Identity() for _ in range(12))
            self.p=torch.nn.Parameter(torch.tensor(1.))
        def get_intermediate_layers(self,x,*,n,**kwargs):
            base=torch.nn.functional.avg_pool2d(x.mean(1,keepdim=True),16)
            return tuple((base+.01*b).expand(-1,384,-1,-1).contiguous() for b in n)
    repo=tmp_path/'repo';repo.mkdir();(repo/'hubconf.py').write_text('# MOCK SOURCE ONLY')
    backbone=MockDINO()
    weights=tmp_path/'fixture.pth';torch.save(backbone.state_dict(),weights)
    monkeypatch.setattr(torch.hub,'load',lambda **kwargs:backbone)
    extractor=DINOv3FeatureExtractor(repo,weights)
    monkeypatch.setattr(builder,'build_online_extractor',lambda **kwargs:extractor)
    plan=[]
    for i,role in enumerate(('train_core','dev_tiny','dev_mixed')):
        source=tmp_path/'fabric/train/good'/f'{i}.png';source.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.full((64,80,3),100,dtype=np.uint8)).save(source)
        for j in range(2):
            plan.append(dict(category='fabric',split=role,image_id=f'{role}_{j}',source_identity=f'fabric/train/good/{i}.png',
                source_path=str(source),make_anomaly=bool(j),defect_type='pinhole',size_bin='sub_patch',
                placement='image_boundary' if j else 'interior',synthetic_seed=17+i*2+j))
    monkeypatch.setattr(builder,'native_source_plan',lambda *args:plan)
    p=yaml.safe_load((ROOT/'configs/full_scale_synthetic.yaml').read_text())
    proto=tmp_path/'synthetic.yaml';proto.write_text(yaml.safe_dump(p))
    args=SimpleNamespace(data_root=tmp_path,output_root=tmp_path/'assets',dinov3_repo=repo,dino_checkpoint=weights,
        backbone='dinov3_vits16',blocks=None,categories='fabric',seed=42,device='cpu',
        synthetic_protocol=proto,data_artifact_root=None,cache_dir=None,export_dev_only=False,dry_run=False)
    result=builder.build(args)
    assert result['status']=='COMPLETE' and result['tile_records']==6
    first_shards={str(path):file_hash(path) for path in (tmp_path/'assets/feature_cache/shards').glob('*.pt')}
    assert builder.build(args)['status']=='COMPLETE'
    assert first_shards=={str(path):file_hash(path) for path in (tmp_path/'assets/feature_cache/shards').glob('*.pt')}
    config=yaml.safe_load((ROOT/'configs/day05_representation.yaml').read_text())
    config['schema']='msila.full_scale.representation.v2';config['locked']['training']['seed']=17
    cfg=tmp_path/'representation.yaml';cfg.write_text(yaml.safe_dump(config))
    protocol=yaml.safe_load((ROOT/'configs/full_scale_train.yaml').read_text())
    protocol['training'].update(epochs=2,batch_size=2,amp={'enabled':False},checkpoint_interval_steps=1)
    protocol['data'].update(num_workers=0,persistent_workers=False,pin_memory=False)
    protocol['evaluation']=dict(synthetic_protocol=p)
    pp=tmp_path/'train.yaml';pp.write_text(yaml.safe_dump(protocol))
    data=Path(result['data_root'])
    run=trainer.train_candidate(SimpleNamespace(day05_config=cfg,training_protocol=pp,r=3,d=7,seed=17,
        category='fabric',candidate='R2',backbone='dinov3_vits16',dino_checkpoint=weights,
        adapter_selection_report=None,train_records=data/'records/train_core.json',val_records=data/'records/dev.json',
        mask_root=data/'masks',cache_dir=result['cache'],output_root=tmp_path/'runs',device='cpu',
        preflight_steps=2,preflight_only=False,resume=True,allow_unverified_cache_provenance=False))
    metrics=evaluate_native_dev(run,data,extractor,'cpu')
    assert metrics['evaluation_unit']=='native_source_image' and metrics['dev_aupro'] is not None
    assert metrics['groups']['dev_tiny']['regions']==1 and metrics['groups']['dev_mixed']['regions']==1
    saved=file_hash(run/'native_dev/map_manifest.json')
    assert evaluate_native_dev(run,data,extractor,'cpu')==metrics
    assert saved==file_hash(run/'native_dev/map_manifest.json')
    assert not backbone.p.requires_grad and backbone.p.grad is None
    changed=json.loads((data/'dev_inputs.json').read_text())['samples'][0]['gt_mask']
    Image.fromarray(np.ones((64,80),dtype=np.uint8)*255).save(changed)
    with pytest.raises(ValueError,match='artifact changed'):
        evaluate_native_dev(run,data,extractor,'cpu')
