from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from torch import nn
import yaml

from src.models.backbone_registry import BACKBONES, adapter_pairs, validate_blocks
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.data.synthetic_anomaly import NativeTinyDefectGenerator
from src.data.tiling import generate_tile_records, crop_with_padding, stitch_tiles_hann
from src.eval.region_stats import component_geometry
from src.eval.full_scale import DEVMetricAccumulator
from src.train.full_scale import expand_grid, rank_models, job_id
from src.train.screen_representation import Day05RepresentationModel, save_checkpoint, load_checkpoint
from scripts.build_day05_cache import native_source_plan
from scripts.run_day05_pipeline import parse_args
from src.train.day05_contract import validate_record_sources, digest

ROOT=Path(__file__).parents[1]


def protocol():
    return yaml.safe_load((ROOT/'configs/full_scale_synthetic.yaml').read_text())


@pytest.mark.parametrize('backbone',list(BACKBONES))
def test_loaded_architecture_and_physical_block_cache_mapping(monkeypatch,tmp_path,backbone):
    spec=BACKBONES[backbone]
    class MockDino(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=nn.ModuleList(nn.Identity() for _ in range(spec.depth))
            self.embed_dim=spec.channels
            self.patch_size=spec.patch_size
            self.p=nn.Parameter(torch.ones(1))
        def get_intermediate_layers(self,x,*,n,**kwargs):
            return tuple(torch.full((x.shape[0],self.embed_dim,x.shape[-2]//16,x.shape[-1]//16),float(b)) for b in n)
    model=MockDino()
    monkeypatch.setattr(torch.hub,'load',lambda **kwargs:model)
    (tmp_path/'hubconf.py').write_text('# fixture')
    extractor=DINOv3FeatureExtractor(tmp_path,'fixture.pth',model_name=backbone)
    assert extractor.blocks==spec.blocks and extractor.out_channels==spec.channels
    cache=extractor.extract_online_cache_features(torch.ones(1,3,32,32),torch.ones(1,3,32,32))
    for slot,actual in zip((4,8,12),spec.blocks):
        assert cache[f'local_b{slot}'].shape==(1,spec.channels,2,2)
        assert cache[f'local_b{slot}'][0,0,0,0]==actual-1
    extractor.train()
    assert not model.training and not model.p.requires_grad
    model.embed_dim += 1
    with pytest.raises(ValueError,match='architecture mismatch'):
        DINOv3FeatureExtractor(tmp_path,'fixture.pth',model_name=backbone)


def test_full_grid_is_not_reduced_and_explicit_pairs_override_ratios():
    config=yaml.safe_load((ROOT/'configs/full_scale_grid.yaml').read_text())
    jobs=expand_grid(config)
    assert len(jobs)==3240
    assert {j['backbone'] for j in jobs}==set(BACKBONES)
    assert {j['seed'] for j in jobs}=={17,42,2026}
    pairs=adapter_pairs(384,config['adapter'])
    assert pairs==[(r,d) for r in (32,64,128) for d in (128,256,384)]
    assert adapter_pairs(1280,{'pairs':[[17,1337]]})==[(17,1337)]
    with pytest.raises(ValueError,match='Duplicate'):
        adapter_pairs(16,{'r_ratios':[.01,.02],'d_ratios':[1.],'round_to':8})
    with pytest.raises(ValueError):validate_blocks([4,8,33],32)
    args=parse_args(['--train-all','--stage','all'])
    assert args.train_all and args.stage=='all'
    assert parse_args(['--train-all']).stage=='all'


def test_exact_local_weights_are_loaded_even_with_identical_basenames(monkeypatch,tmp_path):
    class MockDino(nn.Module):
        embed_dim=384
        patch_size=16
        def __init__(self):
            super().__init__()
            self.blocks=nn.ModuleList(nn.Identity() for _ in range(12))
            self.p=nn.Parameter(torch.tensor(99.))
        def get_intermediate_layers(self,*args,**kwargs):
            return ()
    repo=tmp_path/'repo';repo.mkdir();(repo/'hubconf.py').write_text('# fixture')
    paths=[]
    for i,value in enumerate((2.,7.)):
        path=tmp_path/str(i)/'same_basename.pth';path.parent.mkdir()
        torch.save({'p':torch.tensor(value)},path);paths.append(path)
    def hub(**kwargs):
        assert kwargs['pretrained'] is False
        return MockDino()
    monkeypatch.setattr(torch.hub,'load',hub)
    a=DINOv3FeatureExtractor(repo,paths[0]);b=DINOv3FeatureExtractor(repo,paths[1])
    assert a.backbone.p.item()==2. and b.backbone.p.item()==7.


@pytest.mark.parametrize('channels',[384,768,1024,1280])
@pytest.mark.parametrize('representation',['R0','R1','R2'])
def test_head_uses_C_and_keeps_adapter_d_independent_of_fusion(channels,representation):
    config=yaml.safe_load((ROOT/'configs/day05_representation.yaml').read_text())
    model=Day05RepresentationModel(day05_config=config,candidate=representation,in_channels=channels,
                                  adapter_r=3,adapter_d=7)
    features={f'{view}_b{block}':torch.randn(1,channels,2,2)
              for view in ('local','context') for block in (4,8,12)}
    features.update(output_hw=(9,11),meta=[{'geometry':dict(
        local_input_hw=[512,512],context_input_hw=[512,512],local_box=[128,128,640,640],
        context_box=[0,0,768,768],local_to_context=[[1,0,128],[0,1,128],[0,0,1]],
        context_to_local=[[1,0,-128],[0,1,-128],[0,0,1]])}])
    logits,_=model(features)
    assert logits.shape==(1,1,9,11)
    logits.square().mean().backward()
    assert any(p.grad is not None for p in model.decoder.parameters())
    assert model.projection.fusion_dim==64
    assert model.adapters['b12'].projection_dim==7


@pytest.mark.parametrize('kind',['pinhole','thin_scratch','texture','contamination'])
@pytest.mark.parametrize('placement',['interior','image_boundary'])
def test_native_defects_exact_support_determinism_coverage(kind,placement):
    image=torch.full((3,512,640),.4)
    p=protocol()
    generator=NativeTinyDefectGenerator(p)
    for size in p['size_bins']:
        sample=generator(image,seed=71,defect_type=kind,size_bin=size,placement=placement)
        repeat=generator(image,seed=71,defect_type=kind,size_bin=size,placement=placement)
        assert torch.equal(sample.image,repeat.image) and sample.metadata==repeat.metadata
        assert torch.equal(sample.image[:,sample.mask[0]==0],image[:,sample.mask[0]==0])
        components=component_geometry(sample.mask[0].numpy(),p['boundary_band_px'])
        assert len(components)==1
        lo,hi=p['size_bins'][size]
        assert lo <= components[0]['area'] <= hi
        assert components[0]['is_boundary']==(placement=='image_boundary')
        assert sample.metadata['mean_abs_change'] >= p['min_mean_abs_change']


def test_mask_survives_native_tiling_padding_and_Hann_stitching():
    gt=torch.zeros(1,529,677)
    gt[:,0:2,0:3]=1
    gt[:,256:260,383:386]=1
    records=generate_tile_records(529,677)
    tiles=[crop_with_padding(gt,r.local_xyxy,pad_mode='constant')[0] for r in records]
    restored=stitch_tiles_hann(tiles,records,(529,677))
    assert torch.allclose(restored,gt[0],atol=1e-6)


def test_metrics_keep_original_background_and_return_undefined_for_empty_groups():
    p=protocol()
    metrics=DEVMetricAccumulator(p)
    mask=np.zeros((20,20),dtype=bool);mask[8:10,8:10]=True
    score=mask.astype(np.float32)
    metrics.add(score,mask,split='dev_tiny',image_id='tiny')
    metrics.add(score,mask,split='dev_mixed',image_id='mixed')
    result=metrics.result()
    assert result['dev_aupro']==pytest.approx(1.)
    assert result['groups']['image_boundary_regions']['aupro_0_05'] is None
    assert result['groups']['size_bin/mixed_control']['aupro_0_05'] is None
    assert result['groups']['all']['contour_boundary_f1']==pytest.approx(1.)


def test_train_DEV_sources_disjoint_and_no_TEST_source_is_used(tmp_path):
    from PIL import Image
    for split in ('train','test_public'):
        path=tmp_path/'fabric'/split/'good';path.mkdir(parents=True)
        for i in range(100):Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(path/f'{i:03d}.png')
    p=protocol()
    rows=native_source_plan(tmp_path,['fabric'],p,42)
    assert all('/train/good/' in r['source_path'] for r in rows)
    assert rows==native_source_plan(tmp_path,['fabric'],p,42)
    sources={role:{r['source_identity'] for r in rows if r['split']==role}
             for role in ('train_core','dev_tiny','dev_mixed')}
    assert len(set.union(*sources.values()))==100
    for a,b in itertools_combinations(sources.values()):assert not a & b
    for role in ('dev_tiny','dev_mixed'):
        strata={(r['defect_type'],r['size_bin'],r['placement']) for r in rows
                if r['split']==role and r['make_anomaly']}
        assert len(strata)==4*len(p[role+'_bins'])*2


def itertools_combinations(values):
    import itertools
    return itertools.combinations(values,2)


def test_ranking_requires_full_declared_grid_and_averages_seeds(tmp_path):
    (tmp_path/'protocol_lock.json').write_text('{}')
    jobs=[dict(backbone='dinov3_vits16',r=32,d=128,representation=r,seed=s,category='fabric')
          for r in ('R0','R1') for s in (17,42)]
    completed={job_id(j):dict(run_dir='fixture',dev_aupro=(.9 if j['representation']=='R1' else .7),
                              checkpoint_sha256='a'*64,metrics_sha256='b'*64) for j in jobs}
    with pytest.raises(ValueError,match='incomplete'):
        rank_models(tmp_path,jobs,{})
    result=rank_models(tmp_path,jobs,completed)
    assert result['winners']['fabric']['representation']=='R1'
    assert result['winners']['fabric']['seeds']==[17,42]


@pytest.mark.parametrize('quantized',[False,True])
def test_chunked_native_AUPRO_matches_reference_with_ties(quantized):
    from scipy.ndimage import label
    from src.metrics.aupro import aupro, aupro_from_parts
    rng=np.random.default_rng(42)
    for trial in range(30):
        maps=[rng.random((13,17)).astype(np.float32) for _ in range(3)]
        if quantized:maps=[np.round(s*8)/8 for s in maps]
        masks=[rng.random((13,17)) > .95 for _ in maps]
        normals,components=[],[]
        for score,mask in zip(maps,masks):
            normals.append(np.sort(score[~mask]))
            labels,n=label(mask,structure=np.ones((3,3)))
            components.extend(score[labels==r] for r in range(1,n+1))
        expected=aupro(maps,masks,max_fpr=.05)['aupro']
        actual=aupro_from_parts(normals,components,max_fpr=.05,sorted_normals=True)['aupro']
        assert actual==pytest.approx(expected,abs=1e-12)
