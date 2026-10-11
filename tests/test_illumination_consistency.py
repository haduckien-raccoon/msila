"""Numerical and paired-data regressions, not pretrained accuracy claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.data.diverse_synthetic import DiverseNativeDefectGenerator
from src.data.synthetic_anomaly import NativeTinyDefectGenerator
from src.data.loader import G1NativeDataset, G1TileDataset
from src.losses import AnomalySegmentationLoss, IlluminationConsistencyLoss
from src.train.g2_context import PairedG2Tiles, paired_tile_collate, prepare_pair, paired_record
from src.train.g2_fixed_data import (IlluminatedNativeDataset, smooth_illumination,
                                     validate_illumination_objective)
from src.train.overfit16 import Overfit16Trainer


@pytest.fixture(autouse=True)
def cpu_threads():
    threads, state = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(state)


@pytest.fixture
def protocol():
    value = yaml.safe_load(Path('configs/full_scale_synthetic.yaml').read_text())
    value['train_core_bins'] = ['sub_patch']
    return value


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'normal.png'
    yy, xx = np.mgrid[:96, :128]
    array = np.stack([60+xx, 90+yy, 70+(xx+yy)//2], axis=-1).astype(np.uint8)
    Image.fromarray(array).save(path)
    return SimpleNamespace(image_path=str(path), split='train', defect_type='good')


def options(**kwargs):
    result = dict(enabled=True, probability=1., brightness=[1.1, 1.1],
                  contrast=[1., 1.], gradient=.08)
    result.update(kwargs)
    return result


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_consistency_matches_bernoulli_kl_and_detaches_teacher(dtype):
    clean = torch.tensor([-.7, 1.2], dtype=dtype).reshape(1, 1, 1, 2).requires_grad_()
    lit = torch.tensor([.3, -.9], dtype=dtype).reshape(1, 1, 1, 2).requires_grad_()
    target = torch.tensor([0., 1.]).reshape_as(clean)
    loss = IlluminationConsistencyLoss()(lit, clean, target)
    p, q = clean.detach().double().sigmoid(), lit.detach().double().sigmoid()
    expected = (p*(p/q).log() + (1-p)*((1-p)/(1-q)).log()).mean()
    assert float(loss.detach()) == pytest.approx(float(expected), abs=1e-7)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    loss.backward()
    assert clean.grad is None and torch.isfinite(lit.grad).all()
    assert lit.grad[0, 0, 0, 0] > 0 and lit.grad[0, 0, 0, 1] < 0
    assert float(IlluminationConsistencyLoss()(clean, clean, target)) == 0.


@pytest.mark.parametrize('value', [0., 1.])
def test_single_class_and_saturated_wrong_predictions_still_have_gradient(value):
    clean = torch.full((2, 1, 16, 16), 100. if value else -100.)
    lit = (-clean).requires_grad_()
    loss = IlluminationConsistencyLoss()(lit, clean, torch.full_like(clean, value))
    assert torch.isfinite(loss) and float(loss.detach()) == pytest.approx(100.)
    loss.backward()
    assert float(lit.grad.sum()) == pytest.approx(-1. if value else 1.)


def test_tiny_foreground_consistency_is_not_diluted_by_canvas_size():
    gradients = []
    for size in [16, 512]:
        mask = torch.zeros(1, 1, size, size); mask[..., 2, 2] = 1
        clean = torch.where(mask.bool(), 5., -5.)
        lit = torch.full_like(clean, -5., requires_grad=True)
        IlluminationConsistencyLoss()(lit, clean, mask).backward()
        gradients.append(float(lit.grad[mask.bool()].sum()))
    assert gradients[0] < -.49 and gradients[0] == pytest.approx(gradients[1])


def test_legacy_brightness_can_erase_label_but_smooth_pair_preserves_it(protocol):
    protocol.update(contrast_range=[.04, .04])
    original = torch.full((3, 64, 64), .99)
    sample = NativeTinyDefectGenerator(protocol)(original, seed=42, defect_type='pinhole',
                                                size_bin='sub_patch', placement='interior')
    support = sample.mask.bool().expand_as(original)
    assert (sample.image-original)[support].abs().min() > .015
    assert torch.equal((1.1*sample.image).clamp(0, 1), (1.1*original).clamp(0, 1))
    change = smooth_illumination(sample.image, 1.1, 1., 0., 0.) - smooth_illumination(original, 1.1, 1., 0., 0.)
    assert change[support].abs().min() > .03
    ramp = torch.linspace(0, 1, 256).expand(3, 8, 256)
    lit = smooth_illumination(ramp, 1.1, .9, 0., 0.)
    assert lit.min() == 0 and lit.max() == 1 and (lit[..., 1:] > lit[..., :-1]).all()


def test_diverse_renderer_exact_support_reproducibility_and_polarity(protocol):
    image = torch.full((3, 64, 64), .7)
    generator = DiverseNativeDefectGenerator(protocol)
    appearances, signs, fallback = set(), set(), 0
    for seed in range(36):
        kwargs = dict(seed=seed, defect_type='pinhole', size_bin='sub_patch', placement='interior')
        a, b = generator(image, **kwargs), generator(image, **kwargs)
        support = a.mask.bool().expand_as(image)
        assert torch.equal(a.image, b.image) and torch.equal(a.mask, b.mask)
        assert torch.equal(a.image[~support], image[~support])
        assert ((a.image-image).abs().mean(0)[a.mask[0].bool()] >= .015 - 1e-7).all()
        assert torch.equal(a.mask, NativeTinyDefectGenerator(protocol)(image, **kwargs).mask)
        appearances.add(a.metadata['appearance'])
        if a.metadata['appearance'] == 'intensity':
            signs.add(float((a.image-image)[support].mean().sign()))
        fallback += a.metadata['visibility_fallback_pixels']
    assert appearances == set(generator.appearances) and signs == {-1., 1.} and fallback > 0
    normal = generator(image, **dict(kwargs, normal=True))
    assert torch.equal(normal.image, image) and not normal.mask.any()


def test_new_epoch_schedule_does_not_alias_with_source_count(source, protocol):
    # With one bin: 4 types x 2 placements = 8 combinations, 4 sources x 2 variants.
    dataset = G1NativeDataset([source]*4, protocol, seed=42, role='train', appearance='diverse_v1')
    old = G1NativeDataset([source]*4, protocol, seed=42, role='train')
    def combo(sample):
        s = sample['meta']['synthetic']; return s['defect_type'], s['placement'], s['size_bin']
    before, legacy = combo(dataset[1]), combo(old[1])
    dataset.set_epoch(1); old.set_epoch(1)
    assert combo(dataset[1]) != before and combo(old[1]) == legacy
    dataset.set_epoch(0)
    assert combo(dataset[1]) == before
    with pytest.raises(ValueError, match='TRAIN'):
        G1NativeDataset([source], protocol, seed=1, role='train', fixed=True, appearance='diverse_v1')


def test_clean_anchor_shared_pair_geometry_and_collation(source, protocol):
    dataset = IlluminatedNativeDataset([source], protocol, seed=42, role='train',
        appearance='diverse_v1', illumination=options(), illumination_mode='consistency')
    plain = G1NativeDataset([source], protocol, seed=42, role='train', appearance='diverse_v1')
    sample = dataset[1]
    assert torch.equal(sample['image'], plain[1]['image'])
    assert not torch.equal(sample['illumination_image'], sample['image'])
    assert sample['meta']['illumination']['visibility_ratio_min'] > .5
    tiles = G1TileDataset(dataset)
    paired = PairedG2Tiles(tiles)
    batch = paired_tile_collate([paired[0], paired[1]])
    assert batch['illumination_indices'].tolist() == [0, 1]
    assert torch.equal(batch['mask'], batch['illumination_batch']['mask'])
    local, context, _ = prepare_pair(sample['illumination_image'], paired_record(tiles.records[1][1]))
    assert torch.equal(batch['illumination_batch']['image'][1], local)
    assert torch.equal(batch['illumination_batch']['context'][1], context)
    dataset.set_epoch(1); dataset.set_epoch(0)
    assert torch.equal(dataset[1]['illumination_image'], sample['illumination_image'])


def test_invisible_pair_is_rejected_without_corrupting_supervision(source, protocol):
    # A valid but extreme future gamma setting collapses contrast toward zero.
    Image.fromarray(np.full((96, 128, 3), 102, dtype=np.uint8)).save(source.image_path)
    photometry = options(); photometry['contrast'] = [.01, .01]
    dataset = IlluminatedNativeDataset([source], protocol, seed=42, role='train',
        illumination=photometry, illumination_mode='consistency')
    sample = dataset[1]
    assert sample['illumination_image'] is None and sample['meta']['illumination']['rejected']
    assert sample['mask'].any() and not torch.equal(sample['image'], sample['original'])
    batch = paired_tile_collate([PairedG2Tiles(G1TileDataset(dataset))[1]])
    assert batch['illumination_batch'] is None and len(batch['illumination_indices']) == 0


def test_paired_data_is_identical_with_two_workers(source, protocol):
    from torch.utils.data import DataLoader
    dataset = IlluminatedNativeDataset([source], protocol, seed=42, role='train',
        appearance='diverse_v1', illumination=options(probability=.5), illumination_mode='consistency')
    paired = PairedG2Tiles(G1TileDataset(dataset, positive_tile_fraction=.5))
    for epoch in (0, 1):
        dataset.set_epoch(epoch)
        serial = list(DataLoader(paired, batch_size=2, num_workers=0, collate_fn=paired_tile_collate))
        workers = list(DataLoader(paired, batch_size=2, num_workers=2, collate_fn=paired_tile_collate))
        for a, b in zip(serial, workers):
            assert a['meta'] == b['meta']
            for name in ('image', 'context', 'mask', 'illumination_indices'):
                assert torch.equal(a[name], b[name])
            if a['illumination_batch'] is None:
                assert b['illumination_batch'] is None
            else:
                for name in ('image', 'context', 'mask'):
                    assert torch.equal(a['illumination_batch'][name], b['illumination_batch'][name])


class CountingConv(torch.nn.Conv2d):
    calls = 0
    def forward(self, image):
        self.calls += 1
        return super().forward(image)


@pytest.mark.parametrize('active,weight', [(True, .2), (False, .2), (True, 0.)])
def test_trainer_two_backwards_equal_joint_objective_single_update(active, weight):
    torch.manual_seed(42)
    model = CountingConv(1, 1, 1)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    criterion = AnomalySegmentationLoss(bce_mode='balanced')
    image = torch.rand(2, 1, 8, 8)
    mask = torch.zeros_like(image); mask[1, 0, 2:4, 2:4] = 1
    indices = torch.tensor([1] if active else [], dtype=torch.long)
    illuminated = dict(image=image[indices]*.8, mask=mask[indices]) if active else None
    batch = dict(image=image, mask=mask, illumination_indices=indices, illumination_batch=illuminated)
    clean = reference(image)
    objective = criterion(clean, mask)['loss']
    if active and weight:
        objective = objective + weight*.5*IlluminationConsistencyLoss()(
            reference(illuminated['image']), clean[indices], mask[indices])
    objective.backward(); reference_optimizer.step()
    trainer = Overfit16Trainer(model=model, criterion=criterion, optimizer=optimizer,
                               device='cpu', illumination_loss_weight=weight)
    log = trainer.train_step(batch, step=0, epoch=0)
    assert log.loss == pytest.approx(float(objective.detach()), abs=1e-6)
    assert log.loss == pytest.approx(log.segmentation_loss+log.illumination_contribution, abs=1e-6)
    assert model.calls == 1 + int(active and weight > 0)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b)
    if active and weight:
        assert log.illumination_loss > 0 and log.illumination_pairs == 1


@pytest.mark.parametrize('mode,weight', [('typo', 0), ('off', .1), ('consistency', 0),
                                        ('consistency', float('nan')), ('consistency', -.1)])
def test_invalid_illumination_objective_is_rejected(mode, weight):
    with pytest.raises(ValueError):
        validate_illumination_objective(mode, weight, options())
