"""Regression probes for tiny-mask collapse; no pretrained accuracy claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.loader import G1NativeDataset, G1TileDataset, g1_tile_collate
from src.losses.anomaly_loss import AnomalySegmentationLoss, dice_loss_from_logits
from src.train.g2_context import PairedG2Tiles


@pytest.fixture(autouse=True)
def small_cpu_probe():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def tiny_mask(size=512):
    target = torch.zeros(1, 1, size, size)
    target[:, :, size//2:size//2+2, size//2:size//2+2] = 1
    return target


def test_balanced_loss_rejects_black_solution_and_revives_shared_bias():
    target = tiny_mask()
    black = torch.full_like(target, -10)
    separated = torch.where(target.bool(), 2.0, -2.0)
    legacy = AnomalySegmentationLoss()
    balanced = AnomalySegmentationLoss(bce_mode='balanced')
    assert legacy(black, target)['loss'] < legacy(separated, target)['loss']
    assert balanced(black, target)['loss'] > balanced(separated, target)['loss']
    bias = torch.tensor(-10.0, requires_grad=True)
    balanced(bias.expand_as(target), target)['loss'].backward()
    assert bias.grad < -0.49  # Gradient descent increases a collapsed bias.


def test_foreground_bce_gradient_does_not_dilute_with_image_area():
    gradients = []
    for size in (32, 512):
        target = tiny_mask(size)
        logits = torch.zeros_like(target, requires_grad=True)
        out = AnomalySegmentationLoss(bce_mode='balanced', dice_weight=0)(logits, target)
        out['loss'].backward()
        gradients.append(float(logits.grad[target.bool()].sum()))
        assert float(logits.grad[~target.bool()].sum()) == pytest.approx(.25)
    assert gradients == pytest.approx([-.25, -.25])


@pytest.mark.parametrize('value', [0.0, 1.0])
def test_single_class_masks_keep_full_bce_supervision(value):
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.full_like(logits, value)
    loss = AnomalySegmentationLoss(bce_mode='balanced', dice_weight=0)(logits, target)['loss']
    assert float(loss.detach()) == pytest.approx(np.log(2))
    loss.backward()
    assert float(logits.grad.sum()) == pytest.approx(.5 - value)


def test_mixed_batch_balances_each_sample_instead_of_pooling_normal_images():
    target = torch.cat([tiny_mask(32), torch.zeros(1, 1, 32, 32)])
    logits = torch.cat([torch.full((1, 1, 32, 32), -4.), torch.zeros(1, 1, 32, 32)])
    out = AnomalySegmentationLoss(bce_mode='balanced', dice_weight=0)(logits, target)
    expected = (.5 * (torch.nn.functional.softplus(torch.tensor(4.)) +
                       torch.nn.functional.softplus(torch.tensor(-4.))) + np.log(2)) / 2
    assert float(out['bce']) == pytest.approx(float(expected))
    assert int(out['positive_samples']) == 1


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['mean', 'balanced'])
def test_large_reduced_precision_masks_match_fp32_and_keep_dice_gradient(dtype, mode):
    target = tiny_mask()
    logits = torch.zeros_like(target, dtype=dtype, requires_grad=True)
    criterion = AnomalySegmentationLoss(bce_mode=mode)
    actual = criterion(logits, target)
    reference = criterion(logits.detach().float(), target)
    assert actual['loss'].dtype == torch.float32
    assert torch.equal(actual['loss'], reference['loss'])
    dice_gradient = torch.autograd.grad(actual['dice'], logits, retain_graph=True)[0]
    assert torch.isfinite(dice_gradient).all()
    assert dice_gradient[target.bool()].abs().sum() > 0
    actual['loss'].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[target.bool()].sum() < 0


def test_empty_half_precision_dice_is_differentiable_zero_without_overflow():
    logits = torch.full((1, 1, 512, 512), -10., dtype=torch.float16, requires_grad=True)
    dice, positives = dice_loss_from_logits(logits, torch.zeros_like(logits))
    assert int(positives) == 0 and float(dice) == 0
    dice.backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize('options', [dict(bce_mode='typo'), dict(bce_weight=float('nan')),
    dict(dice_weight=float('inf')), dict(dice_eps=float('nan'))])
def test_invalid_loss_options_fail_before_training(options):
    with pytest.raises(ValueError):
        AnomalySegmentationLoss(**options)


def test_near_binary_float64_masks_are_not_rounded_into_valid_targets():
    target = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    target[0, 0, 0, 0] = 1.0 - 1e-9
    with pytest.raises(ValueError, match='binary'):
        AnomalySegmentationLoss()(torch.zeros_like(target), target)


def test_five_epoch_low_contrast_tiny_signal_learns_instead_of_black_predictions():
    """Pixel readout fixture isolates imbalance from backbone/domain quality."""
    target = tiny_mask(128).expand(4, -1, -1, -1)
    image = .1 * target
    outcomes = {}
    for mode in ('mean', 'balanced'):
        torch.manual_seed(42)
        model = torch.nn.Conv2d(1, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.05)
        criterion = AnomalySegmentationLoss(bce_mode=mode)
        for _ in range(5):
            for _ in range(20):
                optimizer.zero_grad()
                criterion(model(image), target)['loss'].backward()
                optimizer.step()
        prediction = model(image).detach().sigmoid() >= .5
        outcomes[mode] = (int(prediction[target.bool()].sum()), int(prediction[~target.bool()].sum()))
    assert outcomes['mean'][0] == 0
    assert outcomes['balanced'] == (int(target.sum()), 0)


@pytest.fixture
def native(tmp_path):
    image = tmp_path / 'native.png'
    Image.fromarray(np.full((1024, 1536, 3), 128, dtype=np.uint8)).save(image)
    protocol = yaml.safe_load((Path(__file__).resolve().parents[1] /
                               'configs/full_scale_synthetic.yaml').read_text())
    protocol.update(train_core_bins=['sub_patch'])
    source = SimpleNamespace(image_path=str(image), split='train', defect_type='good')
    return G1NativeDataset([source], protocol, seed=42, role='train', variants=1)


def tile_signature(tiles):
    return [(sample['meta']['sample_id'], record.local_xyxy)
            for sample, record in (tiles.native_sample_and_record(i) for i in range(len(tiles)))]


def test_positive_sampling_preserves_budget_normals_and_pair_geometry(native):
    legacy = G1TileDataset(native)
    tiles = G1TileDataset(native, positive_tile_fraction=.5)
    paired = PairedG2Tiles(tiles)
    assert len(tiles) == len(legacy)
    indices = [i for i, (n, _) in enumerate(tiles.records) if n == 1]
    positive = [i for i in indices if bool(tiles[i]['mask'].any())]
    old_positive = [i for i in indices if bool(legacy[i]['mask'].any())]
    assert len(positive) >= len(indices) / 2 and len(old_positive) < len(indices) / 2
    assert any(not bool(tiles[i]['mask'].any()) for i in indices)
    for i, (n, original_record) in enumerate(tiles.records):
        if n == 0:
            assert tiles.native_sample_and_record(i)[1] == original_record
    i = positive[0]
    assert torch.equal(paired[i]['mask'], tiles[i]['mask'])
    assert torch.equal(paired[i]['image'], tiles[i]['image'])
    assert paired[i]['view_meta']['local_native_xyxy'] == tiles[i]['meta']['tile']['local_xyxy']
    first = tile_signature(tiles)
    native.set_epoch(1)
    assert tile_signature(tiles) != first
    native.set_epoch(0)
    assert tile_signature(tiles) == first


def test_positive_sampling_order_and_masks_are_independent_of_worker_count(native):
    tiles = G1TileDataset(native, positive_tile_fraction=.5)
    def observations(workers):
        native.cache.clear()
        tiles._sampling_cache.clear()
        # Spawn avoids inheriting PyTorch thread pools from earlier CPU probes.
        loader = DataLoader(tiles, batch_size=4, num_workers=workers, collate_fn=g1_tile_collate,
                            multiprocessing_context='spawn' if workers else None,
                            timeout=30 if workers else 0)
        return [(meta['sample_id'], meta['tile']['local_xyxy'], int(mask.sum()))
                for batch in loader for meta, mask in zip(batch['meta'], batch['mask'])]
    assert observations(0) == observations(2)


def test_positive_sampling_rejects_dev_and_fixed_sources(native):
    for key in ('role', 'fixed'):
        invalid = deepcopy(native)
        setattr(invalid, key, 'dev' if key == 'role' else True)
        with pytest.raises(ValueError, match='TRAIN'):
            G1TileDataset(invalid, positive_tile_fraction=.5)
