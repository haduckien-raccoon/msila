"""CPU fixtures verify public evaluation contracts, never pretrained performance."""
import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.eval.g1_test_public import evaluate_public, run_public
from src.train.g1_e1 import parse_args, resolve_config, run
from tests.test_g1_e1 import config_and_dataset


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class PixelContractFixture(nn.Module):
    def forward(self, image):
        return 50 * image[:, :1]


def public_fixture(tmp_path, *, missing=False, mismatch=False):
    root = tmp_path / 'data'
    h, w = 37, 53
    for defect in ('good', 'bad'):
        folder = root / 'rice/test_public' / defect
        folder.mkdir(parents=True)
        rgb = np.full((h, w, 3), 40, np.uint8)
        if defect == 'bad': rgb[-4:, -4:, 0] = 240
        Image.fromarray(rgb).save(folder / '0.png')
    if not missing:
        gt = root / 'rice/test_public/ground_truth/bad'; gt.mkdir(parents=True)
        mask = np.zeros((h // 2, w // 2) if mismatch else (h, w), np.uint8)
        mask[-4:, -4:] = 255
        Image.fromarray(mask).save(gt / '0_mask.png')
    cfg = resolve_config(parse_args(['--data-root', str(root)]))
    cfg['data'].update(tile_size=32, overlap=16)
    cfg['evaluation']['tile_batch_size'] = 2
    return cfg


def test_public_native_metric_and_artifacts(tmp_path):
    cfg = public_fixture(tmp_path)
    result = evaluate_public(PixelContractFixture(), cfg, 'cpu', tmp_path / 'out', example_limit=2)
    assert result['split'] == 'test_public' and result['synthetic'] is False
    assert result['test_public_aupro_0_05'] == pytest.approx(1.)
    assert result['n_images'] == 2 and result['n_normal'] == result['n_abnormal'] == 1
    assert result['native_resolution'] and result['qa_status'] == 'PASS'
    maps = sorted((tmp_path / 'out/examples').glob('*/predicted_map.npy'))
    assert len(maps) == 2
    score = np.load(maps[0])  # sorted bad before good
    assert score.shape == (37, 53) and np.isfinite(score).all()
    assert score[-1, -1] > .99 and score[0, 0] < .01
    assert not list((tmp_path / 'out').rglob('synthetic.png'))
    assert (tmp_path / 'out/source_manifest.json').is_file()


@pytest.mark.parametrize('kwargs,pattern', [({'missing': True}, 'pixel GT'), ({'mismatch': True}, 'shape mismatch')])
def test_bad_gt_is_required_in_native_coordinates(tmp_path, kwargs, pattern):
    cfg = public_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=pattern):
        evaluate_public(PixelContractFixture(), cfg, 'cpu', tmp_path / 'out')


def test_public_loads_selected_decoder_and_locks_training_provenance(tmp_path, monkeypatch):
    cfg, _ = config_and_dataset(tmp_path, monkeypatch)
    cfg['training'].update(mode='train', epochs=1, max_steps=2)
    output = Path(cfg['output_root']) / 'rice'; output.mkdir(parents=True)
    run(cfg, device='cpu', output_dir=output)
    data = Path(cfg['data']['root']) / 'rice/TEST_PUBLIC'
    gt = data / 'ground_truth/bad'; gt.mkdir(parents=True)
    mask = np.zeros((96, 128), np.uint8); mask[90:94, 122:126] = 255
    Image.fromarray(mask).save(gt / '0_mask.png')
    good = data / 'good'; good.mkdir()
    Image.fromarray(np.full((96, 128, 3), 77, np.uint8)).save(good / 'normal.png')
    result = run_public(copy.deepcopy(cfg), output / 'best.pt', 'cpu', output / 'test_public', example_limit=1)
    assert result['status'] == 'PASS' and result['n_images'] == 2
    assert result['checkpoint_sha256']
    changed = copy.deepcopy(cfg); changed['training']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='provenance mismatch'):
        run_public(changed, output / 'best.pt', 'cpu', output / 'wrong')
