import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval import compare_representation as m


def make_fixture(tmp_path):
    shape = (96, 128)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(30, 180, shape[1], dtype=np.uint8)
    image[..., 1] = np.linspace(40, 150, shape[0], dtype=np.uint8)[:, None]
    image[..., 2] = 90
    samples = []
    groups = [('00_other', 'other'), ('01_tiny', 'tiny'), ('02_boundary', 'boundary'),
              ('03_both', 'both'), ('04_tiny', 'tiny'), ('05_boundary', 'boundary')]
    for name, group in groups:
        gt = np.zeros(shape, dtype=np.uint8)
        if group == 'tiny': gt[44:46, 61:63] = 255
        elif group == 'boundary': gt[:4, 52:62] = 255
        elif group == 'both': gt[:2, 61:63] = 255
        else: gt[45:50, 55:63] = 255
        np.save(tmp_path / f'{name}_gt.npy', gt)
        visible = image.copy()
        visible[gt != 0] = [200, 100, 80]
        Image.fromarray(visible).save(tmp_path / f'{name}.png')
        for candidate in m.CANDIDATES:
            score = np.full(shape, .1, dtype=np.float32)
            score[gt > 0] = {'R0': .3, 'R1': .6, 'R2': .85}[candidate]
            np.save(tmp_path / f'{name}_{candidate}.npy', score)
        samples.append({'image_id': f'fabric/{name}', 'category': 'fabric', 'original_hw': list(shape),
                        'image_path': f'{name}.png', 'gt_mask': f'{name}_gt.npy',
                        'maps': {c: f'{name}_{c}.npy' for c in m.CANDIDATES}})
    manifest = {'split': 'dev_synthetic', 'categories': ['fabric'], 'seg_f1_threshold': .5,
                'normalization_by_candidate': {c: 'locked_probability' for c in m.CANDIDATES},
                'samples': samples}
    tiny = {'tiny_area_px': 4, 'area_unit': 'original_image_pixels', 'connectivity': 8,
            'max_fpr': .05, 'locked_before_candidate_results': True, 'threshold_basis': 'Synthetic QA only.'}
    boundary = {'boundary_mode': 'image_border_band', 'band_width_px': 3, 'connectivity': 8,
                'max_fpr': .05, 'prediction_threshold': .5, 'tolerance_px': 1,
                'locked_before_candidate_results': True, 'rule_basis': 'Synthetic QA only.'}
    def write():
        for name, data in [('manifest', manifest), ('tiny', tiny), ('boundary', boundary)]:
            (tmp_path / f'{name}.json').write_text(json.dumps(data))
    write()
    return manifest, tiny, boundary, write


def get_plans(tmp_path, manifest, tiny, boundary):
    return [m.sample_plan(s, tmp_path, tiny, boundary, 8) for s in manifest['samples']]


def run(tmp_path, limit=3):
    return m.compare(tmp_path / 'manifest.json', tmp_path / 'tiny.json', tmp_path / 'boundary.json',
                     tmp_path / 'results', limit, 8)


def test_gt_priority_and_interleaving_are_deterministic(tmp_path):
    manifest, tiny, boundary, _ = make_fixture(tmp_path)
    plans = get_plans(tmp_path, manifest, tiny, boundary)
    expected = ['fabric/03_both', 'fabric/01_tiny', 'fabric/02_boundary']
    assert [p['sample']['image_id'] for p in m.select_plans(plans, 3)] == expected
    assert [p['sample']['image_id'] for p in m.select_plans(list(reversed(plans)), 3)] == expected


def test_all_gt_selection_finishes_before_reading_any_candidate_map(tmp_path, monkeypatch):
    manifest, _, _, _ = make_fixture(tmp_path)
    events = []
    original_load = m.load_array
    def tracked(base, value):
        events.append(value)
        return original_load(base, value)
    monkeypatch.setattr(m, 'load_array', tracked)
    report = run(tmp_path, 1)
    first_map = next(i for i, p in enumerate(events) if p.endswith('_R0.npy'))
    assert all(s['gt_mask'] in events[:first_map] for s in manifest['samples'])
    assert report['samples'][0]['image_id'] == 'fabric/03_both'


def test_anomaly_maps_share_normalization_and_keep_values(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    make_fixture(tmp_path)
    figures = []
    original_savefig = plt.Figure.savefig
    def inspect(self, *args, **kwargs):
        figures.append(self)
        maps = [self.axes[i].images[0] for i in (2, 3, 4)]
        assert maps[0].norm is maps[1].norm is maps[2].norm
        assert all(artist.get_clim() == (0., 1.) for artist in maps)
        assert [artist.get_array().max() for artist in maps] == pytest.approx([.3, .6, .85])
        assert [a.get_title() for a in self.axes[:5]] == ['Image', 'GT', 'R0', 'R1', 'R2']
        assert maps[0].cmap(maps[0].norm(.5)) == maps[1].cmap(maps[1].norm(.5))
        return original_savefig(self, *args, **kwargs)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    run(tmp_path, 1)
    assert len(figures) == 2
    assert plt.get_fignums() == []


def test_same_gt_crop_for_all_five_panels(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    make_fixture(tmp_path)
    seen = []
    original_savefig = plt.Figure.savefig
    def inspect(self, *args, **kwargs):
        shapes = [self.axes[i].images[0].get_array().shape[:2] for i in range(5)]
        assert len(set(shapes)) == 1
        seen.append(shapes[0])
        return original_savefig(self, *args, **kwargs)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    report = run(tmp_path, 1)
    assert seen[0] == (96, 128)
    assert seen[1] == (10, 18)
    assert report['samples'][0]['crop_y0_y1_x0_x1'] == [0, 10, 53, 71]


def test_written_pngs_and_manifest_are_diagnostic_only(tmp_path):
    make_fixture(tmp_path)
    report = run(tmp_path)
    assert report['purpose'] == 'diagnostic_only'
    assert report['anomaly_map_range'] == [0., 1.]
    assert len(report['samples']) == 3
    assert len(list((tmp_path / 'results').glob('comparison_*.png'))) == 6
    assert json.loads((tmp_path / 'results' / 'comparison_manifest.json').read_text()) == report
    with Image.open(next((tmp_path / 'results').glob('comparison_*.png'))) as image:
        assert image.size[0] > image.size[1]
    with pytest.raises(FileExistsError): run(tmp_path)


def test_unsigned_rgb_16bit_uses_dtype_range_not_image_minmax(tmp_path):
    image = np.array([[0, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(image).save(tmp_path / 'image16.tiff')
    loaded = m.load_display_image(tmp_path, 'image16.tiff')
    assert loaded.shape == (1, 3, 3)
    assert loaded[0, 1, 0] == pytest.approx(32768 / 65535)
    Image.fromarray(np.full((2, 3, 3), 128, dtype=np.uint8)).save(tmp_path / 'constant.png')
    assert np.allclose(m.load_display_image(tmp_path, 'constant.png'), 128 / 255)


def test_changed_gt_after_selection_is_rejected(tmp_path):
    manifest, tiny, boundary, _ = make_fixture(tmp_path)
    plan = get_plans(tmp_path, manifest, tiny, boundary)[0]
    gt = np.load(tmp_path / plan['sample']['gt_mask'])
    gt[0, 0] = 255
    np.save(tmp_path / plan['sample']['gt_mask'], gt)
    with pytest.raises(ValueError, match='GT changed'):
        m.load_sample(plan, tmp_path)


def test_missing_candidate_file_cannot_influence_selection(tmp_path):
    make_fixture(tmp_path)
    (tmp_path / '03_both_R2.npy').unlink()
    with pytest.raises(FileNotFoundError):
        run(tmp_path, 1)
    assert not (tmp_path / 'results' / 'comparison_manifest.json').exists()


def test_normal_sample_fallback_has_no_fake_zoom(tmp_path):
    manifest, _, _, write = make_fixture(tmp_path)
    manifest['samples'] = [manifest['samples'][0]]
    sample = manifest['samples'][0]
    np.save(tmp_path / sample['gt_mask'], np.zeros(sample['original_hw'], dtype=np.uint8))
    write()
    report = run(tmp_path)
    assert report['samples'][0]['n_regions'] == 0
    assert report['samples'][0]['zoom_figure'] is None
    assert len(list((tmp_path / 'results').glob('comparison_*.png'))) == 1


@pytest.mark.parametrize('case', ['normalization', 'image_path', 'image_shape', 'gt_shape', 'score_range', 'score_nan', 'score_shape', 'tiny_unlocked', 'boundary_rule', 'duplicate', 'missing_map'])
def test_rejects_invalid_inputs(tmp_path, case):
    manifest, tiny, boundary, write = make_fixture(tmp_path)
    sample = manifest['samples'][3]  # Selected both-groups sample.
    if case == 'normalization': manifest['normalization_by_candidate']['R1'] = 'different'
    elif case == 'image_path': sample.pop('image_path')
    elif case == 'image_shape': Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / sample['image_path'])
    elif case == 'gt_shape': sample['original_hw'] = [95, 128]
    elif case == 'score_range': np.save(tmp_path / sample['maps']['R0'], np.full((96, 128), 1.2))
    elif case == 'score_nan': np.save(tmp_path / sample['maps']['R1'], np.full((96, 128), np.nan))
    elif case == 'score_shape': np.save(tmp_path / sample['maps']['R2'], np.ones((3, 4)))
    elif case == 'tiny_unlocked': tiny['locked_before_candidate_results'] = False
    elif case == 'boundary_rule': boundary['R2_band_width_px'] = 88
    elif case == 'duplicate': manifest['samples'].append(sample)
    else: sample['maps'].pop('R2')
    write()
    with pytest.raises(ValueError): run(tmp_path, 1)
    assert not (tmp_path / 'results' / 'comparison_manifest.json').exists()
