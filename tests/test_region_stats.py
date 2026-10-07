import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label

spec = importlib.util.spec_from_file_location('region_stats_delivery', Path(__file__).with_name('region_stats.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_data(tmp_path):
    gt = np.zeros((12, 14), dtype=np.uint8)
    gt[0, 4:6] = 255       # Region 1: area 2, tiny and boundary.
    gt[6:8, 8:11] = 255    # Region 2: area 6, non-tiny and interior.
    np.save(tmp_path / 'gt.npy', gt)
    for candidate in m.CANDIDATES:
        score = np.full(gt.shape, .1)
        score[0, 4:6] = {'R0': [.7, .3], 'R1': [.8, .5], 'R2': [.9, .9]}[candidate]
        score[6:8, 8:11] = {'R0': .2, 'R1': .6, 'R2': .4}[candidate]
        np.save(tmp_path / f'{candidate}.npy', score)
    manifest = {
        'split': 'dev_synthetic', 'categories': ['fabric'], 'seg_f1_threshold': .5,
        'normalization_by_candidate': {c: 'locked_probability' for c in m.CANDIDATES},
        'samples': [{'image_id': 'fabric/a', 'category': 'fabric', 'original_hw': list(gt.shape),
                     'gt_mask': 'gt.npy', 'maps': {c: f'{c}.npy' for c in m.CANDIDATES}}],
    }
    tiny = {'tiny_area_px': 2, 'area_unit': 'original_image_pixels', 'connectivity': 8,
            'max_fpr': .05, 'locked_before_candidate_results': True,
            'threshold_basis': 'Synthetic test fixture; not a recommended research threshold.'}
    boundary = {'boundary_mode': 'image_border_band', 'band_width_px': 2,
                'connectivity': 8, 'max_fpr': .05, 'prediction_threshold': .5,
                'tolerance_px': 0, 'locked_before_candidate_results': True, 'rule_basis': 'Test fixture.'}
    def write():
        for name, value in [('manifest', manifest), ('tiny', tiny), ('boundary', boundary)]:
            (tmp_path / f'{name}.json').write_text(json.dumps(value))
    write()
    return manifest, tiny, boundary, write, gt


def run(tmp_path):
    return m.analyze(tmp_path / 'manifest.json', tmp_path / 'tiny.json', tmp_path / 'boundary.json')


def test_one_wide_row_per_gt_component_with_exact_metrics(tmp_path):
    fixture_data(tmp_path)
    rows = run(tmp_path)
    assert len(rows) == 2  # Not 2x3 candidate rows; not a dataset mean.
    assert [r['region_id'] for r in rows] == [1, 2]
    assert [r['area'] for r in rows] == [2, 6]
    assert [r['is_tiny'] for r in rows] == [True, False]
    assert [r['is_boundary'] for r in rows] == [True, False]
    assert all(r['boundary_flag'] == r['is_boundary'] for r in rows)
    assert rows[0]['R0_metric'] == .5
    assert rows[0]['R1_metric'] == rows[0]['R2_metric'] == 1
    assert rows[1]['R0_metric'] == rows[1]['R2_metric'] == 0
    assert rows[1]['R1_metric'] == 1
    assert rows[0]['R0_mean_score'] == pytest.approx(.5)
    assert rows[0]['R0_max_score'] == pytest.approx(.7)
    assert rows[0]['R0_tp'] == rows[0]['R0_fn'] == 1
    assert all(set(r) == set(m.FIELDS) for r in rows)


def test_matches_direct_per_component_oracle():
    rng = np.random.default_rng(771)
    for _ in range(10):
        gt = rng.random((10, 11)) > .7
        labs, n = label(gt, structure=np.ones((3, 3)))
        areas = np.bincount(labs.ravel(), minlength=n+1)
        scores = rng.integers(0, 5, size=gt.shape) / 4
        result = m.component_score_stats(scores, labs, areas, .5)
        for rid in range(1, n+1):
            values = scores[labs == rid]
            assert result[rid]['metric'] == pytest.approx((values >= .5).mean())
            assert result[rid]['mean_score'] == pytest.approx(values.mean())
            assert result[rid]['max_score'] == pytest.approx(values.max())
            assert result[rid]['tp'] + result[rid]['fn'] == len(values)


def test_eight_connectivity_and_same_ids_across_candidates(tmp_path):
    _, _, _, _, gt = fixture_data(tmp_path)
    gt[:] = 0
    gt[0, 4] = gt[1, 5] = 255
    np.save(tmp_path / 'gt.npy', gt)
    rows = run(tmp_path)
    assert len(rows) == 1 and rows[0]['area'] == 2
    assert rows[0]['is_tiny'] and rows[0]['is_boundary']


def test_supplied_zone_reuses_boundary_definition(tmp_path):
    manifest, _, boundary, write, gt = fixture_data(tmp_path)
    zone = np.zeros_like(gt)
    zone[7, 9] = 255
    np.save(tmp_path / 'zone.npy', zone)
    manifest['samples'][0]['boundary_zone_mask'] = 'zone.npy'
    boundary['boundary_mode'] = 'provided_zone_mask'
    boundary['band_width_px'] = None
    write()
    rows = run(tmp_path)
    assert [r['is_boundary'] for r in rows] == [False, True]
    assert rows[1]['area'] == 6 and rows[1]['boundary_overlap_px'] == 1


def test_region_id_is_local_and_image_id_makes_unique_key(tmp_path):
    manifest, _, _, write, _ = fixture_data(tmp_path)
    another = dict(manifest['samples'][0], image_id='fabric/b')
    manifest['samples'].append(another)
    write()
    rows = run(tmp_path)
    assert len(rows) == 4
    assert len({(r['image_id'], r['region_id']) for r in rows}) == 4


def test_normal_images_have_no_phantom_regions_but_maps_are_validated(tmp_path):
    _, _, _, _, gt = fixture_data(tmp_path)
    np.save(tmp_path / 'gt.npy', np.zeros_like(gt))
    assert run(tmp_path) == []
    np.save(tmp_path / 'R2.npy', np.full(gt.shape, np.nan))
    with pytest.raises(ValueError):
        run(tmp_path)


def test_high_score_everywhere_is_recall_not_precision(tmp_path):
    _, _, _, _, gt = fixture_data(tmp_path)
    np.save(tmp_path / 'R0.npy', np.ones_like(gt, dtype=float))
    assert all(row['R0_metric'] == 1 for row in run(tmp_path))


@pytest.mark.parametrize('case', ['tiny_unlocked', 'tiny_threshold', 'tiny_candidate_override', 'boundary_candidate_override', 'norm', 'duplicate_id', 'missing_R2', 'shape', 'nonbinary_gt', 'nan_score', 'out_of_range', 'hidden_split', 'different_score_threshold'])
def test_invalid_inputs_fail(tmp_path, case):
    manifest, tiny, boundary, write, gt = fixture_data(tmp_path)
    if case == 'tiny_unlocked': tiny['locked_before_candidate_results'] = False
    elif case == 'tiny_threshold': tiny['tiny_area_px'] = None
    elif case == 'tiny_candidate_override': tiny['R2_tiny_area_px'] = 99
    elif case == 'boundary_candidate_override': boundary['R2_band_width_px'] = 99
    elif case == 'norm': manifest['normalization_by_candidate']['R1'] = 'other'
    elif case == 'duplicate_id': manifest['samples'].append(manifest['samples'][0])
    elif case == 'missing_R2': manifest['samples'][0]['maps'].pop('R2')
    elif case == 'shape': manifest['samples'][0]['original_hw'] = [11, 14]
    elif case == 'nonbinary_gt': np.save(tmp_path / 'gt.npy', np.full(gt.shape, 7))
    elif case == 'nan_score': np.save(tmp_path / 'R0.npy', np.full(gt.shape, np.nan))
    elif case == 'out_of_range': np.save(tmp_path / 'R0.npy', np.full(gt.shape, 1.1))
    elif case == 'hidden_split': manifest['split'] = 'test_private'
    else: boundary['prediction_threshold'] = .8
    write()
    with pytest.raises(ValueError):
        run(tmp_path)


def test_cli_exports_region_rows_and_refuses_overwrite(tmp_path, monkeypatch):
    fixture_data(tmp_path)
    out = tmp_path / 'region_stats.csv'
    argv = ['region_stats', '--manifest', str(tmp_path / 'manifest.json'),
            '--tiny-protocol', str(tmp_path / 'tiny.json'),
            '--boundary-protocol', str(tmp_path / 'boundary.json'), '--output', str(out)]
    monkeypatch.setattr('sys.argv', argv)
    m.main()
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[0]['region_id'] == '1'
    assert float(rows[0]['R0_metric']) == .5
    with pytest.raises(FileExistsError):
        m.main()


def test_no_defects_exports_header_only(tmp_path, monkeypatch):
    _, _, _, _, gt = fixture_data(tmp_path)
    np.save(tmp_path / 'gt.npy', np.zeros_like(gt))
    out = tmp_path / 'region_stats.csv'
    monkeypatch.setattr('sys.argv', ['region_stats', '--manifest', str(tmp_path / 'manifest.json'),
                                   '--tiny-protocol', str(tmp_path / 'tiny.json'),
                                   '--boundary-protocol', str(tmp_path / 'boundary.json'), '--output', str(out)])
    m.main()
    with out.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == m.FIELDS
        assert list(reader) == []
