"""E0 math/geometry fixtures; these do not establish pretrained performance."""
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.data.loader import DINOV3_MEAN, DINOV3_STD
from src.data.tiling import generate_tile_records
from src.eval.e0 import predict_native_e0
from src.eval.e0_memory import (E0MemoryModel, NormalFeatureMemoryBank,
                                StreamingNormalMemory, distance_to_score)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("metric,expected", [
    ("euclidean", [5., 0.]), ("squared_euclidean", [25., 0.])])
@pytest.mark.parametrize("chunks", [(1, 1), (2, 10)])
def test_exact_distances_and_feature_matching(metric, expected, chunks):
    bank = NormalFeatureMemoryBank(torch.tensor([[0., 0.], [10., 0.], [10., 0.]]),
                                   distance=metric, query_chunk_size=chunks[0], bank_chunk_size=chunks[1])
    distance, match = bank.nearest(torch.tensor([[3., 4.], [10., 0.]]), return_indices=True)
    assert distance.tolist() == pytest.approx(expected)
    assert match.tolist() == [0, 1]


def test_cosine_and_normalized_euclidean():
    vectors = torch.tensor([[2., 0.], [-3., 0.]])
    query = torch.tensor([[0., 4.], [8., 0.], [-6., 0.]])
    cosine = NormalFeatureMemoryBank(vectors, distance="cosine", bank_chunk_size=1)
    distance, match = cosine.nearest(query, return_indices=True)
    assert distance.tolist() == pytest.approx([1., 0., 0.])
    assert match.tolist() == [0, 0, 1]
    unit_l2 = NormalFeatureMemoryBank(vectors, normalize=True)
    assert unit_l2.nearest(query).tolist() == pytest.approx([2**.5, 0., 0.])


@pytest.mark.parametrize("channels", [384, 768, 1024, 1280])
def test_identical_large_features_have_zero_distance(channels):
    vectors = torch.rand(40, channels, generator=torch.Generator().manual_seed(13)) * 1e4
    bank = NormalFeatureMemoryBank(vectors, query_chunk_size=17, bank_chunk_size=31)
    distance, match = bank.nearest(vectors, return_indices=True)
    assert torch.equal(distance, torch.zeros(40))
    assert torch.equal(match, torch.arange(40))


@pytest.mark.parametrize("vectors", [torch.empty(0, 3), torch.ones(3), torch.ones(2, 2, dtype=torch.int64),
                                       torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
                                       torch.tensor([[1e100]], dtype=torch.float64)])
def test_invalid_memory_features_rejected(vectors):
    with pytest.raises(ValueError):
        NormalFeatureMemoryBank(vectors)


def test_invalid_queries_settings_and_zero_normalization():
    bank = NormalFeatureMemoryBank(torch.ones(3, 2))
    for query in (torch.ones(1, 3), torch.empty(0, 2), torch.tensor([[float("inf"), 1.]])):
        with pytest.raises(ValueError):
            bank.nearest(query)
    for kwargs in (dict(distance="l1"), dict(query_chunk_size=0), dict(bank_chunk_size=True), dict(normalize=1)):
        with pytest.raises(ValueError):
            NormalFeatureMemoryBank(torch.ones(3, 2), **kwargs)
    with pytest.raises(ValueError, match="nonzero"):
        NormalFeatureMemoryBank(torch.zeros(1, 2), distance="cosine")
    with pytest.raises(ValueError, match="nonzero"):
        NormalFeatureMemoryBank(torch.ones(1, 2), normalize=True).nearest(torch.zeros(1, 2))


def sampled(vectors, seed, chunks, cap=13):
    sampler = StreamingNormalMemory(cap, seed=seed)
    offset = 0
    for size in chunks:
        sampler.update(vectors[offset:offset+size])
        offset += size
        assert len(sampler.features) <= cap
    assert offset == len(vectors) and sampler.seen == len(vectors)
    return sampler.finalize()


def test_sampling_bounded_reproducible_independent_of_batch_partition():
    vectors = torch.arange(300, dtype=torch.float32).reshape(100, 3)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    first, ids = sampled(vectors, 2026, [100])
    repeat, repeat_ids = sampled(vectors, 2026, [1] * 100)
    uneven, uneven_ids = sampled(vectors, 2026, [3, 17, 4, 76])
    assert torch.equal(first, repeat) and torch.equal(first, uneven)
    assert torch.equal(ids, repeat_ids) and torch.equal(ids, uneven_ids)
    assert torch.equal(first, vectors[ids]) and torch.all(ids[1:] > ids[:-1])
    assert not torch.equal(ids, sampled(vectors, 2027, [100])[1])
    assert torch.equal(torch_before, torch.get_rng_state())
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])


def test_sampling_keeps_entire_pool_when_below_limit():
    vectors = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    kept, ids = sampled(vectors, 7, [2, 3], cap=100)
    assert torch.equal(kept, vectors) and ids.tolist() == list(range(5))
    sampler = StreamingNormalMemory(2, seed=0)
    with pytest.raises(ValueError, match="empty"):
        sampler.finalize()
    sampler.update(torch.ones(2, 3))
    with pytest.raises(ValueError, match="width"):
        sampler.update(torch.ones(2, 4))


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), True])
def test_invalid_score_scale_rejected(scale):
    with pytest.raises(ValueError):
        distance_to_score(torch.ones(2), scale=scale)


def test_fixed_bounded_distance_transform_has_no_image_rescaling():
    assert distance_to_score(torch.tensor([0., 1., 3.]), scale=1).tolist() == [0., .5, .75]
    assert distance_to_score(torch.tensor([0., 1., 3.]), scale=2).tolist() == pytest.approx([0., 1/3, .6])
    for value in (-1., float("inf"), float("nan")):
        with pytest.raises(ValueError):
            distance_to_score(torch.tensor([value]), scale=1)


class TinyExtractor(nn.Module):
    depth, blocks, out_channels, patch_size = 12, (12,), 3, 16

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, image):
        return {"b12": F.avg_pool2d(image, 16) * self.scale}


def test_model_freeze_patch_match_bilinear_score_and_no_sigmoid():
    extractor = TinyExtractor()
    bank = NormalFeatureMemoryBank(torch.zeros(1, 3))
    model = E0MemoryModel(extractor, bank, score_scale=2)
    model.train()
    assert not extractor.training
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, :16, :16] = 1
    image.requires_grad_(True)
    score, trace = model(image, return_trace=True)
    expected_patch = torch.tensor([[[[3**.5, 0.], [0., 0.]]]])
    expected_distance = F.interpolate(expected_patch, (32, 32), mode="bilinear", align_corners=False)
    assert torch.allclose(trace["patch_distance"], expected_patch)
    assert torch.allclose(trace["tile_distance"], expected_distance)
    assert torch.allclose(score, expected_distance/(expected_distance+2))
    assert score.shape == (1, 1, 32, 32) and not score.requires_grad
    assert score[0, 0, -1, -1] == 0  # A sigmoid would incorrectly turn zero into .5.
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert set(model._modules) == {"extractor"}


def test_model_rejects_wrong_feature_contract():
    extractor = TinyExtractor()
    with pytest.raises(ValueError, match="width"):
        E0MemoryModel(extractor, NormalFeatureMemoryBank(torch.ones(2, 4)))
    extractor.blocks = (4, 8, 12)
    with pytest.raises(ValueError, match="deepest"):
        E0MemoryModel(extractor, NormalFeatureMemoryBank(torch.ones(2, 3)))


class OriginalRedScore(nn.Module):
    def forward(self, tile):
        return tile[:, :1] * DINOV3_STD[0] + DINOV3_MEAN[0]


@pytest.mark.parametrize("hw", [(73, 101), (512, 512), (613, 777)])
def test_native_coordinates_no_final_resize_padding_or_sigmoid(hw):
    cfg = dict(data=dict(tile_size=512, overlap=128), evaluation=dict(tile_batch_size=2))
    image = torch.rand(3, *hw, generator=torch.Generator().manual_seed(12))
    score = predict_native_e0(OriginalRedScore(), image, cfg, "cpu")
    assert score.shape == hw
    assert torch.allclose(score, image[0], atol=3e-7)


def test_overlapping_tiles_use_hann_not_uniform_weights():
    cfg = dict(data=dict(tile_size=512, overlap=128), evaluation=dict(tile_batch_size=2))
    h, w = 613, 777
    image = torch.ones(3, h, w) * .5
    records = generate_tile_records(h, w, context_size=512)

    class TileCounter(nn.Module):
        def __init__(self):
            super().__init__()
            self.index = 0

        def forward(self, tile):
            values = torch.arange(self.index+1, self.index+1+len(tile), dtype=torch.float32) / 10
            self.index += len(tile)
            return values[:, None, None, None].expand(-1, 1, 512, 512)

    score = predict_native_e0(TileCounter(), image, cfg, "cpu")
    window = torch.outer(torch.hann_window(512, periodic=False), torch.hann_window(512, periodic=False)).clamp_min(.001)
    reference, weight = torch.zeros(h, w), torch.zeros(h, w)
    uniform, count = torch.zeros(h, w), torch.zeros(h, w)
    for index, record in enumerate(records):
        x0, y0, x1, y1 = record.local_xyxy
        reference[y0:y1, x0:x1] += (index+1)/10 * window
        weight[y0:y1, x0:x1] += window
        uniform[y0:y1, x0:x1] += (index+1)/10
        count[y0:y1, x0:x1] += 1
    assert torch.allclose(score, reference/weight, atol=1e-7)
    assert not torch.allclose(score, uniform/count, atol=.001)
