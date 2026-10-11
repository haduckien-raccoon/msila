"""Train-only native illumination and locality-preserving tile sampling.

Neither helper generates defects: the existing native synthetic generator owns
the image/mask pair. Both Local and Context subsequently crop this one image.
"""
import hashlib

import numpy as np
import torch
from torch.utils.data import Sampler

from src.data.loader import G1NativeDataset


def validate_illumination(options):
    if set(options) != {'enabled', 'probability', 'brightness', 'contrast', 'gradient'}:
        raise ValueError('Illumination requires enabled/probability/brightness/contrast/gradient')
    if type(options['enabled']) is not bool or not 0 <= options['probability'] <= 1:
        raise ValueError('Invalid illumination probability/enabled')
    for key in ('brightness', 'contrast'):
        bounds = options[key]
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] <= 2:
            raise ValueError(f'Invalid illumination {key} bounds')
    if not 0 <= options['gradient'] <= .3:
        raise ValueError('Illumination gradient must lie in [0, .3]')


class IlluminatedNativeDataset(G1NativeDataset):
    def __init__(self, *args, illumination, **kwargs):
        super().__init__(*args, **kwargs)
        validate_illumination(illumination)
        if self.role != 'train' or self.fixed:
            raise ValueError('Illumination is only permitted on changing TRAIN samples')
        self.illumination = illumination

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if 'illumination' in sample['meta']:
            return sample
        options = self.illumination
        seed = int.from_bytes(hashlib.sha256(
            f'illumination_v1|{self.seed}|{self.epoch}|{index}'.encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        active = options['enabled'] and rng.random() < options['probability']
        image = sample['image']
        parameters = dict(applied=bool(active), seed=seed, scope='one_native_image_both_views')
        if active:
            brightness = float(rng.uniform(*options['brightness']))
            contrast = float(rng.uniform(*options['contrast']))
            gx, gy = rng.uniform(-options['gradient'], options['gradient'], 2).tolist()
            h, w = image.shape[-2:]
            field = 1 + gx * torch.linspace(-1, 1, w)[None, :] + gy * torch.linspace(-1, 1, h)[:, None]
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = ((image - mean) * contrast + mean) * brightness * field
            image = image.clamp(0, 1)
            parameters.update(brightness=brightness, contrast=contrast, gradient_xy=[gx, gy])
        result = {**sample, 'image': image, 'meta': {**sample['meta'], 'illumination': parameters}}
        self.cache[(self.epoch, int(index))] = result
        return result


class NativeTileSampler(Sampler):
    """Visit every tile once; shuffle native samples and tiles within a sample.

    Grouping avoids regenerating the same high-resolution anomaly per randomly
    selected tile. The explicit epoch seed also makes cursor resume repeatable.
    """
    def __init__(self, tiles, seed):
        self.seed = seed
        self.groups = {}
        for index, (native_index, _) in enumerate(tiles.records):
            self.groups.setdefault(native_index, []).append(index)
        self.count = len(tiles)

    def __len__(self):
        return self.count

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        keys = list(self.groups)
        for key_index in rng.permutation(len(keys)):
            yield from rng.permutation(self.groups[keys[key_index]]).tolist()
