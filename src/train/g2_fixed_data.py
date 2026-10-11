"""Train-only native photometry, consistency pairs and grouped tile sampling."""
import hashlib
import math

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


def validate_illumination_objective(mode, weight, options):
    validate_illumination(options)
    if mode not in {'off', 'augmentation', 'consistency'}:
        raise ValueError('Unknown illumination mode')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('illumination_loss_weight must be finite and nonnegative')
    if (mode == 'consistency') != (weight > 0):
        raise ValueError('A positive illumination loss weight requires consistency mode and vice versa')
    if mode == 'consistency' and (not options['enabled'] or options['probability'] == 0):
        raise ValueError('Consistency mode requires enabled illumination with nonzero probability')


def _exposure_gamma(image, gain, contrast):
    base = image.pow(1 / contrast)
    return gain * base / (1 + (gain-1) * base)


def smooth_illumination(image, brightness, contrast, gx, gy):
    """Monotone exposure/gamma field mapping [0,1] into [0,1] without clipping."""
    h, w = image.shape[-2:]
    x = torch.linspace(-1, 1, w, device=image.device)[None, :]
    y = torch.linspace(-1, 1, h, device=image.device)[:, None]
    gain = brightness * (gx*x + gy*y).exp()
    return _exposure_gamma(image, gain, contrast)


class IlluminatedNativeDataset(G1NativeDataset):
    def __init__(self, *args, illumination, illumination_mode='augmentation', **kwargs):
        super().__init__(*args, **kwargs)
        validate_illumination(illumination)
        if self.role != 'train' or self.fixed:
            raise ValueError('Illumination is only permitted on changing TRAIN samples')
        self.illumination = illumination
        if illumination_mode not in {'off', 'augmentation', 'consistency'}:
            raise ValueError('Unknown illumination mode')
        self.illumination_mode = illumination_mode

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if 'illumination' in sample['meta']:
            return sample
        options = self.illumination
        seed = int.from_bytes(hashlib.sha256(
            f'illumination_v1|{self.seed}|{self.epoch}|{index}'.encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        active = self.illumination_mode != 'off' and options['enabled'] and rng.random() < options['probability']
        image = sample['image']
        parameters = dict(applied=bool(active), seed=seed, scope='one_native_image_both_views',
                          mode=self.illumination_mode, rejected=False)
        illuminated = None
        if active:
            brightness = float(rng.uniform(*options['brightness']))
            contrast = float(rng.uniform(*options['contrast']))
            gx, gy = rng.uniform(-options['gradient'], options['gradient'], 2).tolist()
            parameters.update(brightness=brightness, contrast=contrast, gradient_xy=[gx, gy])
            if self.illumination_mode == 'augmentation':
                # Explicit legacy ablation only. New training uses the clean
                # supervised anchor plus a separately weighted consistency pair.
                h, w = image.shape[-2:]
                field = 1 + gx * torch.linspace(-1, 1, w)[None, :] + gy * torch.linspace(-1, 1, h)[:, None]
                mean = image.mean(dim=(-2, -1), keepdim=True)
                image = (((image-mean)*contrast+mean)*brightness*field).clamp(0, 1)
                parameters['policy'] = 'legacy_clamp_v1'
            else:
                illuminated = smooth_illumination(image, brightness, contrast, gx, gy)
                parameters['policy'] = 'smooth_exposure_gamma_v1'
                support = sample['mask'][0].bool()
                if support.any():
                    original = sample['original']
                    # Visibility needs only labelled pixels, not another full
                    # native RGB canvas per worker. Use the identical field.
                    ys, xs = torch.where(support)
                    h, w = image.shape[-2:]
                    x = torch.linspace(-1, 1, w, device=image.device)[xs]
                    y = torch.linspace(-1, 1, h, device=image.device)[ys]
                    gain = brightness * (gx*x + gy*y).exp()
                    clean_lit = _exposure_gamma(original[:, support], gain, contrast)
                    before = (image-original).abs().mean(0)[support]
                    after = (illuminated[:, support]-clean_lit).abs().mean(0)
                    ratio = float((after / before.clamp_min(1e-8)).min())
                    parameters['visibility_ratio_min'] = ratio
                    if ratio < .5 or float(after.min()) < self.generator.protocol['min_mean_abs_change'] * .5:
                        illuminated = None
                        parameters.update(applied=False, rejected=True)
        result = {**sample, 'image': image, 'meta': {**sample['meta'], 'illumination': parameters}}
        if self.illumination_mode == 'consistency':
            result['illumination_image'] = illuminated
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
