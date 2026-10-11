"""TRAIN-only appearance diversity on the locked native defect supports.

This is a synthetic proxy, not a model of the real defect distribution. Keep
the legacy generator for reproducible DEV, and record fallback rendering so
flat images do not masquerade as successful texture transfers.
"""
import hashlib

import numpy as np
import torch

from .synthetic_anomaly import NativeTinyDefectGenerator, SyntheticAnomalySample


class DiverseNativeDefectGenerator(NativeTinyDefectGenerator):
    appearances = ('intensity', 'chromatic', 'texture_transfer')

    def __call__(self, image, **kwargs):
        geometry = super().__call__(image, **kwargs)
        meta = dict(geometry.metadata, appearance_version='diverse_v1')
        if kwargs.get('normal', False):
            return SyntheticAnomalySample(geometry.image, geometry.mask, meta)
        seed = int.from_bytes(hashlib.sha256(
            f"appearance_v1|{kwargs['seed']}".encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        appearance = self.appearances[int(rng.integers(len(self.appearances)))]
        x0, y0, x1, y1 = meta['bbox_xyxy']
        original = image[:, y0:y1, x0:x1]
        support = geometry.mask[:, y0:y1, x0:x1].bool()
        magnitude = float(rng.uniform(*self.protocol['contrast_range']))
        polarity = float(rng.choice([-1., 1.]))
        donor_box, alpha, flipped = None, None, False
        if appearance == 'intensity':
            candidate = original + polarity * magnitude
        elif appearance == 'chromatic':
            # Opposing channel shifts break the shared RGB brightness cue.
            directions = np.array([1., -1., rng.choice([-1., 1.])], dtype=np.float32)
            rng.shuffle(directions)
            shift = directions * rng.uniform(.75, 1., 3).astype(np.float32) * magnitude
            candidate = original + torch.as_tensor(shift, device=image.device)[:, None, None]
        else:
            h, w = image.shape[-2:]
            bh, bw = original.shape[-2:]
            dx, dy = int(rng.integers(w-bw+1)), int(rng.integers(h-bh+1))
            donor_box = [dx, dy, dx+bw, dy+bh]
            donor = image[:, dy:dy+bh, dx:dx+bw]
            flipped = bool(rng.random() < .5)
            if flipped:
                donor = donor.flip(-1)
            # Transfer local structure without turning every pasted patch into
            # an unrelated mean-brightness / colour block.
            donor = donor - donor.mean((-2, -1), keepdim=True) + original.mean((-2, -1), keepdim=True)
            alpha = float(rng.uniform(.35, .85))
            candidate = original.lerp(donor, alpha)
        candidate = candidate.clamp(0, 1)
        # Require each labelled pixel to change in RGB mean. Prefer randomly
        # chosen polarity; reverse only where saturation makes it invisible.
        minimum = self.protocol['min_mean_abs_change']
        weak = (candidate-original).abs().mean(0, keepdim=True) < minimum
        fallback = (original + polarity*magnitude).clamp(0, 1)
        invisible = (fallback-original).abs().mean(0, keepdim=True) < minimum
        fallback = torch.where(invisible, (original-polarity*magnitude).clamp(0, 1), fallback)
        candidate = torch.where(weak, fallback, candidate)
        # Reuse the legacy generator's owned canvas, replacing its entire bbox;
        # avoid a third full-resolution RGB allocation just to reuse its mask.
        output = geometry.image
        output[:, y0:y1, x0:x1] = torch.where(support, candidate, original)
        change = (candidate-original).abs().mean(0, keepdim=True)[support]
        if not bool((change >= minimum - 1e-7).all()):
            raise ValueError('Diverse synthetic visual signal below declared minimum')
        meta.update(appearance=appearance, appearance_seed=seed, contrast=magnitude,
                    requested_polarity=polarity, donor_bbox_xyxy=donor_box,
                    blend_alpha=alpha, donor_flipped=flipped,
                    visibility_fallback_pixels=int((weak & support).sum()),
                    mean_abs_change=float(change.mean()))
        return SyntheticAnomalySample(output, geometry.mask, meta)
