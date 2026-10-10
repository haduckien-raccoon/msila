"""E0 normal-feature memory and chunked exact nearest-neighbor scoring.

Sampling is a seeded streaming priority reservoir: storage is bounded by K
vectors plus one input chunk, without materializing the full TRAIN patch pool.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


DISTANCES = ("euclidean", "squared_euclidean", "cosine")


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _vectors(value, name):
    if (not isinstance(value, Tensor) or value.ndim != 2 or min(value.shape) < 1
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite nonempty floating [N,C] vectors")
    value = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be representable as finite float32 vectors")
    return value


def _unit_vectors(value):
    norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    if not torch.isfinite(norm).all() or (norm <= torch.finfo(value.dtype).tiny).any():
        raise ValueError("Normalized/cosine distance requires nonzero finite-norm vectors")
    return value / norm


class StreamingNormalMemory:
    def __init__(self, max_features: int, *, seed: int):
        _positive(max_features, "max_features")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.max_features, self.seed = max_features, seed
        self.rng = np.random.default_rng(seed)
        self.features = None
        self.priorities = np.empty(0, dtype=np.float64)
        self.indices = np.empty(0, dtype=np.int64)
        self.seen = 0

    def update(self, vectors: Tensor):
        vectors = _vectors(vectors, "normal features").cpu().contiguous()
        if self.features is not None and vectors.shape[1] != self.features.shape[1]:
            raise ValueError("Normal feature width changed during memory construction")
        count = len(vectors)
        priority = self.rng.random(count)
        indices = np.arange(self.seen, self.seen + count, dtype=np.int64)
        self.seen += count
        if self.features is None:
            merged = vectors
        else:
            merged = torch.cat((self.features, vectors))
        priority = np.concatenate((self.priorities, priority))
        indices = np.concatenate((self.indices, indices))
        # Lexicographic index tie-break is deterministic even if priorities tie.
        selected = np.lexsort((indices, priority))[:self.max_features].copy()
        self.features = merged[torch.from_numpy(selected)].contiguous()
        self.priorities, self.indices = priority[selected], indices[selected]

    def finalize(self):
        if self.features is None:
            raise ValueError("Cannot finalize an empty normal memory")
        order = np.argsort(self.indices).copy()
        return (self.features[torch.from_numpy(order)].contiguous(),
                torch.from_numpy(self.indices[order].copy()))


class NormalFeatureMemoryBank:
    """Exact 1-NN; ties keep the earliest bank vector across chunks.

    The bank has an explicit compute device, independent of the frozen DINO
    device. Maximum distance workspace is query_chunk_size * bank_chunk_size.
    """
    def __init__(self, vectors: Tensor, *, distance="euclidean", normalize=False,
                 query_chunk_size=256, bank_chunk_size=2048, device="cpu"):
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {DISTANCES}")
        if type(normalize) is not bool:
            raise ValueError("normalize must be boolean")
        _positive(query_chunk_size, "query_chunk_size")
        _positive(bank_chunk_size, "bank_chunk_size")
        self.distance, self.normalize = distance, normalize
        self.query_chunk_size, self.bank_chunk_size = query_chunk_size, bank_chunk_size
        self.raw_vectors = _vectors(vectors, "memory bank").cpu().contiguous()
        self.vectors = self.raw_vectors.to(device)
        if normalize or distance == "cosine":
            self.vectors = _unit_vectors(self.vectors)

    @torch.no_grad()
    def nearest(self, queries: Tensor, *, return_indices=False):
        queries = _vectors(queries, "query features")
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query and bank feature widths must match")
        target = queries.device
        distances, indices = [], []
        for start in range(0, len(queries), self.query_chunk_size):
            q = queries[start:start+self.query_chunk_size].to(self.vectors.device)
            if self.normalize or self.distance == "cosine":
                q = _unit_vectors(q)
            best = torch.full((len(q),), float("inf"), device=q.device)
            matched = torch.full((len(q),), -1, dtype=torch.int64, device=q.device)
            for offset in range(0, len(self.vectors), self.bank_chunk_size):
                bank = self.vectors[offset:offset+self.bank_chunk_size]
                if self.distance == "cosine":
                    distance = (1 - q @ bank.T).clamp(0, 2)
                else:
                    # Direct cdist avoids cancellation for identical large vectors.
                    distance = torch.cdist(q, bank, p=2, compute_mode="donot_use_mm_for_euclid_dist")
                    if self.distance == "squared_euclidean":
                        distance = distance.square()
                candidate, position = distance.min(dim=1)
                better = candidate < best
                best = torch.where(better, candidate, best)
                matched = torch.where(better, position + offset, matched)
            if not torch.isfinite(best).all():
                raise ValueError("Nearest-neighbor distance overflowed or is nonfinite")
            distances.append(best.to(target))
            indices.append(matched.to(target))
        distance = torch.cat(distances)
        return (distance, torch.cat(indices)) if return_indices else distance


def distance_to_score(distance: Tensor, *, scale: float):
    """Global fixed d/(d+s) in [0,1]; no per-image rescaling or calibration."""
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("score scale must be finite and positive")
    if not torch.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    return distance / (distance + scale)


class E0MemoryModel(nn.Module):
    """Frozen deepest DINO -> 1-NN patch distance -> interpolated tile score.

    Outputs are bounded distance scores, not Decoder logits or calibrated
    probabilities. No sigmoid, trainable parameters, Adapter or Decoder.
    """
    def __init__(self, extractor, bank: NormalFeatureMemoryBank, *, score_scale=1.0):
        super().__init__()
        if extractor.blocks != (extractor.depth,):
            raise ValueError("E0 requires one deepest frozen feature")
        if extractor.out_channels != bank.vectors.shape[1]:
            raise ValueError("Extractor and normal memory width mismatch")
        distance_to_score(torch.zeros(1), scale=score_scale)
        self.extractor = extractor
        self.extractor.requires_grad_(False)
        self.extractor.eval()
        self.bank, self.score_scale = bank, score_scale

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()
        return self

    @torch.no_grad()
    def forward(self, image, *, return_trace=False):
        features = self.extractor(image)
        key = f"b{self.extractor.depth}"
        if set(features) != {key}:
            raise ValueError("E0 must extract exactly the deepest DINO feature")
        feature = features[key]
        b, c, h, w = feature.shape
        query = feature.permute(0, 2, 3, 1).reshape(-1, c)
        distance = self.bank.nearest(query).reshape(b, 1, h, w)
        tile_distance = F.interpolate(distance, size=image.shape[-2:], mode="bilinear", align_corners=False)
        score = distance_to_score(tile_distance, scale=self.score_scale)
        if return_trace:
            return score, dict(feature=feature, patch_distance=distance, tile_distance=tile_distance)
        return score
