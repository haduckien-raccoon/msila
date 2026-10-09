"""Official DINOv3 ViT contracts, checked against each loaded model.

Source: https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py
Registry verified 2026-10-09. Checkpoint paths remain explicit local assets.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BackboneSpec:
    channels: int
    depth: int
    patch_size: int = 16

    @property
    def blocks(self):
        return (round(self.depth / 3), round(2 * self.depth / 3), self.depth)


BACKBONES = {
    "dinov3_vits16": BackboneSpec(384, 12),
    "dinov3_vits16plus": BackboneSpec(384, 12),
    "dinov3_vitb16": BackboneSpec(768, 12),
    "dinov3_vitl16": BackboneSpec(1024, 24),
    "dinov3_vith16plus": BackboneSpec(1280, 32),
}


def backbone_spec(name):
    if name not in BACKBONES:
        raise ValueError(f"Unsupported DINOv3 backbone {name!r}; choose {list(BACKBONES)}")
    return BACKBONES[name]


def validate_blocks(blocks, depth):
    blocks = tuple(blocks)
    if (len(blocks) != 3 or any(type(b) is not int for b in blocks)
            or not (0 < blocks[0] < blocks[1] < blocks[2] <= depth)):
        raise ValueError(f"Expected three increasing 1-based blocks within depth={depth}: {blocks}")
    return blocks


def adapter_pairs(channels, config):
    """Resolve explicit pairs or a declared r/C × d/C Cartesian search.

    Ceiling to a declared multiple preserves positive integer widths. These
    are search candidates, never a claim of optimality.
    """
    if "pairs" in config:
        pairs = [tuple(p) for p in config["pairs"]]
    else:
        multiple = config.get("round_to", 8)
        if type(multiple) is not int or multiple < 1:
            raise ValueError("adapter.round_to must be a positive integer")
        def width(ratio):
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("Adapter ratios must be finite and positive")
            return max(multiple, math.ceil(channels * ratio / multiple) * multiple)
        pairs = [(width(r), width(d)) for r in config["r_ratios"] for d in config["d_ratios"]]
    if not pairs or any(len(p) != 2 or any(type(v) is not int or v <= 0 for v in p) for p in pairs):
        raise ValueError("Adapter grid requires positive integer (r,d) pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate adapter pairs after rounding; declare distinct candidates")
    return pairs
