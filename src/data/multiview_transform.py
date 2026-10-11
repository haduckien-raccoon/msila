"""Paired Local/Context views for high-resolution anomaly localization.

Locked project contract
-----------------------
Local   : crop 512x512 from the source image -> model input 512x512.
Context : crop 768x768 from the same source image -> resize to 512x512.
Geometry: Local is centered inside Context.
Batch   : PyTorch DataLoader turns per-sample [3,512,512] tensors into
          [B,3,512,512].

This file intentionally handles only geometric view construction and optional
backbone normalization. Photometric augmentation should be kept separate so
image/mask geometry remains auditable for anomaly localization.

Scientific references and design rationale are documented in
``multiview_transform.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data.tiling import TileRecord, crop_with_padding, extract_local_context
from src.geometry.view_meta import build_padded_view_meta


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MultiViewConfig",
    "NestedMultiViewTransform",
    "validate_multiview_sample",
]


# DINOv3 LVD-1689M / standard ImageNet normalization.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MultiViewConfig:
    """Configuration for paired Local/Context views.

    Defaults are the locked project setting:
        local source FOV   = 512 px
        context source FOV = 768 px
        network input      = 512 px for both views
    """

    local_size: int = 512
    context_size: int = 768
    input_size: int = 512

    # random: training; center: deterministic validation/debugging.
    sampling: Literal["random", "center"] = "random"

    # DINOv3 LVD-1689M-compatible preprocessing by default.
    normalize: bool = True
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD

    # DINO/DINOv3 training code uses bicubic interpolation for crop resizing.
    interpolation: InterpolationMode = InterpolationMode.BICUBIC
    antialias: bool = True

    # Runtime assertions are cheap compared with backbone inference and catch
    # geometry/data errors early during dataset development.
    validate_output: bool = True

    def __post_init__(self) -> None:
        if min(self.local_size, self.context_size, self.input_size) <= 0:
            raise ValueError("local_size, context_size and input_size must be > 0")
        if self.local_size > self.context_size:
            raise ValueError("local_size must be <= context_size")
        if (self.context_size - self.local_size) % 2 != 0:
            raise ValueError(
                "context_size - local_size must be even for an exact centered crop"
            )
        if self.sampling not in {"random", "center"}:
            raise ValueError("sampling must be 'random' or 'center'")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each contain exactly 3 values")
        if any(s <= 0 for s in self.std):
            raise ValueError("all std values must be > 0")


class NestedMultiViewTransform:
    """Construct a same-center Local/Context pair from one image.

    Coordinates use half-open boxes ``[x0, y0, x1, y1)`` so they map directly
    to PyTorch slicing ``image[:, y0:y1, x0:x1]``.

    Supported input types
    ---------------------
    - PIL.Image
    - numpy.ndarray: HWC/CHW, 1/3/4 channels
    - torch.Tensor: HWC/CHW, 1/3/4 channels

    Floating-point input must already be in [0, 1]. Integer input is scaled to
    [0, 1] using its dtype maximum.
    """

    def __init__(self, config: MultiViewConfig | None = None) -> None:
        self.cfg = config or MultiViewConfig()
        self._mean = torch.tensor(self.cfg.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(self.cfg.std, dtype=torch.float32).view(3, 1, 1)

    @property
    def local_margin(self) -> int:
        """Margin, in source pixels, between Context and centered Local."""
        return (self.cfg.context_size - self.cfg.local_size) // 2

    @staticmethod
    def _to_chw_float01(image: Any) -> Tensor:
        """Convert an image to finite contiguous RGB float32 CHW in [0, 1]."""
        if isinstance(image, Image.Image):
            x = TF.pil_to_tensor(image.convert("RGB"))

        elif isinstance(image, np.ndarray):
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.ndim != 3:
                raise ValueError(
                    f"NumPy image must be HxW, HxWxC or CxHxW; got {arr.shape}"
                )

            arr = np.ascontiguousarray(arr)
            if arr.shape[-1] in (1, 3, 4):  # HWC
                x = torch.from_numpy(arr).permute(2, 0, 1)
            elif arr.shape[0] in (1, 3, 4):  # CHW
                x = torch.from_numpy(arr)
            else:
                raise ValueError(
                    "Cannot infer NumPy channel axis; expected 1/3/4 channels"
                )

        elif isinstance(image, Tensor):
            x = image.detach()
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim != 3:
                raise ValueError(
                    f"Tensor image must be HxW, HxWxC or CxHxW; got {tuple(x.shape)}"
                )
            if x.shape[0] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
                x = x.permute(2, 0, 1)
            if x.shape[0] not in (1, 3, 4):
                raise ValueError("Tensor image must have 1, 3 or 4 channels")
            x = x.contiguous()

        else:
            raise TypeError(
                "image must be PIL.Image, numpy.ndarray or torch.Tensor; "
                f"got {type(image)!r}"
            )

        # Pretrained RGB backbones expect exactly 3 channels.
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] == 4:
            x = x[:3]  # alpha is not part of the pretrained RGB contract

        if torch.is_floating_point(x):
            x = x.to(torch.float32)
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Input image contains NaN or Inf")
            lo = float(x.amin().item())
            hi = float(x.amax().item())
            if lo < -1e-6 or hi > 1.0 + 1e-6:
                raise ValueError(
                    "Floating-point image must already be in [0, 1]; "
                    f"observed range [{lo:.6g}, {hi:.6g}]"
                )
            x = x.clamp(0.0, 1.0)
        else:
            if x.dtype == torch.bool:
                x = x.to(torch.float32)
            else:
                info = torch.iinfo(x.dtype)
                if info.min < 0 and int(x.min().item()) < 0:
                    raise ValueError("Signed integer image contains negative values")
                x = x.to(torch.float32) / float(info.max)

        return x.contiguous()

    def _sample_context_origin(self, h: int, w: int) -> Tuple[int, int]:
        """Sample a valid top-left corner for the Context crop."""
        s = self.cfg.context_size
        if h < s or w < s:
            raise ValueError(
                f"Image too small for exact {s}x{s} Context crop: got HxW={h}x{w}. "
                "Resize/tile upstream if that is an explicit experimental choice."
            )

        if self.cfg.sampling == "center":
            return (h - s) // 2, (w - s) // 2

        # PyTorch seeds each DataLoader worker, so using torch RNG here is
        # reproducible when the experiment seed/worker seeding is controlled.
        top = int(torch.randint(0, h - s + 1, (1,)).item())
        left = int(torch.randint(0, w - s + 1, (1,)).item())
        return top, left

    def _normalize(self, x: Tensor) -> Tensor:
        if not self.cfg.normalize:
            return x
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        # Per-channel normalization: x'_c = (x_c - mu_c) / sigma_c.
        return (x - mean) / std

    def from_tile(
        self, image: Any, record: TileRecord, *, mask: Tensor | None = None,
        source_meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build an E4 sample from ONE already augmented native image and mask.

        Uses the existing tiling resize (bilinear, align_corners=False). RGB
        padding follows tiling's reflect/replicate policy; supervision pads with
        zero so padded pixels never acquire synthetic positive labels. No anomaly
        generator is invoked here. The legacy random/center ``__call__`` API is
        unchanged. Return keys follow G2: image/context/mask/meta/view_meta.
        """
        if (self.cfg.local_size, self.cfg.context_size, self.cfg.input_size) != (512, 768, 512):
            raise ValueError("G2 tile views require Local 512, Context 768, input 512")
        x = self._to_chw_float01(image)
        local_box, context_box = record.local_xyxy, record.context_xyxy
        if (tuple(local_box[i+2]-local_box[i] for i in (0, 1)) != (512, 512)
                or tuple(context_box[i+2]-context_box[i] for i in (0, 1)) != (768, 768)
                or any(local_box[i]-context_box[i] != 128 for i in (0, 1))
                or record.center_xy != ((local_box[0]+local_box[2])/2, (local_box[1]+local_box[3])/2)):
            raise ValueError("TileRecord must contain concentric Local 512 / Context 768 boxes")
        geometry = build_padded_view_meta(
            source_hw=x.shape[-2:], local_box_xyxy=local_box, context_box_xyxy=context_box,
        )
        local, context = extract_local_context(x, record)
        meta = dict(source_meta or {}, tile_id=record.tile_id, native_hw=list(x.shape[-2:]),
                    local_native_xyxy=list(local_box), context_native_xyxy=list(context_box))
        view_meta = dict(meta, geometry=geometry, geometry_source_frame="padded_native",
                         context_resize="bilinear_align_corners_false")
        if "sample_id" in meta:
            view_meta.update(local_sample_id=meta["sample_id"], context_sample_id=meta["sample_id"])
        sample = dict(image=self._normalize(local).contiguous(),
                      context=self._normalize(context).contiguous(), meta=meta, view_meta=view_meta)
        if mask is not None:
            if not isinstance(mask, Tensor):
                raise TypeError("Native mask must be a torch.Tensor")
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape) != (1, *x.shape[-2:]):
                raise ValueError("Native mask must be [1,H,W], matching the native image")
            if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("Native mask must be finite and binary")
            sample["mask"] = crop_with_padding(mask.to(device=x.device, dtype=x.dtype),
                                                local_box, pad_mode="constant", pad_value=0.).contiguous()
        return sample

    def __call__(self, image: Any) -> Dict[str, Any]:
        x = self._to_chw_float01(image)
        _, h, w = x.shape

        c = self.cfg.context_size
        l = self.cfg.local_size
        out_s = self.cfg.input_size
        margin = self.local_margin

        c_top, c_left = self._sample_context_origin(h, w)
        l_top = c_top + margin
        l_left = c_left + margin

        context_box = torch.tensor(
            [c_left, c_top, c_left + c, c_top + c], dtype=torch.int64
        )
        local_box = torch.tensor(
            [l_left, l_top, l_left + l, l_top + l], dtype=torch.int64
        )

        # Exact source-space crops.
        x_context_src = x[:, c_top : c_top + c, c_left : c_left + c]
        x_local_src = x[:, l_top : l_top + l, l_left : l_left + l]

        # Context is explicitly downsampled from a larger source FOV.
        x_context = TF.resize(
            x_context_src,
            [out_s, out_s],
            interpolation=self.cfg.interpolation,
            antialias=self.cfg.antialias,
        )

        # Locked default is 512 -> 512, so Local preserves source pixels.
        if l == out_s:
            x_local = x_local_src
        else:
            x_local = TF.resize(
                x_local_src,
                [out_s, out_s],
                interpolation=self.cfg.interpolation,
                antialias=self.cfg.antialias,
            )

        x_local = self._normalize(x_local).contiguous()
        x_context = self._normalize(x_context).contiguous()

        scale = float(out_s) / float(c)
        local_rel = torch.tensor(
            [margin, margin, margin + l, margin + l], dtype=torch.int64
        )
        local_rel_input = local_rel.to(torch.float32) * scale

        out: Dict[str, Any] = {
            "x_local": x_local,
            "x_context": x_context,
            "meta": {
                "source_hw": torch.tensor([h, w], dtype=torch.int64),
                "context_box_xyxy": context_box,
                "local_box_xyxy": local_box,
                "local_box_in_context_xyxy": local_rel,
                "local_box_in_context_input_xyxy": local_rel_input,
                "context_to_input_scale": torch.tensor(scale, dtype=torch.float32),
            },
        }

        if self.cfg.validate_output:
            validate_multiview_sample(out, self.cfg)

        return out


def validate_multiview_sample(sample: Dict[str, Any], cfg: MultiViewConfig) -> None:
    """Validate the task contract for one transformed sample.

    Raises AssertionError on any geometry/shape/finite violation.
    """
    x_local = sample["x_local"]
    x_context = sample["x_context"]
    meta = sample["meta"]

    expected = (3, cfg.input_size, cfg.input_size)
    assert tuple(x_local.shape) == expected, (
        f"x_local shape {tuple(x_local.shape)} != {expected}"
    )
    assert tuple(x_context.shape) == expected, (
        f"x_context shape {tuple(x_context.shape)} != {expected}"
    )
    assert bool(torch.isfinite(x_local).all()), "x_local contains NaN/Inf"
    assert bool(torch.isfinite(x_context).all()), "x_context contains NaN/Inf"

    c = meta["context_box_xyxy"]
    l = meta["local_box_xyxy"]
    rel = meta["local_box_in_context_xyxy"]

    # Source-space sizes.
    assert int(c[2] - c[0]) == cfg.context_size
    assert int(c[3] - c[1]) == cfg.context_size
    assert int(l[2] - l[0]) == cfg.local_size
    assert int(l[3] - l[1]) == cfg.local_size

    # Required PASS: Local lies completely inside Context.
    assert int(c[0]) <= int(l[0]) < int(l[2]) <= int(c[2])
    assert int(c[1]) <= int(l[1]) < int(l[3]) <= int(c[3])

    # Locked same-center geometry.
    m = (cfg.context_size - cfg.local_size) // 2
    expected_rel = torch.tensor([m, m, m + cfg.local_size, m + cfg.local_size])
    assert torch.equal(rel.cpu(), expected_rel), (
        f"Local is not centered in Context: got {rel.tolist()}"
    )

    # Mapping from Context source coordinates to Context input coordinates.
    expected_scale = cfg.input_size / cfg.context_size
    assert torch.isclose(
        meta["context_to_input_scale"].cpu(),
        torch.tensor(expected_scale, dtype=torch.float32),
    )
    assert torch.allclose(
        meta["local_box_in_context_input_xyxy"].cpu(),
        expected_rel.to(torch.float32) * expected_scale,
        atol=1e-5,
        rtol=0.0,
    )
