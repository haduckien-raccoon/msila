"""Controlled synthetic-anomaly generator for MS-ILA Day-3 Architecture QA.

This module is intentionally a *debug/training-supervision utility*, not a
claim that the generated defects are photorealistic or representative of real
industrial defects.

Scientific contract
-------------------
Input
    Normal RGB image as ``torch.Tensor`` with shape ``[3, H, W]`` and values
    in ``[0, 1]`` *before* DINO normalization.

Output
    ``SyntheticAnomalySample`` containing:

    - ``image``: synthetic RGB tensor ``[3, H, W]`` in ``[0, 1]``;
    - ``mask``: exact binary mask ``[1, H, W]`` in ``{0, 1}``;
    - ``metadata``: reproducibility + anomaly type/size/location statistics.

Hard invariants
---------------
1. Pixels outside the binary mask are unchanged exactly.
2. A fixed explicit seed reproduces image, mask, and metadata.
3. The mask is synchronized with the modified region.
4. The generator can emit a true normal sample (zero mask) for controlled
   Overfit-16 construction.
5. Dataset-level scientific analysis should use the returned metadata rather
   than judging a few hand-picked visual examples.

Supported controlled anomaly proxies
------------------------------------
``intensity``
    Local additive brightness shift.
``color``
    Local per-channel gain/bias perturbation.
``noise``
    Local additive Gaussian noise.
``cutpaste``
    Paste content from another location in the *same image* into the anomaly
    region, followed by mild photometric jitter.

Supported mask shapes
---------------------
``rectangle``, ``ellipse``, ``polygon``.

These proxies are useful for architecture debugging because the ground-truth
support is exact and controllable. They should not be described as realistic
industrial defects without an additional validation study against real-defect
statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence
import math

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor


AnomalyType = Literal["intensity", "color", "noise", "cutpaste"]
ShapeType = Literal["rectangle", "ellipse", "polygon"]

_ALLOWED_ANOMALY_TYPES: tuple[AnomalyType, ...] = (
    "intensity",
    "color",
    "noise",
    "cutpaste",
)
_ALLOWED_SHAPE_TYPES: tuple[ShapeType, ...] = (
    "rectangle",
    "ellipse",
    "polygon",
)


@dataclass(frozen=True)
class SyntheticAnomalyConfig:
    """Configuration for controlled synthetic anomaly generation."""

    anomaly_probability: float = 1.0

    # Mask-area envelope. For 512x512, 0.002 ~= 524 pixels.
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.08

    # Bounding-box aspect ratio = width / height.
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 2.8

    # Minimum bbox side in pixels after rounding/clamping.
    min_side_px: int = 6

    # Fraction of generated anomalies deliberately allowed to touch one image
    # boundary. This enables later boundary-defect QA without claiming realism.
    boundary_probability: float = 0.20

    anomaly_types: tuple[AnomalyType, ...] = _ALLOWED_ANOMALY_TYPES
    shape_types: tuple[ShapeType, ...] = _ALLOWED_SHAPE_TYPES

    # Photometric perturbation ranges.
    intensity_delta_min: float = 0.10
    intensity_delta_max: float = 0.35

    color_gain_min: float = 0.65
    color_gain_max: float = 1.35
    color_bias_abs_max: float = 0.12

    noise_sigma_min: float = 0.04
    noise_sigma_max: float = 0.18

    cutpaste_gain_min: float = 0.85
    cutpaste_gain_max: float = 1.15
    cutpaste_bias_abs_max: float = 0.06

    # If an operation happens to alter the masked region too weakly after
    # clipping, a deterministic fallback perturbation is applied.
    min_mean_abs_change: float = 0.015

    # Polygon complexity.
    polygon_vertices_min: int = 6
    polygon_vertices_max: int = 10

    max_geometry_attempts: int = 64

    def validate(self) -> None:
        if not (0.0 <= self.anomaly_probability <= 1.0):
            raise ValueError("anomaly_probability must be in [0,1]")
        if not (0.0 < self.min_area_ratio <= self.max_area_ratio < 1.0):
            raise ValueError(
                "Require 0 < min_area_ratio <= max_area_ratio < 1"
            )
        if not (0.0 < self.min_aspect_ratio <= self.max_aspect_ratio):
            raise ValueError(
                "Require 0 < min_aspect_ratio <= max_aspect_ratio"
            )
        if self.min_side_px < 1:
            raise ValueError("min_side_px must be >= 1")
        if not (0.0 <= self.boundary_probability <= 1.0):
            raise ValueError("boundary_probability must be in [0,1]")
        if self.intensity_delta_min <= 0 or (
            self.intensity_delta_min > self.intensity_delta_max
        ):
            raise ValueError("Invalid intensity delta range")
        if self.color_gain_min <= 0 or self.color_gain_min > self.color_gain_max:
            raise ValueError("Invalid color gain range")
        if self.noise_sigma_min <= 0 or self.noise_sigma_min > self.noise_sigma_max:
            raise ValueError("Invalid noise sigma range")
        if (
            self.cutpaste_gain_min <= 0
            or self.cutpaste_gain_min > self.cutpaste_gain_max
        ):
            raise ValueError("Invalid cutpaste gain range")
        if self.min_mean_abs_change < 0:
            raise ValueError("min_mean_abs_change must be >= 0")
        if (
            self.polygon_vertices_min < 3
            or self.polygon_vertices_min > self.polygon_vertices_max
        ):
            raise ValueError("Invalid polygon vertex range")
        if self.max_geometry_attempts < 1:
            raise ValueError("max_geometry_attempts must be >= 1")

        unknown_anomaly = set(self.anomaly_types) - set(_ALLOWED_ANOMALY_TYPES)
        if unknown_anomaly:
            raise ValueError(f"Unknown anomaly types: {sorted(unknown_anomaly)}")
        if not self.anomaly_types:
            raise ValueError("anomaly_types must not be empty")

        unknown_shape = set(self.shape_types) - set(_ALLOWED_SHAPE_TYPES)
        if unknown_shape:
            raise ValueError(f"Unknown shape types: {sorted(unknown_shape)}")
        if not self.shape_types:
            raise ValueError("shape_types must not be empty")


@dataclass(frozen=True)
class SyntheticAnomalySample:
    """One generated sample with exact pixel-level supervision."""

    image: Tensor
    mask: Tensor
    metadata: dict[str, Any]


class SyntheticAnomalyGenerator:
    """Generate reproducible synthetic anomalies from a normal RGB tensor.

    Notes
    -----
    ``seed`` is intentionally explicit in :meth:`__call__`. For fixed datasets
    such as Overfit-16, derive it deterministically from a locked base seed and
    sample index (e.g. ``base_seed + index``).
    """

    def __init__(self, config: SyntheticAnomalyConfig | None = None) -> None:
        self.config = config or SyntheticAnomalyConfig()
        self.config.validate()

    @staticmethod
    def _validate_image(image: Tensor) -> None:
        if not isinstance(image, Tensor):
            raise TypeError(f"image must be torch.Tensor, got {type(image)!r}")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                "image must have shape [3,H,W]; "
                f"got {tuple(image.shape)}"
            )
        if image.shape[1] < 2 or image.shape[2] < 2:
            raise ValueError("image spatial dimensions must be >= 2")
        if not image.is_floating_point():
            raise TypeError("image must be floating point in [0,1]")
        if not torch.isfinite(image).all().item():
            raise ValueError("image contains NaN/Inf")
        min_v = float(image.min().item())
        max_v = float(image.max().item())
        if min_v < -1e-6 or max_v > 1.0 + 1e-6:
            raise ValueError(
                "image must be BEFORE DINO normalization and lie in [0,1]; "
                f"observed [{min_v:.6f}, {max_v:.6f}]"
            )

    @staticmethod
    def _np_rng(seed: int) -> np.random.Generator:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)!r}")
        return np.random.default_rng(seed)

    @staticmethod
    def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
        # CUDA Generator is not needed for deterministic synthesis; generate
        # randomness on CPU and move to the image device. This avoids subtle
        # device-specific RNG differences in fixed QA datasets.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
        return gen

    @staticmethod
    def _choice(
        rng: np.random.Generator,
        values: Sequence[str],
    ) -> str:
        idx = int(rng.integers(0, len(values)))
        return str(values[idx])

    def _sample_bbox(
        self,
        *,
        h: int,
        w: int,
        rng: np.random.Generator,
    ) -> tuple[int, int, int, int, float, bool]:
        """Sample ``(x0,y0,x1,y1,target_area_ratio,touch_boundary)``."""
        cfg = self.config

        for _ in range(cfg.max_geometry_attempts):
            target_ratio = float(
                rng.uniform(cfg.min_area_ratio, cfg.max_area_ratio)
            )
            aspect = float(
                np.exp(
                    rng.uniform(
                        math.log(cfg.min_aspect_ratio),
                        math.log(cfg.max_aspect_ratio),
                    )
                )
            )

            target_area = target_ratio * h * w
            bw = int(round(math.sqrt(target_area * aspect)))
            bh = int(round(math.sqrt(target_area / aspect)))
            bw = max(cfg.min_side_px, min(bw, w))
            bh = max(cfg.min_side_px, min(bh, h))

            if bw > w or bh > h:
                continue

            touch_boundary = bool(rng.random() < cfg.boundary_probability)

            if touch_boundary:
                edge = int(rng.integers(0, 4))
                if edge == 0:  # left
                    x0 = 0
                    y0 = int(rng.integers(0, h - bh + 1))
                elif edge == 1:  # right
                    x0 = w - bw
                    y0 = int(rng.integers(0, h - bh + 1))
                elif edge == 2:  # top
                    x0 = int(rng.integers(0, w - bw + 1))
                    y0 = 0
                else:  # bottom
                    x0 = int(rng.integers(0, w - bw + 1))
                    y0 = h - bh
            else:
                x0 = int(rng.integers(0, w - bw + 1))
                y0 = int(rng.integers(0, h - bh + 1))

            x1 = x0 + bw
            y1 = y0 + bh
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1, target_ratio, touch_boundary

        raise RuntimeError(
            "Could not sample a valid anomaly bounding box. "
            "Relax area/aspect/min_side constraints."
        )

    def _mask_from_bbox(
        self,
        *,
        h: int,
        w: int,
        bbox: tuple[int, int, int, int],
        shape_type: ShapeType,
        rng: np.random.Generator,
    ) -> Tensor:
        x0, y0, x1, y1 = bbox
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        # PIL coordinates are inclusive at the right/bottom boundary for many
        # primitives; use x1-1/y1-1 to match our half-open bbox contract.
        xy = (x0, y0, max(x0, x1 - 1), max(y0, y1 - 1))

        if shape_type == "rectangle":
            draw.rectangle(xy, fill=255)
        elif shape_type == "ellipse":
            draw.ellipse(xy, fill=255)
        elif shape_type == "polygon":
            cx = (x0 + x1 - 1) / 2.0
            cy = (y0 + y1 - 1) / 2.0
            rx = max((x1 - x0) / 2.0, 1.0)
            ry = max((y1 - y0) / 2.0, 1.0)

            n = int(
                rng.integers(
                    self.config.polygon_vertices_min,
                    self.config.polygon_vertices_max + 1,
                )
            )
            angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=n))
            radial = rng.uniform(0.58, 1.0, size=n)
            points: list[tuple[int, int]] = []
            for angle, r in zip(angles, radial):
                px = int(round(cx + rx * r * math.cos(float(angle))))
                py = int(round(cy + ry * r * math.sin(float(angle))))
                px = max(x0, min(px, x1 - 1))
                py = max(y0, min(py, y1 - 1))
                points.append((px, py))
            draw.polygon(points, fill=255)
        else:  # pragma: no cover - protected by config validation
            raise ValueError(f"Unsupported shape_type={shape_type!r}")

        arr = np.asarray(mask_img, dtype=np.uint8).copy()
        mask = torch.from_numpy(arr > 0).to(torch.float32).unsqueeze(0)
        if float(mask.sum()) <= 0:
            raise RuntimeError("Generated anomaly mask is empty")
        return mask

    @staticmethod
    def _bbox_from_mask(mask: Tensor) -> tuple[int, int, int, int]:
        ys, xs = torch.where(mask[0] > 0.5)
        if xs.numel() == 0:
            return 0, 0, 0, 0
        x0 = int(xs.min().item())
        x1 = int(xs.max().item()) + 1
        y0 = int(ys.min().item())
        y1 = int(ys.max().item()) + 1
        return x0, y0, x1, y1

    @staticmethod
    def _touches_boundary(mask: Tensor) -> bool:
        m = mask[0].bool()
        return bool(
            m[0, :].any()
            or m[-1, :].any()
            or m[:, 0].any()
            or m[:, -1].any()
        )

    @staticmethod
    def _centroid(mask: Tensor) -> tuple[float, float]:
        ys, xs = torch.where(mask[0] > 0.5)
        if xs.numel() == 0:
            return float("nan"), float("nan")
        return float(xs.float().mean().item()), float(ys.float().mean().item())

    @staticmethod
    def _apply_mask(original: Tensor, candidate: Tensor, mask: Tensor) -> Tensor:
        m = mask.to(device=original.device, dtype=original.dtype)
        return original * (1.0 - m) + candidate * m

    def _candidate_intensity(
        self,
        image: Tensor,
        rng: np.random.Generator,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.config
        magnitude = float(
            rng.uniform(cfg.intensity_delta_min, cfg.intensity_delta_max)
        )
        sign = -1.0 if bool(rng.integers(0, 2)) else 1.0
        delta = sign * magnitude
        return (image + delta).clamp(0.0, 1.0), {"delta": delta}

    def _candidate_color(
        self,
        image: Tensor,
        rng: np.random.Generator,
    ) -> tuple[Tensor, dict[str, Any]]:
        cfg = self.config
        gains = torch.tensor(
            rng.uniform(cfg.color_gain_min, cfg.color_gain_max, size=3),
            dtype=image.dtype,
            device=image.device,
        ).view(3, 1, 1)
        biases = torch.tensor(
            rng.uniform(
                -cfg.color_bias_abs_max,
                cfg.color_bias_abs_max,
                size=3,
            ),
            dtype=image.dtype,
            device=image.device,
        ).view(3, 1, 1)
        out = (image * gains + biases).clamp(0.0, 1.0)
        return out, {
            "gains": [float(v) for v in gains.flatten().detach().cpu()],
            "biases": [float(v) for v in biases.flatten().detach().cpu()],
        }

    def _candidate_noise(
        self,
        image: Tensor,
        rng: np.random.Generator,
        seed: int,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.config
        sigma = float(rng.uniform(cfg.noise_sigma_min, cfg.noise_sigma_max))
        gen = self._torch_generator(seed ^ 0x5A17, image.device)
        noise = torch.randn(
            image.shape,
            generator=gen,
            dtype=torch.float32,
            device="cpu",
        ).to(device=image.device, dtype=image.dtype)
        out = (image + sigma * noise).clamp(0.0, 1.0)
        return out, {"sigma": sigma}

    def _candidate_cutpaste(
        self,
        image: Tensor,
        bbox: tuple[int, int, int, int],
        rng: np.random.Generator,
    ) -> tuple[Tensor, dict[str, Any]]:
        cfg = self.config
        _, h, w = image.shape
        x0, y0, x1, y1 = bbox
        bh = y1 - y0
        bw = x1 - x0

        if bw > w or bh > h:
            raise ValueError("cutpaste bbox larger than image")

        src_x0 = int(rng.integers(0, w - bw + 1))
        src_y0 = int(rng.integers(0, h - bh + 1))

        # Try to avoid near-identical source and destination boxes.
        for _ in range(16):
            if abs(src_x0 - x0) + abs(src_y0 - y0) >= max(2, min(bw, bh) // 2):
                break
            src_x0 = int(rng.integers(0, w - bw + 1))
            src_y0 = int(rng.integers(0, h - bh + 1))

        src_x1 = src_x0 + bw
        src_y1 = src_y0 + bh
        patch = image[:, src_y0:src_y1, src_x0:src_x1].clone()

        gain = float(rng.uniform(cfg.cutpaste_gain_min, cfg.cutpaste_gain_max))
        bias = float(
            rng.uniform(-cfg.cutpaste_bias_abs_max, cfg.cutpaste_bias_abs_max)
        )
        patch = (patch * gain + bias).clamp(0.0, 1.0)

        candidate = image.clone()
        candidate[:, y0:y1, x0:x1] = patch
        return candidate, {
            "source_bbox_xyxy": [src_x0, src_y0, src_x1, src_y1],
            "gain": gain,
            "bias": bias,
        }

    def _fallback_contrast(
        self,
        image: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Deterministically force a visible local change if needed."""
        m = mask.bool().expand_as(image)
        region = image[m]
        mean = float(region.mean().item()) if region.numel() else 0.5
        delta = 0.25 if mean < 0.5 else -0.25
        candidate = (image + delta).clamp(0.0, 1.0)
        return candidate, {"fallback_delta": delta}

    @staticmethod
    def _change_stats(original: Tensor, synthetic: Tensor, mask: Tensor) -> dict[str, float]:
        delta = (synthetic - original).abs()
        m = mask.bool().expand_as(delta)
        inside = delta[m]
        outside = delta[~m]
        mean_inside = float(inside.mean().item()) if inside.numel() else 0.0
        max_outside = float(outside.max().item()) if outside.numel() else 0.0
        changed_inside = float((inside > 1e-6).float().mean().item()) if inside.numel() else 0.0
        return {
            "mean_abs_change_inside": mean_inside,
            "changed_fraction_inside": changed_inside,
            "max_abs_change_outside": max_outside,
        }

    def __call__(
        self,
        image: Tensor,
        *,
        seed: int,
        force_anomaly: bool | None = None,
        anomaly_type: AnomalyType | None = None,
        shape_type: ShapeType | None = None,
    ) -> SyntheticAnomalySample:
        """Generate one controlled sample.

        Parameters
        ----------
        image:
            Normal RGB float tensor ``[3,H,W]`` in ``[0,1]``.
        seed:
            Explicit deterministic seed. Same image + same config + same seed
            reproduces exactly the same output.
        force_anomaly:
            ``True`` always generates an anomaly; ``False`` returns a genuine
            normal sample with zero mask; ``None`` samples according to
            ``config.anomaly_probability``.
        anomaly_type, shape_type:
            Optional explicit overrides, useful for unit tests and controlled
            ablations.
        """
        self._validate_image(image)
        rng = self._np_rng(seed)
        cfg = self.config

        if force_anomaly is None:
            make_anomaly = bool(rng.random() < cfg.anomaly_probability)
        else:
            make_anomaly = bool(force_anomaly)

        _, h, w = image.shape
        original = image.clone()

        if not make_anomaly:
            mask = torch.zeros(
                (1, h, w),
                dtype=torch.float32,
                device=image.device,
            )
            metadata = {
                "seed": int(seed),
                "is_anomaly": False,
                "anomaly_type": "none",
                "shape_type": "none",
                "area_px": 0,
                "area_ratio": 0.0,
                "bbox_xyxy": [0, 0, 0, 0],
                "centroid_xy": [None, None],
                "centroid_norm_xy": [None, None],
                "touches_image_boundary": False,
                "operation": {},
                "mean_abs_change_inside": 0.0,
                "changed_fraction_inside": 0.0,
                "max_abs_change_outside": 0.0,
                "generator_config": asdict(cfg),
            }
            return SyntheticAnomalySample(original, mask, metadata)

        if anomaly_type is None:
            anomaly_type = self._choice(rng, cfg.anomaly_types)  # type: ignore[assignment]
        if anomaly_type not in _ALLOWED_ANOMALY_TYPES:
            raise ValueError(f"Unsupported anomaly_type={anomaly_type!r}")

        if shape_type is None:
            shape_type = self._choice(rng, cfg.shape_types)  # type: ignore[assignment]
        if shape_type not in _ALLOWED_SHAPE_TYPES:
            raise ValueError(f"Unsupported shape_type={shape_type!r}")

        x0, y0, x1, y1, target_area_ratio, _ = self._sample_bbox(
            h=h,
            w=w,
            rng=rng,
        )
        sampled_bbox = (x0, y0, x1, y1)
        mask_cpu = self._mask_from_bbox(
            h=h,
            w=w,
            bbox=sampled_bbox,
            shape_type=shape_type,
            rng=rng,
        )
        mask = mask_cpu.to(device=image.device)

        if anomaly_type == "intensity":
            candidate, op_meta = self._candidate_intensity(image, rng)
        elif anomaly_type == "color":
            candidate, op_meta = self._candidate_color(image, rng)
        elif anomaly_type == "noise":
            candidate, op_meta = self._candidate_noise(image, rng, seed)
        elif anomaly_type == "cutpaste":
            candidate, op_meta = self._candidate_cutpaste(image, sampled_bbox, rng)
        else:  # pragma: no cover
            raise AssertionError(anomaly_type)

        synthetic = self._apply_mask(original, candidate, mask)
        stats = self._change_stats(original, synthetic, mask)

        if stats["mean_abs_change_inside"] < cfg.min_mean_abs_change:
            fallback_candidate, fallback_meta = self._fallback_contrast(original, mask)
            synthetic = self._apply_mask(original, fallback_candidate, mask)
            op_meta = {**op_meta, **fallback_meta, "fallback_applied": True}
            stats = self._change_stats(original, synthetic, mask)
        else:
            op_meta = {**op_meta, "fallback_applied": False}

        # Hard synchronization invariant: no pixel outside GT mask may change.
        if stats["max_abs_change_outside"] != 0.0:
            raise RuntimeError(
                "Synthetic anomaly synchronization failure: pixels outside "
                "the binary mask were modified."
            )

        actual_bbox = self._bbox_from_mask(mask)
        cx, cy = self._centroid(mask)
        area_px = int(mask.sum().item())
        area_ratio = float(area_px / (h * w))

        metadata = {
            "seed": int(seed),
            "is_anomaly": True,
            "anomaly_type": anomaly_type,
            "shape_type": shape_type,
            "target_area_ratio": float(target_area_ratio),
            "area_px": area_px,
            "area_ratio": area_ratio,
            "bbox_xyxy": [int(v) for v in actual_bbox],
            "centroid_xy": [cx, cy],
            "centroid_norm_xy": [
                cx / max(w - 1, 1),
                cy / max(h - 1, 1),
            ],
            "touches_image_boundary": self._touches_boundary(mask),
            "operation": op_meta,
            **stats,
            "generator_config": asdict(cfg),
        }

        # Final numeric contract.
        if synthetic.shape != original.shape:
            raise RuntimeError("synthetic image shape changed unexpectedly")
        if mask.shape != (1, h, w):
            raise RuntimeError("mask shape contract violated")
        if not torch.isfinite(synthetic).all().item():
            raise RuntimeError("synthetic image contains NaN/Inf")
        if float(synthetic.min()) < -1e-6 or float(synthetic.max()) > 1.0 + 1e-6:
            raise RuntimeError("synthetic image escaped [0,1]")
        unique = torch.unique(mask.detach().cpu())
        if not all(float(v) in (0.0, 1.0) for v in unique):
            raise RuntimeError("mask is not binary")

        return SyntheticAnomalySample(
            image=synthetic.contiguous(),
            mask=mask.to(torch.float32).contiguous(),
            metadata=metadata,
        )


def summarize_synthetic_metadata(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize anomaly type, size, and location for scientific QA.

    This intentionally reports descriptive statistics only. It does *not*
    evaluate whether the synthetic distribution is realistic.
    """
    if not records:
        raise ValueError("records must not be empty")

    anomaly_records = [r for r in records if bool(r.get("is_anomaly", False))]
    n_total = len(records)
    n_anom = len(anomaly_records)

    def counts(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in anomaly_records:
            name = str(r.get(key, "unknown"))
            out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items()))

    if not anomaly_records:
        return {
            "n_total": n_total,
            "n_anomaly": 0,
            "anomaly_fraction": 0.0,
            "anomaly_type_counts": {},
            "shape_type_counts": {},
            "area_ratio": None,
            "centroid_norm_xy_mean": None,
            "boundary_fraction": 0.0,
        }

    areas = np.asarray(
        [float(r["area_ratio"]) for r in anomaly_records],
        dtype=np.float64,
    )
    centers = np.asarray(
        [r["centroid_norm_xy"] for r in anomaly_records],
        dtype=np.float64,
    )
    boundary = np.asarray(
        [bool(r["touches_image_boundary"]) for r in anomaly_records],
        dtype=np.float64,
    )

    return {
        "n_total": n_total,
        "n_anomaly": n_anom,
        "anomaly_fraction": float(n_anom / n_total),
        "anomaly_type_counts": counts("anomaly_type"),
        "shape_type_counts": counts("shape_type"),
        "area_ratio": {
            "min": float(areas.min()),
            "mean": float(areas.mean()),
            "std": float(areas.std(ddof=0)),
            "max": float(areas.max()),
        },
        "centroid_norm_xy_mean": [
            float(centers[:, 0].mean()),
            float(centers[:, 1].mean()),
        ],
        "centroid_norm_xy_std": [
            float(centers[:, 0].std(ddof=0)),
            float(centers[:, 1].std(ddof=0)),
        ],
        "boundary_fraction": float(boundary.mean()),
    }


__all__ = [
    "AnomalyType",
    "ShapeType",
    "SyntheticAnomalyConfig",
    "SyntheticAnomalySample",
    "SyntheticAnomalyGenerator",
    "summarize_synthetic_metadata",
]
