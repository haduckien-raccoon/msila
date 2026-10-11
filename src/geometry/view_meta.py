"""Deterministic geometry metadata for paired Local/Context views.

Project contract
----------------
- Local source crop:   typically 512 x 512.
- Context source crop: typically 768 x 768, containing Local.
- Both model inputs:   typically 512 x 512.
- Metadata stores crop boxes, per-view scale, and 3x3 homogeneous transforms.

Coordinate convention
---------------------
Boxes use half-open *pixel-edge* coordinates ``[x0, y0, x1, y1)``.
The homogeneous matrices in this module map continuous pixel-edge coordinates,
not the exact interpolation-kernel sample centers used internally by an image
resizer. This keeps crop geometry deterministic and framework-independent.

Scientific references and the exact formulas are documented in ``view_meta.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "BoxXYXY",
    "HW",
    "ViewGeometryMeta",
    "build_view_meta",
    "build_view_meta_from_transform_meta",
    "build_padded_view_meta",
    "crop_resize_matrix",
    "transform_points_xy",
    "transform_box_xyxy",
    "validate_view_meta",
]


HW = Tuple[int, int]
BoxXYXY = Tuple[int, int, int, int]


def _as_hw(value: Sequence[int], name: str) -> HW:
    if len(value) != 2:
        raise ValueError(f"{name} must be (H, W); got {value!r}")
    h, w = int(value[0]), int(value[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must be positive; got {(h, w)}")
    return h, w


def _as_box(value: Sequence[int], name: str) -> BoxXYXY:
    if len(value) != 4:
        raise ValueError(f"{name} must be [x0, y0, x1, y1); got {value!r}")
    x0, y0, x1, y1 = (int(v) for v in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} must have positive area; got {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _check_box_inside_source(box: BoxXYXY, source_hw: HW, name: str) -> None:
    h, w = source_hw
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            f"{name}={box} is outside source bounds W={w}, H={h}"
        )


def crop_resize_matrix(
    crop_box_xyxy: Sequence[int],
    output_hw: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the 3x3 source -> crop-output affine matrix.

    For source crop ``B=[x0,y0,x1,y1)`` and output size ``(Ho, Wo)``:

        sx = Wo / (x1 - x0)
        sy = Ho / (y1 - y0)

        [u]   [ sx   0  -sx*x0 ] [x]
        [v] = [  0  sy  -sy*y0 ] [y]
        [1]   [  0   0      1   ] [1]

    The mapping is defined in continuous pixel-edge coordinates. Therefore the
    crop rectangle maps exactly to [0, Wo] x [0, Ho].
    """
    x0, y0, x1, y1 = _as_box(crop_box_xyxy, "crop_box_xyxy")
    out_h, out_w = _as_hw(output_hw, "output_hw")

    sx = float(out_w) / float(x1 - x0)
    sy = float(out_h) / float(y1 - y0)

    return torch.tensor(
        [
            [sx, 0.0, -sx * float(x0)],
            [0.0, sy, -sy * float(y0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def transform_points_xy(points_xy: Tensor | Sequence[Sequence[float]], matrix: Tensor) -> Tensor:
    """Apply a 3x3 homogeneous transform to ``[..., 2]`` XY points.

    Supports affine and projective 3x3 matrices. Output is divided by the
    homogeneous coordinate, so projective use remains mathematically correct.
    """
    m = torch.as_tensor(matrix)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3,3); got {tuple(m.shape)}")
    if not torch.is_floating_point(m):
        m = m.to(torch.float64)
    if not bool(torch.isfinite(m).all()):
        raise ValueError("matrix contains NaN/Inf")

    p = torch.as_tensor(points_xy, dtype=m.dtype, device=m.device)
    if p.ndim == 1:
        if p.numel() != 2:
            raise ValueError("a single point must contain exactly 2 values")
        p = p.unsqueeze(0)
        squeeze = True
    else:
        if p.shape[-1] != 2:
            raise ValueError(f"points must have shape [...,2]; got {tuple(p.shape)}")
        squeeze = False

    ones = torch.ones((*p.shape[:-1], 1), dtype=p.dtype, device=p.device)
    ph = torch.cat([p, ones], dim=-1)
    qh = ph @ m.transpose(0, 1)

    w = qh[..., 2:3]
    eps = torch.finfo(qh.dtype).eps * 16
    if bool((w.abs() <= eps).any()):
        raise ValueError("transform maps at least one point to infinity")

    q = qh[..., :2] / w
    return q.squeeze(0) if squeeze else q


def transform_box_xyxy(box_xyxy: Sequence[float] | Tensor, matrix: Tensor) -> Tensor:
    """Transform a box by all four corners and return axis-aligned XYXY bounds."""
    b = torch.as_tensor(box_xyxy, dtype=matrix.dtype, device=matrix.device).flatten()
    if b.numel() != 4:
        raise ValueError("box_xyxy must contain exactly 4 values")
    x0, y0, x1, y1 = b.unbind()
    if not bool((x1 > x0) and (y1 > y0)):
        raise ValueError("box_xyxy must have positive area")

    corners = torch.stack(
        [
            torch.stack([x0, y0]),
            torch.stack([x1, y0]),
            torch.stack([x1, y1]),
            torch.stack([x0, y1]),
        ]
    )
    tc = transform_points_xy(corners, matrix)
    mn = tc.amin(dim=0)
    mx = tc.amax(dim=0)
    return torch.cat([mn, mx])


@dataclass(frozen=True)
class ViewGeometryMeta:
    """Immutable Local <-> Context geometry metadata.

    All matrices are float64 for stable bookkeeping and composition. Model
    features may remain float32/bfloat16; geometry metadata is small enough that
    float64 has negligible cost and avoids unnecessary round-trip drift.
    """

    source_hw: HW
    local_box_xyxy: BoxXYXY
    context_box_xyxy: BoxXYXY
    local_input_hw: HW
    context_input_hw: HW

    local_scale_xy: Tensor
    context_scale_xy: Tensor

    source_to_local: Tensor
    local_to_source: Tensor
    source_to_context: Tensor
    context_to_source: Tensor
    local_to_context: Tensor
    context_to_local: Tensor

    local_box_in_context_source_xyxy: Tensor
    local_box_in_context_input_xyxy: Tensor

    def as_tensor_dict(self, *, dtype: torch.dtype | None = None) -> Dict[str, Tensor]:
        """Return DataLoader-friendly tensors.

        ``dtype=None`` preserves float64 geometry matrices/scales. Integer boxes
        and sizes remain int64. A floating dtype can be requested when a later
        module explicitly requires it.
        """
        float_dtype = dtype or self.source_to_local.dtype
        if not torch.empty((), dtype=float_dtype).is_floating_point():
            raise ValueError("dtype must be a floating-point torch dtype")

        return {
            "source_hw": torch.tensor(self.source_hw, dtype=torch.int64),
            "local_box_xyxy": torch.tensor(self.local_box_xyxy, dtype=torch.int64),
            "context_box_xyxy": torch.tensor(self.context_box_xyxy, dtype=torch.int64),
            "local_input_hw": torch.tensor(self.local_input_hw, dtype=torch.int64),
            "context_input_hw": torch.tensor(self.context_input_hw, dtype=torch.int64),
            "local_scale_xy": self.local_scale_xy.to(dtype=float_dtype),
            "context_scale_xy": self.context_scale_xy.to(dtype=float_dtype),
            "source_to_local": self.source_to_local.to(dtype=float_dtype),
            "local_to_source": self.local_to_source.to(dtype=float_dtype),
            "source_to_context": self.source_to_context.to(dtype=float_dtype),
            "context_to_source": self.context_to_source.to(dtype=float_dtype),
            "local_to_context": self.local_to_context.to(dtype=float_dtype),
            "context_to_local": self.context_to_local.to(dtype=float_dtype),
            "local_box_in_context_source_xyxy": self.local_box_in_context_source_xyxy.to(
                dtype=float_dtype
            ),
            "local_box_in_context_input_xyxy": self.local_box_in_context_input_xyxy.to(
                dtype=float_dtype
            ),
        }


def build_view_meta(
    *,
    source_hw: Sequence[int],
    local_box_xyxy: Sequence[int],
    context_box_xyxy: Sequence[int],
    local_input_hw: Sequence[int] = (512, 512),
    context_input_hw: Sequence[int] = (512, 512),
    validate: bool = True,
) -> ViewGeometryMeta:
    """Build deterministic geometry metadata from source-space crop boxes."""
    src_hw = _as_hw(source_hw, "source_hw")
    lbox = _as_box(local_box_xyxy, "local_box_xyxy")
    cbox = _as_box(context_box_xyxy, "context_box_xyxy")
    lin_hw = _as_hw(local_input_hw, "local_input_hw")
    cin_hw = _as_hw(context_input_hw, "context_input_hw")

    _check_box_inside_source(lbox, src_hw, "local_box_xyxy")
    _check_box_inside_source(cbox, src_hw, "context_box_xyxy")

    lx0, ly0, lx1, ly1 = lbox
    cx0, cy0, cx1, cy1 = cbox
    if not (cx0 <= lx0 < lx1 <= cx1 and cy0 <= ly0 < ly1 <= cy1):
        raise ValueError("Local crop must lie completely inside Context crop")

    s2l = crop_resize_matrix(lbox, lin_hw)
    s2c = crop_resize_matrix(cbox, cin_hw)
    l2s = torch.linalg.inv(s2l)
    c2s = torch.linalg.inv(s2c)

    # Column-vector convention: p_context = S2C @ L2S @ p_local.
    l2c = s2c @ l2s
    c2l = s2l @ c2s

    local_scale_xy = torch.tensor(
        [
            float(lin_hw[1]) / float(lx1 - lx0),
            float(lin_hw[0]) / float(ly1 - ly0),
        ],
        dtype=torch.float64,
    )
    context_scale_xy = torch.tensor(
        [
            float(cin_hw[1]) / float(cx1 - cx0),
            float(cin_hw[0]) / float(cy1 - cy0),
        ],
        dtype=torch.float64,
    )

    local_rel_source = torch.tensor(
        [lx0 - cx0, ly0 - cy0, lx1 - cx0, ly1 - cy0],
        dtype=torch.float64,
    )
    local_in_context_input = transform_box_xyxy(lbox, s2c)

    meta = ViewGeometryMeta(
        source_hw=src_hw,
        local_box_xyxy=lbox,
        context_box_xyxy=cbox,
        local_input_hw=lin_hw,
        context_input_hw=cin_hw,
        local_scale_xy=local_scale_xy,
        context_scale_xy=context_scale_xy,
        source_to_local=s2l,
        local_to_source=l2s,
        source_to_context=s2c,
        context_to_source=c2s,
        local_to_context=l2c,
        context_to_local=c2l,
        local_box_in_context_source_xyxy=local_rel_source,
        local_box_in_context_input_xyxy=local_in_context_input,
    )

    if validate:
        validate_view_meta(meta)
    return meta


def build_view_meta_from_transform_meta(
    transform_meta: Mapping[str, Any],
    *,
    local_input_hw: Sequence[int] = (512, 512),
    context_input_hw: Sequence[int] = (512, 512),
    validate: bool = True,
) -> ViewGeometryMeta:
    """Adapter for ``data/multiview_transform.py`` metadata.

    Required keys are only:
      - ``source_hw``
      - ``local_box_xyxy``
      - ``context_box_xyxy``

    Values may be lists, tuples, NumPy-like arrays, or torch tensors.
    """
    required = ("source_hw", "local_box_xyxy", "context_box_xyxy")
    missing = [k for k in required if k not in transform_meta]
    if missing:
        raise KeyError(f"transform_meta missing required keys: {missing}")

    def values(x: Any) -> list[int]:
        if isinstance(x, Tensor):
            x = x.detach().cpu().flatten().tolist()
        elif hasattr(x, "tolist"):
            x = x.tolist()
        return [int(v) for v in x]

    return build_view_meta(
        source_hw=values(transform_meta["source_hw"]),
        local_box_xyxy=values(transform_meta["local_box_xyxy"]),
        context_box_xyxy=values(transform_meta["context_box_xyxy"]),
        local_input_hw=local_input_hw,
        context_input_hw=context_input_hw,
        validate=validate,
    )


def build_padded_view_meta(
    *,
    source_hw: Sequence[int],
    local_box_xyxy: Sequence[int],
    context_box_xyxy: Sequence[int],
    local_input_hw: Sequence[int] = (512, 512),
    context_input_hw: Sequence[int] = (512, 512),
) -> Dict[str, Tensor]:
    """Describe boundary tiles in a virtual padded-native coordinate frame.

    Native crop boxes keep their full FOV, including negative origins. Translate
    both boxes by the SAME left/top padding before invoking the existing strict
    builder; never clip or shift Context separately. ``source_hw`` in the returned
    geometry is the padded canvas size. ``native_hw``, ``padding_ltrb`` and the
    native/view transforms retain the exact original-image coordinates.
    Padding describes geometry only; image/mask padding is performed by tiling.
    """
    h, w = _as_hw(source_hw, "source_hw")
    local = _as_box(local_box_xyxy, "local_box_xyxy")
    context = _as_box(context_box_xyxy, "context_box_xyxy")
    if min(local[2], w) <= max(local[0], 0) or min(local[3], h) <= max(local[1], 0):
        raise ValueError("Local crop must intersect the native image")
    left = max(0, -min(local[0], context[0]))
    top = max(0, -min(local[1], context[1]))
    right = max(0, max(local[2], context[2]) - w)
    bottom = max(0, max(local[3], context[3]) - h)
    shift = lambda box: (box[0]+left, box[1]+top, box[2]+left, box[3]+top)
    meta = build_view_meta(
        source_hw=(h+top+bottom, w+left+right),
        local_box_xyxy=shift(local), context_box_xyxy=shift(context),
        local_input_hw=local_input_hw, context_input_hw=context_input_hw,
    )
    native_to_source = torch.tensor([[1., 0., left], [0., 1., top], [0., 0., 1.]], dtype=torch.float64)
    geometry = meta.as_tensor_dict()
    geometry.update(
        native_hw=torch.tensor([h, w], dtype=torch.int64),
        padding_ltrb=torch.tensor([left, top, right, bottom], dtype=torch.int64),
        local_native_xyxy=torch.tensor(local, dtype=torch.int64),
        context_native_xyxy=torch.tensor(context, dtype=torch.int64),
        native_to_source=native_to_source,
        source_to_native=torch.linalg.inv(native_to_source),
        native_to_local=meta.source_to_local @ native_to_source,
        native_to_context=meta.source_to_context @ native_to_source,
        local_to_native=torch.linalg.inv(native_to_source) @ meta.local_to_source,
        context_to_native=torch.linalg.inv(native_to_source) @ meta.context_to_source,
    )
    return geometry


def validate_view_meta(meta: ViewGeometryMeta, *, atol: float = 1e-9) -> None:
    """Validate containment, matrices, composition, and round-trip mapping."""
    _check_box_inside_source(meta.local_box_xyxy, meta.source_hw, "local_box_xyxy")
    _check_box_inside_source(meta.context_box_xyxy, meta.source_hw, "context_box_xyxy")

    lx0, ly0, lx1, ly1 = meta.local_box_xyxy
    cx0, cy0, cx1, cy1 = meta.context_box_xyxy
    assert cx0 <= lx0 < lx1 <= cx1
    assert cy0 <= ly0 < ly1 <= cy1

    matrices = (
        meta.source_to_local,
        meta.local_to_source,
        meta.source_to_context,
        meta.context_to_source,
        meta.local_to_context,
        meta.context_to_local,
    )
    for m in matrices:
        assert m.shape == (3, 3)
        assert bool(torch.isfinite(m).all())
        assert abs(float(torch.linalg.det(m))) > torch.finfo(m.dtype).eps

    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(meta.local_to_source @ meta.source_to_local, eye, atol=atol, rtol=0)
    assert torch.allclose(meta.context_to_source @ meta.source_to_context, eye, atol=atol, rtol=0)
    assert torch.allclose(
        meta.local_to_context,
        meta.source_to_context @ meta.local_to_source,
        atol=atol,
        rtol=0,
    )
    assert torch.allclose(
        meta.context_to_local,
        meta.source_to_local @ meta.context_to_source,
        atol=atol,
        rtol=0,
    )

    # Each crop must map exactly to its own model-input edge rectangle.
    expected_local = torch.tensor(
        [0.0, 0.0, float(meta.local_input_hw[1]), float(meta.local_input_hw[0])],
        dtype=torch.float64,
    )
    expected_context = torch.tensor(
        [0.0, 0.0, float(meta.context_input_hw[1]), float(meta.context_input_hw[0])],
        dtype=torch.float64,
    )
    assert torch.allclose(
        transform_box_xyxy(meta.local_box_xyxy, meta.source_to_local),
        expected_local,
        atol=atol,
        rtol=0,
    )
    assert torch.allclose(
        transform_box_xyxy(meta.context_box_xyxy, meta.source_to_context),
        expected_context,
        atol=atol,
        rtol=0,
    )

    # Local box in Context input must remain inside Context input.
    b = meta.local_box_in_context_input_xyxy
    assert -atol <= float(b[0]) <= float(b[2]) <= meta.context_input_hw[1] + atol
    assert -atol <= float(b[1]) <= float(b[3]) <= meta.context_input_hw[0] + atol

    # Round-trip representative source points (crop corners + center).
    pts = torch.tensor(
        [
            [float(lx0), float(ly0)],
            [float(lx1), float(ly1)],
            [(lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0],
        ],
        dtype=torch.float64,
    )
    local_pts = transform_points_xy(pts, meta.source_to_local)
    back = transform_points_xy(local_pts, meta.local_to_source)
    assert torch.allclose(back, pts, atol=atol, rtol=0)
