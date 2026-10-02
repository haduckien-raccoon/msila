# `geometry/view_meta.py` — Local ↔ Context geometry metadata

**Mốc thiết kế:** 09/2026. Module này chỉ lưu **hình học**, không crop/resize ảnh và không học tham số.

## 1. Contract của đề tài

Với pipeline hiện tại:

- Local source crop: `512×512`.
- Context source crop: `768×768`, chứa Local.
- Local input: `512×512`.
- Context input: `512×512`.
- Box dùng chuẩn half-open: `[x0, y0, x1, y1)`.

Output chính của `build_view_meta(...)` gồm:

- `local_box_xyxy`, `context_box_xyxy` — crop box trên ảnh gốc.
- `local_scale_xy`, `context_scale_xy` — scale từ ảnh gốc sang input.
- `source_to_local`, `source_to_context` — ma trận 3×3.
- `local_to_context`, `context_to_local` — mapping trực tiếp giữa 2 view.
- `local_box_in_context_input_xyxy` — vị trí Local trong hệ tọa độ Context sau resize.

## 2. Công thức dùng trong code

Với crop `B=[x0,y0,x1,y1)` và output `(Ho,Wo)`:

$$
s_x=\frac{W_o}{x_1-x_0},\qquad
s_y=\frac{H_o}{y_1-y_0}
$$

Ma trận affine homogeneous từ ảnh nguồn sang view:

$$
T_{src\to view}=
\begin{bmatrix}
s_x&0&-s_xx_0\\
0&s_y&-s_yy_0\\
0&0&1
\end{bmatrix}.
$$

Với điểm homogeneous \(\tilde p=[x,y,1]^T\):

$$
\tilde p_{view}=T_{src\to view}\tilde p_{src}.
$$

Mapping Local → Context được **compose**, không tự đặt công thức riêng:

$$
T_{L\to C}=T_{src\to C}\,T_{L\to src}
=T_{src\to C}\,(T_{src\to L})^{-1}.
$$

Đây là affine geometry tiêu chuẩn; không phải một kiến trúc AI mới.

### Trường hợp khóa 512 / 768 → 512

Nếu Local đồng tâm trong Context:

$$
m=\frac{768-512}{2}=128.
$$

Context scale:

$$
s_C=\frac{512}{768}=\frac23.
$$

Do đó Local chiếm trong Context-input:

$$
[128,128,640,640]\times\frac23
=[85.333\ldots,85.333\ldots,426.666\ldots,426.666\ldots].
$$

Điều này giải thích vì sao **không được fuse Local và Context theo cùng chỉ số pixel/patch nếu chưa alignment**.

## 3. Quy ước quan trọng

Ma trận trong module ánh xạ **continuous pixel-edge coordinates**. Nó dùng để lưu crop geometry, map point/box/feature region và audit pipeline. Nó **không mô phỏng chính xác sample-center của kernel bicubic/bilinear** bên trong torchvision/PIL.

Dùng `float64` cho metadata/matrix vì dữ liệu rất nhỏ nhưng cần composition/inverse ổn định. Feature tensor của backbone vẫn có thể là float32/bfloat16.

`ViewGeometryMeta` là `frozen dataclass`: cùng input box + size ⇒ cùng metadata, không RNG, nên PASS tiêu chí **mapping deterministic**.

## 4. Dùng với `data/multiview_transform.py`

```python
from src.geometry.view_meta import build_view_meta_from_transform_meta

views = multiview_transform(image)
geom = build_view_meta_from_transform_meta(
    views["meta"],
    local_input_hw=(512, 512),
    context_input_hw=(512, 512),
)

print(geom.local_to_context)
print(geom.local_box_in_context_input_xyxy)
```

Nếu cần DataLoader collate:

```python
views["geometry"] = geom.as_tensor_dict()
```

## 5. PASS

Chạy:

```bash
pytest -q tests/test_view_meta.py
```

PASS khi crop/scale/matrix đúng, Local nằm trong Context, transform inverse/composition đúng và hai lần build từ cùng input cho kết quả giống hệt nhau.
