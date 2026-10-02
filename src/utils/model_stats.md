# `utils/model_stats.py` — parameter audit cho Day 04

## Mục tiêu

Task này **không thay đổi kiến trúc Adapter**. Nó chỉ đo và kiểm tra số parameter để 9 candidate `r × d` có thể so sánh khoa học.

Runner nên gọi:

```python
from src.utils.model_stats import adapter_parameter_record

record = adapter_parameter_record(
    adapter,
    run_name="adapter_r64_d256",
)
print(record["trainable_params"])
```

Sau khi có toàn bộ candidate:

```python
from src.utils.model_stats import validate_rd_screen_growth

validate_rd_screen_growth(records)
```

Nếu count không khớp công thức hoặc một biến ngoài `r,d` bị đổi (`C`, `kernel_size`, `bias`), audit sẽ FAIL.

---

## Công thức của Adapter hiện tại

Project Adapter:

\[
C \rightarrow r \rightarrow \mathrm{DWConv}_{k\times k}
\rightarrow d \rightarrow C
\]

và:

\[
F' = F + \gamma \Delta F.
\]

Số weight:

\[
Cr + rk^2 + rd + dC.
\]

Nếu `bias=True`, bias là:

\[
r+r+d+C = 2r+d+C.
\]

`gamma` là một scalar trainable, nên:

\[
\boxed{
P(C,r,d,k)
=
Cr+rk^2+rd+dC+(2r+d+C)+1
}
\]

Nếu `bias=False`:

\[
\boxed{
P(C,r,d,k)=Cr+rk^2+rd+dC+1
}
\]

Đây là **công thức suy ra trực tiếp từ shape của các layer**, không phải công thức trích từ một paper.

PyTorch định nghĩa weight của `Conv2d` có shape:

\[
(C_{out}, C_{in}/groups, k_H, k_W).
\]

Với depthwise conv của project:

```python
in_channels  = r
out_channels = r
groups       = r
kernel       = k x k
```

nên weight chỉ có:

\[
r\times 1\times k\times k = rk^2
\]

parameter.

Nguồn học chính thức:
- PyTorch `nn.Conv2d`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch `nn.Module.named_parameters`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
- PyTorch `torch.numel`: https://docs.pytorch.org/docs/stable/generated/torch.numel.html

---

## Tăng parameter theo `r,d`

Giữ `C,k,d` cố định, tăng `r` một lượng \(\Delta r\):

\[
\Delta P_r
=
\Delta r\,[C+k^2+d+2]
\]

khi `bias=True`.

Giữ `C,k,r` cố định, tăng `d` một lượng \(\Delta d\):

\[
\Delta P_d
=
\Delta d\,[r+C+1].
\]

Vì mọi đại lượng đều dương:

\[
r\uparrow \Rightarrow P\uparrow,
\qquad
d\uparrow \Rightarrow P\uparrow.
\]

Do đó Day 04 không chỉ cần “count ra một số”; ta có thể kiểm tra **exact delta**. Nếu số parameter không tăng đúng công thức, khả năng cao candidate factory/Adapter đã bị thay đổi ngoài ý muốn.

---

## Grid hiện tại nếu `C=384`, `k=3`, `bias=True`

| `r` | `d` | Trainable params |
|---:|---:|---:|
| 32 | 128 | 66,401 |
| 32 | 256 | 119,777 |
| 32 | 384 | 173,153 |
| 64 | 128 | 83,137 |
| 64 | 256 | 140,609 |
| 64 | 384 | 198,081 |
| 128 | 128 | 116,609 |
| 128 | 256 | 182,273 |
| 128 | 384 | 247,937 |

Các số trên **chỉ đúng** khi Adapter của bạn đúng là:

```text
384 -> r -> DWConv3x3 -> d -> 384
bias=True
gamma=trainable scalar
```

Nếu `in_dim` khác 384 thì utility tự tính lại; **không hard-code bảng này vào code production**.

---

## Vì sao phải log trainable parameter?

Adapter là parameter-efficient transfer learning (PEFT), nên capacity/chi phí parameter là một biến quan trọng khi so các cấu hình. Houlsby et al. trực tiếp phân tích trade-off giữa adapter size, số trainable parameters và downstream performance.

Nguồn:
- Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019:  
  https://proceedings.mlr.press/v97/houlsby19a.html

Đây chỉ là căn cứ cho việc **báo cáo parameter efficiency**. Nó không chứng minh `r=64,d=256` hay bất kỳ candidate nào là tối ưu cho bài toán anomaly localization của đề tài.

---

## API nên dùng

### 1. Count model bất kỳ

```python
stats = parameter_stats(model)

print(stats.total_params)
print(stats.trainable_params)
print(stats.frozen_params)
print(stats.trainable_percent)
```

`named_parameters(remove_duplicate=True)` được dùng để không double-count một `nn.Parameter` bị share/tie qua nhiều module path.

### 2. Audit Adapter

```python
record = adapter_parameter_record(adapter)
```

PASS khi:

```text
actual total == analytical expected
trainable == total
run_name == adapter_r{r}_d{d}
```

### 3. Audit toàn grid

```python
records = [
    adapter_parameter_record(candidate_1),
    adapter_parameter_record(candidate_2),
    ...
]

validated = validate_rd_screen_growth(records)
```

Hàm kiểm tra:

```text
[PASS] no duplicate (r,d)
[PASS] same C
[PASS] same kernel_size
[PASS] same bias
[PASS] actual == closed-form expected
[PASS] increasing r -> exact expected Δparams
[PASS] increasing d -> exact expected Δparams
```

---

## Tích hợp vào `screen_adapter.py`

Ngay sau khi factory tạo Adapter:

```python
from src.utils.model_stats import adapter_parameter_record

param_record = adapter_parameter_record(
    adapter,
    run_name=candidate.run_name,
)
```

Ghi vào `config.yaml`:

```yaml
candidate:
  run_name: adapter_r64_d256
  r: 64
  d: 256
  trainable_params: 140609
```

Không dùng tổng params của cả DINO/Fusion/Decoder để thay cho `Adapter trainable params`; hai đại lượng này trả lời hai câu hỏi khác nhau.

---

## Lưu ý

Parameter count **không phải FLOPs, latency hay VRAM**. Candidate nhiều parameter hơn thường có capacity lớn hơn, nhưng không tự động cho AU-PRO tốt hơn.

Ngày 04 nên log ít nhất:

```text
r
d
adapter_trainable_params
validation metric
runtime (nếu đã có)
peak VRAM (nếu đã có)
```

nhưng lựa chọn candidate cuối cùng phải dựa trên validation protocol đã khóa, không dựa riêng vào số parameter.
