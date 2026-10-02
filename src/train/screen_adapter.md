# `train/screen_adapter.py` — Day 04 controlled Adapter screen

## 1. Có cần file runner riêng không?

**Có.** `residual_adapter.py` chỉ định nghĩa kiến trúc Adapter; `adapter_factory.py` tạo candidate `(r,d)`; `cached_dataset.py` đọc feature cache. Day 04 vẫn cần một runner để bảo đảm 9 candidate được train bằng **cùng một protocol**, không sửa code thủ công giữa các run.

Runner này không tự nghĩ ra Fusion/Decoder/loss mới. Nó gọi đúng pipeline hiện có thông qua một hook module của project.

## 2. Biến duy nhất được phép thay đổi

Adapter đã được định nghĩa ở module trước:

\[
F' = F + \gamma R(F)
\]

với project branch:

\[
C \rightarrow r \rightarrow \mathrm{DWConv} \rightarrow d \rightarrow C.
\]

Ở Day 04:

- \(r=\) `bottleneck_dim`
- \(d=\) `projection_dim`

là **hai biến duy nhất được screen**.

Công thức/kiến trúc Adapter không được phát minh trong runner này. Nguồn nền tảng cần đọc ở `models/residual_adapter.md`: ResNet (He et al., CVPR 2016), Houlsby Adapter (ICML 2019), AdaptFormer (NeurIPS 2022), convolutional visual adapter/ConvPass, MobileNet depthwise convolution và ReZero.

## 3. Điểm runner này khóa thêm

Ngoài việc dùng cùng `seed/epoch/loss/split/LR/...`, runner khóa bằng fingerprint:

```text
grid
train records
val records
feature-cache protocol
Fusion config
Decoder config
loss config
optimizer + LR
epochs / batch size
augmentation metadata
hook source
non-Adapter model structure
non-Adapter initial weights
```

Nếu candidate sau làm thay đổi Fusion/Decoder hoặc initialization của phần cố định, runner dừng thay vì âm thầm tiếp tục.

### Vì sao reset RNG sau khi tạo Adapter?

Hai candidate khác kích thước có số parameter khác nhau. Nếu khởi tạo:

```text
Adapter -> Fusion -> Decoder
```

bằng cùng global seed nhưng không reset RNG, Adapter lớn hơn sẽ tiêu thụ nhiều random numbers hơn và có thể làm **Fusion/Decoder nhận initial weights khác**. Khi đó thí nghiệm không còn chỉ thay `(r,d)`.

Runner vì thế:

1. khởi tạo Adapter trong RNG scope riêng;
2. reset cùng seed;
3. mới tạo Fusion/Decoder;
4. hash phần non-Adapter để kiểm chứng.

Đây là kiểm soát thực nghiệm, không phải một kiến trúc neural network mới.

## 4. Hook bắt buộc của project

Do task này không cung cấp code Fusion/Decoder/loss, runner **không giả định API** của chúng. Tạo một module, ví dụ:

```text
train/day04_project_hooks.py
```

có đúng hai hàm:

```python
def build_model(*, adapter, config):
    # Trả về model hiện có:
    # cached features -> adapter -> SAME fusion -> SAME decoder
    return model


def step(*, model, batch, stage, config):
    if stage in {"train", "val"}:
        prediction = model(batch)
        loss = ...  # SAME loss đang dùng của project
        return {
            "loss": loss,
            "metrics": {
                # optional scalar validation/training metrics
            },
        }

    if stage == "predict":
        prediction = model(batch)
        return {"prediction": prediction}
```

`build_model()` phải gắn **chính object `adapter` được runner truyền vào**. Runner từ chối trường hợp hook tự tạo một Adapter khác.

## 5. Protocol cố định

Điền `configs/day04_train_protocol.yaml` **một lần trước candidate đầu tiên**. Các field đang để `null`/`REPLACE_ME` phải lấy từ pipeline hiện có; không được chọn số mới chỉ để runner chạy được.

Đặc biệt:

```text
DINO checkpoint       SAME
DINO frozen           true
blocks                 SAME
Local/Context          SAME
alignment              SAME
cache                  SAME
Fusion                 SAME
Decoder                SAME
loss                   SAME
optimizer              SAME
LR                     SAME
epochs                 SAME
batch size             SAME
seed                    SAME
split                   SAME
augmentation            SAME
```

Feature cache phải đi qua `CachedFeatureDataset`; runner không có DINO inference.

## 6. Chạy

```bash
python train/screen_adapter.py \
    --r 64 \
    --d 256 \
    --category fabric \
    --seed 42
```

Runner xác nhận `(64,256)` thuộc `configs/day04_adapter_grid.yaml`.

Sau lần đầu của `fabric`, một protocol lock được tạo. Candidate tiếp theo mà thay seed/LR/split/model/... sẽ FAIL.

## 7. Output

```text
outputs/day04/
├── _protocol_locks/
│   ├── fabric.yaml
│   └── fabric.sha256
│
├── adapter_r32_d128/
│   ├── config.yaml
│   ├── best.pt
│   ├── train_log.csv
│   └── predictions/
│       ├── manifest.jsonl
│       └── *.pt
│
└── adapter_r32_d256/
    └── ...
```

Runner không overwrite run cũ.

## 8. Best checkpoint

Mặc định protocol dùng:

```yaml
checkpoint:
  monitor: val_loss
  mode: min
```

Nếu pipeline hiện tại đã dùng metric validation khác, ví dụ `val_aupro_005`, đổi **trước candidate đầu tiên**, rồi giữ nguyên cho cả grid. Không đổi criterion giữa các candidate.

## 9. Reproducibility — nguồn kỹ thuật

Các cơ chế seed/determinism bám tài liệu chính thức PyTorch:

- `torch.manual_seed`: seed RNG.
- `torch.use_deterministic_algorithms`: yêu cầu deterministic implementation khi có thể; PyTorch lưu ý setting này **không một mình bảo đảm reproducibility toàn hệ thống**.
- `DataLoader(generator=...)` và worker seed: cùng generator/worker-seeding giúp tái lập sampling order.
- AMP hiện hành: `torch.autocast` + `torch.amp.GradScaler` cho FP16.

Nguồn:
- https://docs.pytorch.org/docs/main/notes/randomness.html
- https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html
- https://docs.pytorch.org/docs/main/data.html
- https://docs.pytorch.org/docs/main/notes/amp_examples.html

### Lưu ý khoa học

`same seed` không có nghĩa kết quả chắc chắn bitwise-identical giữa GPU, CUDA/PyTorch version hoặc operator khác nhau. Vì vậy `config.yaml` cũng ghi `torch_version`, CUDA version và git commit nếu có.

## 10. Điều không nên làm

Không:

```text
candidate 1 -> sửa LR
candidate 2 -> sửa Fusion
candidate 3 -> đổi split
candidate 4 -> đổi augmentation
```

rồi gọi bảng cuối là “Adapter r×d ablation”.

Nếu muốn thử LR/Fusion khác, đó phải là **experiment khác** sau khi Day 04 đã khóa/chọn cấu hình Adapter.
