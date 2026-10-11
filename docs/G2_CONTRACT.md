# G2 contract v2 — D5/D6/D7/D8-TV1 → TV2

Manifest chung: `configs/g2_experiments.yaml`. E1/E2/E3/E4 có trong
`src.models.msila`; E5 chưa được triển khai.
Năm fusion candidate: `mean`, `concat`, `weighted_sum`, `gated`, `attention`;
đây là danh mục cho TV2, không phải năm method đã được triển khai/chạy.

## Model và batch

Giao diện E1–E5: `forward(image, *, return_trace=False)` nhận tensor hoặc batch,
trả raw logits
`[B,1,512,512]`; `return_trace=True` trả `(logits, trace: dict)` với
`trace["decoder_feature"]` là tensor đi vào Decoder. E1/E2/E3/E4 hỗ trợ ngay; TV2 giữ
cùng chữ ký cho E5. E4 bắt buộc nhận batch có Context và geometry. `model(image)` vẫn nhận tensor như runner G1; kích thước
khác 512 chỉ thuộc đường tương thích G1, batch G2 luôn khóa 512.

| Batch key | Contract |
| --- | --- |
| `image` | Local RGB float `[B,3,512,512]`, đã normalize DINOv3 bằng mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` |
| `mask` | Train: float binary `[B,1,512,512]`, 1=anomaly; cùng device với image |
| `meta` | Train: list B dict; giữ source/split/native size và tile coordinates từ loader G1 |
| `context` | Bắt buộc cho E4: cùng shape/device/dtype với image; FOV 768 được resize về 512, cùng tâm Local |
| `view_meta` | E4: list B dict, mỗi item có `geometry` từ pipeline hiện có để align Context→Local |

Inference có thể bỏ `mask/meta`. Dùng `validate_g2_batch(batch)` trước train;
model tự kiểm tra các key có mặt. E1/E2/E3 chỉ trích feature từ `image`, không chạy
Context. Không sửa Data/Evaluator; adaptation key nếu cần nằm ở boundary TV2.

## E2 và gradient

`DINOv3(feature_mode="deepest", frozen/eval) → 1 ResidualAdapter2d → BasicDecoder`.
Lấy duy nhất `b{extractor.depth}: [B,C,32,32]`, với `C=extractor.out_channels`;
ViT-S/16 C=384, ViT-B/16 C=768, cùng block cuối b12. Không Context/Projection/Fusion.
E2 kế thừa đường extract/Decoder của E1, tạo Adapter bằng `ResidualAdapterFactory`:
`C→r→DWConv→d→C`, `F'=F+gamma*ΔF`. `r=adapter_bottleneck_dim`,
`d=adapter_projection_dim`; `d` không đổi C của Decoder.

`gamma=0` phải identity chính xác; nếu Decoder có cùng state, logits E2 bằng E1.
Backward đầu: gamma/Decoder có gradient hữu hạn; các Conv của Adapter có gradient
bằng 0 hợp lệ vì gate=0. Sau gate update, nhánh Conv phải nhận gradient khác 0.
DINO không có gradient và không đổi state sau optimizer step. Optimizer chỉ lấy
`p.requires_grad`; E2 train đúng Adapter+Decoder, E1 chỉ Decoder.
Adapter params: `C*r+r*k²+r*d+d*C+(2*r+d+C nếu bias)+1`.

## Đổi backbone ở một chỗ

YAML mặc định `backbone.name: dinov3_vitb16`; chỉ đổi name để chọn checkpoint
trong `backbone.checkpoints` (cần file thật đúng path). Không khai báo tay
`channels/deepest_block/feature_blocks/patch_size`: `resolve_g2_config` suy ra từ
registry hiện có, extractor kiểm tra lại kiến trúc và state_dict lúc load.
Nếu file ở chỗ khác, đổi mapping hoặc đặt `backbone.weights` làm override.

`adapter.r/d: null` tự tính `C*r_ratio`, `C*d_ratio`, làm tròn lên bội `round_to=8`.
Đặt số nguyên dương để cố định từng width khi muốn cùng ngân sách Adapter.

| Backbone | C | Block cuối | r/d tự tính (C/6, 2C/3) |
| --- | --- | --- | --- |
| `dinov3_vits16`, `dinov3_vits16plus` | 384 | 12 | 64/256 |
| `dinov3_vitb16` | 768 | 12 | 128/512 |
| `dinov3_vitl16` | 1024 | 24 | 176/688 |
| `dinov3_vith16plus` | 1280 | 32 | 216/856 |

`fusion.dim=64`, `decoder.hidden_channels=64`, output 512 và Context FOV 768 px
là cấu hình độc lập với C; không tăng theo backbone. `build_g2_model` dựng
E1/E2/E3/E4; cùng một config dùng được khi so sánh ablation.

```python
import torch, yaml
from src.models.msila import build_g2_model, resolve_g2_config

with open("configs/g2_experiments.yaml") as stream:
    cfg = yaml.safe_load(stream)
model = build_g2_model(cfg, experiment="E2")  # Chọn E1 nếu cần baseline.
resolved = resolve_g2_config(cfg)  # Lưu config đã suy ra cùng run/checkpoint.
logits = model(batch)  # Không sigmoid trước BCEWithLogits.
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=1e-3, weight_decay=1e-4)
```

## Boundary sáu feature → Fusion → Decoder

Với ViT-S/S+/B (depth 12), giữ thứ tự `MULTIVIEW_FEATURE_KEYS`:
`local_b4, local_b8, local_b12, context_b4, context_b8, context_b12`.
Không dùng insertion order của mapping để stack; đọc theo tuple trên.
Sáu tensor đã align/project vào hệ tọa độ Local, cùng shape/device/dtype và
hữu hạn `[B,C,H,W]`; ở boundary này `C=fusion_dim=64`, `H=W=32` cho tile 512.
`fusion_dim` độc lập với Adapter `d` và backbone C. Boundary sáu feature dùng các key b4/b8/b12 làm slot shallow/middle/deep;
E4 ánh xạ block L/H từ registry vào các slot này, ghi block thật trong trace. E1/E2 không dùng boundary này và đã suy ra block cuối tự động.

Mọi fusion candidate đưa **một tensor** `[B,C,H,W]` cho Decoder, giữ C/H/W;
`concat` phải có projection `6C→C` bên trong. `mean` trung bình đều;
`weighted_sum` dùng trọng số nguồn được học; `gated` dùng gate theo input;
`attention` có thể dùng `AttentionFusion` hiện có. Module attention cũ trả dict
`{feature, attention, logits}`: wrapper TV2 lấy `feature` để nối Decoder và giữ
diagnostics trong trace. Kiểm tra bằng `validate_multiview_features` và
`validate_g2_fused_feature`; không đổi API các module Day-2 đang tồn tại.
Decoder dùng `BasicDecoder(C,64)` với `output_size=(512,512)`.

## Protocol và kiểm tra

Giữ 8 category, checkpoint frozen, seed 2026/dev seed 17017, 20 epochs,
batch 4, AdamW lr 1e-3/weight decay 1e-4 và BCE+positive-mask Dice như G1.
Synthetic chỉ sinh từ TRAIN/good (đổi theo epoch); DEV từ VALIDATION/good cố định,
không trùng source. Chọn checkpoint bằng synthetic DEV AU-PRO@0.05, tie lấy epoch
sớm nhất; TEST_PUBLIC chỉ dùng chấm cuối. Mỗi category/experiment có model và
optimizer mới; giữ split, synthetic protocol, seed/order và update budget chung.
Tỷ lệ `r/C=1/6,d/C=2/3` là default khai báo, chưa phải kết quả chọn thực nghiệm.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -rs tests/test_g2_tv1_model.py tests/test_g2_tv1_contract.py
```

Fixture chỉ xác minh kiến trúc/API/autograd. Test integration tự tìm asset local
hoặc nhận `MVTEC_AD2_ROOT`, `DINOV3_REPO`, `DINOV3_WEIGHTS` (tùy chọn
`G2_CATEGORY`, `DINOV3_MODEL`), chạy một batch TRAIN/good qua loader/synthetic/loss
hiện có và một optimizer step. Thiếu asset: SKIP/**NOT RUN**; đường explicit sai:
FAIL. Không chạy full training trong D5-TV1.

## D6-TV1: Colab Adapter screening

`notebooks/G2_02_Adapter_Screening.ipynb` có năm phần: Setup, Preflight,
Smoke một cặp/category, Full screening và xuất kết quả. Notebook gọi
`scripts/g2_colab_screening.py`, dùng lại `run_g2.execute_job`, trainer E2,
validator và selector hiện có. Không chạy E2 main, không đọc TEST để lựa chọn.

Batch mặc định trong YAML vẫn là 4. Notebook có tùy chọn đo batch trên T4/L4/A100
bằng hai TRAIN steps của cặp ViT-B lớn nhất (256,768), giữ 20% VRAM dự phòng.
Batch được khóa chung cho toàn bộ study trong `colab_batch_profile.json` trước
khi chạy; batch lớn đổi số optimizer updates trong 20 epochs, được ghi trong
budget plan. Đặt `BATCH_SIZE=4` để giữ batch của contract. Không đổi batch sau
một run hoặc khi resume; tăng batch/đổi config hoặc commit phải dùng RUN_ID mới.
Capacity probe và smoke không tính vào PASS/72. Preflight trên Colab chưa chạy
thì ghi NOT RUN, kể cả khi kiểm tra CPU fixture đã PASS.

Factory cache chỉ giữ DINO frozen/eval; Adapter, Decoder, optimizer mới cho
mỗi category/r/d. CPU RNG sau khởi tạo DINO được replay để giữ cùng seeded
head initialization với factory không cache. API optional `extractor`,
`model_factory`, `on_checkpoint` phục vụ screening; đường CLI cũ vẫn giữ mặc
định. Checkpoint callback backup best/last + SHA256, log và config sang Drive
mỗi lần trainer lưu (20 steps, cuối epoch); cuối run sync cả metrics và report.

Chọn nhóm bằng `RUN_CATEGORIES` và `RUN_PAIRS=['64:256', '128:512']`; all/all
là 72 jobs và chỉ chạy khi bật riêng `RUN_FULL_SCREENING=True`. Validator
xác minh completed runs trước khi skip; run bị ngắt resume optimizer/RNG/cursor.
Report luôn có 72 hàng và 9 macro; metric thiếu là null, không điền 0.
`adapter_selection_lock.json` chỉ được selector hiện có tạo sau đủ 72 real
full CUDA kết quả hợp lệ. Tie-break: macro cao nhất, ít Adapter parameters,
r nhỏ hơn, d nhỏ hơn. Artifacts/checkpoint được lưu từng run lên Drive.

Kiểm tra code bằng CPU (fixture không phải chứng cứ thực nghiệm):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/msila_g2_mpl .venv/bin/python -m pytest -q tests/test_g2_colab_screening.py tests/test_g2_tv1_runner.py tests/test_g2_tv1_model.py tests/test_g2_tv1_contract.py
```

## D7: E3 và dữ liệu cho TV2

E3: một ảnh Local → frozen DINOv3 (ba layer) → ba ResidualAdapter2d độc lập
cùng r/d → ba projection 1×1 C→`fusion.dim` → Average/MeanFusion → BasicDecoder.
Không tạo Context. S/B lấy 4/8/12; H+ lấy 11/21/32; backbone khác lấy
`backbone_spec(name).blocks`. `FeatureSelector("multi_local")` và
`SixFeatureProjection.project_sources` dùng key b4/b8/b12 làm **slot**
shallow/middle/deep; `trace["source_blocks"]` ghi block thật. Trace cũng có
`dino`, `adapted`, `projected`, `num_sources=3`, `decoder_feature`.

E3 giữ loss, seed/order, TRAIN/DEV sources, synthetic protocol và update budget
của E2; từng source dùng cặp r/d từ `adapter_selection_lock.json`. Width projection
mặc định 64, độc lập với C/d; Decoder E3 nhận width này, E2 nhận C. Do đó params
hai kiến trúc khác nhau và được ghi rõ trong `metrics.json.parameter_report`.
Tại gamma=0, mỗi Adapter identity; gradient Conv bằng 0 ở bước đầu là hợp lệ.

Runner `--stage E3 --categories all --device cuda --resume` yêu cầu lock đủ
72 run hợp lệ như E2. Output: `<output_root>/<backbone>/full/E3/<category>/`.
`best.pt`/`last.pt` chứa `adapters`, `projection`, `decoder`, optimizer, RNG,
cursor và config/hash; DINO frozen được định danh bằng checksum. Resume giữa
epoch; run hoàn thành hợp lệ được skip. Smoke ở thư mục riêng, không tính PASS/8.
Runner đọc lại protocol/checkpoint D6 bất biến cho đúng bản source D6 đã biết;
thay đổi Data/Evaluator/loss/checkpoint hoặc scientific settings vẫn bị chặn.
E3 lưu thêm hash implementation hiện tại trong config để kiểm tra resume.

Sau stage E2/E3, runner xuất `E3_minus_E2_inputs.json` và `.csv` trong thư mục
`full/` (hoặc `smoke/`): luôn có 8 hàng, metric synthetic DEV AU-PRO@0.05 của
từng model, r/d, seed, budget, shared protocol hash, checkpoint/config/metric
paths và hashes. `status=PASS` chỉ khi đủ 8 cặp **real full** hợp lệ; thiếu E2/E3
ghi rõ ở từng hàng. TV2 tính E3−E2 và macro từ các input này; đây là DEV tổng hợp,
không phải kết quả TEST_PUBLIC. Để chấm TEST bằng evaluator của TV2:

```python
cfg = payload["config"]  # Đọc best.pt bằng load_checkpoint_payload(...).
model = build_g2_model(cfg, experiment="E3")
from src.train.g2_e2 import trainable_modules
trainable_modules(model, "E3").load_state_dict(payload["model_state"], strict=True)
model.eval()  # Raw tile logits; sigmoid rồi Hann stitch theo native coordinates.
```

## D8: E4 — Local/Context, align rồi Average Fusion

E4 lấy ba layer mỗi view qua **cùng** frozen DINO (concat batch 2B hoặc sequential).
Ba Adapter và ba projection của E3 được **chia sẻ theo layer giữa hai view**:
r/d giữ cặp đã khóa; trainable params và Decoder initialization bằng E3 với cùng
seed. Context đi qua Adapter → `ContextToLocalAligner` → projection, sau đó
`MeanFusion` trung bình đúng sáu nguồn theo `MULTIVIEW_FEATURE_KEYS`. Fusion
width và Decoder giữ như E3; logits `[B,1,512,512]`.

`src.train.g2_context.PairedG2Tiles` chỉ bọc index của G1TileDataset: đọc một
native synthetic sample, crop Local 512 và Context 768 rồi resize Context về
512 bằng `extract_local_context` đã có. Local/mask/order/budget giống E3; không
sinh anomaly riêng trên từng view. Padding reflect (replicate cho ảnh nhỏ) giữ
hàm tiling hiện có. Geometry dùng pixel-edge coordinates trên canvas padding
ảo; `view_meta` giữ cả box native và padding để audit. Thiếu Context/geometry,
geometry singular hoặc không chứa Local sẽ bị từ chối.

Trace E4: `dino` sáu feature block thật, `adapted` sáu slot, `aligned_context`
ba feature đã đưa về Local, `projected` sáu nguồn cùng width, `source_blocks`,
`num_sources=6`, `decoder_feature`. Alignment dùng deterministic bilinear
sampling để giữ strict CUDA backward; không thêm tham số trainable.

```bash
# Dùng cùng config/backbone/output_root đã tạo selection lock và E3.
.venv/bin/python scripts/run_g2.py --stage E4 --categories all --device cuda --resume
.venv/bin/python scripts/run_g2.py --stage E4 --categories all --device cuda --inference
# Smoke cũng cần lock thực; CPU fixture lock chỉ dùng trong unit tests.
.venv/bin/python scripts/run_g2.py --stage E4 --categories all --device cuda --smoke --resume
```

Checkpoint/results: `<output_root>/<backbone>/full/E4/<category>/` chứa
`best.pt`, `last.pt`, sidecar SHA256, config/hash, train log, metrics/parameter
report. Resume và skip giữ kiểm tra provenance; protocol D6/D7 và checkpoint E3
đúng các bản source đã audit được đọc nguyên trạng, không sửa hash lịch sử.
Thay đổi scientific settings hoặc dependency chưa audit vẫn BLOCKED.

`--inference` chỉ phục hồi best.pt của run hoàn thành hợp lệ, không train;
xuất `inference/metrics.json` và `predictions/*_score.npy`, `*_mask.npy` theo
DEV cố định với metadata/hash. Sigmoid **trước** Hann stitching trên native
coordinates; map cuối giữ nguyên H×W, không resize. TV2 chấm TEST_PUBLIC bằng
evaluator riêng: restore như E3 nhưng `experiment="E4"`; dùng
`predict_native_e4(model, native_image, cfg, device)` cho full native map.

Runner xuất `full/E4_minus_E3_inputs.json` và `.csv`: tám hàng có scalar
synthetic DEV AU-PRO@0.05, r/d/seed/budget/protocol và paths/hashes của E3/E4.
TV2 tính hiệu theo category; thiếu run ghi `MISSING_E3`/`MISSING_E4`. Chỉ PASS
khi đủ tám cặp real full CUDA hợp lệ. Smoke/fixture ghi SMOKE_READY, không
tính vào coverage thực. Thiếu Adapter lock: BLOCKED; thiếu CUDA/assets: NOT RUN.
