# D9-TV1 — E5 Fusion technical preflight

## Kết quả nghiệm thu

- **45/45 tổ hợp PASS trên CPU**: `r=[64,128,256]`, `d=[256,512,768]` × 5 Fusion.
- **462 tests PASS, 4 tests SKIP**, không có failure/error. Bốn tests bỏ qua yêu cầu GPU hoặc checkpoint/dữ liệu thật và được ghi trong `targeted_tests.xml`.
- **GPU / peak VRAM / OOM: NOT RUN**; chưa được cấp phiên Colab. Không chạy GPU local và không suy diễn khả năng chạy trên T4 từ kết quả CPU.
- **Số screening/main training jobs: 0**. Không đo AU-PRO, xếp hạng, chọn `(r,d,F)` hoặc tạo selection lock.

Branch làm việc: `g2/member1`. Đã fetch các branch G2 và fast-forward từ
`3a78b87e247144488d52a703d2bf03c6a99b8a3d` lên integration
`41cc70a681e71d3113a9a59224691f221a6e72c3`, nơi D8-TV2 đã merge. Không có
conflict. Commit này là **base**, chưa chứa thay đổi D9; manifest ghi
`git_dirty=true` cùng SHA-256 của các source đã chạy để phân biệt code hiện tại.

## Code E5

`E5` kế thừa E4, dùng nguyên đường frozen DINOv3 → 3 Adapter chia sẻ giữa
Local/Context → align Context về Local → projection → Fusion → Decoder.
Factory chỉ thay Fusion. Adapter, projection, decoder được khởi tạo trước Fusion
để cùng seed và cùng `(r,d)` có cùng trọng số ban đầu giữa cả năm phương pháp.
Tests kiểm tra trạng thái các thành phần này, sáu gradient nguồn, chiều tensor,
thứ tự nguồn và backbone frozen.

| Tên người dùng | ID canonical | Module | Params Fusion tại D=64 |
| --- | --- | --- | ---: |
| `average` | `mean` | MeanFusion hiện có | 0 |
| `weighted` | `weighted_sum` | WeightedSumFusion TV2 | 6 |
| `concat` | `concat` | ConcatFusion TV2 | 24,640 |
| `gated` | `gated` | GatedFusion TV2 | 390 |
| `attention` | `attention` | PositionAwareAttentionFusion TV1 | 70 |

Attention mới dùng Conv1×1 `D→1` tại từng vị trí và sáu source bias:
logits `[B,6,H,W]` → softmax `dim=1` → weighted sum `[B,D,H,W]`.
Tests dùng hai nguồn có cùng global average nhưng vị trí kích hoạt khác nhau để
chứng minh trọng số thay đổi theo vị trí. Attention khởi tạo đều; không có global
pooling. `AttentionFusion` global cũ và checkpoint/API cũ được giữ nguyên.

Thứ tự cố định:
`local_b4, local_b8, local_b12, context_b4, context_b8, context_b12`.
ViT-B/16 dùng block registry 4/8/12, C=768; `adapter.d` độc lập với `fusion.dim=64`.
Config được chuẩn hóa alias, round-trip YAML và hash riêng cho đủ 45 tổ hợp.
Factory kiểm tra shape/dtype/device/finite đầu vào và đầu ra; tests phủ các đầu
vào thiếu/thừa nguồn, sai chiều, sai dtype và NaN/Inf.

Không sửa Data/Evaluator hoặc các module Fusion TV2. Cấu hình E2 cũ không có
trường `fusion` vẫn resolve và load/resume được; có regression test riêng.
`--stage E5` vẫn trả `BLOCKED: pending_joint_selection_D10` trước khi nạp dữ liệu
hay training, phù hợp contract. Không triển khai D10/D11 trong task này.

## Phạm vi ma trận CPU và artifact

Ma trận dùng frozen mock features `[1,768,4,4]` để giảm tài nguyên CPU; đây không
phải evidence từ pretrained DINOv3. Các thành phần E5 còn lại là module thật.
Mỗi tổ hợp kiểm tra hai lần forward/backward với loss kỹ thuật
`mean(logits²)`. Một bước SGD giữa hai probe kích hoạt gamma, xác nhận nhánh
Conv của Adapter nhận gradient sau khi gate thay đổi; không dùng dataset/epoch
hay objective để lựa chọn hyperparameter. Fusion output `[1,64,4,4]`, Decoder
logits `[1,1,512,512]`. Integration tests bổ sung kiểm tra feature grid **32×32**
qua extractor fixture tại API builder, cũng không tải pretrained.

Artifact tại `outputs/G2/D9/`:

- `fusion_preflight_45.csv`: đủ 45 dòng; PASS/FAIL, shape, trainable params thực,
  số params Adapter/Fusion, config hash, lỗi và trạng thái GPU riêng.
- `cpu/r{r}_d{d}_{fusion}.json`: config và bằng chứng gradient/shape/frozen của
  từng tổ hợp, lưu ngay sau mỗi probe hoàn thành.
- `run_manifest.json`: base/integration commit, seed, source hashes, số PASS và
  trạng thái pending selection. Các tổ hợp lỗi vẫn xuất dòng FAIL kèm lý do.
- `gpu_preflight_plan.json`: shapes/batch probes và chính sách ghi OOM; không có
  số VRAM giả. CSV để trống `peak_vram_if_measured`.
- `targeted_tests.xml`, `code_acceptance.json`: bằng chứng tests và phạm vi nghiệm thu.

## Kế hoạch GPU trên Colab

Chạy smoke kỹ thuật trên T4/L4/A100 khi được cấp runtime, không train 45/360 jobs.
Load checkpoint DINOv3 ViT-B/16 thật một lần, giữ eval/frozen; tuần tự dựng các
head E5 từ cùng config/seed. Record GPU model, dtype, commit/source hashes và
checksum checkpoint. Giải phóng head/optimizer/tensor giữa các probe để không
gộp bộ nhớ của nhiều tổ hợp.

| Tensor | Shape |
| --- | --- |
| Local RGB | `[B,3,512,512]` |
| Context RGB sau resize native FOV 768×768 | `[B,3,512,512]` |
| 3 Local + 3 Context DINO features | 6 × `[B,768,32,32]` |
| 6 features sau align/projection | 6 × `[B,64,32,32]` |
| Spatial Attention logits/weights | `[B,6,32,32]` |
| Fusion output | `[B,64,32,32]` |
| Decoder logits | `[B,1,512,512]` |

Giữ geometry Context→Local của E4 khi resize; không coi hai view là cùng tọa độ.
Thử batch tăng dần `1,2,4,8,16`, float32 theo config hiện có. Với `concat`
paired extraction, backbone nhận batch `2B`; `sequential` dùng hai forward batch B.
Đo peak allocated/reserved sau synchronize và reset peak stats, bao gồm
forward/backward và trạng thái optimizer nếu cần dùng để dự kiến training.
Ghi PASS/FAIL/OOM theo **từng `(r,d,F,batch)`**, kèm lỗi; giữ candidate khi một
batch OOM, thử batch nhỏ hơn. Batch phù hợp là kết quả đo riêng của runtime,
không phải siêu tham số được D9 chọn. D10 phải công bố cùng training budget
trước khi chọn joint `(r,d,F)`.

## Chạy lại

Từ repo root, dùng Python environment đã có dependencies:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/msila-d9-mpl \
  .venv/bin/python scripts/run_g2.py --stage fusion_preflight --device cpu
```

Hoặc gọi helper trực tiếp với model config:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/msila-d9-mpl \
  .venv/bin/python scripts/g2_fusion_preflight.py \
  --config configs/g2_experiments.yaml --output-root outputs/G2/D9
```

Targeted regression đã chạy:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/msila-d9-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests/test_g2_tv1_e5.py tests/test_g2_tv1_e4.py tests/test_g2_tv1_e3.py \
  tests/test_g2_tv1_runner.py tests/test_g2_tv1_model.py tests/test_g2_tv1_contract.py \
  tests/test_g2_adapter_preflight.py tests/test_g2_tv2_fusions.py \
  tests/test_attention_fusion.py tests/test_feature_projection.py \
  tests/test_dinov3_extractor.py --junitxml=/tmp/msila-d9-targeted.xml
```

Files code/config/test thay đổi:
`src/models/msila.py`, `src/models/fusion_factory.py`,
`src/models/position_attention_fusion.py`, `scripts/g2_fusion_preflight.py`,
`scripts/run_g2.py`, `configs/g2_experiments.yaml`, `configs/g2_runner.yaml`,
`tests/test_g2_tv1_e5.py`, `tests/test_g2_tv1_contract.py`.
