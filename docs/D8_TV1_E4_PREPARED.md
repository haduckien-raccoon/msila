# D8-TV1 — E4 Average baseline PREPARED

CODE PASS: **182 targeted tests passed, 4 optional real-backbone/GPU tests skipped**.
Riêng E4: **40 passed, 1 GPU smoke skipped**. Chỉ chạy CPU với fixture của API
DINOv3; không tải pretrained model hoặc chạy GPU local. EXP: **NOT RUN**.

E4 tái sử dụng pipeline hiện có: một frozen DINOv3 trích ba Local và ba Context
features; ba Adapter và ba projection chia sẻ theo tầng giữa hai view.
Context được align về tọa độ Local trước projection và **Mean(6)**.
ViT-B/16 dùng block 4/8/12 (1-based), C=768; sáu tensor sau projection là
`[B,64,32,32]` theo đúng `MULTIVIEW_FEATURE_KEYS`. Decoder trả raw logits
`[B,1,512,512]`. Block vật lý của các backbone khác lấy theo registry.

Sửa lỗi builder đọc `strategy` thay vì `paired_views.pair_strategy`:
`concat` và `sequential` hiện được áp dụng đúng từ YAML. Giữ alias `strategy`
cho config cũ. `concat` chạy một batch 2B; `sequential` chạy hai lượt B để giảm
peak memory. Test xác nhận cùng backbone và logits tương đương giữa hai cách.

Runner/trainer/inference hỗ trợ E4 `debug_pair` không cần lock D6, giới hạn
một epoch và tối đa hai updates. Resume kiểm tra config/checksum, optimizer/RNG
và cursor; run hoàn thành hợp lệ được skip. Kết quả smoke nằm riêng tại
`debug_d8_v3/debug_pair_r128_d512`, không phải evidence chọn r/d theo Average.
Main E4 vẫn BLOCKED chờ joint selection D10 và protocol main D11.

Tests kiểm tra Mean(6) chính xác, source order/shape/finite logits, gradient
cả Local/Context và từng nhánh Adapter sau khi gamma đổi, backbone frozen,
alignment theo tọa độ native ở tâm, góc và ảnh nhỏ có padding, geometry sai,
Hann reconstruction đúng H×W/orientation và sigmoid trước stitching.
Hai view dùng đúng một native synthetic sample; mask Local giữ nguyên đường E3.
Checkpoint loading tái lập logits; resume khớp control liên tục và không lặp
optimizer update khi DEV bị ngắt. Config drift/checkpoint hỏng bị từ chối;
checkpoint và lock legacy được giữ nguyên.

File sửa: `configs/g2_experiments.yaml`, `configs/g2_runner.yaml`,
`scripts/run_g2.py`, `src/models/msila.py`, `src/train/g2_e2.py`,
`tests/test_g2_tv1_e4.py`. Không sửa dataset/evaluator/Fusion TV2.
Không thêm synthetic anomaly riêng cho từng view, không chọn r/d hoặc tạo lock.
`E4−E3` chỉ tính tại D11 với main checkpoint thật và cùng protocol.

Tạo manifest không cần dataset/weights:

```bash
python scripts/run_g2.py --stage E4 --prepare --debug-pair 128 512 --device cpu
```

Artifact: `outputs/G2/D8/dinov3_vitb16/debug_pair_r128_d512/E4_prepared_manifest.json`.
Có resolved config, seed, config/source hashes, commit và trạng thái dirty.
`outputs/G2/D8/code_acceptance.json` và `targeted_tests.xml` lưu bằng chứng tests.
Manifest PREPARED; GPU/VRAM/main training và E4−E3 đều NOT RUN.
Model/training loop sẵn sàng về code; chưa nghiệm thu main training.

Sau khi cấu hình dataset, repo DINOv3 và checkpoint thật trên Colab:

```bash
python scripts/run_g2.py --stage E4 --smoke --categories rice --debug-pair 128 512 --device cuda --output-root /content/drive/MyDrive/msila/G2/D8
```

Thêm `--resume` khi ngắt; thêm `--inference` để load best.pt và xuất native DEV
maps sau smoke hoàn thành. Dùng cùng config/output root. Có thể đổi
`stage_output_roots.E4` trong `configs/g2_runner.yaml`; nếu provenance đổi thì
dùng output root mới, không ghi đè lịch sử. GPU test một batch là opt-in qua
`MSILA_RUN_E4_GPU_SMOKE=1` trên Colab, không cần lock D6.

Lệnh kiểm thử đã chạy:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/msila-d8-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_g2_tv1_e4.py tests/test_g2_tv1_e3.py tests/test_g2_tv1_runner.py tests/test_g2_tv1_model.py tests/test_g2_tv1_contract.py tests/test_feature_projection.py tests/test_dinov3_extractor.py tests/test_context_alignment.py tests/test_tiling.py
```

`py_compile` cho runner/model/trainer/tests và `git diff --check` đều PASS.
