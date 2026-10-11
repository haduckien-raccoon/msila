# D7-TV1 — E3 PREPARED

CODE PASS: 134 targeted tests passed, 3 optional real-backbone tests skipped.
Riêng E3: 27 passed, 1 GPU smoke skipped. CPU dùng fixture của API DINOv3;
không tải pretrained model hoặc chạy GPU local. EXP: NOT RUN.

E3 tái sử dụng `src.models.msila.E3`: Local → ba feature DINOv3 frozen →
ba Adapter độc lập → ba projection 1×1 → Mean(3) → Decoder.
ViT-B/16 dùng width 768, block **4/8/12 (1-based)**, feature grid 32×32,
projection về 64 kênh và raw logits `[B,1,512,512]`.
ViT-L/16 dùng 8/16/24; ViT-H/16+ dùng 11/21/32 theo registry.
Trace lưu block vật lý; tên `local_b4/b8/b12` là slot shallow/middle/deep.

`debug_pair_r128_d512` chỉ dùng kiểm thử, không phải r/d đã chọn.
CPU tests kiểm tra ba nguồn, Mean(3) chính xác, Adapter identity khi gamma=0,
gradient từng nhánh Conv sau cập nhật gamma, projection/Decoder, backbone frozen,
feature sai count/shape/NaN/Inf, config hash, load checkpoint và resume sau ngắt.
Resume khớp weights, optimizer, RNG và cursor với lượt chạy liền mạch;
config drift và kết quả bị gắn nhãn selected sai bị từ chối, checkpoint cũ được giữ.

Runner cho phép E3 smoke không cần `adapter_selection_lock.json`, giới hạn một
epoch và tối đa hai updates. Kết quả debug nằm trong namespace riêng theo pair.
Main E3 vẫn BLOCKED: chờ joint `(r,d,F)` tại D10 và tích hợp protocol main D11.
`E3−E2` chưa đánh giá; chỉ thực hiện tại D11 từ checkpoint main thật cùng protocol.
Không sửa Data, Evaluator hoặc Fusion TV2; không tạo selection lock.

Tạo manifest, không cần dataset/checkpoint:

```bash
python scripts/run_g2.py --stage E3 --prepare --debug-pair 128 512 --device cpu
```

Manifest hiện tại: `outputs/G2/D7/dinov3_vitb16/debug_pair_r128_d512/E3_prepared_manifest.json`.
Có resolved config, config/source hashes, seed, Git commit và trạng thái dirty.
`outputs/G2/D7/code_acceptance.json` và `targeted_tests.xml` lưu bằng chứng test.
Manifest có trạng thái PREPARED; GPU/VRAM/main training đều NOT RUN.

Trên Colab, sau khi cấu hình dữ liệu/DINOv3/checkpoint thật trong YAML hoặc CLI:

```bash
python scripts/run_g2.py --stage E3 --smoke --categories rice --debug-pair 128 512 --device cuda --output-root /content/drive/MyDrive/msila/G2/D7
```

Thêm `--resume` với cùng config và output root khi Colab ngắt; run hoàn thành
hợp lệ được skip. Có thể đổi `stage_output_roots.E3` trong `configs/g2_runner.yaml`.
Nếu provenance thay đổi, dùng output root mới; runner không ghi đè manifest cũ.
GPU smoke một batch của test là opt-in qua `MSILA_RUN_E3_GPU_SMOKE=1` trên Colab.

Lệnh test đã chạy (tắt plugin ROS tự nạp của môi trường local):

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_g2_tv1_e3.py tests/test_g2_tv1_runner.py tests/test_g2_tv1_model.py tests/test_g2_tv1_contract.py tests/test_feature_projection.py tests/test_dinov3_extractor.py
```

`py_compile` cho runner/trainer/tests và `git diff --check` đều PASS.
