# Báo cáo sửa MS-ILA và full-scale training

Đã sửa trực tiếp repository, mở rộng cache builder, trainer và native inference hiện có. Không commit/push, không sửa kết quả hay checkpoint lịch sử. Base commit: `6ad23e66f10af08a70e6f5731de336db7de159c8`.

## Lỗi và thay đổi

- Loader trước đây ghép theo tên và fallback xuyên split. Nay khóa category/split/defect/nested identity; normal nhận zero mask, abnormal thiếu/ambiguous GT bị từ chối khi đánh giá pixel. Giữ native H×W, resize GT bằng nearest.

- Đường Day-05 cũ khóa ViT-S, block 4/8/12 và r32/d384. Đường full-scale kiểm tra C/depth/patch của model đã load theo registry; block theo 1/3, 2/3 và cuối depth, có override. r/d độc lập với fusion 64; backbone frozen; cache/checkpoint có SHA256 và identity. Legacy vẫn đọc được.

- Upstream Torch Hub có thể tái dùng cache theo basename weights. Loader local mới dựng đúng architecture rồi đọc trực tiếp file được chỉ định với strict state dict; test hai file cùng tên nhưng nội dung khác đã PASS.

- DEV cutpaste cũ có bbox ratio envelope 0.002–0.08 trên crop 768; đây chỉ là audit cấu hình. Không có GT lịch sử để đo connected-component distribution, đã ghi rõ trong initial audit và cung cấp CLI audit GT.

- DEV mới là synthetic proxy native: bins 4–32, 33–128, 129–2048 px; có căn cứ theo patch 16×16 và một nhóm control, không tuyên bố khớp lỗi thật TEST. Hash data seed 42 chia nguồn TRAIN normal thành TRAIN core, DEV tiny, DEV mixed riêng biệt. Toàn bộ strata DEV và mọi Local tile được giữ. Pixels ngoài GT giữ nguyên.

- AU-PRO005 giữ semantics evaluator gốc, bổ sung background chunks trên disk và đối chiếu exact với continuous/tied scores. Checkpoint chọn bằng 0.5 tiny + 0.5 mixed trên cached native tiles. Ranking dùng full native Hann maps và mean theo mọi seed đã khai báo; TEST_PUBLIC chỉ sau selection lock hoàn chỉnh.

- Strict deterministic CUDA chặn bilinear backward của PyTorch trong decoder/R2. Backend gather/index-select mới giữ kernel, half-pixel geometry và layout tham số; CPU forward/gradient parity PASS. CUDA vẫn là gate cần GPU thực tế.

## Phạm vi và artifact

Grid YAML giữ đủ 5 backbone × 9 adapter pairs × 3 representation × 3 seed × 8 category = **3.240 jobs**, mỗi job **150 epoch, batch 64**, Local 512 / Context 768 → network 512, overlap 128. Có single mode và `--train-all`; không giảm grid, resolution hay epoch khi thiếu tài nguyên. OOM ghi lỗi và giữ cấu hình.

Mỗi job gọi optimizer loop thực, ghi checkpoint, update/epoch log, DEV AU-PRO005, parameter counts, VRAM/runtime và manifest. Resume kiểm tra checksum/config, khôi phục model/optimizer/RNG/sampler/partial progress; không lặp prefix đã commit. Checkpoint mỗi 100 update và cuối epoch; update chưa commit có thể phải chạy lại. Native map resume cũng xác minh provenance/hash.

## Kiểm thử và giới hạn

**707 passed, 20 skipped, 0 failures, 0 errors; 2 warning Agg của notebook legacy**. Compileall, CLI help và `git diff --check` PASS. JUnit và danh sách skip chính xác ở validation JSON. CPU fixture dùng optimizer thật, 2 epoch để kiểm tra kỹ thuật; không phải kết quả thực nghiệm DINOv3/MVTec.

Audit fixture constant image 512×640 đo đủ 16 strata DEV tiny và 24 strata DEV mixed: một component hợp lệ/stratum, đúng size/edge/signal, background bất biến, Local/Context mask transform và Hann reconstruction PASS. Coverage thật sẽ nằm trong `assets/dataset/gt_statistics.json` khi build cache từ dataset thật.

Chưa xác minh layout/GT MVTec thật, phân bố GT DEV lịch sử, forward/training các weights pretrained thật, CUDA/bfloat16, VRAM/runtime, full grid/ranking/TEST, hoặc bitwise resume CUDA. Workspace không có dataset, pretrained weights hay GPU. Không tuyên bố model/representation thắng hay pipeline đã đạt chuẩn khoa học chỉ từ unit tests.

## Colab Pro

Xem [hướng dẫn đầy đủ](../docs/FULL_SCALE_TRAINING.md) để mount Drive, checkout đúng base, áp dụng [patch](full_scale_changes.patch), cài dependencies, lấy source DINOv3 và đặt đủ năm weights/dataset. Trong notebook, chạy các lệnh shell trong cell `%%bash`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.run_day05_pipeline \
  --train-all --grid-config configs/full_scale_grid.yaml \
  --data-root /content/drive/MyDrive/msila/data/MVTec_AD_2 \
  --dinov3-repo /content/dinov3 \
  --output-root /content/drive/MyDrive/msila/full_scale_v2 \
  --device cuda:0 --resume
```

Chạy lại nguyên lệnh khi reconnect. Sau selection bằng DEV, thêm `--stage test-public` để chỉ đánh giá các family được chọn và đủ seed. Lưu package/source revision, protocol và artifact trong Drive; không tự đổi hyperparameter khi OOM.

## Chính xác file đã sửa/thêm

| File | Trạng thái | Mục đích |
| --- | --- | --- |
| [.gitignore](../.gitignore) | modified | Ignore local QA virtualenv and disposable test artifacts. |
| [README.md](../README.md) | modified | Link full-scale training instructions and scientific limitations. |
| [configs/full_scale_grid.yaml](../configs/full_scale_grid.yaml) | new | Declare every backbone, scaled adapter candidate, representation, seed and category. |
| [configs/full_scale_synthetic.yaml](../configs/full_scale_synthetic.yaml) | new | Lock native-pixel synthetic proxy, bins, placements, signal and contour protocol before evaluation. |
| [configs/full_scale_train.yaml](../configs/full_scale_train.yaml) | new | Declare unchanged 150-epoch/batch-64 update budget, AMP and strict determinism; select by DEV AU-PRO. |
| [docs/FULL_SCALE_TRAINING.md](../docs/FULL_SCALE_TRAINING.md) | new | Document data roles, actual training/resume, Colab setup, selected TEST evaluation and verification limits. |
| [reports/full_scale_changes.patch](../reports/full_scale_changes.patch) | new | Provide a portable uncommitted patch against the recorded base commit; excludes itself. |
| [reports/full_scale_initial_audit.json](../reports/full_scale_initial_audit.json) | new | Record unavailable historic GT and configuration-only legacy generator audit without invented dataset statistics. |
| [reports/full_scale_pytest.xml](../reports/full_scale_pytest.xml) | new | Capture complete pytest evidence, including every skipped external acceptance gate. |
| [reports/full_scale_report.md](../reports/full_scale_report.md) | new | Explain root causes, implemented design, test results and remaining experimental requirements in Vietnamese. |
| [reports/full_scale_synthetic_fixture_coverage.json](../reports/full_scale_synthetic_fixture_coverage.json) | new | Record measured coverage of all declared DEV strata on a clearly labelled constant-image software fixture. |
| [reports/full_scale_validation.json](../reports/full_scale_validation.json) | new | Record verified software results, environments, unverified experiments and exact changed-file inventory. |
| [requirements.txt](../requirements.txt) | new | Declare reproducible dependency constraints compatible with the local tested stack and Colab CUDA installation. |
| [scripts/audit_dataset.py](../scripts/audit_dataset.py) | new | Export loader matching, dimensions and missing/ambiguous GT audit. |
| [scripts/audit_synthetic_gt.py](../scripts/audit_synthetic_gt.py) | new | Audit existing GT component distribution and Local tile geometry before examining model predictions. |
| [scripts/build_day05_cache.py](../scripts/build_day05_cache.py) | modified | Extend the existing builder with source-disjoint native tiny/mixed data, per-backbone caches and GT coverage statistics. |
| [scripts/build_evaluation_manifest.py](../scripts/build_evaluation_manifest.py) | modified | Validate full-scale checkpoint/config/backbone/adapter/blocks identities during native inference. |
| [scripts/day05_full_inference.py](../scripts/day05_full_inference.py) | modified | Keep native resolution, normal zero GT and correct shallow/middle/deep cache slots for every physical backbone depth. |
| [scripts/run_day05_pipeline.py](../scripts/run_day05_pipeline.py) | modified | Route declared full grid or single configuration to the existing trainer and guarded final TEST evaluator. |
| [src/data/feature_cache.py](../src/data/feature_cache.py) | modified | Checksum full-scale feature shards while preserving legacy manifests. |
| [src/data/loader.py](../src/data/loader.py) | modified | Pair GT by category/split/defect/nested identity; prohibit cross-split fallback and missing abnormal zero substitution. |
| [src/data/synthetic_anomaly.py](../src/data/synthetic_anomaly.py) | modified | Add a separately versioned native tiny-defect generator with unchanged pixels outside GT; retain legacy behavior. |
| [src/eval/full_scale.py](../src/eval/full_scale.py) | new | Compute native/tile DEV AU-PRO, declared region groups and contour diagnostics using exact disk-backed backgrounds. |
| [src/eval/region_stats.py](../src/eval/region_stats.py) | modified | Provide reusable native component geometry and explicit image-edge diagnostics. |
| [src/metrics/aupro.py](../src/metrics/aupro.py) | modified | Support NumPy integration versions and exact chunked-background AU-PRO without dropping pixels or regions. |
| [src/models/backbone_registry.py](../src/models/backbone_registry.py) | new | Validate official S/S+/B/L/H+ architecture, thirds blocks and explicitly declared adapter candidates. |
| [src/models/basic_decoder.py](../src/models/basic_decoder.py) | modified | Retain trainable decoder architecture and add equivalent deterministic full-scale bilinear resize. |
| [src/models/bilinear_sampling.py](../src/models/bilinear_sampling.py) | new | Implement fixed-geometry bilinear gather/index-select for strict CUDA backward with unchanged half-pixel semantics. |
| [src/models/context_alignment.py](../src/models/context_alignment.py) | modified | Enable equivalent deterministic sampling only in full-scale mode; preserve legacy alignment default. |
| [src/models/dinov3_extractor.py](../src/models/dinov3_extractor.py) | modified | Validate loaded architecture, freeze backbone, map physical blocks and read exact local checkpoint instead of basename cache aliases. |
| [src/train/day05_contract.py](../src/train/day05_contract.py) | modified | Validate full-scale cache/representation provenance and disjoint TRAIN/DEV roles while retaining legacy checks. |
| [src/train/full_scale.py](../src/train/full_scale.py) | new | Orchestrate complete actual grid training, native DEV seed-mean ranking, selection locks, stage resources and post-selection TEST. |
| [src/train/screen_representation.py](../src/train/screen_representation.py) | modified | Extend the existing optimization loop with dynamic C/r/d/seed, DEV AU-PRO monitoring, exact partial resume and run evidence. |
| [tests/test_bilinear_sampling.py](../tests/test_bilinear_sampling.py) | new | Compare fixed-grid/resize values and gradients with PyTorch reference; expose skipped strict CUDA gates. |
| [tests/test_day05.py](../tests/test_day05.py) | modified | Make the legacy fake ViT-S fixture match official C=384 for strict architecture validation. |
| [tests/test_full_scale_contracts.py](../tests/test_full_scale_contracts.py) | new | Test all five mocked architectures, full 3240 grid, independent adapter/fusion dimensions, synthetic strata and exact metrics/ranking. |
| [tests/test_full_scale_training.py](../tests/test_full_scale_training.py) | new | Execute real CPU fixture optimization, interrupted/resumed equality, native R2 integration, tamper rejection and unchanged OOM configuration. |
| [tests/test_loader_identity.py](../tests/test_loader_identity.py) | new | Test split/defect/nested collisions, missing/ambiguous GT, normal zeros and audit behavior. |
