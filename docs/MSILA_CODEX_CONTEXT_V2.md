# MS-ILA — NGỮ CẢNH BẮT BUỘC CHO CODEX (v2, 2026-10-09)

Repo: https://github.com/haduckien-raccoon/msila
Mốc tham chiếu đã được rà soát: `main` commit `bab1ba74fa00c967f8f599276124e600ed832242` (09/10/2026). **Luôn kiểm tra HEAD hiện tại**, không mặc định code bất biến so với mốc trên.

> Đây là **ngữ cảnh nghiên cứu do nhóm định nghĩa**, KHÔNG phải tuyên bố rằng mọi ký hiệu bên dưới đã tồn tại dưới dạng class, config, checkpoint, hoặc experiment trong repo. Không được báo lỗi vì `rg E1` không tìm thấy. Phải dùng định nghĩa ở đây để đối chiếu chức năng thực tế.

## 1. Bài toán

MS-ILA = **Multi-Scale Illumination-Invariant Lightweight Adapters for Industrial Anomaly Segmentation** trên **MVTec AD 2** (8 category). Đầu ra chính là **dense anomaly score map** (bản đồ điểm bất thường liên tục, H×W), không chỉ nhãn bất thường cả ảnh. Primary evaluation = **AU-PRO@FPR≤0.05**; phụ: SegF1, boundary diagnostics, tiny-region, robustness, params, latency, peak VRAM. Không dùng tập TEST để chọn mô hình, ngưỡng, protocol hay kiến trúc.

Quy trình nghiên cứu: ảnh bình thường TRAIN (`good`) -> tạo synthetic/pseudo anomalies và GT mask từ TRAIN -> DINOv3 pretrained/frozen -> train phần head được phép -> score map -> đánh giá DEV synthetic/normal độc lập -> chỉ dùng TEST sau khi khóa quyết định.

## 2. Ký hiệu của lộ trình nghiên cứu (ngoài repo)

- **TV1**: Thành viên 1, Model & Training (architecture, DINO, gradient, optimizer, training, checkpoint, GPU, resume).
- **TV2**: Thành viên 2, Data & Evaluation (loader, GT, synthetic, source split, tiling, native inference, metric, protocol locks, tổng hợp).
- **G1**: Giai đoạn nghiên cứu 1 trong tài liệu đề bài (`Pasted text(10).txt`): tạo bản baseline dễ train/evaluate; **không** dùng Adapter, multi-layer hoặc Context; gồm 4 ngày **D1–D4 trong lộ trình G1 gốc**.
- **E1**: *tên thí nghiệm đề xuất của G1*, không phải tên class trong repo. Định nghĩa kiến trúc dự kiến: `RGB Local 512×512 -> DINOv3 ViT-S/16 frozen -> một feature tầng sâu nhất [B,384,32,32] -> lightweight segmentation decoder -> logits [B,1,512,512] -> sigmoid score`. Với dữ liệu synthetic mask dùng `BCEWithLogits + Dice`, optimizer chỉ cập nhật decoder; normal mask = 0. DEV synthetic và normal phải độc lập với TRAIN. Inference ảnh gốc bằng tile 512, stride 384, Hann stitching. **E1 không có residual adapter, feature projection đa nguồn, attention, mean-fusion nhiều tầng, hay Context view**. Đây là *đặc tả mục tiêu dự kiến*, KHÔNG khẳng định implementation/checkpoint E1 đã có. Nếu repo chưa có E1 tương đương: ghi `PLANNED_NOT_IMPLEMENTED` hoặc `NOT_VERIFIED`, không tự tạo mới trong một task kiểm toán.
- **G2**: Giai đoạn mở rộng trong lộ trình gốc: Adapter, multi-layer b4/b8/b12, Local/Context alignment/fusion, sau đó có thể Attention/Gated Fusion. Đây là milestone nghiên cứu, **không** đồng nghĩa tên folder hay `Day02/Day05` trong repo.
- **G3**: Giai đoạn nghiên cứu illumination-consistency loss theo tài liệu gốc, không được mặc nhiên bật loss này.
- **D1–D4 của bộ prompt hiện tại**: bốn ngày *kiểm toán/đóng bằng chứng sau khi đã có code Day05*, KHÔNG phải yêu cầu làm lại D1–D4 của G1 gốc. Tên `Day05`/`Day04` của repo là dấu mốc implementation riêng.

## 3. Kiến trúc và thí nghiệm đã có trong repo — KHÁC E1

### A. `MSILA` / Day-1 architectural QA baseline của repository

README_1.md và `src/models/msila.py` mô tả:
`RGB -> frozen DINOv3 features b4,b8,b12 -> 3 ResidualAdapter2d -> MeanFusion -> BasicDecoder -> logits`.
Feature layouts BCHW, ví dụ ViT-S/16 với crop 512: `[B,384,32,32]` mỗi block; logits `[B,1,512,512]` (tùy kích thước input). `gamma=0` cho Adapter identity ở khởi tạo. Đây là architectural QA/control của repo, **không phải E1 G1** vì có 3 tầng/Adapter/Fusion.

### B. Day 05 representation ablation

Có config `configs/day05_representation.yaml`:
- **R0 / deep_only**: chỉ feature `local_b12` (1 nguồn).
- **R1 / multi_layer_local**: `local_b4`, `local_b8`, `local_b12` (3 nguồn).
- **R2 / multi_layer_local_context**: ba nguồn Local + `context_b4`, `context_b8`, `context_b12` được **warp/align** về Local (6 nguồn).

Đây là ablation **thay duy nhất representation source**, giữ kiến trúc và training protocol tương đương: Frozen DINOv3 ViT-S/16; Adapter theo cấu hình đã khóa (ví dụ r32/d384 trong Fabric Day05); downstream feature projection/fusion_dim 64; Mean Fusion; Basic Decoder; BCE+Dice; illumination loss weight=0. R2 phải dùng crop Context 768 được resize 512, không được copy Local thành Context. **R0 không phải E1** vì vẫn thuộc head Adapter/Projection được khóa ở Day05. `src/models/attention_fusion.py` có code nhưng Day05 YAML **cấm bật attention** để đảm bảo so sánh công bằng.

### C. Day05 training evidence vs actual artifacts

`docs/DAY05_TV2_HANDOFF.md` ghi **notebook evidence**: fabric/seed 42/ViT-S16/r32d384, 3 run R0/R1/R2, 150 epochs và 11.550 optimizer updates/run, best epochs báo cáo 28/21/21 theo **minimum val_loss (val_total_loss alias)**. Đây là ghi nhận từ notebook, **chưa tương đương** với việc đọc và strict-verify `best.pt` thực trên Google Drive. `reports/day05_tv2_handoff_qa.json` nói rõ chưa mở Drive checkpoint và chưa chạy native real inference trong workspace QA. Nếu thiếu checkpoint => `BLOCKED_MISSING_CHECKPOINT`, không tự train lại.

`reports/day05_acceptance.json`: 634 PASS/18 SKIP trong CPU fixture ở một mốc, `real_data`/`full_training`/`full_evaluation` BLOCKED tại môi trường đó. `reports/full_scale_validation.json`: một lần kiểm chứng rộng hơn 707 PASS/20 SKIP, **scientific_experimental_status=UNVERIFIED**. Những con số là snapshot khác nhau; không cộng/trộn hoặc coi SKIP là PASS. Chỉ báo kết quả test mới nếu có command/log đúng.

### D. Full-scale v2 — nghiên cứu KHÁC giao thức Day05 Fabric

Có `configs/full_scale_grid.yaml` khai báo 5 DINOv3 backbone × 9 tổ hợp adapter (r,d) × 3 representation × 3 seed × 8 category = **3.240 jobs**, 150 epoch/job. Đây là **grid được khai báo trong code, không phải bằng chứng đã chạy xong**. Có các file `src/train/full_scale.py`, `docs/FULL_SCALE_TRAINING.md`, `configs/full_scale_synthetic.yaml`.

Full-scale v2 có DEV synthetic native và rule checkpoint sử dụng `0.5*DEV_tiny_AU-PRO005 + 0.5*DEV_mixed_AU-PRO005` trong giao thức riêng; **không áp rule này lên checkpoint Day05 Fabric đã train theo minimum val_loss**. Không tự gọi `--train-all`/khởi tạo job GPU tốn tài nguyên, không đổi hyperparameters giữa so sánh để né OOM.

## 4. Data/evaluation conventions

- Local tile `512×512`, overlap 128 => stride `384`; Context FOV `768×768` resize về 512 trước DINO; R2 dùng Context-to-Local alignment từ geometry metadata.
- Native ảnh/mask giữ H×W, ground truth label pixel 0/1, nearest-neighbor khi resize mask; logit và score map phải phân biệt; BCE trên **logits**, Dice trên **sigmoid probability**.
- Stitch float scores bằng Hann weighted mean, mọi pixel có weight > 0; không ép normalize score của mỗi candidate hoặc mỗi ảnh nếu giao thức không cho phép.
- Source disjoint TRAIN vs DEV, test_public/test_private không được dùng selection/tuning. AU-PRO0.05 sử dụng continuous score; segmentation F1 là phụ thuộc threshold.
- `tiny_area_px`/`band_width_px`/`tolerance_px` phải xuất phát từ quy tắc GT/protocol khóa trước khi xem R0/R1/R2 predictions. Skeleton example JSON chưa được xem là scientific lock.
- `docs/G1_decision_log.md` của repo nói **G1 data & feature pipeline PASS**; **không đủ bằng chứng** E1 G1 đã train, vì `G1` có thể dùng khác nghĩa giữa các kế hoạch/tài liệu. Phải trích rõ nguồn của từng status.

## 5. Code đã tồn tại: tái dùng, không sinh class trùng

`src/models/`: `dinov3_extractor.py`, `residual_adapter.py`, `basic_decoder.py`, `msila.py`, `feature_selector.py`, `feature_projection.py`, `context_alignment.py`, `mean_fusion.py`, `attention_fusion.py`, `backbone_registry.py`.

`src/data/`: `loader.py`, `tiling.py`, `synthetic_anomaly.py`, `feature_cache.py`.

`src/train/`: `screen_representation.py`, `day05_contract.py`, `full_scale.py`, `overfit16.py`.

`scripts/`: `run_day05_pipeline.py`, `build_day05_cache.py`, `day05_full_inference.py`, `eval_day05_representation.py`, `aggregate_week06_multiscale.py`, `create_representation_lock.py`.

`src/eval/`: `evaluator.py`, `full_scale.py`, `tiny_analysis.py`, `boundary_analysis.py`, `region_stats.py`, `efficiency.py`.

`src/metrics/aupro.py`, `src/geometry/view_meta.py`, `src/utils/checkpoint.py`, `src/utils/resume.py` cùng bộ `tests/`.

**File path trên là tham chiếu snapshot**: trước khi chạy test/hướng dẫn file, luôn xác nhận file đó có tồn tại (`test -f` / `rg --files`). Nếu không tồn tại, chọn file tương đương đang có; không tạo dummy/alias chỉ vì prompt nhắc tên cũ.

## 6. Cách làm một task; cost và evidence

1. `git status --short && git rev-parse --short HEAD`; không reset/clean/overwrite thay đổi người khác. Đọc file ngữ cảnh này **một lần cho session**, sau đó chỉ đọc file bắt buộc trong task và direct imports khi thật cần. Không dump toàn repository/notebooks.
2. **Audit first**: liệt kê evidence `file:function/test/config/report`. Phân biệt `CODE_PRESENT`, `CODE_TEST_PASS`, `REAL_ASSET_VERIFIED`, `TRAIN_ARTIFACT_VERIFIED`, `EVALUATION_VERIFIED`, `PLANNED_NOT_IMPLEMENTED`, `SKIPPED`, `BLOCKED`. Không dùng một chữ 'PASS' cho mọi mức.
3. Nếu báo cáo/test đã có trong session và không có code mới, **không chạy lại chỉ để lặp số**; sử dụng log hiện có + command/hash + thời điểm, ghi giới hạn. Với bug mới, viết regression test tối thiểu, chạy targeted pytest. Không chạy toàn suite nếu không cần.
4. **Audit task không phải build task**: không tự triển khai E1 mới, không tạo Adapter/Decoder/Fusion trùng; chỉ sửa bug tái hiện trong phần owner. Không tự download dataset/weights, không chạy training 150 epoch/grid/inference toàn bộ MVTec trừ khi task tương ứng và đủ assets/quyền hạn.
5. TV1 sở hữu model/training; TV2 sở hữu data/metric/eval. Khi code chung gây conflict, tạo đề xuất/contract hoặc PR riêng thay vì hai người tự sửa chồng.
6. Output report under `reports/recheck/`; kèm mục `Định nghĩa / Hiện trạng có bằng chứng / Chưa xác minh / Các gate cần assets / Lệnh tái lập / Handoff`. Không bịa metric, checkpoints, expected result.
7. Sau task trả lời **tối đa 12 dòng**: thay đổi nào, lệnh test và PASS/SKIP, blocker, next handoff. Mỗi task phải dừng ở gate, không chủ động chạy task kế tiếp.

## 7. Danh mục gate

- `GATE_ARCH`: API tensor/model, mock shape, frozen flags, adapter identity, gradient.
- `GATE_REAL_DINO`: source+checkpoint real, patch/token layout, weights identity; CUDA nếu cần.
- `GATE_DATA`: GT native, split pairing, source-disjoint, cache and hashes.
- `GATE_TRAIN_ARTIFACT`: checkpoints/logs/config/provenance và budget thật.
- `GATE_NATIVE_INFER`: score map H×W thật + Hann+geometry/provenance.
- `GATE_METRIC`: AU-PRO synthetic DEV + tiny/boundary protocols được khóa, efficiency đo GPU đúng scope.
- `GATE_SELECTION`: report đủ R0/R1/R2 cùng protocol; tạo representation lock chỉ từ DEV hợp lệ.

Nếu chưa có asset, trả `BLOCKED` và **cung cấp exact path/type cần bổ sung**, không giả lập một thí nghiệm đã chạy.
