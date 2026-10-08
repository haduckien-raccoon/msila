# Day 05: ViT-S/16, TV1 → TV2

Dùng `dinov3_vits16`, block 4/8/12, adapter r32/d384, projection 64, MeanFusion, seed 42. Phase 1 chỉ Fabric. `configs/day05_day04_full_v3_protocol.yaml` được lấy từ overlay cell 7 và overrides cell 28 của notebook **Day04_REAL_Pipeline_FULL_CACHE_v3 (3).ipynb**: 150 epochs, batch 64, AdamW lr 0.001, weight decay 0, scheduler null, AMP bfloat16, deterministic warn-only. Checkpoint vẫn chọn minimum `val_loss` (tên cột Day 05: `val_total_loss`). Không dùng protocol smoke/batch 4 thay thế bản này. Nếu Drive còn `configs/full_v3/day04_full_seed_42.yaml`, truyền chính file đó để giữ bản đã materialize.

## Chuẩn bị Colab T4

Notebook gộp để chạy toàn bộ Day 05: [Day05_ALL_IN_ONE_ViTS16_Colab.ipynb](../notebooks/Day05_ALL_IN_ONE_ViTS16_Colab.ipynb). Thứ tự trong một file: setup/audit → export native DEV từ cache Day 04 → khảo sát GT → khóa protocol và nối đường dẫn → preflight → train R0/R1/R2 → inference → evaluation. Phần chọn protocol cần người chạy xem GT và điền căn cứ thật trước candidate results; sau khi khóa, tiếp tục các cell bên dưới. Notebook dùng cùng overlay ZIP đã kiểm tra SHA. Không cần mở hai notebook riêng nếu dùng bản gộp. Khi resume, giữ cùng paths/code/config và tái sử dụng hai lock dưới `OUT/protocols`; để `EXPORT_LOCKED_PROTOCOLS=False` nếu hai lock đã có và giữ các ngưỡng trong cấu hình GT khớp lock.

Bật GPU, mount Drive. Repo/DINO checkout và checkpoint phải có sẵn; pipeline không tự tải dataset/weights. Dùng code của working tree đã sửa: tải `reports/day05_code_overlay.zip` lên Drive, hoặc đưa commit chứa các thay đổi này vào clone. Notebook Day05_TV1C đã bỏ cell nhúng runner cũ và kiểm tra SHA của overlay trước khi dùng.

Chạy từ thư mục repo, luôn dùng `python -m ...`. Cài dependency còn thiếu vào runtime; giữ PyTorch/torchvision đi kèm Colab:

```bash
python -m pip install pytest pyyaml scipy pillow matplotlib
```

Đặt các path thật. Ví dụ tương ứng layout Day 04 đính kèm; nếu đã stage cache vào `/content`, giữ cùng path đó ở các lần resume:

```bash
DAY04="/content/drive/MyDrive/[Q3-4] 2026/[S7] Computer Vision/CV-Nhóm 9/msila_day04"
CACHE="$DAY04/data/full_screen_v3/feature_cache"
TRAIN="$DAY04/data/full_screen_v3/records/train_core.json"
DEV="$DAY04/data/full_screen_v3/records/dev_synthetic.json"
MASKS="$DAY04/data/full_screen_v3/masks"
DINO_REPO="/content/dinov3"
DINO_CKPT="$(dirname "$DAY04")/weights/dinov3_vits16_pretrain_lvd1689m.pth"
SELECTION="$DAY04/e8_full_v3/selection_report.json"
PROTOCOL="$DAY04/configs/full_v3/day04_full_seed_42.yaml"
# Nếu bản YAML materialize chưa được giữ lại, dùng bản trích đúng từ notebook:
# PROTOCOL="configs/day05_day04_full_v3_protocol.yaml"
OUT="$(dirname "$DAY04")/outputs/day05/full_train"
RAW="/content/msila_data/mvtec_ad_2"  # Phải chứa fabric/train/good; sửa theo raw root thật.
DEV_EXPORT="$(dirname "$DAY04")/msila_day05/native_dev"

COMMON=(--cache-dir "$CACHE" --train-records "$TRAIN" --val-records "$DEV"
  --mask-root "$MASKS" --dinov3-repo "$DINO_REPO" --dino-checkpoint "$DINO_CKPT"
  --adapter-selection-report "$SELECTION" --training-protocol "$PROTOCOL"
  --output-root "$OUT" --category fabric --seed 42 --device cuda:0 --resume)
```

### Audit, preflight, training R0/R1/R2

```bash
python -m scripts.run_day05_pipeline --stage audit --dry-run "${COMMON[@]}"
python -m scripts.run_day05_pipeline --stage preflight "${COMMON[@]}"
python -m scripts.run_day05_pipeline --stage train "${COMMON[@]}"
```

`train` chạy R0 → R1 → R2 sequential và chạy preflight trước từng run. Gate kiểm tra loss/tensors/gradients, weight update và đủ sáu nguồn của R2. Ba run dùng cùng protocol lock và cache. DINO không nằm trong optimizer; cache là detached features, aligner/fusion không có parameter. Audit thật kiểm tra shape `(384,32,32)`, mask `(1,512,512)`, provenance/backbone/checkpoint và source disjointness.

Artifact nằm ở `$OUT/seed_42/fabric/{R0,R1,R2}`. Mỗi run có best/last, resolved_config, run_manifest, training/epoch logs, preflight_report, selection_record, sample_anomaly_map. COMPLETE được kiểm tra checksum trước khi skip. Resume từ epoch đã commit trong last.pt; epoch bị ngắt được replay, log chưa commit được bỏ. RNG được khôi phục; sampler có generator riêng, seed `42 + epoch - 1`, để worker persistent không làm thay đổi thứ tự mẫu khi resume. Không đổi path/config/code giữa các lần resume; mismatch phải dùng output root mới.

AMP bfloat16 là cấu hình thực trong notebook Day 04. Preflight sẽ kiểm tra trên GPU thật; nếu runtime không chạy được cấu hình này, gate dừng. Pipeline không tự đổi precision/batch/epochs để vượt gate.

### DEV-synthetic cho inference toàn độ phân giải

Official loader có `image_norm`, nhưng không có split `dev_synthetic`. Cache Day 04 chứa features và mask crop 512, không chứa ảnh anomaly native. Không lấy ảnh normal gốc hoặc Local features thay Context.

Để tái sử dụng cache FULL-v3 đã có, replay đúng hash split, crop, seed và synthetic generator. Lệnh dưới kiểm tra mask crop **và cả sáu DINO features** với cache Day 04 rồi mới xuất `dev_inputs.json`, ảnh float32 `.npy` HWC và GT native. Phải có raw sources của cùng categories như records Day 04 (notebook đã dùng fabric/vial/wallplugs):

```bash
python -m scripts.build_day05_cache --export-dev-only \
  --data-root "$RAW" --categories fabric,vial,wallplugs \
  --cache-dir "$CACHE" --train-records "$TRAIN" --val-records "$DEV" --mask-root "$MASKS" \
  --dinov3-repo "$DINO_REPO" --dino-checkpoint "$DINO_CKPT" \
  --output-root "$DEV_EXPORT" --device cuda:0
```

Replay mismatch sẽ FAIL; không sửa producer_signature để gắn nhãn backbone cho cache cũ. Full native DEV có cùng source IDs/seed/anomaly như crop Day 04, được đặt trở lại source image; evaluation unit mới được ghi rõ là `native_source_image_with_locked_context_synthesis`. AU-PRO native này không phải số AU-PRO crop trong selection Day 04.

Nếu chưa có full cache, builder dùng lại `FeatureCacheWriter/Reader` và đúng source plan FULL-v3, synthesize trước DINO:

```bash
NEW_DATA="$(dirname "$DAY04")/msila_day05/full_data"
python -m scripts.build_day05_cache --data-root "$RAW" --categories fabric \
  --dinov3-repo "$DINO_REPO" --dino-checkpoint "$DINO_CKPT" \
  --output-root "$NEW_DATA" --device cuda:0
```

Layout mới gồm feature_cache/manifest.json + shards/*.pt, records/train_core.json + dev_synthetic.json, masks/, images/, native_masks/, dev_inputs.json. Đổi COMMON sang đúng artifacts mới trước run đầu tiên. Builder reuse committed shards; COMPLETE chỉ ghi sau structural validation. Không trộn cache/records của hai producer.

### TV2 inference

```bash
INPUTS="$DEV_EXPORT/dev_inputs.json"
python -m scripts.run_day05_pipeline --stage inference "${COMMON[@]}" \
  --input-manifest "$INPUTS" --seg-f1-threshold 0.5
```

0.5 là SegF1 threshold trong cell 1 notebook Day 04. TV2 strict-load best.pt của từng run, kiểm tra checkpoint DINO bằng SHA256 và architecture thực 384 channels/patch 16. Local 512, Context 768 → bicubic antialias 512, ImageNet normalization. R2 extract Context riêng và align với geometry của tile thật. Tiling overlap 128 + Hann weighted stitching giữ H×W native; không resize cả ảnh. Map `.npy` float32 và provenance từng ID được lưu dưới `$OUT/maps/dev_synthetic/{R0,R1,R2}`. Handoff và evaluation manifest dùng hai schema riêng.

### Tiny/boundary protocols nằm ở đâu, sinh lúc nào?

Khảo sát GT bằng notebook độc lập [Day05_GT_Tiny_Boundary_Protocol_Review.ipynb](../notebooks/Day05_GT_Tiny_Boundary_Protocol_Review.ipynb): mở trong Colab/Jupyter, sửa cell 3 rồi chạy từ trên xuống. Không cần GPU, repo hoặc prediction. Mode `auto` ưu tiên `dev_inputs.json` native; nếu chưa export native DEV thì đọc records/masks crop Day 04 và ghi rõ phạm vi crop. Báo cáo gồm CSV diện tích/coverage theo category, histogram, preview GT và checksum từng mask. Chỉ GT native mới xuất proposal protocol native; khóa là cell tùy chọn, mặc định tắt. Mask crop không đủ thông tin để suy ra coverage sát mép ảnh gốc.

Notebook Day 04 **không sinh** hai file này. Các module TV2 đọc chúng nhưng repo merge chưa có bản scientific lock. Chúng là tham số nghiên cứu cần lưu trước khi xem kết quả candidate, không phải training artifact.

Copy hai skeleton:

```bash
mkdir -p "$OUT/protocols"
cp -n configs/day05_tiny_protocol.example.json "$OUT/protocols/tiny_protocol.json"
cp -n configs/day05_boundary_protocol.example.json "$OUT/protocols/boundary_protocol.json"
```

Skeleton có null/false nên chưa chạy được. Cần xác định từ protocol hoặc căn cứ GT độc lập với prediction: `tiny_area_px`, `band_width_px`, `tolerance_px`, và ghi `threshold_basis`/`rule_basis`. Giữ connectivity 8, max_fpr 0.05 và prediction_threshold 0.5. Chỉ đặt `locked_before_candidate_results=true` khi đã khóa thật. Nếu chưa thống nhất, giữ trạng thái chưa resolve; training/inference vẫn dùng được, diagnostics/representation lock BLOCKED. Không coi skeleton là kết quả đã khóa.

### TV2 evaluation và representation lock

Sau khi có hai protocol thật:

```bash
TINY="$OUT/protocols/tiny_protocol.json"
BOUNDARY="$OUT/protocols/boundary_protocol.json"
python -m scripts.run_day05_pipeline --stage evaluate "${COMMON[@]}" \
  --input-manifest "$INPUTS" --seg-f1-threshold 0.5 \
  --tiny-protocol "$TINY" --boundary-protocol "$BOUNDARY"
```

Reuse evaluator/AU-PRO Day 04 nguyên vẹn, tiny/boundary/region stats/comparison, efficiency, aggregation và representation lock. GT/maps/checkpoint IDs và hashes được kiểm tra trước metric; không normalize theo candidate. E8 giữ scope cached trainable pipeline FP32, batch 1, warmup 10, 50 iterations × 3 rounds, VRAM warmup 10/1 iteration như utility cũ. Tổng thời gian native tiled inference được ghi riêng trong provenance từng image, không đánh đồng với runtime cached head. CPU không tạo VRAM giả; metric/diagnostics có thể hoàn thành rồi dừng `BLOCKED_MISSING_GPU` trước aggregation.

Có thể truyền `--efficiency-csv /path/efficiency_summary.csv` nếu đã đo bằng API cũ. CSV phải có adjacent efficiency_protocol.json và R0/R1/R2_efficiency.json; checkpoint/config hashes phải khớp TV1. Thiếu/mismatch/stability FAIL sẽ dừng.

Kết quả nằm ở `$OUT/evaluation/dev_synthetic/`: primary/metrics.json, tiny/tiny.csv, boundary/boundary.csv, regions/per_region_stats.csv, comparison/visualizations/, efficiency/efficiency.csv, summary/week06_multiscale.csv và summary/representation_lock.yaml. Stage đã COMPLETE được checksum và skip; attempt bị ngắt được giữ lại dưới `.incomplete.*`, rồi replay. Final TEST dùng `--split test_public --data-root "$RAW" --representation-lock "$OUT/evaluation/dev_synthetic/summary/representation_lock.yaml"`; TEST không tạo lock để chọn winner. Normal `good` không có mask được tạo zero GT từ official normal label với provenance; anomaly thiếu GT phải FAIL.

### Một entrypoint cho tất cả gates

Chỉ khi assets và protocols đầy đủ:

```bash
python -m scripts.run_day05_pipeline --stage all --dry-run "${COMMON[@]}" \
  --input-manifest "$INPUTS" --seg-f1-threshold 0.5 --tiny-protocol "$TINY" --boundary-protocol "$BOUNDARY"
python -m scripts.run_day05_pipeline --stage all "${COMMON[@]}" \
  --input-manifest "$INPUTS" --seg-f1-threshold 0.5 --tiny-protocol "$TINY" --boundary-protocol "$BOUNDARY"
```

Phase 2 fabric/vial/wallplugs chỉ mở sau evidence Phase 1 PASS; CLI này giới hạn Phase 1 Fabric và chưa tự mở rộng khi Phase 1 chưa hoàn thành.

## Acceptance

CODE PASS = tests trên fixture; REAL-DATA PASS = ảnh MVTec AD 2 thật + weights thật; FULL-TRAIN COMPLETE = cả ba run thật đủ update budget; FULL-EVALUATION COMPLETE = native maps + metrics + diagnostics + efficiency + aggregation/lock đủ thật. Không suy ra ba trạng thái sau từ unit fixtures. Báo cáo phiên tích hợp: `reports/day05_integration_audit.md`.
