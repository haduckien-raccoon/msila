# Day 05 — TV2 nhận kết quả TV1-C/TV1-D

Notebook TV1-C có outputs do người dùng gửi xác nhận R0/R1/R2 COMPLETE, mỗi run 150 epochs và 11.550 optimizer steps, Fabric/seed42/ViT-S16/r32d384. TV1-D chọn best epoch 28/21/21 theo minimum val_total_loss, cùng đại lượng val_loss của Day 04. Đây là bằng chứng training/checkpoint selection trong notebook đã gửi, chưa phải AU-PRO native hoặc kết luận representation thắng.

Dùng [Day05_TV2_Receive_Inference_Evaluation_ViTS16_Colab.ipynb](../notebooks/Day05_TV2_Receive_Inference_Evaluation_ViTS16_Colab.ipynb) với [gói code TV2](../reports/day05_tv2_code_overlay.zip). Đây là workflow sau training, không có lệnh train/preflight optimizer. Không chạy lại all-in-one từ đầu.

## Bàn giao

TV1 cung cấp toàn bộ `outputs/day05/full_train/seed_42/fabric/{R0,R1,R2}`: best.pt, last.pt, resolved_config.yaml, run_manifest.json, preflight_report.json, selection_record.json, epoch_log.csv, training_log.csv, sample_anomaly_map.png. Giữ thêm `full_train/protocol_lock.json`, `checkpoint_rule_lock.json` và hai summary TV1-D/completeness nếu có. TV2 cần cache/records/masks Day 04, raw TRAIN sources, đúng DINO weights và DINO source checkout để export native DEV/inference. Preview một sample không thay full inference maps.

## Thứ tự trong notebook TV2

1. Mount Drive, load verified code ZIP, sửa đường dẫn.
2. Nhận/check ba run COMPLETE và strict-load best.pt trên CPU. Lưu handoff receipt. Hiển thị hyperparameters **thực trong resolved_config**; không áp batch64/AMP/epochs của notebook khác vào checkpoint cũ.
3. Export/replay đúng native DEV. So sánh mask crop và cả sáu feature với cache Day 04 trước khi xuất images/native GT/dev_inputs.json. Nguồn DEV vẫn source-disjoint từ raw TRAIN; không dùng test để chọn protocol/representation.
4. Khảo sát GT native và khóa tiny/boundary trước candidate maps/metrics. Nhóm không có GT vùng tương ứng báo undefined; không đổi ngưỡng dựa trên candidate scores.
5. Inference ba best.pt, Local512/Context768 resize512, overlap128/Hann stitch, maps native float32; strict checkpoint identity/provenance. R2 dùng Context thật.
6. Evaluate AU-PRO0.05 và diagnostics, đo E8 cached-head efficiency trên CUDA, tổng hợp và tạo representation lock nếu scientific gates đạt.

## Tương thích với TV1 notebook đã chạy

Notebook TV1 gửi lên dùng original runner chưa nhúng `day05_config`, chưa có checksum index trong run_manifest; TV1-D bổ sung selection evidence. Reader TV2 hỗ trợ đúng cấu trúc đó, giữ nguyên config/checkpoint/hash trên đĩa. Model construction view lấy backbone/input/adapter/projection/fusion/decoder/loss từ resolved config, representation từ canonical R0/R1/R2 contract; không ghi thêm field vào config gốc.

Đối với legacy handoff, TV2 kiểm tra protocol lock theo đúng hash rule cũ, rule lock tồn tại trước training, full epoch/optimizer-step logs, best epoch/min-val-loss, best/last checkpoint schema/config/epoch/update budget, cùng SHA best/config/epoch log do TV1-D lưu. Các SHA còn lại là snapshot **tại thời điểm nhận TV2**, không được gọi là bằng chứng checksum được lưu trước training. Legacy metadata không chứa historical mask checksums; native replay kiểm tra dữ liệu hiện tại với synthetic plan/cache, không thể chứng minh lịch sử mọi file mask bằng checksum thiếu trong TV1.

Reader hiện đại vẫn giữ các checksum/config lock checks. Metadata thiếu hoặc drift sẽ dừng, không tự sửa run_manifest/resolved_config để vượt gate. Strict model load kiểm tra tensor keys/shapes. Mục tiêu tương thích là evaluate đúng checkpoint đã bàn giao, không chứng nhận các hyperparameters của run cũ đã trùng notebook Day 04 khác.

E8 YAML notebook xuất riêng có schema `msila.day05.tv2.efficiency_input.v1`, đọc dataset sample để đo scope đã có (batch1/FP32/warmup10/50×3); không phải training protocol mới. Các thông số training được hiển thị từ TV1 config nguyên gốc.

Kết quả ở `full_train/maps/dev_synthetic` và `full_train/evaluation/dev_synthetic`. Checkpoint/config/log TV1 được đọc, không sửa. Chỉ dùng test_public sau DEV representation lock theo protocol final-test đã khóa.
