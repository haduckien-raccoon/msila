# Hướng dẫn chạy Full Inference Day 5 (TV2-E1)

Tài liệu này hướng dẫn cách gọi script chạy **full inference pipeline** trên các model được bàn giao (R0, R1, R2) qua tập đánh giá DEV evaluation split. Kịch bản xuất output (anomaly score map) tại *full resolution* của ảnh gốc dưới dạng file `.npy` và `.tiff`.

## 1. Yêu cầu môi trường
Đảm bảo bạn đã cài đặt các thư viện tiêu chuẩn cần thiết:
```bash
pip install torch numpy tifffile pyyaml tqdm

python scripts/day05_full_inference.py \
    --r0 path/to/R0 \
    --r1 path/to/R1 \
    --r2 path/to/R2 \
    --data-root path/to/mvtec_ad2 \
    --split dev_synthetic \
    --output-dir outputs/day05_inference \
    --dinov3-repo /content/dinov3

