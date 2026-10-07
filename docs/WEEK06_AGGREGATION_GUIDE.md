# Task 9 — Bảng bằng chứng Day 5: week06_multiscale.csv

## File mới và phạm vi

Đã thêm script `scripts/aggregate_week06_multiscale.py`, tài liệu này và
`tests/test_aggregate_week06_multiscale.py`. Không thay thế evaluator hay các
script tiny/boundary/efficiency đang có.

Script chỉ dùng Python standard library. Không cần PyTorch/CUDA để tổng hợp
**kết quả đã đo**. Nó không đọc anomaly maps, không chạy inference, không tính
lại AU-PRO, không đổi normalization, threshold hoặc GT protocol.

## Đầu vào

1. `metrics.json` từ `scripts/eval_day05_representation.py` (Task E2). File này
   có `candidate_category`, `macro_by_candidate`, manifest hash, evaluator hash,
   split và normalization. Không dùng `reports/day01_report.json` hoặc chỉ
   `comparisons.csv` làm đầu vào thay thế.
2. `tiny_region_metrics.csv` từ `src.eval.tiny_analysis`.
3. `boundary_region_metrics.csv` từ `src.eval.boundary_analysis`.
4. Một hoặc nhiều `efficiency_summary.csv` từ notebook Task 8. Lặp lại option
   `--efficiency-csv` cho mỗi category.

Giữ nguyên các file JSON bên cạnh từng efficiency CSV:

```text
efficiency/<category>/
├── efficiency_summary.csv
├── efficiency_protocol.json
├── R0_efficiency.json
├── R1_efficiency.json
└── R2_efficiency.json
```

Không ghép các CSV bằng cách copy rồi bỏ các JSON. Script đối chiếu CSV với
report gốc để kiểm tra scope/GPU/resolution/batch/precision/warm-up và tránh
nhầm median với mean hoặc peak allocated với peak reserved.

Tất cả nguồn phải có đúng R0/R1/R2 × tập category khai báo trong E2. Nếu E2
chỉ đánh giá `fabric`, đầu vào khác cũng chỉ có `fabric`. Nếu E2 có tám
category, phải cung cấp đủ efficiency của cả tám. Thiếu/duplicate row, metric
NaN/Inf hoặc efficiency FAIL sẽ báo lỗi thay vì điền số giả.

## Cách chạy

Chạy từ thư mục gốc project. Thay các path bên dưới bằng kết quả thật:

```bash
python -B scripts/aggregate_week06_multiscale.py \
  --metrics-json outputs/day05/main/metrics.json \
  --tiny-csv outputs/day05/tiny_region_metrics.csv \
  --boundary-csv outputs/day05/boundary/boundary_region_metrics.csv \
  --efficiency-csv outputs/day05/efficiency/fabric/efficiency_summary.csv \
  --output outputs/week06_multiscale.csv
```

Ví dụ thêm category thứ hai (E2/tiny/boundary phải chứa cả hai):

```bash
python -B scripts/aggregate_week06_multiscale.py \
  --metrics-json outputs/day05/main/metrics.json \
  --tiny-csv outputs/day05/tiny_region_metrics.csv \
  --boundary-csv outputs/day05/boundary/boundary_region_metrics.csv \
  --efficiency-csv outputs/day05/efficiency/fabric/efficiency_summary.csv \
  --efficiency-csv outputs/day05/efficiency/vial/efficiency_summary.csv \
  --output outputs/week06_multiscale.csv
```

Path trong ví dụ là vị trí bạn điền, không phải artifact thật đã có trong
workspace. Không tạo CSV bằng số minh họa rồi dùng nó làm bảng bằng chứng.
Output phải mới. Nếu một trong ba file output đã tồn tại, script từ chối ghi đè;
chọn path mới cho lượt tổng hợp tiếp theo.

## Định nghĩa bảng chính

```csv
candidate,AU-PRO0.05,tiny_AU-PRO0.05,boundary_AU-PRO0.05,params,runtime_ms_per_batch,VRAM_MiB
R0,...
R1,...
R2,...
```

| Cột | Quy tắc tổng hợp đã định nghĩa trong script |
|---|---|
| `AU-PRO0.05` | Trung bình đều theo tất cả category trong E2, khớp macro E2 |
| `tiny_AU-PRO0.05` | Trung bình đều theo category có tiny regions và background hợp lệ |
| `boundary_AU-PRO0.05` | Trung bình đều theo category có boundary regions và background hợp lệ |
| `params` | Số trainable Parameter elements unique; phải bằng nhau giữa category cho mỗi candidate |
| `runtime_ms_per_batch` | Trung bình đều của **median latency mỗi category**; không phải pooled median |
| `VRAM_MiB` | **Maximum** absolute peak allocated qua các category, giữ chi phí peak |

AU-PRO giữ thang `[0,1]`, không nhân 100. Không lấy mean các per-region scores
trong `region_stats.csv` để thay AU-PRO: hai thống kê có ý nghĩa khác nhau.

Nếu chỉ có một category, mỗi cột chính giữ đúng giá trị của category đó.
Không lấy trung bình tham số khi kiến trúc giữa category khác nhau; trường hợp
đó script báo lỗi để bạn xác định lại phạm vi bảng.

Tiny/boundary dùng tập category hợp lệ chung cho **cả R0/R1/R2**, được xác định
bằng GT counts/status. `NO_REGIONS` và `NO_NORMAL_PIXELS` giữ giá trị undefined,
không đổi thành 0/1. Nếu mọi category đều undefined, ô metric tương ứng trong
CSV để trống; terminal hiển thị `NA`.

Quy tắc macro diagnostic này phải được chấp nhận/khóa trước khi xem kết quả
candidate; không thử nhiều cách gộp rồi chọn cách tạo delta tốt nhất. Trong
caption ghi rõ số category được dùng cho từng diagnostic từ provenance.
Nếu protocol nghiên cứu đã khóa một quy tắc khác, cần áp dụng đúng protocol đó
trước khi dùng bảng này làm bằng chứng.

`boundary_AU-PRO0.05` đo localization của nhóm boundary defects; nó không trực
tiếp đo khớp contour. `boundary_f1` vẫn được giữ trong bảng từng category và
macro BF1 trong provenance để đọc cùng diagnostic về khả năng giữ biên.

## Ba output

```text
outputs/week06_multiscale.csv
outputs/week06_multiscale.per_category.csv
outputs/week06_multiscale.provenance.json
```

- CSV chính: đúng ba row R0/R1/R2 theo thứ tự cố định, đủ bảy cột yêu cầu.
- CSV từng category: đầy đủ candidate × category, metric, status, GT counts,
  BF1, seed và efficiency scope hash để truy vết bảng chính.
- Provenance: hash các file đầu vào, split/category, normalization, evaluator
  hashes, danh sách category diagnostic hợp lệ/bị loại cùng lý do, scope
  efficiency và quy tắc tổng hợp.

Script kiểm tra cùng evaluation manifest giữa E2/tiny/boundary; cùng tiny rule,
boundary rule, connectivity=8 và FPR=0.05. Tiny AU-PRO source hash phải khớp E2.
Task boundary hiện không xuất AU-PRO source hash riêng: tổng hợp không tự chứng
nhận lại bản source lịch sử của boundary chỉ từ CSV.

Efficiency CSV phải khớp JSON gốc và có status PASS. Các benchmark phải cùng
hardware (bao gồm UUID nếu được lưu), input shapes/resolution, precision,
batch size và measurement procedure. Sample IDs và cache/DEV record hashes
có thể khác giữa category; trong một category scope phải giống cho cả ba
candidate. Tất cả efficiency rows phải cùng seed.

Benchmark Task 8 hiện là **cached representation pipeline**, không gồm backbone,
tiling, stitching hoặc feature/mask transfer. Bảng tổng hợp giữ phạm vi này.
Với batch=1, `runtime_ms_per_batch` là một tile/sample, không nhất thiết một ảnh
gốc. Nếu GPU UUID không được runtime cung cấp, GPU name không tự chứng minh
hai máy dùng cùng physical GPU; hãy đo cùng một GPU theo protocol Task 8.

E2 hiện chưa ghi checkpoint/seed metadata trong metrics JSON. Vì vậy script
không thể xác minh anomaly maps được tạo từ đúng checkpoint efficiency chỉ
bằng tên R0/R1/R2. Bạn cần giữ map/run provenance từ inference và dùng cùng bộ
artifact; script không tuyên bố đã xác minh mối liên kết chưa được lưu này.

## Cách đọc bằng chứng

Đọc `R1 - R0` và `R2 - R1` trên AU-PRO chính, rồi đối chiếu tiny/boundary và
chi phí. Delta dương là bằng chứng mô tả trong lượt chạy đã khóa; không tự
chứng minh statistical significance hoặc chọn winner từ hình visualization.
Script không chọn winner, không thực hiện significance test và không gộp các
seed khác nhau thành một mean không khai báo.

Task 10 bổ sung option `--lock-output configs/representation_lock.yaml`: khi
có option này, bảng tổng hợp được kiểm tra rồi script tạo lock bằng AU-PRO
DEV cao nhất. Cách dùng và quy tắc tie ở `docs/REPRESENTATION_LOCK_GUIDE.md`.
Chạy Task 9 không có option này vẫn chỉ tổng hợp bằng chứng.

## Kiểm tra

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q -p no:cacheprovider tests/test_aggregate_week06_multiscale.py
```

Tests dùng fixture synthetic chỉ để kiểm tra join/macro/missing-group,
consistency và output protection; không phải kết quả nghiên cứu hoặc benchmark
CUDA. Trong workspace hiện tại chưa có đủ metric/efficiency artifact thật để
xuất `week06_multiscale.csv` với số liệu thật.
