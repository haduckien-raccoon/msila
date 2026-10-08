# Task 10 — Tự động điền representation_lock.yaml sau khi có metric

## Khi nào file được tạo?

**Sau khi có đủ output metric/efficiency hợp lệ và bạn chạy bước tổng hợp cuối.**
Thêm `--lock-output configs/representation_lock.yaml` vào lệnh Task 9: script
tổng hợp xong sẽ tự điền và tạo YAML. Không cần nhập thủ công `selected: R?`
hay copy từng số từ CSV.

Hiện chỉ tạo script và hướng dẫn; chưa tạo YAML kết quả trong project vì chưa
có đủ kết quả thật. Không dùng số fixture test để khóa representation.

File mới: `scripts/create_representation_lock.py`, tài liệu này và
`tests/test_representation_lock.py`. File được cập nhật:
`scripts/aggregate_week06_multiscale.py` thêm `--lock-output` và lưu vai trò các
input trong provenance. Không thay evaluator hoặc model/config training.

## Chạy tổng hợp và tạo lock trong một lệnh

Chạy từ thư mục gốc project, thay đường dẫn bằng output thật:

```bash
python -B scripts/aggregate_week06_multiscale.py \
  --metrics-json outputs/day05/main/metrics.json \
  --tiny-csv outputs/day05/tiny_region_metrics.csv \
  --boundary-csv outputs/day05/boundary/boundary_region_metrics.csv \
  --efficiency-csv outputs/day05/efficiency/fabric/efficiency_summary.csv \
  --output outputs/week06_multiscale.csv \
  --lock-output configs/representation_lock.yaml
```

Lặp `--efficiency-csv` cho mỗi category nếu E2 đánh giá nhiều category. Giữ các
JSON Task 8 cạnh efficiency CSV như hướng dẫn Task 9. Các output phải mới;
script từ chối ghi đè bảng hoặc lock đã có.

Luồng thực hiện:

```text
E2 + tiny CSV + boundary CSV + efficiency reports đầy đủ
  -> kiểm tra cùng candidate/category/protocol, efficiency PASS
  -> kiểm tra điều kiện khóa và quy tắc chọn
  -> tạo week06_multiscale.csv + bảng category + provenance
  -> tự tạo representation_lock.yaml với số liệu thật
```

Thiếu input, protocol lệch hoặc efficiency FAIL: báo lỗi và không tạo lock.
Lock hiện hữu được kiểm tra trước khi ghi bảng mới. Lỗi filesystem trong lúc
ghi có thể để lại một phần output; chọn đường dẫn mới sau khi xử lý nguyên nhân.

Script dùng PyYAML (`yaml`), dependency đã dùng trong trainer của project.
Không cần PyTorch/CUDA cho bước tổng hợp/khóa kết quả đã đo.

## Nếu đã tổng hợp xong trước đó

```bash
python -B scripts/create_representation_lock.py \
  --summary-csv outputs/week06_multiscale.csv \
  --output configs/representation_lock.yaml
```

Script standalone đọc cả `.per_category.csv` và `.provenance.json`, kiểm tra
hash input gốc, chạy lại **phép tổng hợp các report đã đo**, rồi đối chiếu bảng
đã xuất. Nó không chạy lại inference hay AU-PRO. Bảng chỉnh tay hoặc nguồn đã
thay đổi sẽ bị từ chối. Giữ các input gốc ở path đã ghi trong provenance.

Nếu bảng được tạo bằng bản Task 9 cũ chưa có `input_roles`, chạy lại aggregator
đã cập nhật với output path mới trước khi tạo lock.

## Quy tắc chọn đã định nghĩa

- Metric quyết định: **macro AU-PRO0.05**, giữ nguyên thang `[0,1]`.
- Chọn candidate có AU-PRO lớn nhất trên cùng DEV-synthetic/category scope.
- Sai khác <= `1e-12` chỉ được xem là numerical tie, không phải ngưỡng effect
  size khoa học. Khi tie, ưu tiên `R0`, rồi `R1`, rồi `R2` để chọn representation
  đơn giản hơn.
- Tiny, boundary, BF1, runtime, VRAM và params được ghi đầy đủ để đọc tradeoff;
  chúng không tạo thêm điều kiện chọn hoặc thay metric chính.
- Không đọc visualization, không chọn dựa trên hình đẹp.

Quy tắc này cần được giữ cố định trước khi xem kết quả. Nếu nghiên cứu đã khóa
một ngưỡng gain tối thiểu hoặc budget runtime/VRAM, phải bổ sung đúng quy tắc
đó trước khi dùng kết quả; không thử các threshold sau khi xem candidate nào thắng.

Không dùng test_public/test_private để chọn representation. Script yêu cầu
metric chọn đến từ `dev_synthetic`; số test chỉ dùng đánh giá representation đã
khóa. Nếu chỉ đánh giá một subset category, YAML ghi đúng subset đó, không tự
gọi kết quả là bằng chứng cho toàn bộ tám category.

## Các trường được tự điền

```yaml
selected: <R0 hoặc R1 hoặc R2, tính từ metric thật>
representation: <deep_only hoặc multi_layer_local hoặc multi_layer_local_context>
evidence:
  primary: au_pro_0.05
  primary_value: <AU-PRO của candidate được chọn>
  tiny: <tiny AU-PRO0.05 hoặc null nếu không có nhóm GT hợp lệ>
  boundary: <boundary AU-PRO0.05 hoặc null nếu không có nhóm GT hợp lệ>
  boundary_f1: <BF1 hoặc null>
  params: <trainable parameter count>
  runtime: <mean của các category median latency, ms/batch>
  vram: <max peak allocated qua category, MiB>
```

Block trên là mô tả schema, không phải YAML kết quả để chạy. File kết quả còn có:

- `selection_rule`: metric, tie tolerance/order và `visualization_used: false`.
- `candidates`: evidence của cả ba, không chỉ candidate được chọn.
- `comparisons.R0_to_R1` và `.R1_to_R2`: delta primary/tiny/boundary và chi phí.
- `scientific_assessment`: hai bước có cải thiện AU-PRO quan sát được hay không.
- `evaluation_scope`: split, seed, categories và nhóm diagnostic có GT hợp lệ,
  GPU, resolution, batch size và phạm vi inference.
- `aggregation_rules` và `provenance`: cách gộp, hash input, evaluator,
  normalization, hash ba bảng/provenance và source script tạo lock.

Không có tiny/boundary region trong GT là metric **undefined**, không phải
thiếu file hoặc score bằng 0. Nếu report đầy đủ và status/count xác nhận không
có nhóm hợp lệ, YAML giữ `null` cùng scope. Nếu report thiếu thì không khóa.

Runtime giữ phạm vi Task 8 **cached representation pipeline**; không chuyển
thành end-to-end latency toàn ảnh. `boundary` là AU-PRO của boundary defects;
BF1 được ghi riêng để xem độ khớp contour.

## Trả lời câu hỏi khoa học

`R0_to_R1.observed_localization_improvement` là true khi delta AU-PRO chính
R1−R0 > tolerance: multi-layer cải thiện localization quan sát được trên scope
DEV đã khóa. Tương tự R2−R1 cho việc Context có bổ sung lợi ích.

`both_steps_improve` chỉ true khi **cả hai** delta đều dương. Ví dụ R1 tốt nhất
nhưng R2 kém R1 thì chọn R1 và ghi Context không cải thiện metric chính trong
lượt chạy này; không mặc định Context luôn tốt hơn.

Đây là evidence mô tả với một seed. YAML ghi
`significance_test_performed: false`; script không tự kết luận đã chứng minh
statistical significance. Tiny/boundary có thể tăng hoặc giảm trái chiều với
macro chính và vẫn được giữ nguyên trong evidence để bạn giải thích.

Lock là artifact ghi quyết định, không tự cập nhật `day05_representation.yaml`,
không retrain và không thay pipeline deployment.

## Kiểm tra

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q -p no:cacheprovider \
  tests/test_aggregate_week06_multiscale.py tests/test_representation_lock.py
```

Tests dùng dữ liệu synthetic để kiểm tra selection, tie, delta, YAML null,
automation, chống ghi đè và từ chối evidence thay đổi. Không phải số nghiên cứu.
