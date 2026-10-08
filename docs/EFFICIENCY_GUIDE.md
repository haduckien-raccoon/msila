# Task 8 — Efficiency cho R0/R1/R2

## Bạn cần chèn gì?

Không thay thế hoặc chỉnh `src/eval/efficiency.py`.
Day 4 đã có `parameter_report`, `make_benchmark_scope` và
`benchmark_inference_efficiency`. File `efficiency_notebook_cell.txt` là một
cell điều phối để copy vào notebook đánh giá đang có, chạy từ thư mục gốc repo.
Không cần thêm module vào project.

Code và tài liệu được chuẩn bị ngoài project. Chỉ khi bạn chạy cell, cell mới
tạo các artifact kết quả ở thư mục `OUT` do bạn chọn. Nó không sửa source,
checkpoint hoặc config; thư mục output phải mới để tránh ghi đè.

## Đầu vào thật

Chỉnh những biến đường dẫn ở đầu cell:

- `RUN_ROOT`: root output từ `src.train.screen_representation`; bên dưới có
  `seed_42/fabric/R0`, `R1`, `R2`, mỗi thư mục chứa `best.pt` và
  `resolved_config.yaml`.
- `CATEGORY`, `SEED`: cùng category/seed cho cả ba candidate.
- `CACHE_DIR`: cache thật dùng khi train, manifest phải có cùng SHA-256.
- `VAL_RECORDS`: DEV-synthetic records thật dùng khi train. Code kiểm tra hash.
- `MASK_ROOT`: root nếu mask paths là relative; giữ `None` khi paths tự resolve.
- `OUT`: thư mục kết quả mới, tốt nhất ngoài cây source.
- `DAY05_CONFIG`, `TRAIN_PROTOCOL`: cấu hình thật tương ứng các run.

Chạy trên runtime CUDA có PyTorch và các dependency sẵn dùng để train project.
Restart runtime rồi import/chạy cell trước khi nạp model lên GPU. Cell yêu cầu
không còn tensor GPU của lần train/đánh giá trước. `empty_cache()` chỉ giải phóng
cache allocator, không thể giải phóng tensor còn được biến khác giữ.

Cell được viết cho checkpoint của `Day05RepresentationModel` trong repo hiện
tại; không dùng trực tiếp checkpoint Adapter Day 4 với kiến trúc attention.
Checkpoint này chứa RNG/optimizer nên được load trên CPU với
`weights_only=False`; chỉ dùng checkpoint bạn tạo hoặc tin cậy.

## Phạm vi inference

Đây là **cached representation pipeline**, tương ứng phạm vi đo Day 4:

`cached features -> Adapter -> alignment nếu R2 -> projection -> MeanFusion -> Decoder`.

Đầu ra đo là raw logits; existing forward vẫn tạo trace. Code không sửa forward
hoặc tắt validation riêng cho candidate nào.

Không gồm đọc dữ liệu, chuyển feature/mask lên GPU, backbone DINOv3, tiling hoặc
Hann stitching. Geometry được existing forward dựng/chuyển sang GPU cho R2
vẫn nằm trong cửa sổ đo. Đây là chi phí thực của implementation hiện tại.

Không gọi kết quả này là thời gian inference toàn ảnh. Nếu bảng cuối yêu cầu
end-to-end inference toàn ảnh, cần đo callable inference thật bao gồm các bước
đó bằng cùng API Day 4, với cùng ảnh và cùng protocol cho cả ba candidate.
Không cộng backbone timing đo ở môi trường khác vào số này.

`tile_resolution` là kích thước output tile, không phải kích thước ảnh gốc.
Cell lấy nó từ config và kiểm tra GT/output cùng kích thước. `context_size=768`
trong config là context source FOV; không tự diễn giải thành backbone input
768x768. Input contract, shape của cả sáu feature map và hash cache đều được
lưu trong protocol để truy vết preprocessing thật.

## Quy trình chung đã khóa

| Điều kiện | Giá trị/quy tắc |
|---|---|
| GPU | Cùng `cuda:0` trong cùng lượt chạy; lưu GPU name/UUID nếu API hỗ trợ |
| Input | Cùng batch thật; image_id được sắp xếp trước khi xem candidate scores |
| Resolution | Cùng input contract và feature shapes từ cache thật |
| Batch size | 1 cho cả R0/R1/R2 |
| Precision | FP32, autocast và TF32 tắt |
| Mode | `eval()` và `inference_mode()` |
| Latency warm-up | 10 mỗi round |
| Latency measurement | 50 iterations mỗi round, 3 rounds |
| CUDA synchronization | Có trước và sau mỗi lần đo, do API Day 4 thực hiện |
| Timing statistic | Median trên tất cả 150 lần đo; lưu thêm p95 |
| Stability | CV của round medians <= 0.10 |
| VRAM warm-up | 10 |
| VRAM measurement | 1 inference; reset peak sau warm-up |
| Peak VRAM chính | `max_memory_allocated`, đơn vị MiB |

Giữ GPU nhàn rỗi, không có workload khác chạy đồng thời và giữ cùng điều kiện
nguồn điện/clock. Cell chỉ kiểm tra tensor trong process hiện tại; không kiểm
tra tải GPU của process khác. Nếu latency FAIL, ổn định môi trường rồi đo lại
**cả ba** với cùng protocol và thư mục output mới. Không chọn lần tốt nhất riêng
cho từng candidate. Nếu thay batch size/warm-up/iterations thì khóa lại trước
khi xem kết quả và áp dụng cùng giá trị cho cả ba.

## Định nghĩa ba cột chính

- `trainable_params`: số phần tử Parameter unique có `requires_grad=True`,
  đếm trên model được tạo đúng config. `eval()` không đổi cờ này. Không dùng
  `requires_grad_(False)` toàn model để chuẩn bị inference trước khi đếm.
- `inference_ms_per_batch`: median milliseconds cho một batch trong phạm vi
  trên. Batch size=1 tương ứng một tile/sample, không nhất thiết một ảnh gốc.
- `peak_vram_MiB`: absolute peak allocated của PyTorch trong inference, gồm
  model và inputs resident. Không phải peak reserved, phần VRAM tăng thêm hoặc
  bộ nhớ process hiển thị bởi nvidia-smi.

Chỉ một model được nạp lên GPU tại một thời điểm. Sáu feature maps và mask/meta
chung đều resident cho mỗi candidate theo cùng input envelope Day 4. Các module
R0 bị freeze nhưng vẫn được class đăng ký vẫn resident: code đo implementation
thực, không tự prune kiến trúc để làm số đẹp hơn. Chi tiết baseline, incremental
peak và peak reserved có trong JSON Day 4 để kiểm tra.

R0 chỉ có các block active cho deep-only nên số trainable params có thể thấp
hơn R1. R1 và R2 chia sẻ weights local/context; Context alignment và MeanFusion
không có parameter. Vì vậy R1/R2 có thể bằng nhau về trainable params nhưng
khác latency/VRAM. Không nhân số parameter theo số source feature.

## Output và bảng cuối

```text
efficiency_protocol.json
R0_efficiency.json
R1_efficiency.json
R2_efficiency.json
efficiency_summary.csv
```

Protocol được ghi trước khi đo. JSON từng candidate có checkpoint/config hash,
parameter report và efficiency report Day 4. CSV có một row/candidate, kèm
category, seed, GPU, resolution, precision, protocol fields và status. Scope hash
phải giống nhau cho cả ba; khác checkpoint là đúng vì đó là candidate khác.

Nhập ba cột metric vào bảng cuối bằng khóa `(candidate, category, seed)`, không
ghép bằng thứ tự row. Nếu bảng chứa nhiều category/seed, chạy cell cho từng
bộ đó với cùng protocol và output directory khác; không tự gộp mọi category
thành một số không ghi rõ cách gộp.

| Candidate | Trainable params | Inference median (ms/batch) | Peak VRAM (MiB) |
|---|---:|---:|---:|
| R0 | lấy CSV | lấy CSV | lấy CSV |
| R1 | lấy CSV | lấy CSV | lấy CSV |
| R2 | lấy CSV | lấy CSV | lấy CSV |

Chỉ dùng rows có status PASS cho so sánh efficiency. Caption ghi rõ cached
representation pipeline, tile resolution, batch size, GPU và FP32. Efficiency
bổ sung phân tích chi phí cho kết quả localization; không tự chứng minh candidate
có AU-PRO tốt hơn.

## Giới hạn kiểm tra ở workspace hiện tại

Đã đối chiếu interface và protocol với source Day 4 và trainer Day 5, kiểm tra
cú pháp cell. Python hiện tại chưa có PyTorch và workspace chưa cung cấp
checkpoint/cache thật cho lượt đo. Vì vậy chưa có số trainable params,
milliseconds hoặc VRAM thực; không điền số giả vào bảng và không tuyên bố đã
benchmark CUDA.
