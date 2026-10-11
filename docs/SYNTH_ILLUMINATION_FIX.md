# Synthetic và illumination — fix/los v4

## Kết luận từ code và artifact

Run thật đã audit có adapter cập nhật và học được synthetic: `walnuts` đạt
synthetic DEV AU-PRO@0.05 **0,995900**, nhưng TEST_PUBLIC chỉ **0,301619**.
Đây là chênh lệch khả năng khái quát, không phải bằng chứng adapter hoàn toàn
không học. Thư mục Drive này chỉ có `walnuts` hoàn tất; chưa đối chiếu được
artifact `can` trong ảnh người dùng. Số liệu và nguồn ở
[audit trước](LOSS_COLLAPSE_FIX.md) và [JSON tổng hợp](../reports/loss_collapse_audit.json).
`0.05` trong tên AU-PRO là giới hạn FPR của metric, không phải điểm dự đoán.

Đọc luồng thật: `g2_e2` → `IlluminatedNativeDataset` → `G1NativeDataset` →
`NativeTinyDefectGenerator` → tile/Local-Context → DINO frozen → adapter →
projection/fusion/decoder → loss. Generator legacy đầu file có cut-paste/noise
không phải generator native đang được E5 sử dụng.

Các vấn đề xác định được:

1. Bốn tên native `pinhole/thin_scratch/texture/contamination` chủ yếu khác
   hình học và hệ số **cộng/trừ cường độ RGB**. Dấu của perturbation phụ thuộc
   mean vùng gốc: vùng sáng luôn bị làm tối và ngược lại. Đây là dấu hiệu dễ
   học của generator; chưa có ablation chứng minh mức đóng góp vào lỗi TEST.
2. TRAIN và DEV cũ cùng họ renderer, DEV mặc định chỉ bốn ảnh nguồn good mỗi
   category. Seed và source độc lập vẫn không bảo đảm đa dạng lỗi thật.
3. Tăng sáng sau khi synth rồi `clamp(0,1)` có thể xóa tín hiệu lỗi mà giữ mask.
   Regression tái hiện: nền 0,99, contrast 0,04, pinhole giảm 0,036; nhân 1,1
   làm cả ảnh sạch lẫn ảnh lỗi thành 1,0. Cấu hình cũ cho phép các tham số này.
4. Lịch loại lỗi dùng `epoch * n_sources * variants` modulo số combinations.
   Nếu stride chia hết chu kỳ, từng ảnh giữ nguyên loại/size/placement qua các
   epoch. Seed chỉ đổi tọa độ/biên độ. Ví dụ 12 sources × 2 variants với 24
   combinations gây lặp; đây là điều kiện từ code, không phải khẳng định số
   sources trong run Drive bằng 12.

Chênh lệch DEV/TEST không đủ để kết luận illumination là nguyên nhân duy nhất.
Mất cân bằng pixel/tile đã sửa ở v3 cũng là một vấn đề độc lập.

## Thay đổi v4

### Synthetic TRAIN

`synthetic_appearance: diverse_v1` dùng ba kiểu appearance, chọn độc lập với
tên hình học: intensity có dấu ngẫu nhiên, channel shifts khác dấu, và texture
transfer từ chính ảnh TRAIN/good. Texture donor được khớp mean từng kênh với
vùng đích trước khi blend để giảm tín hiệu chênh sáng/màu trung bình của patch.
Mọi thay đổi vẫn chỉ nằm trong mask native; normal variant giữ nguyên ảnh.

Mỗi pixel có nhãn phải có RGB mean absolute change tối thiểu 0,015. Nếu texture
phẳng hoặc bị saturation làm tín hiệu yếu, renderer dùng perturbation nhìn
thấy được; số pixel fallback và tham số donor/alpha/polarity ghi trong metadata.
Fallback không được coi là texture transfer thành công. Dấu chỉ đảo khi hướng
đã chọn không đủ khả kiến; không dùng quy tắc mean > 0,5 cho mọi defect.

Lịch epoch mới nhảy qua một block loại lỗi và thêm một slot, tránh stride phụ
thuộc số nguồn. Với bốn loại, hai placement và 1–3 bins của protocol, stride
nguyên tố cùng nhau với độ dài chu kỳ. Source/seed/epoch vẫn tái lập chính xác.
Chế độ `legacy` giữ renderer và lịch cũ cho đối chứng.

Area bins, placement, số synthetic variants và positive tile sampling giữ như
v3. Bản sửa chưa mở rộng mô hình hình học ra tất cả các loại lỗi thực tế.

### Loss illumination riêng

```text
x = ảnh TRAIN sạch hoặc ảnh có lỗi synthetic, y = mask
x_lit = T(x), p = sigmoid(f(x)), q = sigmoid(f(x_lit))
L_seg = BCE_balanced(f(x), y) + Dice_positive(f(x), y)
L_illum = mean_pairs[balanced_pixels(KL(Bern(stopgrad(p)) || Bern(q)))]
L_total = L_seg + 0.1 * (n_pairs / batch_size) * L_illum
```

BCE và Dice **luôn** giám sát ảnh gốc `x`. Loss illumination ràng buộc bản đồ
dự đoán của cùng một ảnh qua đổi sáng; không dùng brightness làm nhãn anomaly.
KL tính từ raw logits bằng BCEWithLogits trừ entropy teacher, ít nhất FP32.
Foreground/background được trung bình riêng trên từng tile, tránh pha loãng
tín hiệu lỗi nhỏ. Tile chỉ có background hoặc foreground dùng trọng số đầy đủ
cho class hiện diện. Teacher detach, chỉ nhánh ảnh đổi sáng nhận gradient KL.

Consistency một mình vẫn chấp nhận nghiệm toàn nền đen; nó **không thay thế**
BCE/Dice cân bằng. Weight 0,1 là giá trị khởi đầu khai báo trước, chưa được
xác nhận tối ưu trên pretrained data. Không chọn weight bằng TEST_PUBLIC.

`T` là gamma và trường exposure trơn:

```text
u = x ** (1 / contrast)
gain = brightness * exp(gx * X + gy * Y)
T(x) = gain * u / (1 + (gain - 1) * u)
```

Không hard-clamp; cả hai view crop từ cùng một native canvas. Trước khi nhận
pair, so sánh `T(synthetic)` với `T(original good)` tại pixel có nhãn. Nếu bất
kỳ pixel nào còn dưới 50% contrast gốc hoặc dưới nửa mức visibility tối thiểu,
bỏ pair illumination, vẫn giữ nguyên supervised sample. Kiểm tra visibility
chỉ dùng dữ liệu TRAIN. Transform không thể khôi phục lỗi đã mất thông tin
trong ảnh thật bị overexposed; loss này không phải giải pháp cho mọi lighting.

Trainer backward supervised trước, giải phóng activations, rồi forward/backward
các pair active và thực hiện **một** optimizer step. DINO vẫn frozen, gradient
đi qua adapter/projection/fusion/decoder. Số updates không đổi; compute tăng do
forward/backward bổ sung, mặc định probability 0,5. Chưa đo throughput/VRAM GPU.

Log riêng `segmentation_loss`, `illumination_loss`, `illumination_contribution`,
`illumination_pairs`, `illumination_rejected`; tổng hợp theo epoch trong
`learning_diagnostics.json`, cùng p_fg/p_bg và gradient adapter đã có ở v3.

## Chạy và đối chứng

[Notebook Fixed E5](../notebooks/G2_05_E5_Fixed_Attention.ipynb) và
[config](../configs/g2_fixed_attention.yaml) mặc định dùng
`e5_r128_d512_attention_consistency_v4`, 5 epochs, batch 32, seed 42/dev 17017.
Tạo run mới; không resume weights/optimizer của v3 hoặc run 5 epochs cũ.
Config và source hashes bao gồm renderer/mode/weight mới.

Khai báo bốn cấu hình trước khi chạy, mỗi cấu hình một `RUN_NAME` riêng:

| Run | SYNTHETIC_APPEARANCE | ILLUMINATION_MODE | ILLUMINATION_LOSS_WEIGHT | So sánh |
| --- | --- | --- | ---: | --- |
| A | legacy | augmentation | 0 | Baseline v3 với augmentation clamp cũ |
| B | legacy | off | 0 | A→B: bỏ augmentation cũ |
| C | diverse_v1 | off | 0 | B→C: đổi appearance và lịch synth |
| D | diverse_v1 | consistency | 0.1 | C→D: thêm consistency |

Cả bốn giữ BCE balanced, positive fraction 0,5, cùng seeds, nguồn, budget và
native DEV/metric. Chọn best bằng synthetic DEV như protocol hiện tại. TEST
chỉ báo cáo sau train, không dùng để tìm weight/threshold/checkpoint. Đây là
đối chứng cùng số optimizer updates, không phải cùng FLOPs hay thời gian GPU.

DEV cũ nay là một họ renderer khác với TRAIN mới nhưng vẫn là synthetic proxy.
Điểm DEV cao không bảo đảm lỗi thật. Decoder hiện tạo logits từ grid DINO 32×32
rồi bilinear lên 512×512; loss không tự bổ sung chi tiết ảnh ở cấp pixel đã mất.
Các giới hạn này cần được xét khi đọc Boundary-F1 và lỗi rất nhỏ.

## Kiểm chứng và giới hạn bằng chứng

Kết quả bộ hồi quy: **234 passed, 26 skipped** trên CPU (67,09 giây), gồm 21
tests mới về illumination/synth. Các skip là điều kiện GPU/integration chưa
đáp ứng. `git diff --check` sạch. Lệnh chạy:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_anomaly_loss.py tests/test_loss_collapse.py \
  tests/test_illumination_consistency.py tests/test_overfit16_trainer.py \
  tests/test_g2_fixed_attention.py tests/test_g2_fusion_amp.py \
  tests/test_g2_tv1_runner.py tests/test_g2_tv1_model.py tests/test_g2_tv1_e4.py \
  tests/test_g1_e1.py tests/test_g1_test_public.py tests/test_aupro_reference.py \
  tests/test_full_scale_contracts.py tests/test_day03_cached_train_step.py \
  --disable-warnings
```

Tests kiểm tra KL theo công thức Bernoulli độc lập, teacher detach, saturated
student vẫn có gradient, tiny foreground không bị pha loãng, FP16/BF16/FP32/FP64,
normal masks, augmentation cũ xóa lỗi, transform mới giữ lỗi, visibility reject,
mask support/epoch schedule/geometry, zero-weight không forward lần hai, và
hai backward cho kết quả tương đương objective gộp trong một optimizer update.
Integration dùng CPU backbone fixture để kiểm tra checkpoint/resume và frozen
backbone. Không có CUDA, dataset hay pretrained weights trong workspace để
train lại năm epochs hoặc công bố mức tăng TEST_PUBLIC.

Ý tưởng invariance theo augmentation có liên quan tới
[AugMix](https://arxiv.org/abs/1912.02781); đa dạng proxy patch có liên quan tới
[CutPaste](https://arxiv.org/abs/2104.04015). Implementation này là KL Bernoulli
có teacher detach và renderer riêng, không phải tái hiện nguyên bản các paper
hay bằng chứng các kết quả công bố sẽ chuyển sang MSILA.
