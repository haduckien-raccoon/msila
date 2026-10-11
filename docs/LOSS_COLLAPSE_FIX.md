# Kiểm tra loss và bản đồ nền đen — fix/los

Bản này ghi audit và thay đổi v3. Điều tra tiếp về synthetic shortcut,
augmentation xóa nhãn và loss illumination mới nằm ở
[SYNTH_ILLUMINATION_FIX.md](SYNTH_ILLUMINATION_FIX.md); notebook hiện dùng v4.

## Bằng chứng từ run thật

Đã đọc `notebook_environment.json`, toàn bộ 7.090 dòng `loss_history.json`,
training metrics, public metrics và QA 150 ảnh từ [thư mục người dùng cung cấp](https://drive.google.com/drive/folders/1-gREy47QE9XVuRmaUeBnyubY2ozcgkX2).
Thư mục này chứa run `walnuts`, commit `bfb2ee5`, 5 epochs, batch 32,
inference batch 16, BF16 trên NVIDIA L4. Bảy category còn lại ghi NOT RUN;
kết quả `can` trong tin nhắn chưa được đối chiếu độc lập với artifact của nó.

Chỉ 8.101 lượt tile có mask dương, trên khoảng 226.880 lượt tile: **3,57%**.
Mẫu số là số batches × 32, nên là xấp xỉ do batch cuối có thể nhỏ hơn.
Khoảng 34–35% batches hoàn toàn không có lỗi. Batches có lỗi thường chỉ có
1–5 tile dương, không phải cả batch đều dương.

| Epoch | BCE trung bình | Dice loss trên batches có lỗi | Tile dương, xấp xỉ | Recall trung bình trên batches có lỗi |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,002149 | 0,652203 | 3,57% | 0,559656 |
| 2 | 0,001114 | 0,573668 | 3,60% | 0,653214 |
| 3 | 0,001452 | 0,560559 | 3,50% | 0,668925 |
| 4 | 0,001032 | 0,542879 | 3,59% | 0,675500 |
| 5 | 0,001092 | 0,546861 | 3,60% | 0,666501 |

Adapter, projection, decoder và fusion có cập nhật; backbone giữ frozen.
Best là epoch 2: synthetic DEV AU-PRO 0,995900, synthetic Dice 0,720140.
Epoch cuối DEV AU-PRO giảm còn 0,879269. TEST_PUBLIC của best đạt 0,301619,
contour Boundary-F1 chỉ 0,000310. Có 44/150 ảnh có max probability dưới 0,5;
4 ảnh dưới 0,001; không có ảnh nào hoàn toàn zero trong QA.

Như vậy model đã học một phần tín hiệu synthetic. Điểm DEV cao chưa chứng minh
generalization với lỗi thật. Synthetic hiện chỉ tạo nhiễu cường độ/texture nhỏ
4–2.048 pixel, độ tương phản 0,04–0,12; đây là proxy, không mô phỏng đầy đủ lỗi
thật. Chênh lệch DEV/TEST phù hợp với vấn đề generalization, nhưng chưa xác định
được đóng góp nhân quả của từng thành phần nếu chưa train lại.

## Lỗi và điều chỉnh

1. **BCE bị nền lấn át.** Mean BCE chia cho mọi pixel. Với 4 pixel lỗi trên
   512×512, loss cũ giảm từ 1,693117 xuống 1,000175 khi toàn bộ logits giảm
   từ 0 xuống −10, và gradient bias vẫn đẩy về phía âm. Dice gần 1 và tín hiệu
   phục hồi nhỏ. BCE cân bằng mới làm loss của logits −10 tăng lên 6,000023,
   gradient bias khoảng −0,49996: gradient descent kéo logits lên để sửa lỗi.
2. **Tile có lỗi quá hiếm.** Mỗi synthetic native image chỉ có một vùng nhỏ,
   nhưng trước đây train toàn bộ grid. Chính sách mới giữ nguyên số slots và
   updates, chọn ít nhất 50% slots của mỗi synthetic variant từ tile thực sự
   chứa mask dương. Slots còn lại chọn tile âm nếu có; ảnh normal giữ toàn bộ
   grid. Với hai synthetic variants và một normal variant, tỷ lệ tile dương
   tổng thể thường ít nhất khoảng 1/3. Local/Context dùng chung record đã chọn.
3. **Loss trực tiếp chưa an toàn với reduced precision.** Dice trước đây có
   thể overflow tổng FP16 ở 512×512. Sigmoid/BCE/Dice nay dùng ít nhất FP32;
   FP64 được giữ cho kiểm tra số học. Fixed trainer của run thật đã ép FP32,
   nên đây là hardening, không phải nguyên nhân của run BF16 đã kiểm tra.
4. **PASS không xác nhận chất lượng.** Thêm log positive pixels/tiles,
   p_foreground/p_background và `learning_diagnostics.json` theo epoch với gamma,
   gradient nhánh adapter của batch cuối, cảnh báo không có positive tiles hoặc
   zero recall ở ngưỡng 0,5. Gradient batch cuối có thể zero trên batch normal;
   phải đọc cùng thống kê cả epoch. Recall 0,5 là diagnostic; AU-PRO vẫn quét
   continuous scores. Hình tối một mình không chứng minh model không có tín hiệu.

Loss mới cho từng tile dương:

```text
BCE_balanced = 0.5 * mean(BCE trên foreground)
             + 0.5 * mean(BCE trên background)
L = BCE_balanced + Dice_positive_masks
```

Tile chỉ có một class dùng nguyên mean BCE của class đó. Loss không resize
mask, không sigmoid trước BCEWithLogits, không điều chỉnh score từ TEST.
`bce_mode: mean` vẫn là mặc định API cho các protocol cũ; Fixed E5 khai báo
`balanced` rõ ràng trong config. Synthetic DEV, native Hann stitching,
AU-PRO@0.05 và quy tắc chọn best giữ nguyên. Loss/sampling được đưa vào config
hash; checkpoint loss cũ không được resume dưới cấu hình mới.

## Chạy bản sửa

Dùng [notebook Fixed E5](../notebooks/G2_05_E5_Fixed_Attention.ipynb) sau khi đưa
nhánh `fix/los` lên remote. Notebook và `configs/g2_fixed_attention.yaml` dùng
namespace `e5_r128_d512_attention_consistency_v4`, 5 epochs, batch 32, inference 16,
seed 42/dev seed 17017 và cùng DINO commit để đối chiếu với run đã audit. Trên L4, `auto`
resolve BF16. Sửa đường dẫn Drive, chạy preflight → smoke → full. Run mới không
restore namespace cũ. Nếu chủ động đổi budget, đặt RUN_NAME mới.

Theo dõi positive tiles, p_fg/p_bg, recall và gamma/gradient; không chỉ nhìn
total loss hoặc Training PASS. Đánh giá lại toàn bộ TEST_PUBLIC sau train,
giữ lựa chọn checkpoint trên synthetic DEV. Chưa có bằng chứng điểm TEST sẽ
tăng đến mức nào; môi trường hiện tại không có CUDA, dataset hoặc DINO weights.

## Kiểm chứng

213 tests passed, 26 skipped: loss cũ/mới, FP16/BF16 reductions, masks rỗng,
gradient foreground không bị chia loãng theo diện tích, toy low-contrast sau
5 epochs, tile sampling/Local-Context, nhiều worker, checkpoint resume,
G1/G2 contracts và metric reference. Tests GPU và một số integration gates
skip vì thiếu điều kiện; CPU fixtures không phải kết quả accuracy pretrained.
Lệnh kiểm thử bật `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` để tránh plugin ROS ngoài
project. Test nhiều worker cần socket IPC nội bộ của PyTorch.

Số liệu audit tổng hợp ở [reports/loss_collapse_audit.json](../reports/loss_collapse_audit.json).
