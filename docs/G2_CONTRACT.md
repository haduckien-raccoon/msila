# G2 MS-ILA — Shared Contract v3 (pipeline D5–D11, fusion-aware joint selection)

> **Phạm vi:** Hợp đồng tích hợp giữa TV1 (Model/Training) và TV2 (Data/Evaluation/Fusion). Đây là **đặc tả mục tiêu đã cập nhật**, không phải xác nhận rằng mọi tính năng đã được triển khai hoặc thực nghiệm đã PASS.
>
> **Thay đổi quan trọng so với v2:** D6 chỉ kiểm thử kỹ thuật 9 cặp `(r,d)`; D9 kiểm thử kỹ thuật 45 tổ hợp `(r,d,Fusion)`; D10 huấn luyện screening E5 để lựa chọn **chung** `(r*,d*,F*)` trên synthetic DEV; D11 mới main-train và nghiệm thu E0–E5. **Không yêu cầu `adapter_selection_lock.json` từ D6.**

## 1. Phạm vi, quyền sở hữu và trạng thái

- Repo: `haduckien-raccoon/msila`; nhánh `g2/member1` (TV1), `g2/member2` (TV2), `g2/integration` (tích hợp).
- Manifest cấu hình chung hiện có: `configs/g2_experiments.yaml`. TV1 chịu trách nhiệm thay đổi cấu hình/Model/Training/Runner; TV2 sở hữu Data/Geometry/Evaluator/Benchmark và ba module Fusion độc lập `weighted_sum`, `concat`, `gated` (tên file/lớp cụ thể tuân theo repository).
- Tình trạng đã ghi trong contract v2: E1–E4 đã có trong `src.models.msila`; **E5 chưa được xác nhận triển khai**. Trước khi dùng E5, phải kiểm tra thực tế trên commit tích hợp; không tuyên bố PASS dựa trên kế hoạch.
- Category TV1: `can`, `fabric`, `fruit_jelly`, `rice`. Category TV2: `sheet_metal`, `vial`, `wallplugs`, `walnuts`. Tổng kết khoa học phải bao gồm cả 8/8 category.
- **Tách hai loại nghiệm thu:** `CODE PASS` = các test/contract tương ứng đạt; `EXP PASS` = checkpoint, log, config, AU-PRO và coverage từ lượt chạy thật hợp lệ. Test fixture/smoke không được tính là kết quả khoa học.

### 1.1. Kiến trúc thí nghiệm

| ID | Kiến trúc | Khi nào huấn luyện/đánh giá chính thức |
|---|---|---|
| **E0** | Frozen DINOv3 + Normal Feature Memory Bank (không train Adapter/Decoder) | D5 có thể chạy Colab độc lập; hoàn tất chậm nhất D11 |
| **E1** | Frozen DINOv3 deepest feature → Decoder | D11; có thể tái sử dụng checkpoint G1 nếu protocol tương đương |
| **E2** | E1 + một Residual Adapter cho deepest feature | D6 chỉ test kỹ thuật; D11 main training với `(r*,d*)` |
| **E3** | 3 layer Local → 3 Adapter → projection → **Mean(3)** → Decoder | D7 code/test; D11 main training |
| **E4** | 3 Local + 3 Context → alignment/projection → **Mean(6)** → Decoder | D8 code/test; D11 main training (baseline) |
| **E5** | Cùng **6 nguồn** của E4, thay Fusion bằng một trong **5 candidate** | D9 integration; D10 joint search; D11 main training và so sánh 5 Fusion |

`E3 Mean(3)` khác `E4/E5 Average Mean(6)` về số nguồn. `E5-Average` chỉ tái sử dụng checkpoint E4 khi kiến trúc **và toàn bộ resolved config, weights, protocol, provenance tương đương**; không mặc định tương đương do trùng tên Fusion.

## 2. API Model và hợp đồng batch — giữ tương thích code cũ

Giao diện đích thống nhất: `forward(image, *, return_trace=False)`, trong đó `image` có thể là tensor hoặc dict batch theo API thực tế hiện hành. Đầu ra là **raw logits** `[B,1,512,512]`. Với `return_trace=True`, trả `(logits, trace: dict)`; `trace["decoder_feature"]` là tensor truyền vào Decoder. `model(image_tensor)` được giữ cho đường G1/E1–E3; **E4/E5 yêu cầu batch có `context`, `view_meta`**. Nếu E5 chưa tồn tại, đây là yêu cầu để TV1 triển khai, không phải tính năng đã xác nhận.

| Batch key | Contract |
|---|---|
| `image` | Local RGB float `[B,3,512,512]`, đã chuẩn hóa theo DINOv3 mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` |
| `mask` | Khi train/eval có ground truth: float/binary `[B,1,512,512]`, `1=anomaly`, cùng device |
| `meta` | Danh sách B dict; source, split, native size, tile coordinates |
| `context` | **Bắt buộc E4/E5**: RGB `[B,3,512,512]` sau khi lấy FOV 768×768 cùng tâm và resize về 512; cùng dtype/device và chuẩn hóa nhất quán |
| `view_meta` | **Bắt buộc E4/E5**: danh sách B dict, có `geometry` đủ để align Context → Local |

Dùng `validate_g2_batch(batch)` ở boundary theo chữ ký hiện có; inference không cần `mask`, và có thể không cần `meta` nếu pipeline dự đoán native đã có thông tin tọa độ riêng. E1/E2/E3 chỉ dùng Local. Batch G2 cố định tile 512; xử lý kích thước khác 512 thuộc đường tương thích G1/native tiling.

## 3. DINOv3, Adapter và các ký hiệu r/d

### 3.1. E2 — không có Fusion

`Frozen DINOv3(feature_mode="deepest") → ResidualAdapter2d → BasicDecoder`.

- Deepest feature: `[B,C,32,32]` với patch size 16 và tile 512. ViT-S/16: `C=384`; **ViT-B/16: `C=768`**. Backbone luôn `eval()/frozen`; tham số backbone không nằm trong optimizer.
- Theo contract v2, Adapter là `C → r → DWConv → d → C` và `F_out = F + gamma * ΔF`.
- `r = adapter_bottleneck_dim`; `d = adapter_projection_dim`. **`d` không phải `fusion.dim`**, không đổi chiều C tại đầu ra Adapter hay E2 Decoder. `fusion.dim` mặc định hiện có bằng `64`, độc lập với cả `r`, `d` và C.
- `gamma=0` phải cho identity chính xác; nếu Decoder weights giống nhau, logits E1/E2 trùng nhau trước cập nhật. Bước backward đầu tiên có thể chỉ có gradient ở gamma/Decoder; gradient nhánh Conv có thể bằng 0 khi gamma=0; kiểm tra nhánh Conv nhận gradient sau khi gamma thay đổi.
- E1 train Decoder; E2 train Adapter + Decoder; DINOv3 frozen. Đếm trainable params từ **model thật**, không chỉ dựa vào công thức lý thuyết.

### 3.2. Search space ViT-B/16 **đề xuất**, chưa được chọn

| Tên ứng viên | r (bottleneck) | Tên ứng viên | d (projection) |
|---|---:|---|---:|
| `r0` | **64** | `d1` | **256** |
| `r1` | **128** | `d2` | **512** |
| `r2` | **256** | `d3` | **768** |

- Tổng 3×3 = **9 cặp** `(r,d)`; khi kết hợp 5 Fusion là **45 cấu hình E5**, trên 8 category là **360 ô huấn luyện screening D10**. Đây là số lượng theo thiết kế; các ô OOM/thiếu dữ liệu không được tính thành kết quả AU-PRO.
- Trước khi dùng grid, TV1 phải xác minh 9 cặp có tác động thực sự đến kiến trúc và trainable params; nếu có ứng viên vô hiệu/trùng nghĩa, **dừng để sửa search protocol trước khi chạy D10**, không tự thay đổi grid sau khi xem DEV.
- `adapter.r/d: null` có thể dùng tỉ lệ mặc định theo registry/cấu hình cũ (ViT-B/16 thường suy ra `128/512`); **đây chỉ là `debug/default`, không phải `r*,d*`**. Khi chạy preflight/search/main, phải ghi **giá trị số nguyên đã resolve** cùng checkpoint.
- Chọn backbone qua `backbone.name` và mapping weights hiện có; `resolve_g2_config`/registry suy ra C/depth/blocks/patch size, không ghi đè bằng hằng số tùy ý. Không áp grid ViT-B cho backbone khác mà không định nghĩa protocol riêng.

Ví dụ **để kiểm thử**, không được coi là cấu hình thắng:

```yaml
backbone:
  name: dinov3_vitb16
adapter:
  r: 128     # debug_pair, không phải selected_r
  d: 512     # debug_pair, không phải selected_d
fusion:
  dim: 64   # KHÁC adapter.d
```

## 4. Feature contract E3/E4/E5 và Fusion

- Backbone ViT-B/16 depth 12 dùng ba block **4/8/12 (1-based)**; các backbone khác lấy theo registry, đồng thời ghi block thực trong trace.
- E3: 3 feature từ Local; 3 Adapter riêng theo tầng; projection 1×1 về `fusion.dim=64`; **Mean(3)**; Decoder output 512. **Chưa cần selection lock để tạo model, test CPU, smoke**; full main D11 đòi `joint_selection_lock.json` hợp lệ.
- E4: cùng frozen DINOv3 trích 3 Local + 3 Context; Adapter/projection **chia sẻ theo tầng giữa hai view** theo thiết kế v2; Context phải align về tọa độ Local trước Fusion. Trên E4 dùng Mean(6), không tối ưu r/d bằng Mean đơn lẻ.
- Sáu slot theo **đúng thứ tự** `MULTIVIEW_FEATURE_KEYS` (không dựa insertion order):

```python
MULTIVIEW_FEATURE_KEYS = (
    "local_b4", "local_b8", "local_b12",
    "context_b4", "context_b8", "context_b12",
)
```

  Các hậu tố `b4/b8/b12` là **tên slot** shallow/middle/deep; với backbone khác, block thực phải lấy từ registry và lưu trong `trace["source_blocks"]`.
- Sau align/projection: mỗi nguồn `[B,D,H,W]`, cùng dtype/device, hữu hạn; **`D=fusion.dim=64`, `H=W=32`** với tile 512/ViT-B16. C trong Adapter là `768`, **không nhầm C với D**. Fusion phải trả đúng **một tensor** `[B,D,H,W]` cho Decoder.
- Giữ trace nếu API hiện có hỗ trợ: `dino`, `adapted`, `projected`, `aligned_context`, `source_blocks`, `num_sources`, `decoder_feature`.

### 4.1. Tên Fusion: tên kế hoạch ≠ nhất thiết là ID trong code

| Tên trong kế hoạch | ID đã được contract v2 dùng | Bản chất bắt buộc |
|---|---|---|
| Average | `mean` | Trung bình đều 6 nguồn, không có trọng số học |
| Weighted | `weighted_sum` | Học 6 global scalar logits, softmax theo nguồn, cố định theo vị trí |
| Concat | `concat` | Nối `[B,6D,H,W]`, Conv1×1 xuống `[B,D,H,W]` |
| Gated | `gated` | Spatial sigmoid gate theo nguồn, chuẩn hóa ổn định qua 6 nguồn |
| Position-aware Attention | `attention` | **Spatial** logits `[B,6,H,W]`, softmax qua nguồn ở từng pixel/token, weighted sum ra `[B,D,H,W]` |

**QUAN TRỌNG:** Contract v2 cho phép `AttentionFusion` cũ trả `{feature,attention,logits}`. Điều này **không chứng minh** attention đó position-aware. D9-TV1 bắt buộc xác minh/tạo scorer `[B,6,H,W]`, không được dùng global logits `[B,6]` thay thế. Có thể viết adapter wrapper cho kết quả `{feature,...}` nhưng wrapper **không biến global attention thành spatial attention**. TV1 làm Fusion Factory/E5; TV2 chỉ bàn giao ba module độc lập, benchmark/evaluator.

Để giảm thay đổi code, **ưu tiên giữ ID đã có** `mean|weighted_sum|concat|gated|attention` và ánh xạ nhãn `average|weighted` tại boundary CLI/config **nếu cần**. Không tự đổi tên class/flag hiện có; trước hết kiểm tra registry/runtime. Các module phải giữ output width, gradient hợp lệ, finite values; dùng validator tương ứng sẵn có.

## 5. Local/Context geometry và native anomaly map

- Local crop 512×512; Context crop 768×768 cùng tâm, resize 512×512. Cả hai dùng **một synthetic anomaly** được tạo trên native image; mask giám sát nằm trong Local và không được tạo lại riêng cho Context.
- Kiểm tra padding ảnh biên, transform pixel-edge, metadata (`view_meta.geometry`), điểm có tọa độ ground truth, `align_corners=False`, tiny anomaly và tính tương ứng giữa hai view. E4/E5 từ chối geometry thiếu/singular hoặc Context không bao được Local nếu validator yêu cầu.
- Pipeline native inference dùng tiling/Hann hiện có; trả score map đúng H×W ảnh gốc và đúng orientation. **Tuân theo thứ tự sigmoid/stitching đã khóa ở evaluator hiện có** (contract v2 nêu sigmoid trước Hann); không tự thay đổi để so sánh dễ đẹp hơn.
- TV2 là chủ Data/Geometry/Metric; TV1 không sửa các module này trừ khi đã thỏa thuận merge/chuyển giao.

## 6. Giao thức thực nghiệm và lựa chọn cấu hình

### 6.1. Split, protocol và provenance

- TRAIN: synthetic anomaly từ `TRAIN/good`, có thể đổi theo epoch. Synthetic DEV: cố định từ `VALIDATION/good`, không dùng trùng source ảnh với TRAIN. Held-out `TEST_PUBLIC`/real defects: chỉ dùng **sau khi khóa** lựa chọn và chỉ khi hợp lệ theo quyền truy cập/giao thức của dataset.
- Contract v2 từng dùng seed 2026, DEV seed 17017, 20 epochs, batch 4, AdamW lr `1e-3`, weight decay `1e-4`, `BCEWithLogits + positive-mask Dice`. **Giữ làm main default nếu config đã xác nhận; không coi đây là budget screening D10.** Trước D10 phải định nghĩa và khóa **hai budget riêng**: `screening_budget` (ngắn, giống nhau giữa các ứng viên) và `main_budget` (full); Codex không tự bịa số epoch hay chọn lại sau khi xem DEV.
- Giữ cùng backbone pretrained/checksum, split, input/native protocol, seed, loss, optimizer, augmentation distribution, update budget, evaluation threshold rule giữa các candidate để so sánh. Với khác biệt batch do VRAM, phải công bố/chốt trước phương án công bằng về optimizer updates và báo các thay đổi; không hạ batch riêng cho ứng viên có điểm kém.
- Checkpoint được chọn trong mỗi lượt training theo synthetic DEV AU-PRO@0.05, tie-break epoch sớm nhất **nếu giao thức đã khóa như v2**. Việc dùng DEV cho cả chọn checkpoint và hyperparameter gây selection bias: **không gọi DEV là đánh giá tổng quát hóa độc lập**.
- Mỗi job lưu `run_id`, `experiment`, `category`, `r`, `d`, `fusion`, `seed`, `training_budget`, backbone weights checksum, split/dataset hash, config hash, Git commit SHA, best/last checkpoint, AU-PRO và trạng thái. Retry/resume phải kiểm chứng đúng hash, optimizer/RNG/cursor và không ghi đè kết quả lịch sử khi đổi protocol.

### 6.2. D5–D11: điểm chặn và bằng chứng nghiệm thu

| D | TV1 | TV2 | Đầu ra/điểm chặn |
|---|---|---|---|
| **D5** | E2 identity/gradient/contracts | E0 Memory Bank + `G2_01_E0_Baseline.ipynb` | Code tests; E0 8 AU-PRO khi Colab thực chạy (có thể bổ sung đến D11) |
| **D6** | **9 cặp `(r,d)` technical preflight**; notebook `G2_02_Adapter_Preflight.ipynb` | Evaluator G2 + `G2_04_Evaluation.ipynb` | 9 trạng thái CPU; GPU VRAM/OOM khi có; **không chọn/khóa r,d, không 72 E2 train** |
| **D7** | E3 Local Mean(3), shape/grad/checkpoint code | QA Local/Context/geometry 8 category | Code/QA tests; không ép E3 main train hoặc E3−E2 khi chưa có joint lock |
| **D8** | E4 Local/Context Mean(6) baseline | 3 Fusion Weighted/Concat/Gated | Six-source integration & gradient/geometry tests; **TV2 D8 merge trước D9 factory** |
| **D9** | E5 Position-aware Attention + Factory; **45 tổ hợp preflight** | Benchmark/evaluator & manifest cho 360 ô | 45 cấu hình ghi PASS/FAIL/NOT RUN; chưa chọn `(r,d,F)` |
| **D10** | Joint E5 short-budget train 45×4 = **180** ô | Joint E5 short-budget train **180** ô, tổng hợp | 360 trạng thái, AU-PRO trên synthetic DEV, khóa **`joint_selection_lock.json`** chỉ nếu lựa chọn có căn cứ |
| **D11** | Main train E1–E5 trên 4 category | Main train 4 category, evaluation/ablation 8 category | 8 E0, E1–E4 4×8, E5 5×8; report & checkpoints thực; không tái chọn từ D11 |

#### D6 — chỉ technical preflight

- ViT-B/16: test 9 cặp trên input fixture (instantiate, forward/backward, shape, finite loss/grad, params, config serialization); nếu chạy Colab, GPU smoke trên backbone thật và VRAM/OOM. **Không lấy AU-PRO từ model chưa train để xếp hạng.**
- Artifact gợi ý: `outputs/G2/D6/adapter_preflight.csv`, đủ 9 cặp. `PASS CPU` không ngụ ý `PASS GPU`.
- Nếu repo đang có `scripts/g2_colab_screening.py` hoặc notebook `G2_02_Adapter_Screening.ipynb` theo kế hoạch cũ, **tái sử dụng/sửa nhãn và luồng chạy**, không tự động thực hiện 72 training jobs. Giữ lịch sử/lock cũ ở trạng thái **legacy**, không xóa dữ liệu, không dùng để chốt E5.

#### D7–D8 — code/test độc lập selection

- E3/E4 cho phép `debug_pair` rõ ràng chỉ để test CPU/smoke. **Không yêu cầu `adapter_selection_lock.json`** khi tạo model, test gradient, hoặc smoke; main train vẫn **bị chặn** nếu chưa có `joint_selection_lock.json` D10.
- Các runner cũ có thể đang yêu cầu lock 72 kết quả; TV1 cần cập nhật **đường preflight/debug tách khỏi đường main** mà không làm hỏng checkpoint G2 cũ. Mọi thay đổi semantic phải có test và commit mới; không sửa hash/provenance lịch sử.
- E3−E2, E4−E3 chỉ tính sau D11 từ **main checkpoint được train cùng `(r*,d*)`**, cùng protocol.

#### D9 — kiểm thử 45 cấu hình

- Sau khi D8-TV2 Fusion modules đã merge, chạy `9 (r,d) × 5 Fusion = 45` bài test instantiate/forward/backward/config/gradient, ít nhất trên CPU mock; GPU preflight VRAM/OOM nếu có tài nguyên.
- Artifact: `outputs/G2/D9/fusion_preflight_45.csv`; *không có AU-PRO hiệu năng* nếu model chưa train. `Attention` chỉ PASS khi spatial source logits đúng `[B,6,H,W]`.
- TV2 chuẩn bị report manifest D10 đủ **45×8 = 360 ô**, các ô chưa chạy mang `NOT RUN/BLOCKED`, score null.

#### D10 — **joint trained search** và lock duy nhất

- Search đầy đủ nếu tài nguyên cho phép: 9 `(r,d)` × 5 Fusion × 8 category = **360 job E5**, chia mỗi TV 180 ô. Mỗi job **train ngắn có kiểm soát** và đo synthetic DEV AU-PRO@0.05. Không áp kết quả E2-only hoặc Average-only để tự loại cấu hình r/d. Nếu tiết giảm theo protocol định trước, chỉ kết luận *best among tested subset*.
- Report tối thiểu: `joint_search_360.csv`, `joint_search_coverage.json`, `joint_search_macro.csv` (45 dòng). Với mọi `(r,d,F)`, chỉ tính macro 8 category nếu **đủ 8 AU-PRO hợp lệ**. Không điền missing bằng 0 và không dùng `mean(skipna=True)` để làm như đủ coverage.
- **Chọn chung**:

  \[
  (r^*,d^*,F^*) = \arg\max_{(r,d,F)\in\mathcal V}\frac{1}{8}\sum_{c=1}^{8}\mathrm{AU\text{-}PRO@0.05}^{\mathrm{DEV}}(c;r,d,F)
  \]

  với `V` chỉ chứa cấu hình thỏa điều kiện kỹ thuật và có 8/8 metric DEV hợp lệ. Tie-break **phải được chốt trước khi xem DEV**; đề xuất: macro cao hơn → ít trainable params hơn → `r` nhỏ hơn → `d` nhỏ hơn → thứ tự Fusion đã khai báo. Đây là **đề xuất protocol**, chưa phải quy tắc chắc chắn đã được áp dụng.
- `joint_selection_lock.json` chỉ được tạo khi **mọi ô trong search space đã quyết định được trạng thái cuối** theo quy tắc định trước, có đủ 8 metric cho từng candidate đủ điều kiện lựa chọn, và có thể kiểm tra provenance; nếu còn chưa chạy thì báo `INCOMPLETE`/`PROVISIONAL`, không tự tạo lock. Ứng viên OOM được ghi `INFEASIBLE_ON_HARDWARE` và phải có log/trial thực; không được gán điểm 0 rồi so sánh.
- Nếu muốn kết luận tốt nhất trong **toàn bộ 45 cấu hình**, mọi candidate khả thi đều phải có 8/8 kết quả; cấu hình không khả thi phải có bằng chứng/rule loại trừ được tuyên bố trước. Full screening một seed/ngân sách ngắn **không bảo đảm** cùng ranking sau main train; nêu hạn chế trong báo cáo.
- **Không tạo `adapter_selection_lock.json` mới từ D6/D10** và **không tự chọn lại** theo kết quả D11.

#### D11 — main training, ablation, final evaluation

- D11 đọc duy nhất `(r*,d*,F*)` từ joint lock để **xác định lựa chọn đã khóa**. Dùng **chung `(r*,d*)`** để main-train E2, E3, E4 và tất cả 5 phương pháp E5; E1 không có Adapter nên không dùng r/d. E5 giữ nguyên nguồn/adapter/projection/decoder giữa 5 Fusion, chỉ khác Fusion.
- Cho 8 category: E0 có **8** AU-PRO; E1–E4 có **4×8=32** ô; E5 có **5×8=40** ô, trong đó `E5-Average` có thể reuse E4 **khi thật sự tương đương**, tránh train thừa.
- Báo cáo `metrics_g2.csv`, `fusion_comparison.csv`, `ablation_g2.csv`, `docs/G2_FINAL_REPORT.md`, `docs/G2_SOLO_RUNBOOK.md`; chênh lệch **E2−E1**, **E3−E2**, **E4−E3**, **mỗi E5 Fusion−E5 Average** theo 8 category và macro. E0 là baseline riêng, không gộp E0 với ablation supervised khi diễn giải.
- AU-PRO@0.05 là metric chính; tiny/mixed AU-PRO, Synthetic Dice, normal P99, trainable params, peak VRAM, latency và native anomaly maps là bằng chứng bổ sung. Nếu thiếu `Dice threshold` đã khóa thì đánh dấu metric chưa đủ điều kiện, không tự chọn theo TEST/DEV hậu nghiệm.
- `F*` được lựa chọn từ D10; kết quả D11 được ghi là **validation của lựa chọn có điều kiện**, không tái-tune `(r,d,F)` sau khi thấy main scores. Nếu có TEST/real defects held-out hợp lệ thì đánh giá **một lần sau lock** theo protocol; không dùng TEST để thay đổi kiến trúc.
- `G2 EXP PASS` chỉ khi đủ các ô bắt buộc và provenance hợp lệ. Nếu Colab ngắt, bảo toàn manifest/checkpoint để resume; thiếu dữ liệu/GPU thì báo `BLOCKED/NOT RUN`.

## 7. Notebook Colab và lưu artifact

| Notebook | Nhiệm vụ | Có GPU bắt buộc để hoàn thành code? |
|---|---|---|
| `notebooks/G2_01_E0_Baseline.ipynb` | D5 E0 Memory Bank 8 category | Không; GPU cần để nghiệm thu thực nghiệm E0 |
| `notebooks/G2_02_Adapter_Preflight.ipynb` | D6 9 cặp r/d, GPU VRAM/OOM smoke tùy tài nguyên | Không; CPU test 9 cặp là bắt buộc |
| `notebooks/G2_03_Training.ipynb` | D10 screening 360 ô + D11 main training theo stage | Không; nghiệm thu training cần GPU |
| `notebooks/G2_04_Evaluation.ipynb` | D6 evaluator E1/E2; D10/D11 aggregation và final eval | Không; nghiệm thu metric thật cần checkpoint/dữ liệu |

- Notebook **chỉ gọi** CLI/module hiện có; không duplicate mô hình/trainer/evaluator. Nếu đường dẫn/flag chưa tồn tại, Codex phải kiểm tra trước rồi bổ sung tối thiểu.
- Lưu outputs/Drive sau từng job (manifest, JSON, CSV, best/last checkpoints, resolved config, logs, seed, commit SHA, checksums), có resume/skip kiểm chứng hash; không đợi 360 jobs mới sao lưu.
- Smoke, capacity probe và fixture không được tính `EXP PASS`; GPU model có thể OOM dù fixture CPU PASS.

## 8. Kiểm thử, merge và tương thích ngược

1. TV1 chỉ sửa Model/Training/Runner/Config/Factory; TV2 chỉ sửa Data/Geometry/Eval/Benchmark và các Fusion được giao. Chỉ merge code qua `g2/integration` khi targeted tests PASS; TV2 D8 phải merge trước TV1 D9 integration.
2. Sau merge, chạy targeted tests hai nhánh và integration tests liên quan. **Không cần chờ GPU để merge code đã được CPU test phù hợp**, nhưng phải gắn `EXP NOT RUN` khi chưa có metrics.
3. Các entry points được nêu trong v2 như `build_g2_model`, `resolve_g2_config`, `trainable_modules`, `run_g2.py`, `predict_native_e4` vẫn là **API kỳ vọng theo code cũ**. Không tự tuyên bố chúng hỗ trợ E5, `joint_search` hay `--stage` mới nếu chưa được implement/test; bổ sung backward-compatible và tests theo task TV1.
4. Các checkpoint và `adapter_selection_lock.json` cũ (nếu có) phải giữ nguyên để audit, nhưng đánh dấu `legacy/E2_only`; **không dùng lock này để nghiệm thu E5 và không sửa lại metadata lịch sử**. Tách output_root hoặc RUN_ID khi đổi protocol để không skip nhầm.
5. Unit test chạy local CPU; preflight GPU và training thực chạy Colab. Ghi chính xác `PASS / FAIL / BLOCKED / NOT RUN` cùng bằng chứng. Bất kỳ thay đổi nào đến scientific settings, source data, checkpoint hoặc metric đều phải cập nhật config/protocol hash và không được reuse kết quả không tương thích.

---

**Điều kiện quyết định cuối G2:** Không được tuyên bố `(r*,d*,F*)` tốt nhất cho E5 chỉ vì E2/Mean cho AU-PRO cao. Quyết định phải dựa vào **trained E5 joint search trên cùng synthetic DEV**, bao quát các Fusion candidate hợp lệ trong phạm vi tìm kiếm đã công bố, sau đó khóa trước khi đánh giá giữ lại.
