# TV1-D1 — Architecture crosswalk và evidence MS-ILA

Ngày: **09/10/2026 (Asia/Ho_Chi_Minh)**. HEAD: `bab1ba74fa00c967f8f599276124e600ed832242` (`bab1ba7`).

**Kết luận:** các contract được kiểm thử bằng CPU/mock có **189 PASS / 18 SKIP / 0 FAIL / 0 ERROR**. E1 G1 là **PLANNED_NOT_IMPLEMENTED trong phạm vi audit / NOT_VERIFIED**, không phải `MSILA` baseline hay Day05 R0. Các gate DINO thật, dữ liệu thật, CUDA và checkpoint huấn luyện thật vẫn **BLOCKED**. Không train, không implement E1, không sửa model/config/optimizer/protocol, không chuyển sang D2.

## 1. Định nghĩa nghiên cứu và phạm vi

- Đường dẫn yêu cầu `docs/CODEX_PROJECT_CONTEXT.md` **không tồn tại**. File ngữ cảnh hiện có là [docs/MSILA_CODEX_CONTEXT_V2.md](../../docs/MSILA_CODEX_CONTEXT_V2.md), §2–3, đọc theo lý do cụ thể này. Định nghĩa trong file và prompt bổ sung được dùng làm nguồn nghiên cứu ngoài tên class/code; không tạo alias hoặc đổi tên file.
- **G1**: giai đoạn baseline trong lộ trình nghiên cứu, không Adapter/multi-layer/Context. **E1**: thí nghiệm đề xuất của G1, frozen DINOv3 ViT-S/16 → duy nhất deep feature → lightweight decoder; optimizer dự kiến chỉ cập nhật decoder. **G2**: giai đoạn mở rộng Adapter/multi-layer/Local–Context, không đồng nghĩa Day02/Day05 hay tên folder. Các định nghĩa này không chứng minh một run đã tồn tại.
- Đối chiếu source trong năm file model được giao, `configs/day05_representation.yaml`, `README_1.md` và các báo cáo Day05/Full-scale. Chỉ đọc thêm test liên quan và cấu hình pytest để hiểu/chạy gate; không đọc implementation trainer, không tìm toàn repo để suy ra alias E1.
- Trước ghi báo cáo đã chạy `git status --short` và `git rev-parse --short HEAD`: có `?? docs/MSILA_CODEX_CONTEXT_V2.md` do nhóm bổ sung; giữ nguyên. Model/config không thay đổi so với lần chạy test trong session, đã đối chiếu SHA256. Thay đổi của task chỉ là báo cáo này.

## 2. Crosswalk — ba hệ khác nhau

| Hệ | Mục đích | Source features | Adapter | Projection | Fusion | Decoder | Pretrained/frozen | Trạng thái có bằng chứng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **E1 G1 (planned)** | Baseline đề xuất, chỉ train decoder | Duy nhất tầng sâu nhất; ViT-S/16 là b12 `[B,384,32,32]` | Không | Không | Không có multi-source fusion | Lightweight segmentation decoder dự kiến | Pretrained frozen theo đặc tả | **PLANNED_NOT_IMPLEMENTED trong phạm vi audit; NOT_VERIFIED**. Không thấy assembly/forward tương đương trong các file được giao, không có artifact E1 được xác minh |
| **Repo Day1 baseline `MSILA`** | Architectural QA/control của repo | Một RGB view → b4, b8, b12 | Ba `ResidualAdapter2d` độc lập theo block | Không có projection downstream; `d` chỉ nằm trong adapter | Mean của ba feature đã adapt, giữ C | `BasicDecoder(C)` | `from_dinov3` dùng frozen extractor; generic constructor không tự freeze mọi extractor được truyền vào | **CODE_PRESENT**; hai test forward/freeze trực tiếp với DINO thật **SKIP**, chưa có PASS runtime cho toàn assembly này trong lần chạy |
| **Day05 R0 / deep_only** | Control một nguồn trong ablation representation | `local_b12` | Có, theo head khóa chung; b12 là nhánh active | Có, C → fusion_dim 64 | Mean một nguồn; identity đối với feature đã project | `BasicDecoder(64)` | Frozen ViT-S/16 theo YAML; mock freeze PASS | **CODE_PRESENT / CODE_TEST_PASS** trên CPU fixture; real asset **NOT_VERIFIED** |
| **Day05 R1 / multi_local** | Đo thêm tầng Local | `local_b4`, `local_b8`, `local_b12` | Có, ba block active | Có, C → 64; projector theo block | Mean ba nguồn | Cùng `BasicDecoder(64)` | Như R0 | **CODE_PRESENT / CODE_TEST_PASS** trên CPU fixture; real asset **NOT_VERIFIED** |
| **Day05 R2 / multi_local_context** | Đo thông tin Context sau alignment | Ba Local + `context_b4`, `context_b8`, `context_b12` đã về hệ Local | Có; harness Day05 dùng cùng adapter theo block cho Local/Context | Có, C → 64, `share_across_views: true` | Mean sáu nguồn | Cùng `BasicDecoder(64)` | Như R0 | **CODE_PRESENT / CODE_TEST_PASS** trên CPU fixture; alignment trên ảnh/checkpoint thật **NOT_VERIFIED** |

Bằng chứng source:

- [README_1.md](../../README_1.md):7 và :55 phân biệt architectural QA với gate DINO thật. [msila.py](../../src/models/msila.py):45, :137, :151, :219, :256, :290 có assembly ba adapter → MeanFusion → decoder C. Không dùng assembly này làm E1.
- [feature_selector.py](../../src/models/feature_selector.py):41, :60, :194 xác định chính xác thứ tự 1/3/6 nguồn. Selector chỉ chọn tensor đã chuẩn bị; không tự adapt/align/project/fuse/decode.
- [day05_representation.yaml](../../configs/day05_representation.yaml):39 khóa adapter; :48 khóa projection64/share_views; :56 khóa MeanFusion; :62 khóa decoder; :83 định nghĩa R0/R1/R2; :115 cấm attention, illumination loss và override theo candidate.
- Test Day05 `test_backward_routes_gradients_only_through_active_day05_path` chạy đủ R0/R1/R2 với adapter/projection thật và input surrogate frozen. Các node `test_head_uses_C_and_keeps_adapter_d_independent_of_fusion` import trực tiếp `Day05RepresentationModel` hiện có, kiểm tra forward/backward của assembly ở C=384/768/1024/1280. Đây là kiểm thử runtime với fixture; source trainer chứa class này không được đọc trong audit.

**R0 không phải E1:** chọn một nguồn không bỏ adapter/projection. `gamma=0` tạo identity lúc khởi tạo nhưng adapter vẫn có tham số trainable. Mean một nguồn không loại bỏ các layer đứng trước nó. R1 cũng không đồng nhất baseline repo: head Day05 project xuống 64, baseline repo giữ C.

`MSILADay2Head` và `MSILADay2Integrated` cũng đã có trong `msila.py`:337 và :587. Mặc định chúng nhận sáu feature projected, dùng **AttentionFusion**, rồi decoder; trace có attention `[B,6]`. Đây là head Day2 riêng, không phải E1 và không được bật mặc định thay MeanFusion của Day05.

## 3. GATE_ARCH — source contract và mức kiểm chứng

| Contract | Bằng chứng code/test | Kết quả và giới hạn |
| --- | --- | --- |
| BCHW, block IDs | Extractor :193, :367, :495; test mapping 4/8/12 → 3/7/11 và shape/API | **PASS mock**: mỗi block `[B,C,H/16,W/16]`; 512 → `[B,384,32,32]` cho ViT-S/16. Real feature gates **SKIP** |
| CLS/register stripping | Extractor :373 gọi `reshape=True`, `return_class_token=False`, `return_extra_tokens=False` | **PASS mock về arguments/output shape**. Code ủy quyền bỏ CLS/register cho API DINO, không tự slice token. Chưa xác minh patch/token layout bằng backbone pretrained thật |
| Frozen backbone | Extractor :261 `requires_grad_(False)` + `eval()`; :266 giữ backbone eval khi parent gọi train; :372 dùng `torch.no_grad()` | **PASS mock** frozen/eval và downstream backward. DINO thật **SKIP**. `MSILA(extractor=...)` nhận generic module nên không tự bảo đảm freeze; cần extractor đáp ứng contract |
| Features dùng được cho head trainable | `test_no_grad_features_can_feed_trainable_adapter`; `test_dinov3_extractor_enforces_frozen_eval_mode` | **PASS CPU**. `no_grad` không biến feature thành inference tensor; không có gradient về backbone/input qua extractor |
| Local/Context extraction | Extractor :342, :400; concat/sequential tests | **PASS mock**: cùng shape/device/dtype; concat batch `[2B,3,512,512]` rồi split về B; sequential hai call. Cùng network size không chứng minh cùng FOV hoặc alignment |
| Adapter/identity | ResidualAdapter :164, :200, :242, :267; identity/screening tests | **PASS CPU**: `C→r→DWConv→GELU→d→GELU→C`, `F_out=F+gamma*delta`, giữ BCHW; `gamma=0` exact identity trên tensor finite; out_proj không bị zero cùng gamma |
| Adapter gradients | Screening :160/:176; gradient-flow harness; Day05 active-path tests | **PASS CPU**: gamma nhận gradient finite; khi gate khác 0 mọi branch parameter được kiểm tra gradient finite. Tại gamma=0 branch grad có thể bằng 0; `grad=None` của nhánh active mới là đứt graph, nhánh inactive R0 được kỳ vọng `grad=None` |
| Head ownership/gradient | Day05 ownership :680 và backward :839; projection tests; TV1→TV2 integrated backward | **PASS CPU fixture**: adapter/projector/decoder trainable; selector/aligner/MeanFusion không có tham số; source được chọn có gradient finite. Real-backbone ownership **SKIP** |
| Decoder/logits | BasicDecoder :29/:57; Day2 shape/wiring, multiview tests và Day05/full-scale fixtures | **PASS CPU**: Conv3×3→GELU→Conv1×1→bilinear → `[B,1,H,W]`; trả logits, không sigmoid trong decoder. Baseline `MSILA` real forward **SKIP** |
| Deterministic resize | BasicDecoder :69; `test_bilinear_sampling.py` | **23 PASS CPU**, forward/input-gradient parity với PyTorch. Hai CUDA FP32/BF16 cases **SKIP**; không có GPU determinism PASS |
| Checkpoint identity / physical block slots | Extractor :168/:188; ba nhóm node full-scale đã chọn | **PASS mock/fixture**: local checkpoint đọc trực tiếp strict state_dict, hai file cùng basename không alias; registry architecture mismatch bị chặn. Cache b4/b8/b12 là shallow/middle/deep slots khi physical blocks đổi. Không chứng minh weights pretrained thật |

Các giới hạn cần giữ rõ:

1. `FeatureSelector` :217 kiểm tra BCHW, shape/dtype/device tương thích. Shape bằng nhau **không chứng minh Context đã warp đúng**; cần geometry và gate ảnh thật. Fixture R2 backward Day05 dùng identity geometry để test autograd, không đại diện Context FOV 768 thật.
2. YAML chỉ ghi `adapter.source: day04_locked_selected_candidate`, **không chứa số r,d**. R32/d384 là giá trị được ghi nhận cho Fabric trong báo cáo, không phải default khoa học để tự áp mọi run. `FeatureSelector.from_config` chỉ đọc mode/sources, không thực thi toàn bộ scientific lock hay kế thừa base config.
3. `MSILA` baseline hiện gắn theo ba key b4/b8/b12; extractor có hỗ trợ physical blocks theo backbone rộng hơn. Không suy từ extractor tests rằng toàn baseline legacy hỗ trợ mọi bộ blocks full-scale.
4. Hai test Day05 `test_real_batch_representation_gate_file_exists` và `test_trainable_parameter_report_gate_file_exists` **PASS chỉ về sự tồn tại/cấu trúc test gate**, không chứng minh JSON real-data đã được sinh hay acceptance đã PASS.
5. **API drift tĩnh trong gate bị SKIP:** `tests/test_full_forward.py`:47 vẫn gọi `adapter_reduction=4`, trong khi `MSILA.from_dinov3` :178 yêu cầu `adapter_bottleneck_dim` và `adapter_projection_dim`, không nhận keyword cũ. Lần chạy đã SKIP ở asset check trước constructor; không có test đỏ runtime tái hiện trong session, nên không sửa. Cần regression nhỏ và sửa fixture API trước khi dùng gate này để tuyên bố baseline PASS; không đổi model hoặc tự chọn lại r,d.

## 4. Kết quả targeted tests đã chạy — không chạy lại

Môi trường: `.venv/qa/bin/python`, Python 3.12.3, torch **2.6.0+cpu**, pytest 7.4.4, NumPy 1.26.4, SciPy 1.11.4, PyYAML 6.0.1. `torch.cuda.is_available()=False`, CUDA version `None`, device count 0.

Lệnh thực tế đã chạy trước khi bổ sung ngữ cảnh; code không đổi nên giữ evidence này:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/msila-tv1-d1-mpl PYTHONPATH=. .venv/qa/bin/python -m pytest -q \
  tests/test_dinov3_extractor.py \
  tests/test_dino_features.py \
  tests/test_residual_adapter_identity.py \
  tests/test_residual_adapter_screening.py \
  tests/test_day05.py \
  tests/test_representation_shapes.py \
  tests/test_trainable_parameters.py \
  tests/test_gradient_flow.py \
  tests/test_msila_day02_head.py \
  tests/test_tv1_integration.py \
  tests/test_full_forward.py \
  tests/test_day02_contracts.py \
  tests/test_multiview_forward.py \
  tests/test_feature_projection.py \
  tests/test_context_alignment.py \
  tests/test_bilinear_sampling.py \
  tests/test_feature_cache.py::test_real_built_cache_matches_fresh_online_extraction \
  tests/test_full_scale_contracts.py::test_loaded_architecture_and_physical_block_cache_mapping \
  tests/test_full_scale_contracts.py::test_exact_local_weights_are_loaded_even_with_identical_basenames \
  tests/test_full_scale_contracts.py::test_head_uses_C_and_keeps_adapter_d_independent_of_fusion \
  --tb=short -rs --junitxml=/tmp/msila-tv1-d1-architecture.xml
```

Exit code **0**; stdout: **`189 passed, 18 skipped in 8.07s`**. JUnit: 207 cases, time 8.071s, timestamp `2026-10-09T12:23:18.584467` (XML không gắn timezone), failures=0, errors=0. XML session nằm ở `/tmp/msila-tv1-d1-architecture.xml`, SHA256 `4f128f20a8d4f0001e9e812089dc7e9815c815bc33baa1a31b0fcb278eeed06d`.

| Test module / phần đã chọn | PASS | SKIP |
| --- | ---: | ---: |
| test_dinov3_extractor | 12 | 1 |
| test_dino_features | 0 | 9 |
| test_residual_adapter_identity | 4 | 0 |
| test_residual_adapter_screening | 24 | 0 |
| test_day05 | 57 | 1 |
| test_representation_shapes | 0 | 1 |
| test_trainable_parameters | 0 | 1 |
| test_gradient_flow | 6 | 0 |
| test_msila_day02_head | 3 | 0 |
| test_tv1_integration | 3 | 0 |
| test_full_forward | 0 | 2 |
| test_day02_contracts | 12 | 0 |
| test_multiview_forward | 13 | 0 |
| test_feature_projection | 9 | 0 |
| test_context_alignment | 5 | 0 |
| test_bilinear_sampling | 23 | 2 |
| test_feature_cache: chỉ real cache-vs-online node | 0 | 1 |
| test_full_scale_contracts: chỉ 3 nhóm node kiến trúc nêu trên | 18 | 0 |
| **Tổng** | **189** | **18** |

18 SKIP được phân biệt theo nguyên nhân thực tế:

- **10**: extractor real smoke (1) và toàn bộ real DINO features (9), thiếu DINO source/checkpoint.
- **2**: `test_full_forward` và `test_parent_train_keeps_dino_frozen`, không có `/content/dinov3`.
- **1**: real-backbone Day05 gradient ownership, không có DINO source.
- **1**: real cache-vs-online, chưa đặt `DINOV3_REPO` / `DINOV3_WEIGHTS`.
- **1**: representation trên ảnh TRAIN thật, thiếu root MVTec.
- **3**: CUDA device mismatch (1) và strict deterministic backward FP32/BF16 (2), không có CUDA.

Không chạy suite `test_full_scale_training.py`, các test pipeline có vòng train theo epoch, trainer CLI, native inference thật hoặc full-scale grid. Kết quả trên là software/fixture evidence, không phải accuracy hay real-data/GPU PASS.

## 5. Day05 notebook evidence và artifact thật

[day05_tv2_handoff_qa.json](../day05_tv2_handoff_qa.json) ghi nhận từ notebook: **fabric / seed42 / dinov3_vits16 / r32,d384**, ba run R0/R1/R2, **150 epoch và 11.550 optimizer updates/run**, best epoch **28 / 21 / 21**. Ngữ cảnh §3C mô tả selection bằng **minimum val_loss**, `val_total_loss` là alias.

Trạng thái: **RECORDED_NOT_INDEPENDENTLY_VERIFIED**. Chính JSON đặt `drive_artifacts_read_in_this_workspace=false` và `native_real_inference_executed_in_this_workspace=false`. Audit này không mở notebook nguồn ngoài phạm vi, không đọc `best.pt` Drive, không strict-load checkpoint huấn luyện thật, không kiểm log/budget/hash độc lập. Không kết luận notebook đã bịa hoặc chưa train; cũng không nâng thông tin đã ghi nhận thành TRAIN_ARTIFACT_VERIFIED.

Các snapshot cũ không phải kết quả mới: [day05_acceptance.json](../day05_acceptance.json) **634 PASS / 18 SKIP**, real-data/full-training/full-evaluation BLOCKED; handoff QA **643 PASS / 18 SKIP**; [full_scale_validation.json](../full_scale_validation.json) **707 PASS / 20 SKIP**, `scientific_experimental_status=UNVERIFIED`. Không cộng/trộn các số này với 189/18 của task.

Full-scale v2 là protocol riêng: 3.240 jobs chỉ là grid khai báo; rule `0.5*DEV_tiny_AU-PRO005 + 0.5*DEV_mixed_AU-PRO005` không được áp ngược lên checkpoint Day05 Fabric chọn theo min val_loss. Audit không đổi hay xác minh lại các rule này.

## 6. Blocker, file đã có và phần cần làm tiếp

| Gate | Trạng thái hiện tại | Asset/đầu việc cụ thể còn cần |
| --- | --- | --- |
| GATE_ARCH E1 | **PLANNED_NOT_IMPLEMENTED / NOT_VERIFIED** trong phạm vi audit | Chưa có forward/training/artifact tương đương được chỉ ra. Nếu sau này muốn build E1 phải có task build riêng; TV1-D1 không implement |
| GATE_ARCH software hiện có | **CODE_TEST_PASS** cho các contract/assembly fixture đã nêu; toàn baseline MSILA chưa runtime PASS | Xử lý API drift của fixture full-forward bằng test đỏ nhỏ trước khi mở real gate |
| GATE_REAL_DINO | **BLOCKED_MISSING_SOURCE_CHECKPOINT; tests SKIP** | `DINOV3_REPO=/path/to/official/dinov3` chứa `hubconf.py`; weights ViT-S/16 pretrained thật, SHA và source revision. Đặt cả `DINOV3_WEIGHTS` và `DINOV3_CHECKPOINT` trỏ cùng checkpoint vì các test dùng tên env khác nhau |
| GATE_DATA / real representation | **BLOCKED_MISSING_DATA; test SKIP** | `MSILA_MVTEC_AD2_ROOT=/path/to/mvtec_ad_2`, category mặc định fabric, có ảnh TRAIN thật đủ FOV 768; raw image/GT/cache/records và source identities/hashes khớp. `FEATURE_CACHE_DIR`/`FEATURE_TRAIN_INDEX` chưa cấu hình; không chạy real-cache training gate ở task này |
| Real adapter ownership | **BLOCKED; test SKIP** | DINO thật cùng numeric Day04 lock đã xác minh; YAML thiếu r,d nên gate ownership cần `MSILA_DAY04_ADAPTER_R` / `MSILA_DAY04_ADAPTER_D`. Không tự điền để biến SKIP thành PASS |
| GPU/AMP/determinism/resources | **BLOCKED_MISSING_GPU; 3 tests SKIP** | CUDA GPU; BF16-capable GPU cho BF16 gate; kiểm strict deterministic backward và VRAM/runtime đúng batch/resolution/protocol. CPU parity không thay được gate GPU |
| GATE_TRAIN_ARTIFACT Day05 | **BLOCKED_MISSING_CHECKPOINT** tại môi trường audit | Ba `best.pt` R0/R1/R2 thật từ thư mục run Fabric trên Drive, resolved config/manifest, logs/selection record, SHA256 và provenance. Đường dẫn Drive chính xác chưa có trong các file được đọc; cần nhóm cung cấp, không đoán đường dẫn hoặc train lại |
| GATE_NATIVE_INFER / GATE_METRIC / GATE_SELECTION | **NOT_VERIFIED / BLOCKED**, nằm ngoài D1 | Checkpoint verified, ảnh/GT native, tiled Hann maps/provenance, DEV source-disjoint và tiny/boundary locks thật; skeleton không thay scientific lock. Chưa có căn cứ chọn representation/threshold |

File có sẵn có thể tái sử dụng: năm model được audit, Day05 YAML, `README_1.md`, ba JSON evidence, các test ở §4. Các module projection/alignment/MeanFusion/Day05RepresentationModel đã được test import và sử dụng; không cần sinh module/class trùng. Việc cần tiếp theo là đóng các gate asset/contract trong scope được giao, không tự chạy D2 hoặc grid.

## 7. Contract TV2 cần giữ khi handoff

1. Phân biệt `E1 planned`, repo Day1 baseline và Day05 R0/R1/R2 trong tên run/manifest. Không chuyển R0 thành E1 bằng rename; không áp protocol full-scale vào Day05 Fabric.
2. Local `[B,3,512,512]`; R2 có **Context crop độc lập FOV 768→network512**, geometry metadata đúng, warp về frame Local. Không copy Local làm Context. Mỗi selected feature sau projection có BCHW `[B,64,32,32]` cho lock ViT-S/16; thứ tự canonical R0/R1/R2 ở §2.
3. Head Day05 khóa Adapter/Projection + **MeanFusion** + BasicDecoder; logits `[B,1,H,W]`. Dùng sigmoid ở boundary score; giữ semantics score continuous, native geometry và Hann stitching theo protocol. Tensor cùng shape không đủ chứng minh alignment đúng.
4. Handoff checkpoint kèm candidate/seed/category, backbone/physical blocks/checkpoint SHA, adapter r,d, fusion_dim, config/protocol/selection rule và log/budget. Strict-load đúng assembly; receipt/notebook record chưa thay kiểm checkpoint thật.
5. TRAIN/DEV source-disjoint; selection và threshold chỉ từ DEV theo lock. Không dùng TEST để chọn model/threshold hoặc chỉnh protocol. Chỉ chuyển scientific status sang PASS khi gate thật tương ứng đã có evidence.

## 8. Fingerprint source giữ nguyên

| File | SHA256 lúc chạy test và khi chốt báo cáo |
| --- | --- |
| src/models/dinov3_extractor.py | `f2c394541090fa0d3c15161dd5ed719b7d77f9111cc838ef2f3b9127e097ebbd` |
| src/models/residual_adapter.py | `6374d93f0cfe30e862e24ae72d0f8391ff5a8c576defad825856dc07b8e078b7` |
| src/models/basic_decoder.py | `105526ad0f76af105d1bcdf5b4674d2c47f2ecc31af710dab85514f6649f7dc6` |
| src/models/msila.py | `d202cae962cf00bfb15c75d8f60e94a4b002f54176376a350354c24695055614` |
| src/models/feature_selector.py | `2fb812b5f7ec655d860112d39f745bbcd7e828980a7ecf2380068df9ae52c91e` |
| configs/day05_representation.yaml | `3cf7584e9551ce59a999a1cc1ac151f2a58eebadbe99cfa1d147b1d8cbe2d7ec` |

Ngữ cảnh nhóm `docs/MSILA_CODEX_CONTEXT_V2.md` SHA256 `163896613637992b08f681b583227a25537a8d27829474302cdb41ea18afae8c`, được giữ nguyên.
