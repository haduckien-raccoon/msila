# TV2-D1 — Dataset, GT và source-leakage audit

Ngày: **09/10/2026 (Asia/Ho_Chi_Minh)**. HEAD: **`bab1ba7`**.

**Kết quả:** targeted tests hiện có **81 PASS / 0 SKIP / 1 DESELECTED**. Phát hiện bug padding mask, thêm regression **1 FAIL trước sửa → PASS sau sửa**; kiểm tra lại các đường bị ảnh hưởng **10 PASS / 0 FAIL**. Real cache-vs-online giữ evidence **1 SKIP từ TV1-D1**, không chạy lại. Dataset/GT/cache/checkpoint thật tại các đường dẫn được cấu hình/tài liệu hóa **BLOCKED**, không có real-data/GPU PASS.

Chỉ sửa bug trong `crop_with_padding` và thêm một regression; không làm lại loader, không sửa model/training/generator/cache format, không đổi splits/protocol, không train hay tự chạy D2.

## 1. Nguồn định nghĩa và worktree

`docs/CODEX_PROJECT_CONTEXT.md` vẫn **MISSING**; dùng [docs/MSILA_CODEX_CONTEXT_V2.md](../../docs/MSILA_CODEX_CONTEXT_V2.md) §1–4 đã đọc trong session. **G1/E1 là kế hoạch nghiên cứu**, TRAIN good → synthetic mask → DEV synthetic/normal source-disjoint; không đồng nhất với cache Day05 hoặc chữ G1 PASS của một decision log cũ.

Đã xác nhận tồn tại trước khi đọc toàn bộ file được giao: bốn module data, hai audit scripts, `data_card.md`, `G1_decision_log.md`, `DAY05_TV2_HANDOFF.md`, `FULL_SCALE_TRAINING.md`, `full_scale_synthetic.yaml` và các tests liên quan. Không mở thêm source model/training hoặc builder/inference; một số test import trực tiếp implementation đó để kiểm boundary bằng fixture.

Trước sửa đã kiểm `git status --short` / `git rev-parse --short HEAD`: có `?? docs/MSILA_CODEX_CONTEXT_V2.md` và `?? reports/recheck/`. Giữ nguyên ngữ cảnh nhóm và báo cáo TV1-D1; không reset/clean/commit/push.

## 2. Dataset contract và implementation đang có

| Contract | Implementation / evidence | Kết luận |
| --- | --- | --- |
| Image ↔ GT identity | [loader.py](../../src/data/loader.py):93 `_layout_identity`, :108 `scan_mvtec_ad2`: `(category, canonical split, defect, nested directories + canonical stem)` | **CODE_PRESENT / CPU fixture PASS**: không fallback theo basename xuyên split/defect; camera/nested-ID được giữ. Hai layout GT được test: `category/test_public/ground_truth/bad/ID_mask.png` và `category/ground_truth/test_public/bad/ID.png` |
| Normal GT=0 | Loader :89, :135, :278; normal aliases `good/normal/ok` | **PASS fixture**: mask `uint8 [H,W]` toàn 0 theo label normal, không cần file GT |
| Abnormal thiếu/mơ hồ GT | Loader :136/:140; Dataset mặc định `require_pixel_gt=True` :246 | **PASS fixture** cho missing/ambiguous rejection. `require_pixel_gt=False` cho phép inventory/score-only với `mask=None`, không âm thầm tạo zero mask cho abnormal. GT abnormal rỗng bị Dataset từ chối tại :281; nhánh empty-GT này được đối chiếu source, chưa có test riêng đã chạy |
| Native resolution/channel/range | Loader :176/:206/:272; loader tests | **PASS fixture**: raw `image=float32 [3,H,W]` trong `[0,1]`, `image_norm` cùng H×W; grayscale được replicate thành RGB ngay trong loader; uint8/uint16 scale theo dtype. Không resize toàn ảnh thành hình vuông |
| Mask nearest | Loader :219; tiling :306/:337; mask tests | **PASS fixture**: mask load/resize nearest → binary; local mask `[512,512]`; stitch mask bằng max/OR, không dùng Hann cho label |
| Tile coverage | [tiling.py](../../src/data/tiling.py):19/:40; coverage/reconstruction tests | **PASS fixture**: Local512, overlap128, stride chuẩn384; thêm start cuối `length-512` khi cần, nên bước cuối có thể nhỏ hơn384. 100% coverage ở các resolution được test, gồm nhỏ hơn512 và không chia hết stride |
| Padding | Tiling :81; regression ở §5 | Image reflect fallback replicate khi reflect không hợp lệ; mask constant0 phải giữ 0 ngoài ảnh. **Bug đã sửa và targeted PASS** |
| Context FOV/metadata | Tiling :40/:141; multiview-transform/view-meta/R2 independent-Context tests | **PASS fixture**: box Context768 cùng tâm Local512; network Context512; Local/Context không được copy nhau. Metadata roundtrip/crop-resize matrix được test qua module hiện có; `TileRecord` riêng chỉ chứa boxes/center, không tự chứa đầy đủ transform/native metadata |
| Hann score reconstruction | Tiling :203/:219; tiling/ramp/native-mask tests | **PASS fixture**: weighted average, Hann clamp tối thiểu1e-3, assert mọi pixel có weight>0; map được cắt về native H×W; max reconstruction error được test <1e-5 |
| Legacy synthetic image + mask | [synthetic_anomaly.py](../../src/data/synthetic_anomaly.py):81/:189/:524 | **6 PASS fixture**: raw RGB `[3,H,W]` trước normalize → cùng shape/range, mask float32 `[1,H,W]` binary; explicit seed tái lập image/mask/metadata, background ngoài GT bất biến, force-normal trả mask0 |
| Native full-scale proxy | Synthetic :771/:815; [full_scale_synthetic.yaml](../../configs/full_scale_synthetic.yaml); full-scale native-defect tests | **8 PASS fixture** cho bốn type × hai placement, mỗi case kiểm các bins, determinism/support/visibility. Không chứng minh phân bố lỗi công nghiệp thật |
| Cache format/integrity | [feature_cache.py](../../src/data/feature_cache.py):232/:347/:447/:540/:592; cache/cached-dataset tests | **PASS fixture**: sáu nguồn, geometry, category/image-ID; producer signature/hash, duplicate rejection, shards/roundtrip, finite tensors và downstream mask-shape guard. Real cache-vs-online **SKIP**, không chuyển thành PASS |

Các contract có thể tái sử dụng trực tiếp: `scan_mvtec_ad2`, `MVTecAD2HighResDataset`, `audit_mvtec_ad2`, `generate_tile_records`, `crop_mask_tiles`, `stitch_mask_tiles`, `stitch_tiles_hann`, hai synthetic generators, `FeatureCacheWriter/Reader` và hai audit CLI. Không cần sinh loader/generator mới.

## 3. Không trộn ba protocol dữ liệu

| Hệ | TRAIN/DEV và generator | Đơn vị/GT và điều kiện dùng |
| --- | --- | --- |
| **G1/E1 dự kiến** | TRAIN good; synthetic supervision và DEV synthetic + normal tách nguồn TRAIN. Chưa có split ratio/generator manifest thật được xác minh trong task | Đây là dataset contract dự kiến, không phải artifact cache Day05. Không tự áp generator/bins/tỷ lệ full-scale để "hoàn thiện" G1 |
| **Day05 Fabric replay** | R0/R1/R2 phải dùng cùng source plan, records, generator/config/seed và masks/cache của Day04 đã train. `prepare_sample` test xác nhận cutpaste trước extraction, Local/native mask tương thích | Native DEV export/replay phải so mask Local và cả sáu feature với cache cũ. Không tạo DEV mới để evaluate checkpoint cũ. Test replay phát hiện thay mask; đây vẫn là mock backbone/fixture, chưa replay artifact Drive thật |
| **Full-scale v2 native proxy** | Theo [FULL_SCALE_TRAINING.md](../../docs/FULL_SCALE_TRAINING.md):48, hash với **data seed42**, khoảng80% TRAIN-core /10% DEV-tiny /10% DEV-mixed từ official TRAIN normal, ba nguồn disjoint. Hai source-plan tests kiểm deterministic/disjoint/không lấy TEST bằng fixture | Synthesize ở native trước tiling; giữ mọi Local tile kể cả negative. Bins foreground native pixels4–32 /33–128 /129–2048; DEV tiny16 strata, DEV mixed24 strata mỗi nguồn, thêm normal. Proxy declared trước evaluation, không fit theo TEST. Không relabel cache legacy thành full-scale |

Legacy `SyntheticAnomalyConfig` defaults ratio0.002–0.08 điều khiển **bbox được sample**; ellipse/polygon và clipping/crop có thể làm diện tích foreground thực khác. Metadata `area_px/area_ratio` đo mask thực, không phải distribution connected components của DEV cũ. Full-scale bins là **foreground native area**, không phải bbox/crop ratio. Chưa có historical DEV GT thật để đối chiếu phân bố.

Day05 checkpoint chọn bằng **minimum val_loss/val_total_loss**; full-scale chọn bằng rule DEV-tiny/mixed AU-PRO riêng. Không dùng TEST chọn model, threshold, bins hoặc protocol. Source-disjoint phải kiểm ở **raw source identity**, không chỉ synthetic image-ID/tile-ID; cùng ảnh nguồn sinh nhiều seed/tile vẫn chỉ là một source.

## 4. Risk và giới hạn audit

1. **Không suy completeness từ scanner PASS.** Loader bỏ qua path không có một split hợp lệ; GT không scoped theo split không được guess. `audit_mvtec_ad2` không đối chiếu inventory mọi file, orphan GT, tám category hoặc expected dataset count. D2 cần manifest SHA256 và count/ignored-layout audit thật. Data card ghi8004 ảnh và integrity `PASS / FAIL` là thông tin tài liệu/placeholder, chưa được xác minh ở workspace.
2. **Private GT hợp lệ có thể hidden.** Dataset pixel-mode phải chặn abnormal không GT; inventory cho phép `mask=None`. Audit script hiện cộng missing/ambiguous abnormal vào errors cả các split private, nên raw audit FAIL không tự chứng minh private dataset hỏng. Tách việc kiểm inventory/score export với pixel-evaluation eligibility khi diễn giải; không tạo zero GT cho abnormal private.
3. **Docs cũ khác contract hiện tại.** [data_card.md](../../docs/data_card.md):65 ghi không tạo additional split nhưng :169 lại có TRAIN-core/DEV-synthetic; full-scale định nghĩa partition TRAIN-derived riêng. Data card :156 nói grayscale chỉ replicate trước DINO, trong khi loader đã trả RGB. [G1_decision_log.md](../../docs/G1_decision_log.md) ghi G1 data/feature PASS và hook/block choices lịch sử; không đủ chứng minh E1 G1 đã train, cache Day05 hiện tại hợp lệ, hoặc split rule hiện tại đã được kiểm bằng data thật. Giữ nguyên docs, ghi provenance thay vì đổi splits.
4. **Resize kernel phải lấy từ producer thật.** Utility `tiling.extract_local_context` dùng **bilinear**, trong khi data card ghi bicubic. Shape/FOV tests không chứng minh preprocessing equality với Day04. Replay cần đúng kernel/antialias/normalization/padding trong persisted config/signature; không tự đổi utility thành bicubic trong audit.
5. **Cache schema không tự chứng minh leakage hoặc GT.** Cache v1 chỉ lưu category/image-ID, geometry và sáu feature; `normalize_record` bỏ extra top-level `split/meta/mask/source_identity`. `sample_key` hash category+image-ID, không có split riêng. Cần records/masks/source manifest ngoài cache, hoặc source fields được giữ đúng trong geometry/signature. Storage chỉ kiểm tensor float/nonempty/finite, không tự enforce mọi BCHW/shape, matrix đúng hình học hay source split. Cùng category/image-ID phải unique qua các tile/role.
6. **Provenance hash phải có kỳ vọng độc lập.** Reader chỉ đối chiếu expected producer khi caller truyền signature; hash bằng nhau không tự chứng minh bộ source sạch. Full-scale schema `msila.full_scale.cache.v2` có shard SHA; legacy không được giả định có historical mask/shard checksums. Thiếu checksum lịch sử không thể chữa bằng receipt SHA hiện tại.
7. **Source-disjoint PASS hiện chỉ trên fixture.** Source-plan tests kiểm relative source identities/partition; chưa đối chiếu bytes/hash/copies/symlinks trên data thật. Test `test_source_leakage_is_rejected` của Day05 phụ thuộc fixture chạy trainer, nên không chạy trong audit này. Không tuyên bố đã runtime kiểm hết validator của training bằng 81 tests.
8. **GT-only script không đóng tất cả scientific gates.** [audit_synthetic_gt.py](../../scripts/audit_synthetic_gt.py):12 đọc list hoặc `samples`, mask path, connected components/native tiles; absent GT chỉ được bỏ qua khi top-level `is_anomaly=false`. Script không tự enforce DEV-only/source-disjoint/hash/native-image equality, không tự khóa bins và không từ chối riêng abnormal GT toàn0. Caller phải cung cấp manifest DEV đã kiểm; không truyền TEST để chọn protocol. Không inspect predictions trong script (`metrics_inspected=false`).

## 5. Bug có regression — giữ padding mask bằng 0

**Trigger:** ảnh/mask2×3, foreground duy nhất ở góc dưới phải, crop Local512. `crop_with_padding(..., pad_mode='constant', pad_value=0)` trước sửa chuyển sang replicate khi padding lớn dù caller không yêu cầu reflect.

Probe CPU trong memory, **không phải dữ liệu thật**: native foreground1; padded foreground ngoài native **260609**; tile foreground tổng260610. Native roundtrip vẫn đúng vì stitch cắt padding, nên ba mask roundtrip tests cũ không phát hiện false supervision ở padding.

- Regression [tests/test_masks.py](../../tests/test_masks.py):84 `test_small_image_mask_padding_stays_zero` kiểm native crop giữ nguyên, tổng foreground tile bằng native và reconstruction đúng.
- **RED:** `assert 260610 == 1`, exit1, một test FAIL trước sửa.
- Sửa [src/data/tiling.py](../../src/data/tiling.py):113: fallback replicate chỉ khi **`effective_mode == 'reflect'`** và reflect không hợp lệ. Constant pad_value được giữ; không đổi default image reflect, resolution, split, generator hoặc protocol.
- **GREEN:** 10 targeted tests liên quan PASS, gồm regression, coverage/Context/Hann, mask stitch, native tiling và Day04 replay. Không rebuild hoặc sửa mask/cache đã tồn tại. Nếu artifact cũ có ảnh đủ nhỏ để kích hoạt lỗi, D2 cần kiểm padding/GT của artifact đó trước dùng; chưa có assets để đo phạm vi ảnh hưởng thật.

## 6. Test evidence và lệnh tái lập

Môi trường: Python3.12.3, `.venv/qa/bin/python`, torch2.6.0+cpu, pytest7.4.4, NumPy1.26.4, SciPy1.11.4; CUDA=False. Existing fixtures được tạo trong pytest tmp để kiểm software, không được gọi dataset/GT/DINO thật. Hai audit CLI `--help` exit0; không chạy real dataset/GT audit vì thiếu assets.

Lần targeted trước sửa:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/msila-tv2-d1-mpl PYTHONPATH=. .venv/qa/bin/python -m pytest -q \
  tests/test_loader.py tests/test_loader_identity.py \
  tests/test_tiling.py tests/test_masks.py tests/test_synthetic_anomaly.py \
  tests/test_feature_cache.py tests/test_cached_dataset.py \
  tests/test_multiview_transform.py tests/test_view_meta.py \
  tests/test_full_scale_contracts.py::test_native_defects_exact_support_determinism_coverage \
  tests/test_full_scale_contracts.py::test_mask_survives_native_tiling_padding_and_Hann_stitching \
  tests/test_full_scale_contracts.py::test_train_DEV_sources_disjoint_and_no_TEST_source_is_used \
  tests/test_day05_pipeline.py::test_backbone_metadata_has_no_filename_or_substring_bypass \
  tests/test_day05_pipeline.py::test_r2_tile_has_independent_context_and_true_alignment \
  tests/test_day05_pipeline.py::test_stitching_preserves_pixel_coordinates \
  tests/test_day05_pipeline.py::test_day04_source_plan_is_deterministic_and_excludes_test \
  tests/test_day05_pipeline.py::test_synthesis_precedes_extraction_and_native_masks_match \
  tests/test_day05_pipeline.py::test_builder_reuses_shards_and_verifies_day04_replay \
  tests/test_day05_pipeline.py::test_source_plan_accepts_official_archive_wrapper \
  --deselect=tests/test_feature_cache.py::test_real_built_cache_matches_fresh_online_extraction \
  --tb=short -rs --junitxml=/tmp/msila-tv2-d1-data.xml
```

**81 PASS / 0 SKIP / 1 DESELECTED**, exit0,7.66s. Command được viết gộp các file độc lập trên cùng dòng cho dễ tái lập; collection giống command đã chạy. Chạy lại command trên code hiện tại sẽ thêm regression mới; số81 là evidence trước thay đổi.

| Existing tests đã chọn | PASS |
| --- | ---: |
| test_loader / test_loader_identity | 4 / 3 |
| test_tiling / test_masks | 3 / 3 |
| test_synthetic_anomaly | 6 |
| test_feature_cache, trừ real node | 8 |
| test_cached_dataset | 7 |
| test_multiview_transform / test_view_meta | 12 / 13 |
| full-scale: native defect/support, mask tiling, source-plan | 10 |
| Day05: metadata, Context, coordinates, source-plan, synthesis, replay, archive wrapper | 12 |
| **Tổng** | **81** |

RED command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/qa/bin/python -m pytest -q tests/test_masks.py::test_small_image_mask_padding_stays_zero --tb=short --junitxml=/tmp/msila-tv2-d1-padding-red.xml` → **1 FAIL**,2.03s trước sửa.

Sau sửa, chỉ chạy lại đường bị ảnh hưởng:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/msila-tv2-d1-mpl PYTHONPATH=. .venv/qa/bin/python -m pytest -q \
  tests/test_tiling.py tests/test_masks.py \
  tests/test_full_scale_contracts.py::test_mask_survives_native_tiling_padding_and_Hann_stitching \
  tests/test_day05_pipeline.py::test_synthesis_precedes_extraction_and_native_masks_match \
  tests/test_day05_pipeline.py::test_builder_reuses_shards_and_verifies_day04_replay \
  --tb=short -rs --junitxml=/tmp/msila-tv2-d1-padding-green.xml
```

**10 PASS / 0 SKIP / 0 FAIL**, exit0,5.83s; gồm9 test đã có và1 regression mới. Không cộng81+10 thành91 test độc lập.

Real `test_feature_cache.py::test_real_built_cache_matches_fresh_online_extraction`: **SKIP tái sử dụng từ TV1-D1**, lý do thiếu `DINOV3_REPO`/`DINOV3_WEIGHTS`; XML `/tmp/msila-tv1-d1-architecture.xml`. Deselect ở lần này là quyết định không chạy lại, không phải SKIP mới và không phải real PASS. Gate còn cần `REAL_SAMPLE_PT`.

JUnit SHA256:

| Evidence session | SHA256 |
| --- | --- |
| /tmp/msila-tv2-d1-data.xml | `7cc48eadda39d3500d060df6de3b03ddfcf5a46afaa85c839cb216cced3fa87d` |
| /tmp/msila-tv2-d1-padding-red.xml | `6ebac8d59a3a5e9c1a8b4939afa98d085a2eb00139811ee9cbf36e809f3a025b` |
| /tmp/msila-tv2-d1-padding-green.xml | `280a67463d9f64bae8122e4072c4901cc3604ae68d11c069aaba31629cdb7171` |

## 7. BLOCKED — exact paths/manifests cần cho D2 và TV1

Kiểm read-only hiện tại: env `MSILA_MVTEC_AD2_ROOT`, `FEATURE_CACHE_DIR`, `FEATURE_TRAIN_INDEX`, `DINOV3_REPO`, `DINOV3_WEIGHTS`, `DINOV3_CHECKPOINT`, `REAL_SAMPLE_PT` đều unset. Các path được tài liệu hóa bên dưới không có trong workspace; không tìm toàn máy/Drive để đoán root.

| Gate / owner nhận | Path hoặc artifact cần cung cấp | Trạng thái / điều kiện |
| --- | --- | --- |
| GATE_DATA inventory — TV2 D2 | Root thật trực tiếp chứa category folders. Data card: `/content/data/mvtec_ad2/`; full-scale guide: `/content/drive/MyDrive/msila/data/MVTec_AD_2/`; hoặc root explicit của nhóm | **BLOCKED_MISSING_DATA**; cả hai documented roots và `data/` local đều MISSING. Root Colab ghi trong docs không phải chứng cứ dataset mounted hiện tại |
| Integrity/source-disjoint — TV2→TV1 | `metadata/dataset_sha256.csv`: relative_path,file_size,sha256,split,category; persisted source plan kèm normalized raw source identity, source SHA, role, data seed, split version/hash; manifests TRAIN/DEV thật | **BLOCKED_MISSING_MANIFEST**; CSV không tồn tại. Cần kiểm same-content copies và overlap raw sources, không chỉ unique synthetic IDs |
| Day04/05 cache replay — TV2 D2 | `<DAY04_CACHE_DIR>/manifest.json`, `shards/shard-*.pt`; records `train_core.json`/`dev_synthetic.json`, mask root, raw TRAIN images, generator config/code/seed/crop metadata. Test builder dùng `<EXPORT_ROOT>/records/...` và `<EXPORT_ROOT>/masks/...` | **BLOCKED_MISSING_CACHE_RECORDS**; absolute paths Drive chưa được cung cấp. Đặt `FEATURE_CACHE_DIR`/`FEATURE_TRAIN_INDEX` khi bàn giao; không tự tạo cache thay thế |
| Native DEV Day05 — TV2→TV1 | `<DAY05_EXPORT_ROOT>/dev_inputs.json`, native image/GT files; schema `msila.day05.inference_inputs.v1`, split `dev_synthetic`, samples có image_id/category/image/gt_mask/original_hw/source identity/image+GT SHA và synthetic provenance | **BLOCKED**; cần replay mask Local và sáu feature trước export; same records/mask/generator cho R0/R1/R2. Không tự áp bins native-v2 vào replay |
| Old DEV GT audit — TV2 D2 | `OLD_DEV_RECORDS.json` + `OLD_MASK_ROOT`, rows có image_id,split,gt_mask hoặc mask_path,is_anomaly; GT binary2D ở native H×W | **BLOCKED_MISSING_HISTORICAL_GT**; chưa đo connected-component distribution. Cần phân biệt local-crop GT với native GT trước gọi CLI |
| Full-scale native coverage — TV2 D2 | `configs/full_scale_synthetic.yaml` hiện có; output thực `assets/dataset/gt_statistics.json`, full-scale source plan/records/native inputs và cache producer schema `msila.full_scale.cache.v2` | **Protocol CODE_PRESENT; REAL_ASSET_NOT_VERIFIED**. Stats file chưa có; root cache/build output thật phải được cung cấp. Không dùng fixture coverage thay real statistics |
| Native inference checkpoint — TV1→TV2 | `outputs/day05/full_train/seed_42/fabric/{R0,R1,R2}/`: best.pt,last.pt,resolved_config.yaml,run_manifest.json,selection_record.json,epoch_log.csv,training_log.csv; `full_train/protocol_lock.json`, `checkpoint_rule_lock.json` theo handoff doc | **BLOCKED_MISSING_CHECKPOINT** tại môi trường audit. Local path trên MISSING; cần exact Drive run directories cùng checksum. Notebook COMPLETE chỉ là record, không phải strict artifact verification |
| Real feature replay — TV1/TV2 | `DINOV3_REPO` có official `hubconf.py`, weights thật cùng SHA/source revision, `REAL_SAMPLE_PT` export bởi preprocessing thật (image_id/category,x_local,x_context,geometry) | **SKIP / BLOCKED_MISSING_DINO_SAMPLE**; không tự download hoặc dùng mock fixture làm real |
| Scientific diagnostics/selection — TV2 | Day05 tiny/boundary JSON locks thật, DEV native GT/maps provenance, explicit threshold từ DEV, representation lock hợp lệ trước TEST; GPU cho efficiency | **BLOCKED / NOT_VERIFIED**. Actual Day05 lock paths chưa được cung cấp. Full-scale native threshold0.5/band2/tolerance2 là protocol riêng, không tự điền vào Day05 |

Hai lệnh audit chỉ để handoff sau khi assets có sẵn, **chưa chạy trên data thật**:

```bash
python -m scripts.audit_dataset --data-root REAL_DATA_ROOT --output dataset_audit.json
python -m scripts.audit_synthetic_gt --inputs OLD_DEV_RECORDS.json --mask-root OLD_MASK_ROOT --output OLD_DEV_GT_AUDIT.json
```

Các script đọc source assets và ghi output audit riêng; không sửa ảnh/GT/split. DEV manifest phải kiểm source-disjoint trước; audit TEST inventory/pairing không cho phép dùng TEST statistics/predictions để tune.

## 8. Handoff và file thay đổi

TV1 cần nhận **cùng source plan/records/mask hashes và producer signature** cho R0/R1/R2; freeze protocol trước training/evaluation. TV2 D2 cần assets ở §7 và replay/padding checks trước cấp GATE_DATA PASS. Không chứng nhận E1 data/training, Day05 replay thực hay full-scale DEV thực chỉ từ software tests này.

- `src/data/tiling.py`: bug fix4 dòng thay1 dòng; SHA256 sau sửa `e8dec3dfdb16843d72d34c33a279ded2fd58fe0998b19911b28bc5aa364bd8e6` (trước sửa `a0da173a68c9c7193ee002e183ea492efdea986b7a41438b0d26ce18d52f365e`).
- `tests/test_masks.py`: thêm regression13 dòng; SHA256 sau sửa `c8e0932150ac37d60d60d49ff5138bdc8d2df4f6f7db22f99bfc36371a59df16`.
- `reports/recheck/tv2_d1_data.md`: báo cáo mới.

Hai file code/test giữ CRLF như bản gốc. Plain `git diff --check` exit2 vì coi CR cuối các dòng thêm là trailing whitespace; `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check` **PASS**, không sửa Git config hoặc normalize toàn file. Relative links và số test trong báo cáo đã đối chiếu JUnit.

Loader/generator/cache format, configs/protocols, ngữ cảnh nhóm và `reports/recheck/tv1_d1_architecture.md` được giữ nguyên. Dừng ở audit TV2-D1; chưa chạy D2, training, real inference hay selection.
