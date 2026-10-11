# D11-TV1 — main training E1–E5

Code chuẩn bị cho bốn category `can,fabric,fruit_jelly,rice` trên `g2/member1`.
Không merge commit D11 vào `g2/integration`; chủ repo tạo nhánh integration D11 sau.
Thực nghiệm hiện **BLOCKED / NOT RUN: 0/36 ô, 0 checkpoint thực**.
Workspace thiếu `joint_selection_lock.json` D10; GPU thực chỉ chạy trên Colab.

## Điểm vào và điều kiện

Notebook dùng chung: `notebooks/G2_03_Training.ipynb`, chọn `TRAINING_STAGE='main'`.
CLI: `scripts/run_g2.py --stage main --config configs/g2_main.yaml`.
Runner gọi lại `_train_segmentation` qua `train_main`; không có training loop trong notebook.
Data/loss/native inference/AU-PRO/Dice/QA dùng các API hiện có. Không sửa TV2 evaluator,
Data, Geometry hoặc Fusion. Đường screening D10 và debug D7/D8 vẫn được giữ.

D11 chỉ chấp nhận lock **`g2_joint_selection_lock_v1`** được TV2 xuất, hai CSV 180 ô
và toàn bộ bằng chứng `train/`, `evaluation/`, `training_protocol.json` của hai owner.
Runner kiểm tra SHA của lock, rule được khai báo trước kết quả, spec, ranking/selected,
360 ô PASS và parity DEV. Sau đó gọi `g2_joint_selection.evidence_auditor` trên đủ
360 checkpoint/native exports. Không có đường chỉ tin giá trị r/d trong JSON.
Các evidence có thể nằm trên Drive: runner relocate đường dẫn bằng API TV2 nhưng
không sửa nội dung đã ký. Audit là CPU và thực hiện ở đầu mỗi invocation.

Main budget lấy **nguyên** từ `main_budget` trong protocol D10 đã audit; phải khác
screening budget và không cắt ngắn bằng wall time. Không có flag ghi đè r/d, seed,
loss hoặc batch trong D11. D10 config hiện khai báo screening 10 epoch/batch 32;
main default ở model config là 20 epoch/batch 4. Số chính thức là số trong protocol
đi kèm lock thực. Batch 32/64 phải được khai báo làm main budget trước lựa chọn D10;
không tự đổi batch sau lock để vừa VRAM hoặc sau khi nhìn điểm main.

## Coverage và reuse

Mỗi category có 9 ô logical: E1, E2, E3, E4 và E5 với đủ
`mean|weighted_sum|concat|gated|attention`. Tổng TV1 36 ô.
Tất cả E2–E5 dùng cùng locked `(r*,d*)`; `F*` chỉ là metadata, không giới hạn các E5.
E2 không có Fusion; E3 Mean(3) Local; E4 Mean(6) aligned Local/Context.

E4 và E5-Mean được canonicalize về **cùng resolved config**, physical run ID,
checkpoint/config checksum, nguồn, Adapter/projection, decoder và protocol.
CPU test đối chiếu state, logits và gradient giữa hai model factory; Mean không có
tham số học. Với `reuse_e4_average: true` có **32 physical runs**, 36 ô báo cáo.
Ô alias chỉ PASS sau khi checkpoint/native export của physical E4 được kiểm tra.
Coverage ghi riêng số physical hoàn thành, số reuse G1 và số train mới thực sự hoàn thành.
`reuse_e4_average: false` trước main tạo protocol hash mới và 36 physical runs;
không dùng alias khi configs khác nhau.

`g1_checkpoints: {category: /path/to/G1/category/best.pt}` là tùy chọn. Reuse E1 đòi
G1 train thật, cùng backbone/checksum/source implementation, sources/split/synthetic
protocol, seed, optimizer, loss, full budget, metric/threshold và decoder.
Giữ best/last + SHA + metrics đầy đủ; thiếu provenance hoặc khác protocol ⇒ ghi
`g1_reuse_decision.json` với RETRAIN. Checkpoint legacy thiếu DINO source hashes
không đủ điều kiện. Reuse hợp lệ bọc lại best/last vào schema D11, giữ số update/
weights/optimizer tương ứng, không train lại và không sửa G1 gốc.

## Resume, Drive và GPU

`--resume` kiểm tra config/checksum trước load model, optimizer, RNG và batch cursor.
Skip chỉ sau khi full budget, trained heads/frozen backbone, config, loss history,
best/last và native score/mask/QA đều hợp lệ. CPU fixture và smoke không được tính
science PASS hoặc đưa metric vào bảng khoa học. FAIL/OOM giữ log, không gán điểm 0;
report-only vẫn giữ trạng thái lỗi. Không tự hạ batch từng ứng viên.

Frozen backbone cache được chia theo deepest/multilayer, tái dùng giữa architecture
khi mode tương thích; heads và optimizer được khởi tạo mới theo cùng seed cho mỗi
category/run. Chỉ giữ một frozen backbone trên GPU. Local/Context dùng strategy
trong protocol D10 (thường `sequential`, mỗi forward batch B). Ghi peak allocated/
reserved bytes và GPU tên thật trong `gpu_memory.json` khi đã đo; CPU không đo VRAM.

Checkpoint trung gian chỉ ghi local theo interval hiện có. Drive sync **mỗi epoch**,
kết thúc job và khi lỗi; không sync mỗi step hoặc bằng timer. Native maps được sync
sau job. Notebook restore `<MAIN_RUN_ID>/<main_protocol_sha>/<owner>` từ Drive
về Colab trước resume; input archives được copy lên Colab rồi mới giải nén local.

## Chạy tiếp trên Colab

1. Push commit D11 trên `g2/member1`, đặt `GIT_COMMIT` đúng SHA; không cần merge
   integration. Chọn T4/L4/A100 và chạy Setup. Dependencies giữ CUDA PyTorch hiện có,
   cài `safetensors>=0.8`.
2. Trong INPUT, đặt Drive dataset archives/weights, `JOINT_LOCK_ON_DRIVE`,
   `TV1_D10_MANIFEST`, `TV2_D10_MANIFEST` đến evidence D10 thật. Pin DINOv3 commit
   đúng source SHA của D10. Cell OUTPUT chứa riêng các folder output D11.
3. Chạy prepare data và D11 preflight. Khi lock chưa đủ điều kiện, xem BLOCKED reason.
4. Bật `RUN_MAIN_GPU_SMOKE=True`: E5 Attention, cùng locked pair và batch,
   tối đa 1 epoch/2 updates, namespace riêng; không tính vào main coverage.
5. Bật `RUN_MAIN_TRAINING=True`; `MAIN_MAX_JOBS=1` để chia phiên hoặc `None` để chạy
   phần còn lại. Có thể lọc category/experiment/Fusion để chia công việc. Giữ RESUME.
6. Chạy coverage. TV2 dùng đúng runner commit, lock, DINO source và main protocol,
   đổi `OWNER='tv2'`. D11 không tự chọn lại và không tự công bố E5 thắng.

CLI tương đương, chạy **trên Colab** (đường dẫn ví dụ cần thay):

```bash
python scripts/run_g2.py --stage main --config configs/g2_main.yaml \
  --joint-lock /content/drive/MyDrive/msila/outputs/G2/D10/selection/joint_selection_lock.json \
  --tv1-manifest /content/drive/MyDrive/msila/outputs/G2/D10/RUN_ID/PROTOCOL_SHA/tv1/tv1_joint_search_180.csv \
  --tv2-manifest /content/drive/MyDrive/msila/outputs/G2/D10/RUN_ID/PROTOCOL_SHA/tv2/tv2_joint_search_180.csv \
  --data-root /content/datasets/mvtec_ad2 --repo-dir /content/dinov3 \
  --weights /content/msila_d10/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output-root /content/msila_d11/outputs \
  --drive-output /content/drive/MyDrive/msila/outputs/G2/D11 \
  --owner tv1 --device cuda --preflight
```

Giữ các flag đường dẫn, thay `--preflight` bằng
`--smoke --categories can --experiments E5 --fusions attention --max-jobs 1 --resume`,
sau đó `--categories can fabric fruit_jelly rice --max-jobs 1 --resume`.
`--report-only` audit coverage, không train.

## Artifact và handoff TV2

`outputs/G2/D11/<RUN_ID>/<main_protocol_sha>/<owner>/` chứa:

- `main_training_36.csv`, `main_coverage.json`, `run_manifest.json`, `training_protocol.json`,
  `selection_audit.json` và config/hash của từng physical run trong `plan/`.
- `main/<physical_run_id>/`: best/last + SHA, resolved config/hash, metrics + SHA,
  loss history, train log, GPU memory nếu đã đo, reuse/failure evidence khi áp dụng.
- `native_dev/native_maps.json`, maps score/mask native resolution và QA, gắn checksum.
- `smoke/` riêng; thiếu lock tạo `prepared/<owner>/` với 36 BLOCKED và metric null.

Restore D11 head state qua `trainable_modules(model, experiment)`; E1 là ModuleDict
`decoder`, E2 `adapter/decoder`, E3/E4 `adapters/projection/decoder`, E5 thêm `fusion`.
E5-Mean alias dùng chính E4 config/checkpoint; E5-Mean factory load cùng state strict
vì Fusion không có weights. Metadata checkpoint có lock SHA, main protocol SHA,
Git SHA, run ID, optimizer/RNG/cursor và history.

CLI `eval_g2.py` hiện chưa intake D11 main E3/E4/E5 và E1 schema mới; **không sửa
CLI đó trong task TV1**. Main runner đã dùng public TV2 native capture/Dice/QA và
metric hiện có để lưu synthetic DEV. Final held-out evaluation, các báo cáo ablation
8 category và nghiệm thu G2 tổng thể cần TV2 nối intake D11 sau; không coi DEV này
là kiểm định generalization độc lập. E0/G3 và retuning nằm ngoài task.
