# D10-TV1 — Joint E5 screening train/resume

## Trạng thái

Code E5 screening và notebook dùng chung đã được triển khai. Targeted tests và
XML nghiệm thu nằm ở `outputs/G2/D10/code_acceptance.json` và `targeted_tests.xml`.
**Thực nghiệm Colab: NOT RUN, PASS 0/180 TV1**; chưa có checkpoints/logs/metrics
từ lượt GPU thật. CPU training fixtures/smoke không được tính là scientific PASS.

Đã fetch và kiểm tra `g2/integration` sau D9: base
`ab64c0d1b7133815a94dbbb0826196e75bdc9f2b`. `g2/member1` bắt đầu cùng commit,
không có thay đổi chưa lưu hay conflict. Không sửa evaluator, metric, Data,
Geometry hoặc Fusion modules TV2.

## Ngân sách được chốt

Người dùng yêu cầu 20 epochs, AdamW, batch 32 hoặc 64 và Drive sync sau epoch.
`configs/g2_joint_search.yaml` khai báo riêng **screening_budget: 20 epochs,
batch 32, max_steps=null, max_minutes=null**. `main_budget` được ghi riêng từ
config chính (mặc định 20 epochs, batch 4); D10 không chạy main training.
Seed TRAIN 2026, seed DEV 17017; AdamW lr 1e-3, weight decay 1e-4, loss BCE +
positive-mask Dice và synthetic protocol/evaluator giữ nguyên.

Toàn bộ candidate dùng cùng batch, ngân sách, input, pretrained checksum,
split/source checksums và loss/optimizer. Pipeline float32, AMP/TF32 tắt.
`sequential` trích Local/Context qua hai forward batch B của cùng backbone,
giảm bộ nhớ so với packed batch 2B; strategy được giữ chung cho toàn search.
Data workers mặc định 2, có thể cấu hình trước khi chạy protocol.

T4/L4/A100 phải kiểm tra bằng smoke thật; chưa xác nhận batch 32/64 vừa VRAM.
OOM được ghi `FAIL`, reason/failure_kind `OOM`, candidate/budget/hardware và
attempt history. Không tự hạ batch riêng, gán score 0 hay bỏ Fusion. Nếu cần
batch nhỏ hơn, đổi batch **chung cho tất cả candidate** trước một protocol mới;
không gộp điểm của hai ngân sách. Thay đổi source data/code/budget/weights tạo
protocol hash và thư mục mới, giữ nguyên kết quả cũ.

## Runner và bằng chứng mỗi job

Entry point: `scripts/run_g2.py --stage joint_search`. Coordinator tái sử dụng
`src/train/g2_e2.py` qua `train_e5`, cùng training loop, paired dataset, train
step, loss và native synthetic DEV evaluation hiện có. Factory tái sử dụng
**frozen backbone**; mỗi category/r/d/Fusion có head và optimizer mới, không
dùng chung model đã train giữa category. Chỉ train Adapter, projection,
Decoder và tham số Fusion (Average không có tham số học).

ViT-B/16 C=768, block registry 4/8/12, grid r=[64,128,256], d=[256,512,768].
Runner kiểm tra đủ 45 D9 CPU PASS và chín số params Adapter khác nhau; nếu D6
hiện có hoặc D9 báo invalid/missing thì chặn trước khi lập/running search.
Không dùng E2-only hay Average-only AU-PRO để chọn r/d.

Một job chỉ vào PASS khi đủ budget, pretrained thật, CUDA và evaluator TV2
chấp nhận/audit evidence. Training checkpoint được chọn theo synthetic DEV
AU-PRO@0.05, exact ties giữ epoch sớm nhất. Sau training, gọi nguyên
`scripts/eval_g2.py` API để xuất native score/mask, QA, metrics và identity;
không viết metric/evaluator khác. Frozen training cache được park CPU trong
lúc evaluator restore model để tránh hai backbone cùng ở GPU.

Trong folder protocol/owner:

- `train/E5/r{r}_d{d}/{fusion}/{category}/seed2026/`: best/last checkpoint và SHA,
  optimizer/RNG/cursor, planned/resolved config + hash, loss history, train log,
  training metrics, epoch progress; `attempts/` giữ lỗi lịch sử.
- `evaluation/E5/...`: evaluator metrics/config, QA, native maps/exports và SHA.
- `joint_search_spec.json`, `joint_search_360.csv`: schema TV2, đủ 360 ô; ô của
  worker còn lại chưa chạy để `NOT RUN`.
- `tv1_joint_search_180.csv` hoặc `tv2_joint_search_180.csv`: đúng 180 ô của owner.
- `training_protocol.json`, `run_manifest.json`, `training_coverage.json`:
  seed/budgets/config/commit/source and dataset hashes, hardware và PASS/N/180.

Skip kiểm chứng toàn bộ candidate/config hash/category/r/d/F/seed/budget,
best/last SHA, finite weights/optimizer, RNG/cursor, params và evaluator exports.
Resume dùng optimizer/RNG/cursor checkpoint hiện có, không lặp update ở batch
đã commit. Config hoặc checksum sai bị chặn, không ghi đè checkpoint cũ.

Checkpoint giữa epoch chỉ ghi local theo interval hiện có (20 steps).
**Drive sync sau mỗi epoch và sau job/khi lỗi**; không dùng timer/per-step sync.
Report/skip không ghi Drive 180 lần theo từng dòng. Khi Colab ngắt giữa epoch,
có thể mất phần tiến độ local chưa sync; Drive giữ checkpoint epoch trước.

## Chạy trên Colab

Mở `notebooks/G2_03_Training.ipynb`. Notebook không chứa trainer/model/metric.
Setup checkout SHA chứa D10, resolve/pin DINO SHA, giữ CUDA torch Colab và cài
`safetensors>=0.8`. INPUT cell chứa archives/checkpoint; OUTPUT cell riêng chứa
tất cả output paths/RUN_ID. Mount Drive → copy đủ tám `.tar.gz` local → extract
TRAIN/VALIDATION good trên Colab → copy checkpoint local → preflight → smoke
một job nhỏ → cell full screening riêng. Không dùng TEST.

Chọn `OWNER='tv1'` (bốn category đầu) hoặc `OWNER='tv2'` (bốn category cuối).
Mặc định full search tắt và max jobs/session là 1. Sau smoke PASS, bật
`RUN_FULL_SEARCH=True`; đặt `MAX_JOBS_THIS_SESSION=None` để chạy phần còn lại,
hoặc giữ giới hạn/chọn category, pair, Fusion để chia phiên. `RESUME=True`
khôi phục Drive trước khi chạy và kiểm chứng skip/resume. Không thay budget
riêng sau khi xem điểm. Notebook không tạo selection lock.

Ví dụ CLI, chạy từ repo root trong môi trường có dependencies:

```bash
# Asset-free preparation: không GPU, không train, output NOT RUN.
CUDA_VISIBLE_DEVICES='' python scripts/run_g2.py --stage joint_search \
  --config configs/g2_joint_search.yaml --prepare --device cpu

# Colab: chỉnh input paths trong model/base runner config trước.
python scripts/run_g2.py --stage joint_search --config /content/msila_d10/config/joint_search.yaml \
  --preflight --owner tv1 --device cuda --output-root /content/msila_d10/outputs

python scripts/run_g2.py --stage joint_search --config /content/msila_d10/config/joint_search.yaml \
  --smoke --owner tv1 --categories can --r 256 --d 768 --fusions attention \
  --resume --device cuda --output-root /content/msila_d10/outputs \
  --drive-output /content/drive/MyDrive/msila/outputs/G2/D10

python scripts/run_g2.py --stage joint_search --config /content/msila_d10/config/joint_search.yaml \
  --owner tv1 --resume --device cuda --max-jobs 1 \
  --output-root /content/msila_d10/outputs \
  --drive-output /content/drive/MyDrive/msila/outputs/G2/D10
```

CLI exit 2 có thể là INCOMPLETE khi chỉ chạy một nhóm; xem coverage/FAIL reason.
PREPARED/READY là code/setup status, không phải nghiệm thu thực nghiệm.

## Handoff TV2

Dùng **cùng commit chứa D10**, DINO SHA/checkpoint, tám archives và resolved
config/budget; copy thông tin từ `notebook_environment.json`, shared config và
`training_protocol.json`. TV2 chỉ đổi OWNER thành tv2, hoặc `--owner tv2`.
Hai owner có thư mục riêng dưới cùng protocol hash, tránh ghi đè manifest khi
chạy đồng thời. Mỗi owner bàn giao `*_joint_search_180.csv`, spec, checkpoints
và evaluator exports. TV2 ghép đủ 360 ô, relocate path khi cần rồi dùng
`scripts/summarize_g2_joint.py` / evaluator hiện có để audit/tổng hợp.
Không tính macro tám category từ chỉ bốn category, không lock ở TV1.

Files thay đổi: `scripts/run_g2.py`, `src/models/msila.py`, `src/train/g2_e2.py`,
`src/train/g2_joint_search.py`, `configs/g2_joint_search.yaml`,
`notebooks/G2_03_Training.ipynb`, `tests/test_g2_tv1_d10.py`, report và artifact
nghiệm thu D10. Không mở rộng sang main training D11.
