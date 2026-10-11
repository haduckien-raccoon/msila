# D6-TV1 / PROMPT 03 — nghiệm thu kỹ thuật

Worktree bền vững: `.worktrees/g2-member1-d6`, branch `g2/member1`.
Workspace chính giữ nguyên branch `g2/member2` và các thay đổi TV2 chưa commit.

**CPU: 9/9 PASS. GPU real backbone / peak VRAM / OOM: NOT RUN.**
Hai bước kiểm tra trên mock feature `[2,768,4,4]`, E2/loss/optimizer/train-step thật,
logits `[2,1,512,512]`. `gamma=0` identity; bước đầu gradient ở gamma/Decoder;
bước sau mọi Conv Adapter nhận gradient khác 0. Backbone frozen/không đổi,
optimizer chỉ chứa Adapter + Decoder. Tám cặp bất hợp lệ bị từ chối;
config YAML round-trip giữ hash; cả chín cấu hình và số tham số khác nhau.

| r | d | CPU | Adapter params | GPU / VRAM |
|---:|---:|---|---:|---|
| 64 | 256 | PASS | 263873 | NOT RUN |
| 64 | 512 | PASS | 477121 | NOT RUN |
| 64 | 768 | PASS | 690369 | NOT RUN |
| 128 | 256 | PASS | 330113 | NOT RUN |
| 128 | 512 | PASS | 559745 | NOT RUN |
| 128 | 768 | PASS | 789377 | NOT RUN |
| 256 | 256 | PASS | 462593 | NOT RUN |
| 256 | 512 | PASS | 724993 | NOT RUN |
| 256 | 768 | PASS | 987393 | NOT RUN |

`params` trong CSV đếm Adapter; Decoder và tổng trainable params ở cột riêng.
VRAM chưa đo để trống, không gán 0. Không có AU-PRO/ranking/top-3/selection lock.
Không chạy 72 jobs hoặc full E2 training. CPU fixture không phải nghiệm thu GPU.

## File thay đổi

- `docs/G2_CONTRACT.md`: đồng bộ nguyên văn contract v3 từ workspace người dùng.
- `configs/g2_runner.yaml`: runner v2; `pending_joint_selection_D10`, legacy lock audit-only.
- `scripts/run_g2.py`: stage `adapter_preflight`; alias `adapter_screen` chạy CPU;
  `E2 --smoke --debug-pair R D` không cần lock D6, dùng output `debug_d6_v3` riêng.
  Giữ trainer/API main hiện có; CLI main chính thức chờ joint lock/protocol D10/D11.
- `scripts/g2_adapter_preflight.py`: orchestration CPU/GPU, checksum/config provenance,
  actual parameter counts, OOM/headroom backoff, GPU subset/resume, lưu từng cặp lên Drive.
- `scripts/g2_colab_screening.py`: shim chuyển sang preflight, loại bỏ đường 72 jobs.
- `notebooks/G2_02_Adapter_Screening.ipynb` → `G2_02_Adapter_Preflight.ipynb`;
  không giữ hai notebook duplicate. Cell OUTPUT riêng; safetensors>=0.8.
- `tests/test_g2_colab_screening.py` → `tests/test_g2_adapter_preflight.py`;
  cập nhật `tests/test_g2_tv1_runner.py` theo semantics D6-v3, bảo toàn legacy artifacts.
- `outputs/G2/D6/`: CSV, manifest/config/seed/source hashes, logs,
  9 JSON CPU evidence có checksum, báo cáo code acceptance.

Adapter, E2 model, training loop, loss, Evaluator TV2 và checkpoint cũ không sửa.

## Kiểm thử

- 145 PASS, 1 deselected: preflight, runner, Adapter factory, E2 model và G2 contract.
  Test real-pretrained integration được loại khỏi local để không chạy GPU/tải model.
- 31 PASS sau chỉnh sửa cuối: toàn bộ test preflight, gồm cache backbone trên CPU fixture,
  giảm batch/OOM bằng unit fixture và từ chối evidence checksum sai.
- nbformat validate, AST toàn bộ code cell, Python syntax, git diff --check: PASS.
- CLI CPU thật: `python scripts/run_g2.py --stage adapter_preflight --device cpu`: 9/9 PASS.
- GPU không chạy; không tạo metric hay VRAM giả. Bằng chứng test mock nằm trong temp pytest.

## Cách chạy Colab

1. Commit/push các thay đổi trong worktree `g2/member1` lên GitHub; code hiện chưa commit.
2. Upload `notebooks/G2_02_Adapter_Preflight.ipynb` lên Colab; đặt `GIT_COMMIT` bằng SHA mới.
3. Chỉnh Drive OUTPUT/RUN_ID ở cell OUTPUT; chạy setup và CPU preflight.
4. Muốn smoke GPU: chọn T4/L4/A100, bật `RUN_GPU_SMOKE=True`, đặt đường dẫn checkpoint
   ViT-B/16 trên Drive, chạy cell GPU riêng. Checkpoint được copy về `/content` và verify SHA256.
5. Mặc định GPU chỉ đo `256:768`; `GPU_PAIRS=['all']` đo cả 9 cặp. `MAX_BATCH` và
   `MEMORY_FRACTION` ở cell Input; probe giảm batch theo VRAM/OOM, không cam kết trước mức batch.
6. Cell tổng hợp xuất bảng và evidence trên Drive. Resume kiểm tra source/config/checkpoint,
   GPU/runtime và checksum. Đổi protocol/hardware dùng RUN_ID mới để giữ lịch sử.

D6 này không cần MVTec AD2/tar.gz vì chỉ dùng input fixture, không đọc DEV/TEST.
Main chính thức BLOCKED theo contract v3: pending joint selection D10 và tích hợp protocol D11.
