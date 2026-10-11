# D10-TV2 — Hợp nhất screening và khóa joint selection

TV1 D10 đã được kiểm tra trên `g2/integration` tại
`bb2df3a911fb5725bcd7b59ef8618691162a264c` và fast-forward vào `g2/member2`.
TV2 giữ nguyên Model, `scripts/run_g2.py`, training modules, config và
`notebooks/G2_03_Training.ipynb` của commit đó. Không có training loop,
evaluator hay metric mới: phần bổ sung chỉ đọc manifest, gọi audit hiện có,
tổng hợp và kiểm soát publication của lock.

Người dùng xác nhận **chưa chạy D10-TV1**, sẽ chạy sau khi D10-TV2 hoàn tất.
Artifact CPU preparation tại `outputs/G2/D10/tv2_acceptance/` ghi
**TV1 0/180 + TV2 0/180; 360/360 NOT RUN; 0/45 macro hợp lệ; lock BLOCKED**.
Có đủ 360 dòng trạng thái không có nghĩa là đủ 360 kết quả. Không có checkpoint
screening thật, metric GPU hoặc số đo latency/VRAM. Fixtures kiểm thử được lưu
trong thư mục tạm pytest, không đưa vào bảng thực nghiệm.

## Quy tắc trước khi chạy

Quy tắc cố định trong `src/eval/g2_joint_selection.py`, khai báo trước khi xem
kết quả, được ghi thành `joint_selection_rule.json` gắn với SHA của spec:

1. Xét các candidate có **8/8** AU-PRO@0.05 hợp lệ được evaluator audit.
2. Macro là tổng tám AU-PRO chia **8**; không skip missing, không điền zero.
3. Chọn macro cao nhất. Exact ties: ít trainable parameters hơn → r nhỏ hơn →
   d nhỏ hơn → `mean,weighted_sum,concat,gated,attention`.
4. Trainable parameter count của một candidate phải giống nhau trên tám
   category. Không dùng tiny/mixed, Dice, P99, latency hoặc VRAM để đổi tie-break.
5. Average giữ nguyên đủ chín cặp × tám category = **72 ô đối chứng**.

Policy tài nguyên **đang có trong config TV1** là fixed batch cho tất cả
candidate và `record_failure_no_automatic_batch_change`. OOM được ghi FAIL;
policy này chưa định nghĩa quy tắc chuyển failure thành một exclusion cuối.
Vì vậy **protocol hiện tại cần 360 audited PASS để khóa**. Không đổi OOM thành
score 0 hoặc tự gán INFEASIBLE chỉ từ trạng thái CSV. Những ô
`INFEASIBLE_ON_HARDWARE` chưa có quy tắc nghiệm thu được liệt kê là unresolved,
không được hạch toán final. Muốn hỗ trợ exclusion khác cần khai báo protocol
tài nguyên trước một lượt search mới; không bổ sung hậu nghiệm cho search này.

CPU preparation rule chỉ gắn với prepared spec chưa có hashes assets. Với
thực nghiệm, **khai báo lại tại namespace protocol thật sau preflight, trước
full screening**. Hàm khai báo từ chối manifest đã có screening attempt hoặc
thư mục `train/` có evidence; file rule đã có được giữ nguyên khi resume.

## Colab: cùng notebook, hai owner

Sau khi merge D10-TV2, cả hai owner phải checkout **cùng SHA cuối đã merge**,
có cả code TV1 và TV2. SHA `bb2df3a` ở trên chỉ là base đã kiểm tra, chưa chứa
report/selection TV2. Pin `GIT_COMMIT` trong notebook, đồng thời pin DINO SHA
chung; giữ cùng checkpoint checksum, archives và resolved config. Không dùng
một branch HEAD thay đổi giữa các phiên.

Mở `notebooks/G2_03_Training.ipynb`, chỉnh INPUT/OUTPUT theo Drive thực tế.
Notebook đã copy **đủ tám `.tar.gz` từ Drive sang Colab trước khi giải nén**,
chỉ dùng TRAIN/good và VALIDATION/good; không dùng TEST. Giữ nguyên ngân sách
TV1: ViT-B/16, seed TRAIN **2026**, DEV **17017**, 20 epochs, batch **32**,
AdamW, pair strategy sequential. Nếu phải đổi ngân sách, đổi chung trước một
protocol mới; không gộp các protocol khác nhau.

Sau cell preflight và trước cell full, chạy lệnh khai báo rule dưới đây trong
một code cell. `PROTOCOL` là tên thư mục hash thật được preflight in ra, **không
dùng `prepared`**. `--resource-config` phải là resolved joint config notebook.

```python
PROTOCOL = 'HASH_THAT_PREFLIGHT_IN_RA'
STUDY = LOCAL_OUTPUT_BASE / RUN_ID / PROTOCOL
SELECTION = STUDY / 'selection'
RULE = SELECTION / 'joint_selection_rule.json'
subprocess.run([
    PYTHON, 'scripts/summarize_g2_joint.py',
    '--manifest', str(STUDY / OWNER / 'joint_search_360.csv'),
    '--declare-rule', '--rule', str(RULE),
    '--resource-config', str(JOINT_CONFIG),
], cwd=REPO, env=RUN_ENV, check=True)
# Persist declaration BEFORE starting full search. Resume restores the same file.
rule_on_drive = DRIVE_OUTPUT_BASE / RUN_ID / PROTOCOL / 'selection' / RULE.name
rule_on_drive.parent.mkdir(parents=True, exist_ok=True)
if rule_on_drive.exists():
    assert rule_on_drive.read_bytes() == RULE.read_bytes(), 'Immutable rule drift'
else:
    shutil.copy2(RULE, rule_on_drive)
```

Khi resume sau Colab ngắt, cell chuẩn bị hiện có khôi phục cả namespace Drive.
Rule và training/evaluation evidence của owner cũng được giữ nguyên.

TV2 đặt `OWNER='tv2'`, `MAX_JOBS_THIS_SESSION=None`,
`SELECTED_CATEGORIES=[]`, `PAIR_FILTER=None`, `FUSION_FILTER=[]` và bật
`RUN_FULL_SEARCH=True` sau smoke PASS. Đây là đúng **180 ô** cho
`sheet_metal,vial,wallplugs,walnuts`. Đặt max jobs nhỏ để chia phiên nếu cần;
`RESUME=True` kiểm chứng checkpoint trước khi skip. Smoke không tính coverage.

Lệnh runner tương đương cho TV2 (chạy từ repo trên Colab sau setup):

```bash
python scripts/run_g2.py --stage joint_search \
  --config /content/msila_d10/config/joint_search.yaml \
  --owner tv2 --categories sheet_metal vial wallplugs walnuts \
  --resume --device cuda --output-root /content/msila_d10/outputs \
  --drive-output /content/drive/MyDrive/msila/outputs/G2/D10
```

Để chạy cả 360 ô trên một Colab đã chuẩn bị, dùng cùng lệnh với `--owner tv1`
cho bốn category đầu, rồi `--owner tv2` cho bốn category cuối. Giữ nguyên
protocol và rule; có thể chia nhiều phiên với resume. Không thêm trainer riêng.
Drive namespace là `RUN_ID/PROTOCOL/{tv1,tv2}`; runner sync theo epoch/job.

## Hợp nhất và khóa

Mỗi owner phải bàn giao nguyên thư mục: manifest 180 ô, sibling spec,
`training_protocol.json`, `train/**/{best,last}.pt` + SHA/metrics/optimizer/RNG,
`evaluation/**` gồm native maps, masks, metrics và signed exports. Chỉ CSV là
chưa đủ. Specs phải giống hoàn toàn, gồm config hashes, Git SHA, seed/budget,
backbone và synthetic protocol; training protocol/source hashes cũng phải khớp.

Chạy sau khi có hai manifest thật (thay `PROTOCOL_HASH` bằng hash thực tế):

```bash
python scripts/summarize_g2_joint.py \
  --tv1-manifest /content/drive/MyDrive/msila/outputs/G2/D10/e5_screen_e20_b32_v1/PROTOCOL_HASH/tv1/tv1_joint_search_180.csv \
  --tv2-manifest /content/drive/MyDrive/msila/outputs/G2/D10/e5_screen_e20_b32_v1/PROTOCOL_HASH/tv2/tv2_joint_search_180.csv \
  --relocate-evidence \
  --rule /content/drive/MyDrive/msila/outputs/G2/D10/e5_screen_e20_b32_v1/PROTOCOL_HASH/selection/joint_selection_rule.json \
  --output-root /content/drive/MyDrive/msila/outputs/G2/D10/e5_screen_e20_b32_v1/PROTOCOL_HASH/selection \
  --lock
```

`--relocate-evidence` thay prefix đường dẫn theo đúng layout runner dưới folder
owner, giữ nguyên nội dung checkpoint/export và kiểm lại hashes. Không sửa
identity JSON để hợp thức hóa checkpoint khác. Tổng hợp cũng chạy được trên
CPU sau khi đồng bộ evidence: đó là audit **kết quả GPU đã lưu**, không phải
benchmark GPU trên CPU. Không overwrite manifest của owner.

Output: `joint_search_360.csv`, `joint_search_spec.json`, schema,
`joint_search_coverage.json`, `joint_search_macro.csv` (45 dòng),
`joint_search_matrix_45x8.csv`, `joint_selection_decision.json/.md`.
Khi đủ evidence và rule hợp lệ, thêm `joint_selection_lock.json` có hash,
ranking 45 candidate và provenance 360 ô; bảng `fusion_comparison_selected.csv`
vẫn hiển thị đủ năm Fusion trên tám category tại r*/d*. Lock đã có chỉ được
reuse nếu nội dung/evidence giống hệt, không overwrite bằng kết quả khác.

Exit 0: declaration thành công hoặc coverage complete (và lock PASS nếu yêu
cầu). Exit 2: incomplete/BLOCKED, xem coverage và reason; không tính là kết
quả khoa học PASS. Không có lock khi chưa chạy GPU.

## Giới hạn và kiểm thử

Latency/peak VRAM hiện **NOT RUN**, không có số đo T4/L4/A100 giả. Nếu đo bằng
benchmark evaluator hiện có, công bố GPU, CUDA/Torch, dtype, native H×W,
tile batch, warmup/repeats và phạm vi latency; không dùng để thay quy tắc chọn.
Budget screening và một seed có thể đổi thứ hạng khi full training. DEV được
reuse để chọn checkpoint/candidate có selection bias. Xác nhận nhiều seed/top
candidates nếu có tài nguyên phải công bố riêng; không sửa rule hậu nghiệm.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/msila-mpl .venv/bin/python -m pytest -q \
  tests/test_g2_tv2_d10_selection.py tests/test_g2_tv2_joint_search.py \
  tests/test_g2_tv1_d10.py tests/test_g2_tv2_evaluator.py \
  tests/test_g2_tv2_evaluation_notebook.py
```

Tests kiểm chứng mean có ground truth tính được, thiếu category/duplicate ID,
tie-break, rule/config/protocol parity, checkpoint thật bắt buộc, relocation,
OOM không được tự final, publication/immutability, runner resume và evaluator
regression. Số test thực tế/lệnh/coverage được ghi trong
`outputs/G2/D10/tv2_acceptance/code_acceptance.json`; không tính các checkpoint
fixture là checkpoint thực nghiệm.
