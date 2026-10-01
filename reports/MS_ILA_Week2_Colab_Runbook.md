# MS-ILA — Week 2 / Day-2 Colab Runbook

> Mục tiêu: clone repository trên Google Colab, dựng môi trường, chạy toàn bộ QA/test cho pipeline Day-2, lưu log + report + thông tin môi trường vào Google Drive để không mất khi Colab reset.
>
> Phạm vi QA:
>
> `Feature Contract → Attention Fusion → Debug Attention → Fusion→Decoder → Gradient Flow → Numerical Stability → Batch/Shape → TV1 Integration → Day-2 Smoke Test`

---

## 0. Gate cần đạt

Pipeline Day-2:

```text
TV1
Local / Context
      ↓
DINOv3 frozen
      ↓
Adapter
      ↓
Context→Local Alignment
      ↓
Projection
      ↓
6 tensors [B,d,h,w]
      ↓
Attention Fusion v0
      ↓
F_fused [B,d,h,w]
      ↓
Decoder
      ↓
Anomaly logits [B,1,512,512]
```

PASS cuối cùng phải thỏa:

```text
Feature contract                  PASS
Attention Fusion                  PASS
Attention debug                   PASS
Fusion → Decoder                  PASS
Gradient flow                     PASS
Numerical stability              PASS
Batch B=1 / B=2                  PASS
TV1 → TV2 integration            PASS
Full Day-2 smoke test            PASS

NaN                               0
Inf                               0

DINO                              frozen / grad=None
Adapter                           gradient
Projection                        gradient
Fusion                            gradient
Decoder                           gradient
```

---

# 1. Các file bắt buộc trong repository

Trước khi chạy Colab, repository nên có tối thiểu:

```text
models/
├── attention_fusion.py
├── basic_decoder.py
├── contracts.py
├── dinov3_extractor.py
├── mean_fusion.py
├── msila.py
└── residual_adapter.py

tests/
├── test_day02_contracts.py
├── test_attention_fusion.py
├── test_msila_day02_head.py
├── test_gradient_flow.py
├── test_numerical_stability.py
├── test_multiview_forward.py
└── test_tv1_integration.py

scripts/
└── day02_smoke_test.py
```

Nếu TV1 đã hoàn thiện module thật, repository có thể thêm:

```text
data/
└── multiview_transform.py

geometry/
└── view_meta.py

models/
├── context_alignment.py
└── feature_projection.py
```

> `MSILADay2Integrated` không được `detach()` feature ở ranh giới TV1→TV2. Nhờ đó gradient từ Decoder/Fusion có thể quay về Projection/Adapter. DINO phải được freeze bằng `requires_grad=False`.

---

# 2. Colab — mount Google Drive

Chạy cell đầu tiên:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Kiểm tra:

```python
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive")
print("Drive mounted:", DRIVE.exists())
```

Kỳ vọng:

```text
Drive mounted: True
```

---

# 3. Khai báo cấu hình chạy

Chỉnh đúng `REPO_URL` và `REPO_NAME`.

```python
from pathlib import Path
from datetime import datetime

REPO_URL = "https://github.com/<OWNER>/<REPOSITORY>.git"
REPO_NAME = "MS-ILA"
BRANCH = "main"

REPO_DIR = Path("/content") / REPO_NAME

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVE_ROOT = (
    Path("/content/drive/MyDrive")
    / "MS-ILA"
    / "week02"
    / timestamp
)

SAVE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

print("Repository :", REPO_DIR)
print("Save folder:", SAVE_ROOT)
```

Ví dụ output:

```text
Repository : /content/MS-ILA
Save folder: /content/drive/MyDrive/MS-ILA/week02/20261001_110000
```

Mỗi lần chạy tạo một folder mới nên kết quả cũ không bị ghi đè.

---

# 4. Clone project

Nếu repository public:

```python
import subprocess
import shutil

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(
    [
        "git",
        "clone",
        "--branch",
        BRANCH,
        REPO_URL,
        str(REPO_DIR),
    ],
    check=True,
)

print("Clone PASS")
```

Sau đó:

```python
%cd /content/MS-ILA
```

Kiểm tra:

```python
!git status
!git rev-parse HEAD
!find models tests scripts -maxdepth 2 -type f | sort
```

> Nếu repository private, không hard-code GitHub token vào notebook hoặc commit token lên Git. Dùng Colab Secrets/GitHub credential phù hợp.

---

# 5. Clone DINOv3 chính thức nếu pipeline thật cần backbone

Các unit test/synthetic integration có thể không cần checkpoint thật.  
Nhưng full pipeline dùng `DINOv3FeatureExtractor` thì cần source DINOv3 + checkpoint.

```python
from pathlib import Path
import subprocess

DINO_DIR = Path("/content/dinov3")

if not DINO_DIR.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/facebookresearch/dinov3.git",
            str(DINO_DIR),
        ],
        check=True,
    )

print("DINOv3:", DINO_DIR)
```

Khuyến nghị khóa commit để tái lập thí nghiệm:

```python
import subprocess

dino_commit = subprocess.check_output(
    ["git", "-C", str(DINO_DIR), "rev-parse", "HEAD"],
    text=True,
).strip()

print("DINOv3 commit:", dino_commit)
```

Checkpoint nên lưu bền vững trên Drive, ví dụ:

```text
/content/drive/MyDrive/MS-ILA/checkpoints/
└── dinov3_vits16_pretrain_lvd1689m.pth
```

Không lưu checkpoint lớn trực tiếp vào Git repository.

---

# 6. Cài dependencies

```python
import subprocess
import sys
from pathlib import Path

requirements = REPO_DIR / "requirements.txt"

if requirements.exists():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(requirements),
        ],
        check=True,
    )

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pytest",
    ],
    check=True,
)

print("Dependencies PASS")
```

Kiểm tra PyTorch/GPU:

```python
import torch

print("torch :", torch.__version__)
print("CUDA  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
```

---

# 7. Pre-flight: kiểm tra đủ file Day-2

```python
from pathlib import Path

required_files = [
    "models/contracts.py",
    "models/attention_fusion.py",
    "models/basic_decoder.py",
    "models/msila.py",
    "models/residual_adapter.py",

    "tests/test_day02_contracts.py",
    "tests/test_attention_fusion.py",
    "tests/test_msila_day02_head.py",
    "tests/test_gradient_flow.py",
    "tests/test_numerical_stability.py",
    "tests/test_multiview_forward.py",
    "tests/test_tv1_integration.py",

    "scripts/day02_smoke_test.py",
]

missing = [
    rel
    for rel in required_files
    if not (REPO_DIR / rel).is_file()
]

if missing:
    print("PRE-FLIGHT: FAIL")
    for rel in missing:
        print(" - missing:", rel)
    raise RuntimeError(
        "Thiếu file Day-2. Không chạy smoke test."
    )

print("PRE-FLIGHT: PASS")
```

---

# 8. Chạy từng nhóm test Day-2

Có thể chạy riêng từng nhóm để biết lỗi nằm ở module nào.

## 8.1 Feature contract

```python
!python -m pytest tests/test_day02_contracts.py -v
```

PASS cần chứng minh:

```text
6 source đúng tên
same [B,d,h,w]
same dtype/device
finite
sai shape/source/channel → ContractError
```

---

## 8.2 Attention Fusion + debug attention

```python
!python -m pytest tests/test_attention_fusion.py -v
```

PASS:

```text
F_fused             [B,d,h,w]
attention            [B,6]
attention logits     [B,6]
weights finite
weights >= 0
sum(weights)=1
gradient tới Fusion
```

---

## 8.3 Fusion → Decoder

```python
!python -m pytest tests/test_msila_day02_head.py -v
```

PASS:

```text
F_fused              [B,d,h,w]
        ↓
Decoder
        ↓
anomaly logits       [B,1,512,512]
```

---

## 8.4 Gradient-flow

```python
!python -m pytest tests/test_gradient_flow.py -v
```

PASS:

```text
DINO          requires_grad=False / grad=None
Adapter       gradient finite
Projection    gradient finite
Fusion        gradient finite
Decoder       gradient finite
```

Lưu ý: Adapter có `gamma_init=0`, vì vậy không ép mọi parameter bên trong residual branch phải có gradient khác 0 ngay tại initialization. Gate đúng là graph không bị đứt, gradient finite và module có learning signal.

---

## 8.5 Numerical stability

```python
!python -m pytest tests/test_numerical_stability.py -v
```

PASS:

```text
input feature          NaN=0 Inf=0
attention logits       NaN=0 Inf=0
attention weights      NaN=0 Inf=0
F_fused                NaN=0 Inf=0
decoder output         NaN=0 Inf=0
backward gradients     NaN=0 Inf=0
```

---

## 8.6 Batch / shape

```python
!python -m pytest tests/test_multiview_forward.py -v
```

Bắt buộc chạy:

```text
B=1
B=2
```

PASS:

```text
input             [B,d,h,w] × 6
F_fused           [B,d,h,w]
attention          [B,6]
logits             [B,6]
output             [B,1,512,512]
```

---

## 8.7 TV1 → TV2 integration

```python
!python -m pytest tests/test_tv1_integration.py -v
```

Gate:

```text
TV1 output
    ↓
6 aligned/projected features
    ↓
validate_multiview_features
    ↓
AttentionFusion
    ↓
Decoder
    ↓
loss
    ↓
backward
```

PASS:

```text
full forward       PASS
full backward      PASS
DINO frozen        PASS
Adapter grad       PASS
Projection grad    PASS
Fusion grad        PASS
Decoder grad       PASS
```

> Nếu `test_tv1_integration.py` hiện còn dùng contract-faithful TV1 test pipeline thay vì module TV1 thật, kết quả này chỉ chứng minh integration boundary/autograd. Chỉ gọi là **real TV1 integration PASS** sau khi thay bằng `context_alignment.py`, `feature_projection.py` và pipeline TV1 thật.

---

# 9. Chạy Smoke Test Day-2 bằng một lệnh

```python
%cd /content/MS-ILA
!python scripts/day02_smoke_test.py
```

Kết quả mong muốn:

```text
================================================================
MS-ILA — DAY 02 ARCHITECTURE QA
================================================================

...
================================================================
DAY 02: PASS
Report: .../day02_report.json
================================================================
```

Script phải sinh:

```text
day02_report.json
```

Nếu một test lỗi:

```text
DAY 02: FAIL
```

Không được đổi FAIL thành PASS thủ công.

---

# 10. Chạy TOÀN BỘ test repository và lưu kết quả

Đây là bước khuyến nghị sau smoke test để phát hiện regression ở code cũ.

```python
import subprocess
import sys
from pathlib import Path

full_log = SAVE_ROOT / "full_pytest.log"
junit_xml = SAVE_ROOT / "full_pytest_junit.xml"

cmd = [
    sys.executable,
    "-m",
    "pytest",
    "tests",
    "-v",
    f"--junitxml={junit_xml}",
]

result = subprocess.run(
    cmd,
    cwd=REPO_DIR,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

print(result.stdout)

full_log.write_text(
    result.stdout,
    encoding="utf-8",
)

FULL_TEST_RETURN_CODE = result.returncode

print(
    "\nFULL TEST:",
    "PASS" if FULL_TEST_RETURN_CODE == 0 else "FAIL",
)
```

Không dùng chỉ mỗi terminal output làm bằng chứng, vì runtime Colab có thể mất.

---

# 11. Chạy smoke test + tự động lưu console

```python
import subprocess
import sys

smoke_log = SAVE_ROOT / "day02_smoke_console.log"

smoke = subprocess.run(
    [
        sys.executable,
        "scripts/day02_smoke_test.py",
    ],
    cwd=REPO_DIR,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

print(smoke.stdout)

smoke_log.write_text(
    smoke.stdout,
    encoding="utf-8",
)

SMOKE_RETURN_CODE = smoke.returncode

print(
    "\nDAY-2 SMOKE:",
    "PASS" if SMOKE_RETURN_CODE == 0 else "FAIL",
)
```

---

# 12. Lưu `day02_report.json` lên Google Drive

```python
import shutil
from pathlib import Path

src_report = REPO_DIR / "day02_report.json"

if src_report.exists():
    shutil.copy2(
        src_report,
        SAVE_ROOT / "day02_report.json",
    )
    print("Saved day02_report.json")
else:
    print("WARNING: day02_report.json not found")
```

---

# 13. Lưu môi trường + Git commit để tái lập

```python
import subprocess
import sys
import torch
from pathlib import Path

git_commit = subprocess.check_output(
    [
        "git",
        "-C",
        str(REPO_DIR),
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

git_status = subprocess.check_output(
    [
        "git",
        "-C",
        str(REPO_DIR),
        "status",
        "--short",
    ],
    text=True,
)

(SAVE_ROOT / "git_state.txt").write_text(
    f"commit={git_commit}\n\n"
    f"status:\n{git_status}",
    encoding="utf-8",
)

pip_freeze = subprocess.check_output(
    [
        sys.executable,
        "-m",
        "pip",
        "freeze",
    ],
    text=True,
)

(SAVE_ROOT / "pip_freeze.txt").write_text(
    pip_freeze,
    encoding="utf-8",
)

env_text = [
    f"torch={torch.__version__}",
    f"cuda_available={torch.cuda.is_available()}",
]

if torch.cuda.is_available():
    env_text.append(
        f"gpu={torch.cuda.get_device_name(0)}"
    )

(SAVE_ROOT / "environment.txt").write_text(
    "\n".join(env_text),
    encoding="utf-8",
)

print("Reproducibility metadata saved.")
```

---

# 14. Tạo summary cuối cùng

```python
import json
from datetime import datetime, timezone

summary = {
    "project": "MS-ILA",
    "stage": "Week-2 / Day-2 architecture QA",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "git_commit": git_commit,

    "day02_smoke": (
        "PASS"
        if SMOKE_RETURN_CODE == 0
        else "FAIL"
    ),

    "full_pytest": (
        "PASS"
        if FULL_TEST_RETURN_CODE == 0
        else "FAIL"
    ),

    "final_status": (
        "PASS"
        if (
            SMOKE_RETURN_CODE == 0
            and FULL_TEST_RETURN_CODE == 0
        )
        else "FAIL"
    ),
}

summary_path = (
    SAVE_ROOT
    / "week02_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)
```

Chỉ chốt:

```text
FINAL STATUS = PASS
```

khi:

```text
day02_smoke == PASS
AND
full_pytest == PASS
```

---

# 15. Backup các source/test quan trọng

Để lần sau biết chính xác code nào đã tạo ra kết quả này:

```python
import shutil
from pathlib import Path

snapshot = SAVE_ROOT / "snapshot"

files_to_backup = [
    "models/contracts.py",
    "models/attention_fusion.py",
    "models/msila.py",

    "tests/test_day02_contracts.py",
    "tests/test_attention_fusion.py",
    "tests/test_msila_day02_head.py",
    "tests/test_gradient_flow.py",
    "tests/test_numerical_stability.py",
    "tests/test_multiview_forward.py",
    "tests/test_tv1_integration.py",

    "scripts/day02_smoke_test.py",
]

for rel in files_to_backup:
    src = REPO_DIR / rel

    if not src.exists():
        continue

    dst = snapshot / rel
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )

print("Snapshot:", snapshot)
```

---

# 16. Cấu trúc kết quả cuối trên Drive

Sau khi chạy xong:

```text
MyDrive/
└── MS-ILA/
    └── week02/
        └── YYYYMMDD_HHMMSS/
            ├── week02_summary.json
            ├── day02_report.json
            ├── day02_smoke_console.log
            ├── full_pytest.log
            ├── full_pytest_junit.xml
            ├── environment.txt
            ├── git_state.txt
            ├── pip_freeze.txt
            └── snapshot/
                ├── models/
                │   ├── contracts.py
                │   ├── attention_fusion.py
                │   └── msila.py
                ├── tests/
                │   ├── test_day02_contracts.py
                │   ├── test_attention_fusion.py
                │   ├── test_msila_day02_head.py
                │   ├── test_gradient_flow.py
                │   ├── test_numerical_stability.py
                │   ├── test_multiview_forward.py
                │   └── test_tv1_integration.py
                └── scripts/
                    └── day02_smoke_test.py
```

Đây là bộ bằng chứng đủ để biết:

```text
code commit nào đã chạy
môi trường nào đã chạy
test nào PASS/FAIL
console output là gì
Day-2 report là gì
source/test tại thời điểm chạy là version nào
```

---

# 17. Cell chạy nhanh cuối cùng

Sau khi đã clone + install dependencies, có thể chạy toàn bộ QA và backup bằng workflow:

```text
1. Mount Drive
2. Tạo SAVE_ROOT
3. Pre-flight
4. python scripts/day02_smoke_test.py
5. python -m pytest tests -v
6. Copy report/log → Drive
7. Save git SHA + pip freeze + environment
8. Save source snapshot
9. Chốt week02_summary.json
```

Không nên chỉ nhìn:

```text
pytest ... PASSED
```

rồi kết luận hoàn thành. Kết quả chính thức phải gắn với:

```text
Git commit
+ test report
+ environment
+ source snapshot
```

để có thể tái lập.

---

# 18. Tiêu chí chốt Week-2 / Day-2

```text
[ ] 6 feature contract đúng
[ ] Context/Local sau TV1 cùng [B,d,h,w]
[ ] Attention weights finite
[ ] sum(attention)=1
[ ] F_fused đúng shape
[ ] Decoder output [B,1,512,512]
[ ] DINO frozen
[ ] Adapter có gradient
[ ] Projection có gradient
[ ] Fusion có gradient
[ ] Decoder có gradient
[ ] NaN = 0
[ ] Inf = 0
[ ] B=1 PASS
[ ] B=2 PASS
[ ] TV1→TV2 integration forward PASS
[ ] TV1→TV2 integration backward PASS
[ ] Day-2 smoke PASS
[ ] Full repository pytest PASS
[ ] Report đã copy lên Drive
[ ] Git/environment/source snapshot đã lưu
```

Khi toàn bộ checkbox trên đạt, có thể khóa checkpoint:

```text
WEEK-2 / DAY-2 ARCHITECTURE QA = PASS
```

Sau đó mới chuyển sang giai đoạn tiếp theo như feature cache, synthetic anomaly và Overfit-16.
