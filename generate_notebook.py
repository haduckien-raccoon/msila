import json

notebook = {
    "cells": [],
    "metadata": {
        "colab": {
            "name": "Day04_QA_Report.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_markdown("""
# MS-ILA — Day-04 QA Report — Google Drive version

Notebook này dùng để kiểm tra lại toàn bộ công việc Ngày 4 trên Colab, với dữ liệu/checkpoint lưu trên Google Drive.

**Yêu cầu Ngày 4:**
- **TV1**: Chuẩn hóa Adapter, Grid screen, Candidate factory, Feature cache, Train candidate, Log params, Kiểm tra gradient.
- **TV2**: Implement AU-PRO_0.05, Anomaly maps+masks, Đo Params/VRAM/Latency, Summary 3 pilot category.
- **Cả hai**: Fairness audit, Full screen, Chốt candidate.

**Lưu ý quan trọng về Data thật:**
Ở Ngày 3, quá trình test dùng data dummy để kiểm tra flow. Ở Ngày 4, ta sử dụng data thật. Notebook này sẽ mount Google Drive, giải nén category thật để test.
""")

add_code("""
# ============================================================
# 0. MOUNT GOOGLE DRIVE + CONFIG
# ============================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import os
import sys
import json
import shutil
import subprocess
import tarfile

# ------------------------------------------------------------
# THAY ĐỔI ĐƯỜNG DẪN DRIVE CỦA BẠN TẠI ĐÂY
# ------------------------------------------------------------
PROJECT_DRIVE = Path("/content/drive/MyDrive/[Q3-4] 2026/[S7] Computer Vision/CV-Nhóm 9")

ARCHIVE_DIR = PROJECT_DRIVE / "data"
WEIGHTS_DIR = PROJECT_DRIVE / "weights"

# DINOv3 Checkpoint path
DINOV3_CHECKPOINT = WEIGHTS_DIR / "dinov3_vits16_pretrain_lvd1689m.pth"

# ------------------------------------------------------------
# MS-ILA / Colab local
# ------------------------------------------------------------
REPO_URL = "https://github.com/haduckien-raccoon/msila.git"
BRANCH = "main"
PROJECT_DIR = Path("/content/msila")

DINO_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
DINOV3_REPO = Path("/content/dinov3")

# ------------------------------------------------------------
# CHỌN 1 CATEGORY ĐỂ CHẠY DATA THẬT
# ------------------------------------------------------------
CATEGORY = "fabric"  # Thay đổi thành category bạn muốn test (ví dụ: can, fruit_jelly...)
ALL_CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
assert CATEGORY in ALL_CATEGORIES

LOCAL_DATA_ROOT = Path("/content/msila_data/mvtec_ad_2")
CATEGORY_LOCAL_ROOT = LOCAL_DATA_ROOT / CATEGORY
CATEGORY_ARCHIVE = ARCHIVE_DIR / f"{CATEGORY}.tar.gz"

# Persistent Day-04 artifacts trên Drive
DAY04_DRIVE_ROOT = PROJECT_DRIVE / "msila_day04"
ARTIFACT_ROOT = DAY04_DRIVE_ROOT / "artifacts"
REPORT_DRIVE_DIR = DAY04_DRIVE_ROOT / "reports"

# Feature cache và index (Load data thật từ artifacts Day 03 hoặc Day 04 tùy nơi lưu)
DAY03_DRIVE_ROOT = PROJECT_DRIVE / "msila_day03"
DAY03_ARTIFACT_ROOT = DAY03_DRIVE_ROOT / "artifacts"
FEATURE_CACHE_DIR = DAY03_ARTIFACT_ROOT / "feature_cache"
FEATURE_TRAIN_INDEX = DAY03_ARTIFACT_ROOT / "overfit16" / "train_index.json"

print("================================================================================")
print("PROJECT_DRIVE      :", PROJECT_DRIVE)
print("CATEGORY           :", CATEGORY)
print("CATEGORY_ARCHIVE   :", CATEGORY_ARCHIVE)
print("DINOV3_CHECKPOINT  :", DINOV3_CHECKPOINT)
print("FEATURE_CACHE_DIR  :", FEATURE_CACHE_DIR)
print("================================================================================")
""")

add_markdown("""
## 1. Load data thật từ Drive (Dataset)

Chỉ extract category được chọn để I/O nhanh hơn, không extract tất cả.
""")

add_code("""
# ============================================================
# 1. EXTRACT CATEGORY TỪ GOOGLE DRIVE
# ============================================================
def extract_selected_category(archive_path: Path, destination: Path):
    if not archive_path.is_file():
        print(f"[ERROR] Không tìm thấy archive: {archive_path}")
        return
        
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Đang giải nén {archive_path} vào {destination}...")
    
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.lower().endswith('.png'):
                tf.extract(member, path=destination.parent)
    
    print("Giải nén hoàn tất!")

extract_selected_category(CATEGORY_ARCHIVE, CATEGORY_LOCAL_ROOT)
print("LOCAL CATEGORY ROOT:", CATEGORY_LOCAL_ROOT)
""")

add_markdown("""
## 2. Clone MS-ILA và DINOv3
""")

add_code("""
# ============================================================
# 2. CLONE REPO & CAI ĐẶT MÔI TRƯỜNG
# ============================================================
os.chdir("/content")

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(PROJECT_DIR)], check=True)

if not DINOV3_REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", DINO_REPO_URL, str(DINOV3_REPO)], check=True)

os.environ["DINOV3_REPO"] = str(DINOV3_REPO)
os.environ["DINOV3_CHECKPOINT"] = str(DINOV3_CHECKPOINT)
os.environ["DINOV3_WEIGHTS"] = str(DINOV3_CHECKPOINT)
os.environ["DINOV3_MODEL"] = "dinov3_vits16"

os.chdir(PROJECT_DIR)
print("Current working directory:", Path.cwd())

# Install requirements
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "numpy", "pillow", "matplotlib", "torchmetrics", "safetensors", "einops", "scipy", "pyyaml"], check=True)
if (PROJECT_DIR / "requirements.txt").is_file():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

print("Môi trường và Dependencies đã sẵn sàng.")
""")

add_code("""
# ============================================================
# UTILS: RUN PYTEST
# ============================================================
def run_pytest(args, env=None):
    command = [sys.executable, "-m", "pytest", "-q", "-s", *args]
    print("$", " ".join(command))
    
    completed = subprocess.run(command, cwd=PROJECT_DIR, text=True, capture_output=True, env=env)
    print(completed.stdout)
    if completed.stderr:
        print("STDERR:")
        print(completed.stderr)
        
    return completed.returncode == 0
""")

add_markdown("""
## 3. Kiểm tra công việc TV1 - Feature / Adapter

Bao gồm:
- **Chuẩn hóa Adapter**: `tests/test_residual_adapter_screening.py`
- **Candidate factory**: `tests/test_adapter_factory.py`
- **Train protocol / Cached Dataset**: `tests/test_cached_dataset.py`, `tests/test_screen_adapter.py`
- **Log params / Gradient check**: `tests/test_model_stats.py`, `tests/test_adapter_candidates.py`
""")

add_code("""
# ============================================================
# TV1: TEST SUITE
# ============================================================
print("--- Đang chạy Test cho TV1 ---")
tv1_tests = [
    "tests/test_residual_adapter_screening.py",
    "tests/test_adapter_factory.py",
    "tests/test_cached_dataset.py",
    "tests/test_screen_adapter.py",
    "tests/test_model_stats.py",
    "tests/test_adapter_candidates.py"
]

existing_tv1 = [p for p in tv1_tests if (PROJECT_DIR / p).exists()]

if existing_tv1:
    tv1_pass = run_pytest(existing_tv1)
    print("TV1 Tests PASSED:", tv1_pass)
else:
    print("Không tìm thấy file test cho TV1.")
""")

add_markdown("""
## 4. Kiểm tra công việc TV2 - Evaluation

Bao gồm:
- **AU-PRO_0.05**: `tests/test_aupro_reference.py`
- **Anomaly Maps + Masks (Evaluator)**: `tests/test_evaluator.py`
- **Đo Params, peak VRAM, inference latency**: `tests/test_efficiency.py`
""")

add_code("""
# ============================================================
# TV2: TEST SUITE
# ============================================================
print("--- Đang chạy Test cho TV2 ---")
tv2_tests = [
    "tests/test_aupro_reference.py",
    "tests/test_evaluator.py",
    "tests/test_efficiency.py"
]

existing_tv2 = [p for p in tv2_tests if (PROJECT_DIR / p).exists()]

if existing_tv2:
    tv2_pass = run_pytest(existing_tv2)
    print("TV2 Tests PASSED:", tv2_pass)
else:
    print("Không tìm thấy file test cho TV2.")
""")

add_markdown("""
## 5. Fairness Audit (Cả Hai)

Kiểm tra config chỉ thay đổi r, d giữa các run (Test fairness config audit).
""")

add_code("""
# ============================================================
# FAIRNESS AUDIT TEST
# ============================================================
fairness_test = "tests/test_day04_fairness.py"
if (PROJECT_DIR / fairness_test).exists():
    fair_pass = run_pytest([fairness_test])
    print("Fairness Audit PASSED:", fair_pass)
else:
    print(f"Không tìm thấy file: {fairness_test}")
""")

add_markdown("""
## 6. Tổng hợp & Chạy Full Screen (Cả Hai)

Kiểm tra script tổng hợp kết quả (`scripts/eval_day04.py`).
Vì script này yêu cầu artifacts đã được tạo từ quá trình huấn luyện, cell dưới đây gọi lệnh help để đảm bảo script import thành công và thực thi được.
Để chạy full experiment trên data thật, hãy gọi script bash / python tương ứng trong repository.
""")

add_code("""
# ============================================================
# AGGREGATION SCRIPT SMOKE TEST
# ============================================================
eval_script = PROJECT_DIR / "scripts/eval_day04.py"
if eval_script.exists():
    print(f"Chạy: python {eval_script} -h")
    result = subprocess.run([sys.executable, str(eval_script), "-h"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("Script Day 04 tồn tại và thực thi được.")
else:
    print(f"Không tìm thấy script {eval_script}.")
""")

add_markdown("""
## 7. Demo Train 1 Candidate bằng DATA THẬT (Optional)

Sử dụng `screen_adapter.py` để train 1 candidate với DATA THẬT (nếu feature cache đã tồn tại trên Drive).
Thay vì dummy random input, model sẽ tiêu thụ cache thật.
""")

add_code("""
# ============================================================
# DEMO SCREEN ADAPTER WITH REAL CACHE
# ============================================================
if FEATURE_CACHE_DIR.exists() and FEATURE_TRAIN_INDEX.exists():
    env = os.environ.copy()
    env["FEATURE_CACHE_DIR"] = str(FEATURE_CACHE_DIR)
    env["FEATURE_TRAIN_INDEX"] = str(FEATURE_TRAIN_INDEX)
    
    print("Sử dụng Feature Cache THẬT tại:", FEATURE_CACHE_DIR)
    # Nếu file test dưới đây hỗ trợ đọc real cache qua env biến (như Day 3), nó sẽ train trên data thật.
    test_file = "tests/test_day03_cached_train_step.py::test_external_real_cache_one_training_step"
    if (PROJECT_DIR / "tests/test_day03_cached_train_step.py").exists():
        run_pytest([test_file], env=env)
    else:
        print("Không tìm thấy bài test kiểm tra cache thật.")
else:
    print("Không tìm thấy Feature Cache thật trên Drive. Bỏ qua chạy test real data.")
""")

with open("/media/haduckien/E/Studying/HK7/ComputerVision /projects/msila/notebooks/Day04_QA_Report.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Notebook Day04_QA_Report.ipynb created successfully.")
