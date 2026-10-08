#!/usr/bin/env python3
"""
Module: build_evaluation_manifest.py
Purpose: Kiểm tra 3 artifact R0, R1, R2 do TV1 bàn giao và tạo evaluation_manifest.json.
Verify: candidate, category, seed, git commit, resolved config và checkpoint.
Nếu mismatch -> fail ngay.
"""

import sys
import json
import yaml
import hashlib
from pathlib import Path

def get_file_sha256(path: Path) -> str:
    """Tính SHA256 cho checkpoint để verify toàn vẹn file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def verify_and_build_manifest(
    r0_path: Path, 
    r1_path: Path, 
    r2_path: Path, 
    output_path: Path
):
    artifacts = [r0_path, r1_path, r2_path]
    runs_metadata = []

    shared_fields = {
        "candidate": set(),
        "category": set(),
        "git_commit": set(),
        "resolved_config_hash": set()
    }
    
    seeds = []

    for idx, path in enumerate(artifacts):
        if not path.is_dir():
            raise FileNotFoundError(f"Thư mục artifact không tồn tại: {path}")

        config_path = path / "config.yaml"
        checkpoint_path = path / "best.pt"

        # Verify sự tồn tại của file cấu hình và checkpoint
        if not config_path.is_file():
            raise FileNotFoundError(f"Thiếu file config.yaml tại: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Thiếu file checkpoint best.pt tại: {checkpoint_path}")

        config_data = load_yaml(config_path)

        # Trích xuất các field cần verify từ config
        candidate = config_data.get("candidate", {}).get("run_name")
        if not candidate:
            candidate = config_data.get("candidate_id")
            
        category = config_data.get("category")
        seed = config_data.get("seed")
        git_commit = config_data.get("environment", {}).get("git_commit")
        
        # Trích xuất resolved config (scientific_config)
        resolved_config = config_data.get("scientific_config")
        config_json = json.dumps(resolved_config, sort_keys=True)
        
        # Verify checkpoint bằng cách tính hash
        checkpoint_hash = get_file_sha256(checkpoint_path)

        # Đưa vào set để kiểm tra sự đồng nhất giữa 3 bản
        shared_fields["candidate"].add(candidate)
        shared_fields["category"].add(category)
        shared_fields["git_commit"].add(git_commit)
        shared_fields["resolved_config_hash"].add(config_json)
        
        seeds.append(seed)

        runs_metadata.append({
            "artifact_id": f"R{idx}",
            "artifact_path": str(path.resolve()),
            "candidate": candidate,
            "category": category,
            "seed": seed,
            "git_commit": git_commit,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash
        })

    # VERIFY 1: candidate, category, git_commit, resolved_config phải giống nhau trên 3 bản R0/R1/R2
    for field, values in shared_fields.items():
        if len(values) != 1:
            raise ValueError(f"Mismatch ({field}) giữa các artifacts! Các giá trị hiện có: {values}. Fail ngay.")

    # VERIFY 2: seed phải là 3 seed khác biệt (đại diện cho 3 run replicate).
    # (Ghi chú: Nếu pipeline của bạn yêu cầu 3 artifact có CÙNG 1 seed, 
    # hãy đổi điều kiện dưới đây thành len(set(seeds)) != 1)
    if len(set(seeds)) != len(artifacts):
        raise ValueError(f"Mismatch (seed)! Yêu cầu 3 seed phân biệt cho 3 artifacts, nhưng nhận được: {seeds}. Fail ngay.")

    # Tạo evaluation_manifest.json
    manifest = {
        "manifest_version": "1.0",
        "verified_shared_attributes": {
            "candidate": runs_metadata[0]["candidate"],
            "category": runs_metadata[0]["category"],
            "git_commit": runs_metadata[0]["git_commit"]
        },
        "runs": runs_metadata
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"SUCCESS: Đã verify 3 artifacts. Manifest được tạo tại: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python build_evaluation_manifest.py <R0_DIR> <R1_DIR> <R2_DIR> <OUTPUT_JSON>")
        sys.exit(1)
        
    try:
        verify_and_build_manifest(
            Path(sys.argv[1]), 
            Path(sys.argv[2]), 
            Path(sys.argv[3]), 
            Path(sys.argv[4])
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)