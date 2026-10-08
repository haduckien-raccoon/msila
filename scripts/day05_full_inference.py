#!/usr/bin/env python3
"""
Day 5 Full Inference Script (TV2-E1)
Yêu cầu:
- Không train, chỉ inference.
- Load lần lượt best.pt của R0, R1, R2.
- Chạy trên một DEV split duy nhất, đảm bảo cùng preprocess/postprocess.
- Xuất map full resolution (.npy, .tiff).
- Assert đúng HxW ảnh gốc và đủ số lượng sample.
"""

import argparse
import torch
import numpy as np
import tifffile
import yaml
from pathlib import Path
from tqdm import tqdm

# Import các components từ inference pipeline đã có
from src.data.loader import MVTecAD2HighResDataset
from src.models.dinov3_extractor import DINOv3FeatureExtractor
from src.train.screen_representation import Day05RepresentationModel

def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_inference(args):
    device = torch.device(args.device)
    artifacts = [args.r0, args.r1, args.r2]
    
    print(f"Loading DEV Dataset (split={args.split}) từ {args.data_root}")
    # Đảm bảo chung preprocessing (mean/std DINOv3) và load đúng ảnh gốc
    dataset = MVTecAD2HighResDataset(data_root=args.data_root, split=args.split)
    num_samples = len(dataset)
    print(f"Dataset có tổng cộng {num_samples} samples.")
    
    if num_samples == 0:
        raise ValueError("Dataset trống, vui lòng kiểm tra split và data_root.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Khởi tạo DINOv3 Backbone (Dùng chung cho tất cả candidates để đảm bảo chuẩn preprocess)
    print(f"Khởi tạo DINOv3 Feature Extractor...")
    backbone = DINOv3FeatureExtractor(
        repo_dir=args.dinov3_repo,
        model_name=args.model_name,
        blocks=(4, 8, 12),
        norm=True
    ).eval().to(device)
    
    # Duyệt lần lượt qua các artifact (R0, R1, R2)
    for run_dir in artifacts:
        print(f"\n--- Bắt đầu xử lý artifact: {run_dir.name} ---")
        candidate_id = run_dir.name
        candidate_out_dir = args.output_dir / candidate_id
        candidate_out_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Thiếu file config.yaml tại {run_dir}")
            
        run_cfg = load_config(config_path)
        
        candidate_meta = run_cfg.get("candidate", {})
        adapter_r = candidate_meta.get("r", 128)
        adapter_d = candidate_meta.get("d", 512)
        in_channels = run_cfg.get("in_channels", 384) 
        
        # 2. Khởi tạo Day05RepresentationModel từ config của candidate tương ứng
        day05_model = Day05RepresentationModel(
            day05_config=run_cfg,
            candidate=candidate_id,
            in_channels=in_channels,
            adapter_r=adapter_r,
            adapter_d=adapter_d
        )
        
        checkpoint_path = run_dir / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Thiếu file best.pt tại {run_dir}")
            
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state:
            day05_model.load_state_dict(state["model"], strict=True)
        else:
            day05_model.load_state_dict(state, strict=True)
            
        day05_model.to(device).eval()
        
        processed_count = 0
        
        with torch.inference_mode():
            for idx in tqdm(range(num_samples), desc=f"Inference {candidate_id}"):
                sample = dataset[idx]
                
                # Image đã được preprocess & norm theo tiêu chuẩn DINOv3
                image_norm = sample["image_norm"].unsqueeze(0).to(device) 
                
                # Lấy kích thước ảnh gốc HxW
                original_h, original_w = sample["image"].shape[1], sample["image"].shape[2]
                
                # Trích xuất đặc trưng Backbone
                features = backbone(image_norm)
                
                H, W = image_norm.shape[2], image_norm.shape[3]
                
                # Tạo mock batch cho Day 5 Model để chạy full pipeline như training pipeline
                mock_batch = {
                    "local_b4": features["b4"],
                    "local_b8": features["b8"],
                    "local_b12": features["b12"],
                    "context_b4": features["b4"],
                    "context_b8": features["b8"],
                    "context_b12": features["b12"],
                    "meta": [{
                        "geometry": {
                            "local_to_context": torch.eye(3).tolist(),
                            "local_input_hw": [H, W],
                            "context_input_hw": [H, W],
                        }
                    }],
                    # Truyền mask dummy để BasicDecoder tự động nội suy (restore) về kích thước ảnh gốc
                    "mask": torch.empty((1, original_h, original_w), device=device)
                }
                
                # Inference model TV2-E1 
                logits, _ = day05_model(mock_batch)
                
                # Chuyển đổi thành xác suất (Probability) - chung post-processing
                anomaly_prob = torch.sigmoid(logits)
                anomaly_map_np = anomaly_prob.squeeze().cpu().numpy() 
                
                # KIỂM TRA ĐẦU RA 1: Output phải đúng kích thước HxW của ảnh/GT gốc
                assert anomaly_map_np.shape == (original_h, original_w), \
                    f"Mismatch shape: Map {anomaly_map_np.shape} vs Original {(original_h, original_w)}"
                
                # Lưu dưới dạng .npy và .tiff
                meta = sample["meta"]
                category = meta["category"]
                
                out_base = candidate_out_dir / f"{category}_{idx:04d}"
                np.save(f"{out_base}.npy", anomaly_map_np)
                tifffile.imwrite(f"{out_base}.tiff", anomaly_map_np)
                
                processed_count += 1
                
        # KIỂM TRA ĐẦU RA 2: Đảm bảo không bị sót sample nào trong DEV split
        assert processed_count == num_samples, \
            f"Thiếu sample! Đã xử lý {processed_count}/{num_samples} cho candidate {candidate_id}"
            
        print(f"Hoàn thành {candidate_id}: {processed_count} samples. Kết quả xuất tại {candidate_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", type=Path, required=True, help="Đường dẫn đến thư mục R0")
    parser.add_argument("--r1", type=Path, required=True, help="Đường dẫn đến thư mục R1")
    parser.add_argument("--r2", type=Path, required=True, help="Đường dẫn đến thư mục R2")
    parser.add_argument("--data-root", type=Path, required=True, help="Thư mục gốc chứa MVTec AD 2")
    parser.add_argument("--split", type=str, default="dev_synthetic", help="DEV split cần inference")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day05_inference"))
    parser.add_argument("--dinov3-repo", type=Path, default=Path("/content/dinov3"))
    parser.add_argument("--model-name", type=str, default="dinov3_vits16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    run_inference(args)