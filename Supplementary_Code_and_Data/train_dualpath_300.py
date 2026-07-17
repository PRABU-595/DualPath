"""
DualPath-VLM: Unified SMART Trainer (300 Epochs)
Auto-detects environment (GPU/CPU) and adapts strategy accordingly.

COMMAND TO RUN:
    python train_dualpath_300.py --image_dir data/coco/val2017

ONE CODE. ONE COMMAND. SMART HARDWARE HANDLING.
"""

import os
import torch
import argparse
import yaml
from pathlib import Path

# Local imports
from utils.device_utils import get_device, get_dtype
from scripts.extract_features import extract as run_extraction
from training.laptop_trainer import main as run_training

def start_pipeline():
    parser = argparse.ArgumentParser(description="DualPath-VLM Smart Orchestrator")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to COCO images")
    parser.add_argument("--feature_dir", type=str, default="data/extracted_features", help="Where to save/load features")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file")
    parser.add_argument("--laptop_config", type=str, default="configs/laptop_config.yaml", help="Laptop training config")
    parser.add_argument("--sanity_check", action="store_true", help="Run a 1-epoch sanity check with mock data")
    args = parser.parse_args()

    # 1. SMART DEVICE DETECTION
    device = get_device()
    dtype = get_dtype(device)
    print(f"\n[SMART ENVIRONMENT DETECTION]")
    print(f"  > Detected Hardware : {device.upper()}")
    print(f"  > Target Precision : {dtype}")

    # 2. DECIDE MODE: Extraction (GPU Server) vs Training (Laptop/Anywhere)
    feature_path = Path(args.feature_dir)
    has_features = feature_path.exists() and any(feature_path.glob("*.pt"))
    
    # If we have a real GPU, we can handle the extraction if needed
    is_gpu = (device == "cuda")

    import subprocess
    import sys
    python_exe = sys.executable

    if args.sanity_check:
        print(f"\n[SANITY CHECK MODE] Running 1 epoch on your laptop...")
        print(f"  > Creating dummy features for testing...")
        # Always recreate dummy data for sanity check to ensure compatibility
        dummy_data = [{
            "hidden_states": torch.randn(1, 1, 3584),
            "image_embeddings": torch.randn(1, 4, 3584),
            "fab_signal": torch.randn(1, 28, 16),
            "drift_signal": torch.randn(1, 1),
            "labels": torch.tensor([[100, 200]]),
            "fabricated_tokens": [[100]],
            "omitted_tokens": [[200]]
        }]
        torch.save(dummy_data, f"{args.feature_dir}/fabrication_dummy.pt")
        torch.save(dummy_data, f"{args.feature_dir}/omission_dummy.pt")
        
        args.laptop_config = "configs/laptop_config.yaml"
        # Force tiny batch for laptop sanity
        subprocess.check_call([
            python_exe, "training/laptop_trainer.py",
            "--config", args.laptop_config,
            "--feature_dir", args.feature_dir
        ])
        return

    if (not has_features or args.force_extract) and is_gpu:
        print(f"\n[PHASE 1] Starting 'Turbo' Feature Extraction on GPU...")
        subprocess.check_call([
            python_exe, "scripts/extract_features.py",
            "--config", args.config,
            "--image_dir", args.image_dir,
            "--output_dir", args.feature_dir
        ])
        print(f"[✓] Features extracted and saved to {args.feature_dir}")
        has_features = True
    elif not has_features and not is_gpu:
        print(f"\n[ERROR] No pre-extracted features found at {args.feature_dir} and no GPU detected to run extraction.")
        print("Please run this command first on your GPU server to generate the features.")
        return

    # 3. RUN THE 300-EPOCH TRAINING (SMART ADAPTS TO GPU OR CPU)
    if has_features:
        print(f"\n[PHASE 2] Starting Smart 300-Epoch Training...")
        print(f"  > Mode: Training on {device.upper()}")
        print(f"  > Strategy: Feature-Caching (Lightning Fast)")
        
        # We run the training. The LaptopTrainer is designed to adapt to CUDA/MPS/CPU
        subprocess.check_call([
            python_exe, "training/laptop_trainer.py",
            "--config", args.laptop_config,
            "--feature_dir", args.feature_dir
        ])
        print(f"\n[✓] ALL STEPS COMPLETE.")
        print(f"Check outputs/results/ for the final training curve.")

if __name__ == "__main__":
    start_pipeline()
