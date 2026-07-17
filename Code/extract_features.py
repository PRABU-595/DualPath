"""
DualPath-VLM: Feature Extraction Script (Turbo Mode)
Extracts hidden states and attention signals from the 7B model to enable
rapid training (300+ epochs) on a laptop without the full backbone.

Usage:
    python scripts/extract_features.py --config configs/base_config.yaml --image_dir data/coco/val2017
"""

import os
import sys

# Ensure the project root is in PYTHONPATH when executing this file directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import json
import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from model.dualpath_vlm import DualPathVLM
from data.dataset_loader import DualPathLocalDataset, collate_fn_v2
from utils.device_utils import get_device, get_dtype

def extract():
    parser = argparse.ArgumentParser(description="Feature Extraction for Laptop Training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/extracted_features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device()
    dtype = get_dtype(device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Initializing DualPath-VLM on {device}...")
    model = DualPathVLM(
        model_id=config["model"]["backbone_id"],
        device=device,
        use_4bit=config["model"].get("use_4bit", True),
        gradient_checkpointing=False # Not needed for extraction
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(config["model"]["backbone_id"], trust_remote_code=True)

    # Setup datasets
    fab_jsonl = config["data"]["datasets"]["fabrication_triplets"]
    om_jsonl = config["data"]["datasets"]["omission_triplets"]

    datasets = {
        "fabrication": DualPathLocalDataset(fab_jsonl, args.image_dir, processor),
        "omission": DualPathLocalDataset(om_jsonl, args.image_dir, processor)
    }

    for name, ds in datasets.items():
        print(f"\nExtracting {name} features...")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_v2)
        
        extracted_data = []
        count = 0
        
        for i, batch in enumerate(tqdm(loader)):
            if args.limit and count >= args.limit:
                break
                
            with torch.no_grad():
                # We need to run the forward pass up to the point where 
                # hidden states and attention weights are available.
                # DualPathVLM.forward already handles the monitor setup.
                
                # Move batch to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["inputs"].items()}
                
                # Run forward
                # We ensure output_attentions=True to catch the signals
                outputs = model.backbone(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # The attention monitor is updated via hooks during the forward pass
                fab_signal = model.monitor.get_fabrication_signal() # [batch, num_layers, num_heads]
                drift_signal = model.monitor.get_drift_signal()     # [batch, 1]
                
                # Last hidden state (all tokens)
                full_hidden_states = outputs.hidden_states[-1] 
                
                # Identify image tokens
                start, end = model.monitor.image_token_range
                image_embeddings = full_hidden_states[:, start:end, :]
                
                # Final hidden states for the current prediction step (usually the last token)
                hidden_states = full_hidden_states[:, -1:, :]
                
                # Save essential data
                sample = {
                    "hidden_states": hidden_states.cpu(),
                    "image_embeddings": image_embeddings.cpu(),
                    "fab_signal": fab_signal.cpu(),
                    "drift_signal": drift_signal.cpu(),
                    "labels": inputs["labels"].cpu(),
                    "fabricated_tokens": batch.get("fabricated_tokens"),
                    "omitted_tokens": batch.get("omitted_tokens")
                }
                extracted_data.append(sample)
                count += args.batch_size
                
                # Save in chunks of 500 to prevent OOM in system memory
                if len(extracted_data) >= 500:
                    chunk_path = os.path.join(args.output_dir, f"{name}_chunk_{i//500}.pt")
                    torch.save(extracted_data, chunk_path)
                    extracted_data = []

        # Save remaining
        if extracted_data:
            chunk_path = os.path.join(args.output_dir, f"{name}_final.pt")
            torch.save(extracted_data, chunk_path)

    print(f"\n[✓] Extraction complete. Features saved to {args.output_dir}")

if __name__ == "__main__":
    extract()
