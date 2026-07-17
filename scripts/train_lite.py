import torch
import yaml
import argparse
import os
from model.dualpath_vlm import DualPathVLM
from training.trainer import Trainer
from training.dual_contrastive_loss import DualContrastiveLoss
from data.dataset_loader import DualPathDataset, collate_fn
from transformers import Qwen2_5_VLProcessor
from torch.utils.data import DataLoader, Subset

def train_lite(config_path: str):
    """
    Launches a tiny training run on a laptop for verification.
    Forces a subset of data and single-step logging.
    """
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = config.get("device", "cpu")
    print(f"--- Starting Lite Training on {device.upper()} ---")
    
    # 2. Init Model and Processor
    # Note: On a laptop with limited RAM, this still might take time.
    processor = Qwen2_5_VLProcessor.from_pretrained(config["model_id"])
    model = DualPathVLM(model_id=config["model_id"], device=device)
    
    # 3. Load Dataset (Subset for verification)
    full_dataset = DualPathDataset(processor=processor)
    # Select only top 5 samples
    lite_dataset = Subset(full_dataset, range(min(5, len(full_dataset))))
    
    dataloader = DataLoader(
        lite_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 4. Init Trainer
    loss_fn = DualContrastiveLoss(
        lambda_f=config["lambda_f"],
        lambda_o=config["lambda_o"],
        lambda_reg=config["lambda_reg"]
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=None,
        optimizer=None, # Trainer will init if None
        lr_scheduler=None,
        loss_fn=loss_fn,
        config=config
    )
    
    # 5. Execute 1 Epoch on the subset
    trainer.train()
    print("--- Lite Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/laptop_config.yaml")
    args = parser.parse_args()
    
    train_lite(args.config)
