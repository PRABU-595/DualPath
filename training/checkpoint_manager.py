import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    Saves and loads DualPathVLM checkpoints, including LoRA adapters and pathway weights.
    Ensures that the entire modular architecture is persisted.
    """
    def __init__(self, save_dir: str = "./checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def find_latest_checkpoint(self):
        if not os.path.exists(self.save_dir): return None
        ckpt_dirs = [d for d in os.listdir(self.save_dir) if d.startswith("checkpoint-")]
        if not ckpt_dirs: return None
        # Sort by the number at the end
        latest = max(ckpt_dirs, key=lambda x: int(x.split("-")[-1]))
        return os.path.join(self.save_dir, latest)

    def save_checkpoint(self, 
                        model: nn.Module, 
                        step: int, 
                        optimizer: Optional[torch.optim.Optimizer] = None, 
                        scheduler: Optional[Any] = None, 
                        is_best: bool = False):
        """
        Saves a model checkpoint.
        
        Args:
            model: The DualPathVLM model.
            step: Current training step.
            optimizer: Optional optimizer state.
            scheduler: Optional LR scheduler state.
            is_best: True to save as the best performing model.
        """
        checkpoint_name = f"checkpoint-{step}" if not is_best else "best_model"
        save_path = os.path.join(self.save_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Save LoRA adapters (uses PEFT's save_pretrained)
        # Note: model.backbone is the PeftModel
        model.backbone.save_pretrained(save_path)
        
        # 2. Save DualPath module weights (custom modules)
        dualpath_weights = {
            "pathway_f": model.pathway_f.state_dict(),
            "pathway_o": model.pathway_o.state_dict(),
            "gating": model.gating.state_dict()
        }
        torch.save(dualpath_weights, os.path.join(save_path, "dualpath_weights.pt"))
        
        # 3. Save training states
        states = {"step": step}
        if optimizer:
            states["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            states["scheduler_state_dict"] = scheduler.state_dict()
            
        torch.save(states, os.path.join(save_path, "training_states.pt"))
        
        print(f"--- Saved DualPath-VLM checkpoint: {save_path} ---")

    def load_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """
        Loads a checkpoint into the model.
        
        Args:
            model: The DualPathVLM model.
            checkpoint_path: Path to the checkpoint folder.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
            
        # 1. Load LoRA weights
        # This will load the adapters into the backbone
        model.backbone.load_adapter(checkpoint_path, adapter_name="default")
        
        # 2. Load DualPath module weights
        dualpath_path = os.path.join(checkpoint_path, "dualpath_weights.pt")
        if os.path.exists(dualpath_path):
            weights = torch.load(dualpath_path, map_location=next(model.parameters()).device)
            model.pathway_f.load_state_dict(weights["pathway_f"])
            model.pathway_o.load_state_dict(weights["pathway_o"])
            model.gating.load_state_dict(weights["gating"])
            
        print(f"--- Successfully loaded DualPath-VLM checkpoint from {checkpoint_path} ---")

if __name__ == "__main__":
    # Internal test for manager logic
    manager = CheckpointManager(save_dir="./test_checkpoints")
    print("CheckpointManager initialized.")
