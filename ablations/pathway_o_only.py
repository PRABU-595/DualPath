import torch
import torch.nn as nn
from model.dualpath_vlm import DualPathVLM
from evaluation.eval_pope import evaluate_pope
from transformers import Qwen2_5_VLProcessor
from typing import Dict, Any, List

class PathwayOOnly(DualPathVLM):
    """
    Ablation model: only Pathway O (Omission Correction) is active.
    This is achieved by overriding the forward/generate logic to force gate g = 1.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        super().__init__(model_id, device)
        print("--- Initialized Ablation: Pathway O Only (Omission Correction) ---")
        
    def generate(self, *args, **kwargs):
        """
        Intercepts gated output and forces gate g = 1 internally.
        """
        # For ablation, we temporarily mock the gating MLP to return ones
        original_gating = self.gating
        mock_gating = lambda h, sw, iw, fo, oo: (oo, torch.ones_like(iw))
        self.gating = mock_gating
        
        results = super().generate(*args, **kwargs)
        
        # Restore original gating
        self.gating = original_gating
        return results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PathwayOOnly(device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    print("Evaluating Pathway O Only Ablation on POPE...")
    evaluate_pope(model, processor, device)
