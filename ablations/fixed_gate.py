import torch
import torch.nn as nn
from model.dualpath_vlm import DualPathVLM
from evaluation.eval_pope import evaluate_pope
from transformers import Qwen2_5_VLProcessor
from typing import Dict, Any, List

class FixedGateAblation(DualPathVLM):
    """
    Ablation model: Forces gate g = 0.5 (equal weighting of both pathways).
    This tests the hypothesis that a dynamic gate is better than fixed arbitration.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        super().__init__(model_id, device)
        print("--- Initialized Ablation: Fixed Gate (g=0.5) ---")
        
    def generate(self, *args, **kwargs):
        """
        Intercepts gated output and forces gate g = 0.5 internally.
        """
        original_gating = self.gating
        mock_gating = lambda h, sw, iw, fo, oo: (0.5 * fo + 0.5 * oo, torch.full_like(iw, 0.5))
        self.gating = mock_gating
        
        results = super().generate(*args, **kwargs)
        
        # Restore original gating
        self.gating = original_gating
        return results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FixedGateAblation(device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    print("Evaluating Fixed Gate Ablation on POPE...")
    evaluate_pope(model, processor, device)
