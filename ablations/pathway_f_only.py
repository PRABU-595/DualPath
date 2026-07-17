import torch
import torch.nn as nn
from model.dualpath_vlm import DualPathVLM
from evaluation.eval_pope import evaluate_pope
from transformers import Qwen2_5_VLProcessor
from typing import Dict, Any, List

class PathwayFOnly(DualPathVLM):
    """
    Ablation model: only Pathway F (Fabrication Suppression) is active.
    This is achieved by overriding the forward/generate logic to force gate g = 0.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        super().__init__(model_id, device)
        print("--- Initialized Ablation: Pathway F Only (Fabrication Suppression) ---")
        
    def generate(self, *args, **kwargs):
        """
        Intercepts gated output and forces gate g = 0 internally.
        We can achieve this by mocking the gating module or by 
        wrapping the generate call to force a fixed gate.
        """
        # For ablation, we temporarily mock the gating MLP to return zeros
        original_gating = self.gating
        mock_gating = lambda h, sw, iw, fo, oo: (fo, torch.zeros_like(iw))
        self.gating = mock_gating
        
        results = super().generate(*args, **kwargs)
        
        # Restore original gating
        self.gating = original_gating
        return results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PathwayFOnly(device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    print("Evaluating Pathway F Only Ablation on POPE...")
    evaluate_pope(model, processor, device)
