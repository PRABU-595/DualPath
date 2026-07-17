import torch
import torch.nn as nn
from model.dualpath_vlm import DualPathVLM
from evaluation.eval_pope import evaluate_pope
from transformers import Qwen2_5_VLProcessor
from typing import Dict, Any, List

class NoGateBaseline(DualPathVLM):
    """
    Ablation model: No DualPath modules are active.
    This is identical to the frozen backbone (vanilla model).
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        super().__init__(model_id, device, baseline_mode=True)
        print("--- Initialized Ablation: No Gate (Vanilla Baseline) ---")
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NoGateBaseline(device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    print("Evaluating No Gate (Vanilla) Baseline on POPE...")
    evaluate_pope(model, processor, device, baseline_mode=True)
