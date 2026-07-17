import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os
import re
from typing import Dict, Any, List

class MMHalEvaluator:
    """
    Evaluates the model on MMHal-Bench.
    Measures halluncination rates across different question categories.
    """
    def __init__(self, model: nn.Module, processor: Any, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
    def evaluate(self, dataset_path: str = "openbmb/MMHal-Bench", split: str = "test") -> Dict[str, Any]:
        """
        Runs the MMHal-Bench evaluation suite.
        """
        self.model.eval()
        mmhal_data = load_dataset(dataset_path, split=split)
        
        results = []
        pbar = tqdm(mmhal_data, desc="Evaluating MMHal-Bench")
        
        for item in pbar:
            image = item["image"]
            question = item["question"]
            category = item.get("category", "general")
            ground_truth = item.get("ground_truth", "")
            
            # 1. Run Generation
            inputs = self.processor(text=[question], images=[image], return_tensors="pt").to(self.device)
            # Use model custom generate
            out = self.model.generate(**inputs, max_new_tokens=100)
            pred_text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            
            results.append({
                "id": str(item.get("id", "0")),
                "category": category,
                "question": question,
                "prediction": pred_text,
                "ground_truth": ground_truth
            })
            
        # 2. Output Analysis
        # MMHal-Bench usually compares predictions against ground truth using GPT-4 scores.
        # Here we save formatted results for the metric calculation script.
        output_file = "mmhal_predictions.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"\nSaved MMHal predictions to {output_file}. Analysis of open-ended generation usually requires GPT-eval.")
            
        return {"samples": len(results), "output_file": output_file}

if __name__ == "__main__":
    from model.dualpath_vlm import DualPathVLM
    from transformers import Qwen2_5_VLProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    evaluator = MMHalEvaluator(model, processor, device)
    evaluator.evaluate()
