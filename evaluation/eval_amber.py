import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os
from PIL import Image
from typing import Dict, Any, List

class AmberEvaluator:
    """
    Evaluates the model on the AMBER benchmark (Adversarial Multi-modal Benchmark).
    Measures hallucination rate, attribute recall, and logical consistency.
    """
    def __init__(self, model: nn.Module, processor: Any, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
    def evaluate(self, dataset_path: str = "openbmb/AMBER", split: str = "test") -> Dict[str, Any]:
        """
        Runs the full AMBER evaluation suite.
        """
        self.model.eval()
        amber_data = load_dataset(dataset_path, split=split)
        
        results = []
        pbar = tqdm(amber_data, desc="Evaluating AMBER")
        
        for item in pbar:
            image = item["image"]
            query = item["query"]
            # AMBER uses specific tasks (existence, attribute, etc.)
            task = item.get("task", "general")
            ground_truth = item.get("ground_truth", "")
            
            # 1. Run Generation
            inputs = self.processor(text=[query], images=[image], return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=50)
            pred_text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            
            results.append({
                "id": item.get("id", "0"),
                "task": task,
                "query": query,
                "prediction": pred_text,
                "ground_truth": ground_truth
            })
            
        # 2. Save results for AMBER off-line script
        # AMBER typically uses a specialized GPT-4 or rule-based evaluator on the predictions.
        # Here we save formatted output for the AMBER evaluation script.
        output_file = "amber_predictions.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"\nSaved AMBER predictions to {output_file}. Please run the official AMBER scoring script for final metrics.")
            
        return {"predictions_count": len(results), "output_file": output_file}

if __name__ == "__main__":
    from model.dualpath_vlm import DualPathVLM
    from transformers import Qwen2_5_VLProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    evaluator = AmberEvaluator(model, processor, device)
    evaluator.evaluate()
