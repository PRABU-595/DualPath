import torch
import torch.nn as nn
from transformers import Qwen2_5_VLProcessor
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os
import argparse
from model.dualpath_vlm import DualPathVLM

class RegressionEvaluator:
    """
    Standard Benchmark Runner to verify DualPath-VLM does not regress
    on general multimodal tasks like VQAv2, MMBench, and TextVQA.
    """
    def __init__(self, model: nn.Module, processor: Any, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def evaluate_vqa(self, dataset_id: str, split: str = "validation", limit: int = 500) -> float:
        """Evaluates accuracy on VQA datasets."""
        print(f"Loading {dataset_id}...")
        ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
        limit = min(limit, len(ds))
        
        correct = 0
        pbar = tqdm(range(limit), desc=f"Evaluating {dataset_id}")
        
        for i in pbar:
            item = ds[i]
            image = item["image"]
            question = item["question"]
            answer = item.get("multiple_choice_answer", item.get("answer", ""))
            
            # Formulate prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question} Answer in one or two words."}
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                    max_new_tokens=10
                )
            pred = self.processor.batch_decode(out, skip_special_tokens=True)[0].lower().strip()
            
            if pred == str(answer).lower().strip():
                correct += 1
            
            pbar.set_postfix({"Acc": f"{correct/(i+1):.4f}"})
            
        return correct / limit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(model_id=args.model_id, device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id)
    
    evaluator = RegressionEvaluator(model, processor, device)
    
    results = {}
    # 1. VQAv2
    results["VQAv2"] = evaluator.evaluate_vqa("HuggingFaceM4/VQAv2", limit=args.limit)
    # 2. TextVQA
    results["TextVQA"] = evaluator.evaluate_vqa("textvqa", limit=args.limit)
    
    print("\n[Regression Analysis Summary]")
    for bm, acc in results.items():
        print(f"{bm:<15}: {acc*100:.2f}%")
        
    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/regression_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
