import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os
from typing import Dict, Any, List

def evaluate_pope(model: nn.Module, 
                  processor: Any, 
                  device: str, 
                  pope_dataset_path: str = "RUCAIBox/POPE",
                  baseline_mode: bool = False,
                  sample_limit: int = 1000) -> Dict[str, Any]:
    """
    Evaluates the model on the POPE benchmark.
    Focuses on 'adversarial' as the most rigorous split.
    """
    if hasattr(model, "baseline_mode"):
        model.baseline_mode = baseline_mode
        
    model.eval()
    # High-impact split: adversarial
    splits = ["adversarial", "popular", "random"]
    results = {}
    
    for split in splits:
        pope_data = load_dataset(pope_dataset_path, split=split, trust_remote_code=True)
        # Limit samples for faster turnaround while maintaining statistical significance
        data_to_eval = pope_data.select(range(min(sample_limit, len(pope_data))))
        
        tp, fp, tn, fn = 0, 0, 0, 0
        pbar = tqdm(data_to_eval, desc=f"POPE {split} [DualPath={not baseline_mode}]")
        
        for item in pbar:
            image = item["image"]
            question = item["question"]
            answer = item.get("answer", item.get("label", ""))
            
            # Map ground truth to strict binary
            if isinstance(answer, (int, float)):
                answer = "yes" if int(answer) == 1 else "no"
            else:
                answer = str(answer).lower()
                answer = "no" if answer in ["0", "no", "false"] else "yes"
            
            # Use Qwen2.5-VL prompt format if possible
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question + " Please answer with 'yes' or 'no'."}
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            
            # Custom generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                    max_new_tokens=10
                )
            
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
            
            # Strict prediction extraction
            pred = "yes" if "yes" in output_text else "no"
            
            if pred == "yes" and answer == "yes": tp += 1
            elif pred == "yes" and answer == "no": fp += 1
            elif pred == "no" and answer == "no": tn += 1
            elif pred == "no" and answer == "yes": fn += 1
                
        # Metrics
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / (total + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        results[split] = {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        }

    # Summary Output
    print(f"\nPOPE Results [Baseline={baseline_mode}]")
    print("-" * 50)
    for split, m in results.items():
        print(f"{split.upper():<12} | Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f}")
    
    os.makedirs("outputs/evaluation", exist_ok=True)
    suffix = "baseline" if baseline_mode else "dualpath"
    with open(f"outputs/evaluation/pope_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    import argparse
    from model.dualpath_vlm import DualPathVLM
    from transformers import Qwen2_5_VLProcessor
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(model_id=args.model_id, device=device, baseline_mode=args.baseline)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id)
    
    evaluate_pope(model, processor, device, baseline_mode=args.baseline, sample_limit=args.limit)
