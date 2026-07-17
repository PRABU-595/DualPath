import torch
import json
import os
import argparse
from tqdm import tqdm
from model.dualpath_vlm import DualPathVLM
from transformers import Qwen2_5_VLProcessor
from evaluation.eval_chair import ChairEvaluator
from evaluation.eval_object_recall import ObjectRecallEvaluator

def main():
    parser = argparse.ArgumentParser(description="Sweep tau_O threshold sensitivity")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(model_id=args.model_id, device=device)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    sweep_results = []

    for tau in thresholds:
        print(f"\n--- Sweeping Threshold tau_O = {tau} ---")
        model.drift_tracker.threshold = tau
        
        # Run mini-evals
        chair_eval = ChairEvaluator(model, processor, device)
        chair_metrics = chair_eval.evaluate(num_samples=args.limit)
        
        recall_eval = ObjectRecallEvaluator(model, processor, device)
        recall_metrics = recall_eval.evaluate("data/vg_objects_val.json", "data/vg/images", num_samples=args.limit)
        
        sweep_results.append({
            "threshold": tau,
            "CHAIRi": chair_metrics["CHAIRi"] * 100,
            "Recall": recall_metrics["mean_recall"] * 100
        })

    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/threshold_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
        
    print(f"Sweep complete. Results saved to outputs/evaluation/threshold_sweep.json")

if __name__ == "__main__":
    main()
