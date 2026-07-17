import torch
import torch.nn as nn
from transformers import Qwen2_5_VLProcessor
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os
import argparse
from PIL import Image
from typing import Dict, Any, List
from model.dualpath_vlm import DualPathVLM

class InferenceEngine:
    """
    Core engine to run DualPathVLM inference and extract mechanistic data.
    Saves a comprehensive JSON log for plotting and metrics.
    """
    def __init__(self, model_id: str, checkpoint_path: str = None, device: str = "cuda"):
        self.device = device
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = DualPathVLM(model_id=model_id, device=device, use_4bit=True)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            # Use CheckpointManager to load LoRA adapters and custom weights correctly from the directory
            from training.checkpoint_manager import CheckpointManager
            manager = CheckpointManager()
            manager.load_checkpoint(self.model, checkpoint_path)
        
        self.model.to(device).eval()

    @torch.no_grad()
    def run_inference(self, 
                      dataset_name: str, 
                      split: str, 
                      output_file: str, 
                      limit: int = 1000, 
                      ablation_mode: str = "learned"):
        """
        Runs inference on a dataset and logs all internal signals.
        """
        print(f"Running inference on {dataset_name} ({split}) | Mode: {ablation_mode}")
        
        # Load dataset (standard HF paths for COCO/VG)
        if dataset_name == "coco":
            ds = load_dataset("detection-datasets/coco", split=split, trust_remote_code=True)
        elif dataset_name == "pope":
            ds = load_dataset("RUCAIBox/POPE", split=split, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        limit = min(limit, len(ds))
        results = []

        for i in tqdm(range(limit), desc="Inference"):
            item = ds[i]
            image = item["image"]
            
            # Format Prompt
            if dataset_name == "pope":
                question = item["question"]
                text = f"User: <|vision_start|><|image_pad|><|vision_end|>{question} Please answer with 'yes' or 'no'.\nAssistant: "
            else:
                text = "User: <|vision_start|><|image_pad|><|vision_end|>Describe this image in detail.\nAssistant: "
            
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            
            # Run DualPath Inference
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=100,
                ablation_mode=ablation_mode
            )
            
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract internal history from model
            sample_data = {
                "sample_index": i,
                "dataset": dataset_name,
                "generated_text": output_text,
                "gate_history": self.model._last_gate_history,
                "drift_history": self.model._last_drift_history,
                "interventions_f": self.model._intervention_triggers_f,
                "interventions_o": self.model._intervention_triggers_o
            }
            
            if "objects" in item:
                sample_data["gt_objects"] = item["objects"]
            if "answer" in item:
                sample_data["gt_answer"] = item["answer"]
            elif "label" in item:
                sample_data["gt_answer"] = item["label"]

            results.append(sample_data)

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="outputs/dualpath_vlm_final.pt")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--mode", type=str, default="learned", choices=["learned", "force_f", "force_o", "baseline"])
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    engine = InferenceEngine(args.model_id, args.checkpoint)
    engine.run_inference(args.dataset, args.split, args.out, limit=args.limit, ablation_mode=args.mode)
