import os
import sys
import json
import csv
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import precision_recall_curve, confusion_matrix

def check_yes_no_tokens(processor, text):
    """
    Qwen2.5-VL tokenization is sensitive to leading spaces and capitalization.
    We identify all potential ID mappings for Yes/No variants.
    """
    variants = ["Yes", " Yes", "yes", " yes", "No", " No", "no", " no"]
    token_ids = {}
    for v in variants:
        # Extract the last token ID which represents the word
        ids = processor.tokenizer.encode(v, add_special_tokens=False)
        token_ids[v] = ids[-1] if ids else None
        
    yes_ids = set([token_ids["Yes"], token_ids[" Yes"], token_ids["yes"], token_ids[" yes"]])
    no_ids = set([token_ids["No"], token_ids[" No"], token_ids["no"], token_ids[" no"]])
    
    # Remove Nones
    yes_ids = {i for i in yes_ids if i is not None}
    no_ids = {i for i in no_ids if i is not None}
    
    return yes_ids, no_ids

def load_dataset(file_path):
    """Robustly load JSON or JSONL datasets."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    data = []
    # Try reading as JSONL first
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except json.JSONDecodeError:
        pass # Not a valid JSONL, fallback to standard JSON
        
    # If JSONL fails, read as standard JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        if isinstance(content, list):
            return content
        elif isinstance(content, dict):
            # Try to find the array in common keys
            for key in ["data", "items", "examples", "questions"]:
                if key in content and isinstance(content[key], list):
                    return content[key]
            raise ValueError(f"JSON object doesn't contain a recognizable array. Keys: {list(content.keys())}")
        else:
            raise ValueError("JSON file doesn't contain a list or recognizable dict.")

def run_evaluation(model, processor, data, yes_ids, no_ids, device, dry_run=False):
    """
    Runs the forward pass on the dataset and extracts raw Yes/No probabilities.
    """
    results = []
    
    limit = 10 if dry_run else len(data)
    if limit == 0:
        print("WARNING: Dataset is empty! No examples to evaluate.")
        return results
        
    print(f"Running evaluation on {limit} examples...")
    
    for i, item in enumerate(tqdm(data[:limit])):
        image_path = item.get("image", item.get("image_path"))
        question = item.get("text", item.get("question"))
        gt_label = str(item.get("label")).lower() == "yes"  # True if Yes, False if No
        
        # Prepare input following Qwen2.5-VL chat template
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question + " Please answer Yes or No."}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=None if not os.path.exists(image_path) else [image_path], 
            padding=True,
            return_tensors="pt"
        )
        
        if os.path.exists(image_path):
            from PIL import Image
            inputs = processor(
                text=[text],
                images=[Image.open(image_path).convert("RGB")],
                padding=True,
                return_tensors="pt"
            )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits for the next token (the first token of the answer)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Extract sum of probabilities for Yes and No variants
            probs = torch.softmax(next_token_logits, dim=-1)
            
            p_yes = sum([probs[vid].item() for vid in yes_ids])
            p_no = sum([probs[vid].item() for vid in no_ids])
            
            # Normalize over just Yes/No to get a clean confidence score in [0,1]
            total = p_yes + p_no
            if total == 0:
                score_yes = 0.5 # Fallback if model outputs something entirely different
            else:
                score_yes = p_yes / total
                
        results.append({
            "question_id": item.get("question_id", i),
            "image": image_path,
            "target": gt_label,
            "score_yes": score_yes,
            "p_yes_raw": p_yes,
            "p_no_raw": p_no
        })
        
        if dry_run:
            print(f"Q: {question} | GT: {gt_label} | Score(Yes): {score_yes:.4f}")
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pope_file", type=str, required=True, help="Path to original POPE eval JSON")
    parser.add_argument("--vg_file", type=str, required=True, help="Path to original VG recall JSON")
    parser.add_argument("--dry_run", action="store_true", help="Run on 10 examples for sanity check")
    parser.add_argument("--target_precision", type=float, default=0.956, help="Target precision from DualPath-VLM")
    parser.add_argument("--pope_limit", type=int, default=500, help="Number of POPE examples to evaluate (matches paper)")
    args = parser.parse_args()

    print("==================================================")
    print("DualPath-VLM: Real GPU Threshold Sweep Evaluation")
    print("==================================================\n")
    
    if not torch.cuda.is_available():
        print("\n" + "!"*70)
        print("CRITICAL WARNING: CUDA is NOT available. PyTorch is running on CPU.")
        print("This will take a VERY long time and indicates a broken environment.")
        print("If you are on a GPU server, ensure you installed torch with CUDA:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("!"*70 + "\n")
        
        response = input("Continue on CPU anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            sys.exit(1)
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen2.5-VL-7B-Instruct on {device}...")
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    
    yes_ids, no_ids = check_yes_no_tokens(processor, "Yes/No")
    print(f"Extracted Yes IDs: {yes_ids}")
    print(f"Extracted No IDs: {no_ids}")
    
    if args.dry_run:
        print("\n--- DRY RUN SANITY CHECK (10 EXAMPLES) ---")
        
    print(f"\nLoading POPE dataset: {args.pope_file}")
    pope_data = load_dataset(args.pope_file)
    
    # Enforce dataset limit to match paper N=500
    if len(pope_data) > args.pope_limit:
        print(f"Subsampling POPE from {len(pope_data)} down to {args.pope_limit} examples (matching paper)...")
        pope_data = pope_data[:args.pope_limit]

    pope_results = run_evaluation(model, processor, pope_data, yes_ids, no_ids, device, dry_run=args.dry_run)
    
    if not args.dry_run:
        print("\nComputing Precision-Recall curve to find exact threshold...")
        y_true = [r["target"] for r in pope_results]
        y_scores = [r["score_yes"] for r in pope_results]
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find threshold where precision >= target_precision (95.6%)
        target_idx = np.where(precisions >= args.target_precision)[0]
        if len(target_idx) > 0:
            idx = target_idx[0]
            chosen_threshold = thresholds[min(idx, len(thresholds)-1)]
            achieved_precision = precisions[idx]
            achieved_recall = recalls[idx]
        else:
            # If 95.6% is unachievable, take max precision
            idx = np.argmax(precisions)
            chosen_threshold = thresholds[min(idx, len(thresholds)-1)]
            achieved_precision = precisions[idx]
            achieved_recall = recalls[idx]
            print(f"WARNING: Target precision {args.target_precision} unreachable. Max achieved: {achieved_precision:.4f}")

        print(f"\n[Vanilla VLM Calibration]")
        print(f"Chosen Threshold T = {chosen_threshold:.4f}")
        print(f"Achieved POPE Precision: {achieved_precision:.4f}")
        print(f"Achieved POPE Recall: {achieved_recall:.4f}")
        
        y_pred = [1 if s >= chosen_threshold else 0 for s in y_scores]
        cm = confusion_matrix(y_true, y_pred)
        print("\n--- POPE Confusion Matrix at T ---")
        print(f"TN: {cm[0,0]:<5} | FP: {cm[0,1]:<5}")
        print(f"FN: {cm[1,0]:<5} | TP: {cm[1,1]:<5}")
        
        # Save raw log
        log_path = "outputs/raw_pope_confidence_log.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["QuestionID", "Image", "Target", "Score_Yes", "P_Yes_Raw", "P_No_Raw", "Prediction"])
            for i, r in enumerate(pope_results):
                writer.writerow([r["question_id"], r["image"], r["target"], r["score_yes"], r["p_yes_raw"], r["p_no_raw"], y_pred[i]])
        print(f"Saved full raw POPE audit log to {log_path}")
        
        print(f"\nLoading VG Recall dataset: {args.vg_file}")
        vg_data = load_dataset(args.vg_file)
            
        vg_results = run_evaluation(model, processor, vg_data, yes_ids, no_ids, device, dry_run=False)
        
        vg_true = [r["target"] for r in vg_results]
        vg_scores = [r["score_yes"] for r in vg_results]
        vg_pred = [1 if s >= chosen_threshold else 0 for s in vg_scores]
        
        vg_recall = sum([1 for t, p in zip(vg_true, vg_pred) if t and p]) / sum(vg_true) if sum(vg_true) > 0 else 0
        print(f"\n[VG Recall Evaluation]")
        print(f"Applying identical threshold T = {chosen_threshold:.4f} to VG Dataset")
        print(f"Achieved VG Object Recall: {vg_recall:.4f}")
        
        vg_log_path = "outputs/raw_vg_confidence_log.csv"
        with open(vg_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["QuestionID", "Image", "Target", "Score_Yes", "Prediction"])
            for i, r in enumerate(vg_results):
                writer.writerow([r["question_id"], r["image"], r["target"], r["score_yes"], vg_pred[i]])
        print(f"Saved full raw VG audit log to {vg_log_path}")
        
if __name__ == "__main__":
    main()
