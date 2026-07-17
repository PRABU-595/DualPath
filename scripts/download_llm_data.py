"""
DualPath-LLM: HaluEval Dataset Downloader
Fetches the official HaluEval dataset from HuggingFace and processes it into
the DualPath JSONL format (fabrications and omissions).
"""

import json
import os
import sys
import subprocess

def ensure_dependencies():
    try:
        import datasets
    except ImportError:
        print("Installing 'datasets' library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])

def main():
    ensure_dependencies()
    from datasets import load_dataset
    
    output_dir = r"D:\Dataset\DualPath_Text_Data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading HaluEval (QA subset) from HuggingFace...")
    try:
        # Load the QA subset of HaluEval
        ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have internet access or try running this on the University server.")
        return

    fab_pairs = []
    om_pairs = []
    
    print(f"Processing {len(ds)} samples...")
    
    # We will split the dataset: 
    # Even indices -> Fabrication pairs (using the provided hallucinated answer)
    # Odd indices -> Omission pairs (we artificially truncate the right answer to simulate omission)
    
    for i, item in enumerate(ds):
        # HaluEval QA schema:
        # knowledge, question, right_answer, hallucinated_answer
        
        context = item.get("knowledge", "")
        question = item.get("question", "")
        good_ans = item.get("right_answer", "")
        bad_ans = item.get("hallucinated_answer", "")
        
        prompt = f"Context: {context}\nQuestion: {question}"
        
        if i % 2 == 0:
            # Fabrication
            fab_pairs.append({
                "prompt": prompt,
                "good_caption": good_ans,
                "bad_caption": bad_ans,
                "fabricated_objects": ["hallucination_simulation"], # In practice, extract diff
                "pathway": "F"
            })
        else:
            # Omission
            # Simulate omission by splitting the good answer and dropping the second half
            words = good_ans.split()
            if len(words) > 4:
                omitted_ans = " ".join(words[:len(words)//2]) + "..."
            else:
                omitted_ans = "I don't know."
                
            om_pairs.append({
                "prompt": prompt,
                "good_caption": good_ans,
                "bad_caption": omitted_ans,
                "omitted_objects": ["omitted_context_simulation"],
                "pathway": "O"
            })
            
    # Save the pairs
    fab_path = os.path.join(output_dir, "text_fabrication_pairs.jsonl")
    om_path = os.path.join(output_dir, "text_omission_pairs.jsonl")
    
    with open(fab_path, "w", encoding="utf-8") as f:
        for p in fab_pairs:
            f.write(json.dumps(p) + "\n")
            
    with open(om_path, "w", encoding="utf-8") as f:
        for p in om_pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"[✓] Saved {len(fab_pairs)} fabrication pairs to {fab_path}")
    print(f"[✓] Saved {len(om_pairs)} omission pairs to {om_path}")
    print("\nDataset is ready for DualPath-Unified training!")

if __name__ == "__main__":
    main()
