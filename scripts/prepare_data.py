import os
from datasets import load_dataset
import argparse
from tqdm.auto import tqdm

def prepare_datasets(subset_mode=False):
    """
    Downloads and caches all datasets for DualPath-VLM.
    If subset_mode is True, only loads the first 10 samples to verify access.
    """
    manifest = {
        "POPE": ("RUCAIBox/POPE", ["random", "popular", "adversarial"]),
        "CHAIR-COCO": ("detection-datasets/coco", ["val"]),
        "AMBER": ("openbmb/AMBER", ["test"]),
        "MMHal-Bench": ("openbmb/MMHal-Bench", ["test"]),
        "LRV-Instruction": ("VictorSanh/LrvInstruction", ["train"]),
        "RLHF-V": ("HaoyeZhang/RLHF-V-Dataset", ["train"])
    }

    print("--- Starting Data Preparation Suite ---")
    print(f"Mode: {'Subset (Verification)' if subset_mode else 'Full Download'}")
    
    for name, (path, splits) in manifest.items():
        print(f"\nProcessing {name} ({path})...")
        for split in splits:
            try:
                # Use trust_remote_code for detection datasets (COCO)
                ds = load_dataset(path, split=split, trust_remote_code=True)
                
                if subset_mode:
                    print(f"  [Verified] {split} split loaded ({len(ds)} total). Selecting subset...")
                    ds = ds.select(range(min(10, len(ds))))
                else:
                    print(f"  [Success] {split} split cached ({len(ds)} samples).")
                    
            except Exception as e:
                print(f"  [Error] Failed to load {path} split {split}: {e}")

    print("\n--- Data Preparation Complete ---")
    print("All datasets are now cached in ~/.cache/huggingface/datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualPath-VLM Data Preparation")
    parser.add_argument("--subset", action="store_true", help="Load only a subset for verification")
    args = parser.parse_args()
    
    prepare_datasets(subset_mode=args.subset)
