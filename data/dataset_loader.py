"""
DualPath-VLM: Dataset Loader
Loads locally-generated .jsonl triplets and COCO images for training.

Expects .jsonl files produced by scripts/generate_real_pairs.py.
Each line: {image_id, image_path, good_caption, bad_caption, fabricated_objects/omitted_objects, pathway}
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoProcessor
from PIL import Image
import json
import os
import random
from typing import List, Dict, Any, Optional


class DualPathLocalDataset(Dataset):
    """
    Loads training pairs from local .jsonl files + COCO images.
    Each sample contains a good caption and a bad caption for contrastive training.
    """
    
    def __init__(self,
                 jsonl_path: str,
                 image_dir: str,
                 processor: AutoProcessor,
                 max_length: int = 512,
                 image_size: int = 448):
        """
        Args:
            jsonl_path: Path to .jsonl file (fabrication or omission pairs)
            image_dir: Path to COCO images directory (None for pure text datasets)
            processor: The VLM processor for tokenization
            max_length: Max token sequence length
            image_size: Target image size for resizing
        """
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load all samples
        self.samples = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Training data not found: {jsonl_path}\n"
                f"Run: python scripts/generate_real_pairs.py"
            )
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Text vs VLM Formatting
        is_text_only = self.image_dir is None or not sample.get("image_path")
        
        if not is_text_only:
            # Load image
            img_path = os.path.join(self.image_dir, sample["image_path"])
            try:
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
            
            prompt = "Describe this image in detail."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": good_caption},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        else:
            # Pure Text LLM Formatting
            prompt = sample.get("prompt", "Analyze the given context.")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": good_caption},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        
        # Squeeze batch dimension
        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
        if not is_text_only and "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0) if "image_grid_thw" in inputs else torch.zeros(3, dtype=torch.long)
        
        # Create labels (same as input_ids, but mask the prompt tokens)
        labels = result["input_ids"].clone()
        # Mask everything before the assistant response with -100
        # Simple heuristic: find the last occurrence of the assistant token
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        result["labels"] = labels
        
        # Metadata for the loss function
        if pathway == "F":
            # Tokenize fabricated object names to get their token IDs
            fab_objects = sample.get("fabricated_objects", [])
            fab_token_ids = []
            for obj_name in fab_objects:
                tokens = self.processor.tokenizer.encode(obj_name, add_special_tokens=False)
                fab_token_ids.extend(tokens)
            result["fabricated_token_ids"] = fab_token_ids
            result["omitted_token_ids"] = []
        else:
            # Tokenize omitted object names
            om_objects = sample.get("omitted_objects", [])
            om_token_ids = []
            for obj_name in om_objects:
                tokens = self.processor.tokenizer.encode(obj_name, add_special_tokens=False)
                om_token_ids.extend(tokens)
            result["fabricated_token_ids"] = []
            result["omitted_token_ids"] = om_token_ids
        
        result["pathway"] = pathway
        
        return result


def collate_fn_v2(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that produces the exact format the Trainer expects.
    
    Returns:
        {
            "inputs": {input_ids, attention_mask, pixel_values, image_grid_thw, labels},
            "fabricated_tokens": List[int],
            "omitted_tokens": List[int],
            "pathway": str
        }
    """
    # Stack tensor fields
    input_ids = torch.stack([s["input_ids"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])
    labels = torch.stack([s["labels"] for s in batch])
    
    labels = torch.stack([s["labels"] for s in batch])
    
    # Handle optional image inputs for text-only mode
    has_pixels = "pixel_values" in batch[0]
    
    pixel_values = None
    image_grid_thw = None
    if has_pixels:
        pixel_values = torch.stack([s["pixel_values"] for s in batch])
        image_grid_thw = torch.stack([s["image_grid_thw"] for s in batch])
    
    # Merge token ID lists across the batch
    all_fab_tokens = []
    all_om_tokens = []
    for s in batch:
        all_fab_tokens.extend(s["fabricated_token_ids"])
        all_om_tokens.extend(s["omitted_token_ids"])
    
    # Determine dominant pathway in this batch
    pathways = [s["pathway"] for s in batch]
    dominant_pathway = max(set(pathways), key=pathways.count)
    
    result = {
        "inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        "fabricated_tokens": list(set(all_fab_tokens)),  # deduplicate
        "omitted_tokens": list(set(all_om_tokens)),
        "pathway": dominant_pathway,
    }
    
    if has_pixels:
        result["inputs"]["pixel_values"] = pixel_values
        result["inputs"]["image_grid_thw"] = image_grid_thw
        
    return result


# Keep legacy collate_fn for backward compatibility
def collate_fn(batch):
    return collate_fn_v2(batch)
