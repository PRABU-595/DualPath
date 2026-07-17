import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
from PIL import Image
import json
import os
import re
import spacy
from typing import Dict, Any, List, Set

class ChairEvaluator:
    """
    Computes CHAIR (Caption Hallucination Assessment with Image Relevance).
    Calculates CHAIRs (sentence level) and CHAIRi (instance level).
    Improved with spaCy lemmatization for precise object matching.
    """
    def __init__(self, model: nn.Module, processor: Any, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Load spaCy for lemmatization
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Official COCO 80 categories
        self.coco_objects = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        # Pre-lemmatize COCO objects for faster matching
        self.coco_lemmas = {self.nlp(obj)[0].lemma_: obj for obj in self.coco_objects}

    def evaluate(self, num_samples: int = 500, baseline_mode: bool = False) -> Dict[str, float]:
        if hasattr(self.model, "baseline_mode"):
            self.model.baseline_mode = baseline_mode

        self.model.eval()
        # Load HF detection-datasets/coco (standard for hallucination research)
        coco = load_dataset("detection-datasets/coco", split="val", trust_remote_code=True)
        
        chairs_count = 0
        chairi_count = 0
        total_sentences = 0
        total_instances = 0
        
        limit = min(num_samples, len(coco))
        pbar = tqdm(range(limit), desc=f"CHAIR [DualPath={not baseline_mode}]")
        
        results_log = []

        for i in pbar:
            item = coco[i]
            image = item["image"]
            
            # Ground truth objects from annotations
            gt_objects = set()
            for obj in item["objects"]:
                cat_id = obj["category"]
                # In detection-datasets/coco, category is often the index or has a mapping
                # We'll use the category_name if available, else fallback to index
                if "category_name" in obj:
                    gt_objects.add(self.nlp(obj["category_name"].lower())[0].lemma_)
                elif cat_id < len(self.coco_objects):
                    gt_objects.add(self.nlp(self.coco_objects[cat_id])[0].lemma_)
            
            # 1. Generate Caption (with Chat Template)
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                    max_new_tokens=50
                )
            generated_caption = self.processor.batch_decode(out, skip_special_tokens=True)[0].lower()
            
            # 2. Extract objects from generated caption using spaCy
            extracted_lemmas = self._extract_object_lemmas(generated_caption)
            
            # 3. Compute Metrics
            hallucinated = [lemma for lemma in extracted_lemmas if lemma not in gt_objects]
            
            if hallucinated:
                chairs_count += 1
            
            chairi_count += len(hallucinated)
            total_instances += len(extracted_lemmas)
            total_sentences += 1
            
            pbar.set_postfix({
                "CHAIRs": f"{chairs_count/total_sentences:.4f}",
                "CHAIRi": f"{chairi_count/(total_instances + 1e-8):.4f}"
            })
            
            results_log.append({
                "image_id": i,
                "caption": generated_caption,
                "gt_objects": list(gt_objects),
                "pred_objects": list(extracted_lemmas),
                "hallucinated": hallucinated
            })
            
        final_metrics = {
            "CHAIRs": chairs_count / (total_sentences + 1e-8),
            "CHAIRi": chairi_count / (total_instances + 1e-8)
        }
        
        os.makedirs("outputs/evaluation", exist_ok=True)
        suffix = "baseline" if baseline_mode else "dualpath"
        with open(f"outputs/evaluation/chair_{suffix}.json", "w") as f:
            json.dump({"metrics": final_metrics, "samples": results_log}, f, indent=4)
            
        return final_metrics

    def _extract_object_lemmas(self, caption: str) -> Set[str]:
        """Extracts coco object lemmas from caption using spaCy."""
        doc = self.nlp(caption.lower())
        found_lemmas = set()
        for token in doc:
            lemma = token.lemma_
            if lemma in self.coco_lemmas:
                found_lemmas.add(lemma)
        return found_lemmas

if __name__ == "__main__":
    import argparse
    from model.dualpath_vlm import DualPathVLM
    from transformers import Qwen2_5_VLProcessor
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(model_id=args.model_id, device=device, baseline_mode=args.baseline)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id)
    
    evaluator = ChairEvaluator(model, processor, device)
    evaluator.evaluate(num_samples=args.limit, baseline_mode=args.baseline)
