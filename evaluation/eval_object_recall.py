import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os
import spacy
from typing import Dict, Any, List, Set

class ObjectRecallEvaluator:
    """
    Evaluates Object Recall on Visual Genome.
    Novel Metric: Recall@K = (Mentioned GT Objects) / (Total GT Objects).
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

    def evaluate(self, vg_objects_path: str, image_dir: str, num_samples: int = 500, baseline_mode: bool = False) -> Dict[str, float]:
        if hasattr(self.model, "baseline_mode"):
            self.model.baseline_mode = baseline_mode

        self.model.eval()
        
        # Load VG objects mapping
        # Expected format: {image_id: [object_names...]}
        print(f"Loading VG objects from {vg_objects_path}...")
        with open(vg_objects_path, 'r') as f:
            vg_data = json.load(f)
            
        recall_scores = []
        pbar = tqdm(list(vg_data.keys())[:num_samples], desc=f"VG Recall [DualPath={not baseline_mode}]")
        
        results_log = []

        for img_id in pbar:
            image_path = os.path.join(image_dir, f"{img_id}.jpg")
            if not os.path.exists(image_path):
                continue
                
            image = Image.open(image_path).convert("RGB")
            gt_objects_raw = vg_data[img_id]
            # Lemmatize all GT objects
            gt_lemmas = set()
            for obj in gt_objects_raw:
                doc = self.nlp(obj.lower())
                if len(doc) > 0:
                    gt_lemmas.add(doc[0].lemma_)
            
            if not gt_lemmas:
                continue

            # 1. Generate Caption
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "List all the visible objects in this image."}
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                    max_new_tokens=64
                )
            generated_text = self.processor.batch_decode(out, skip_special_tokens=True)[0].lower()
            
            # 2. Extract Mentioned Lemmas
            mentioned_lemmas = self._extract_lemmas(generated_text)
            
            # 3. Compute Recall
            hits = gt_lemmas.intersection(mentioned_lemmas)
            recall = len(hits) / len(gt_lemmas)
            recall_scores.append(recall)
            
            pbar.set_postfix({"Recall": f"{sum(recall_scores)/len(recall_scores):.4f}"})
            
            results_log.append({
                "image_id": img_id,
                "caption": generated_text,
                "gt_objects": list(gt_lemmas),
                "hits": list(hits),
                "recall": recall
            })
            
        final_recall = sum(recall_scores) / (len(recall_scores) + 1e-8)
        
        os.makedirs("outputs/evaluation", exist_ok=True)
        suffix = "baseline" if baseline_mode else "dualpath"
        with open(f"outputs/evaluation/vg_recall_{suffix}.json", "w") as f:
            json.dump({"mean_recall": final_recall, "samples": results_log}, f, indent=4)
            
        return {"mean_recall": final_recall}

    def _extract_lemmas(self, text: str) -> Set[str]:
        doc = self.nlp(text.lower())
        return {token.lemma_ for token in doc if not token.is_stop and not token.is_punct}

if __name__ == "__main__":
    from PIL import Image
    import argparse
    from model.dualpath_vlm import DualPathVLM
    from transformers import Qwen2_5_VLProcessor
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--vg_json", type=str, default="data/vg_objects_val.json")
    parser.add_argument("--img_dir", type=str, default="data/vg/images")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathVLM(device=device, baseline_mode=args.baseline)
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    evaluator = ObjectRecallEvaluator(model, processor, device)
    evaluator.evaluate(args.vg_json, args.img_dir, num_samples=args.limit, baseline_mode=args.baseline)
