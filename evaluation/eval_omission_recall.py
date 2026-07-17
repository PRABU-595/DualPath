import argparse
import json
import spacy
from tqdm import tqdm
from typing import List, Set, Dict

# Load spaCy for extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_nouns(text: str) -> Set[str]:
    """Extract lemmatized root nouns from text to represent mentioned objects."""
    doc = nlp(text.lower())
    return {chunk.root.lemma_ for chunk in doc.noun_chunks}

class OmissionRecallEvaluator:
    """
    Evaluates Omission rate (Recall@K) for VLM generated captions.
    Recall = len(mentioned ∩ gold) / len(gold)
    """
    def __init__(self, ground_truth_json: str):
        print(f"Loading ground truth objects from {ground_truth_json}...")
        self.gt_objects: Dict[str, Set[str]] = {}
        
        try:
            with open(ground_truth_json, 'r') as f:
                data = json.load(f)
                
            if 'annotations' in data:
                # COCO format
                categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
                for ann in data['annotations']:
                    if 'category_id' in ann:
                        img_id = str(ann['image_id']).zfill(12)
                        self.gt_objects.setdefault(img_id, set()).add(categories[ann['category_id']])
            else:
                # Assuming VG or flattened dictionary representation for mock/testing
                for img_id, objs in data.items():
                    self.gt_objects[str(img_id)] = set(objs)
                    
        except Exception as e:
            print(f"Warning: Mocking GT Objects due to load error: {e}")
            self.gt_objects = {"000000000001": {"dog", "car", "frisbee", "person", "tree"}}
            
    def evaluate(self, predictions_jsonl: str) -> Dict[str, float]:
        """
        Evaluate generated predictions.
        predictions_jsonl format: {"image_id": "...", "caption": "..."}
        """
        print(f"Evaluating predictions from {predictions_jsonl}...")
        recalls = []
        total_gt_objects = 0
        total_mentioned_gt_objects = 0
        
        with open(predictions_jsonl, 'r') as f:
            for line in tqdm(f):
                pred = json.loads(line)
                img_id = str(pred['image_id']).zfill(12)
                
                if img_id not in self.gt_objects:
                    continue
                    
                gold_set = self.gt_objects[img_id]
                if len(gold_set) == 0:
                    continue
                    
                caption = pred['caption']
                mentioned_set = extract_nouns(caption)
                
                # Intersection
                hits = gold_set.intersection(mentioned_set)
                recall = len(hits) / len(gold_set)
                recalls.append(recall)
                
                total_gt_objects += len(gold_set)
                total_mentioned_gt_objects += len(hits)
                
        if len(recalls) == 0:
            print("No matching image IDs found for evaluation.")
            return {"mean_recall": 0.0, "corpus_recall": 0.0}
            
        mean_recall = sum(recalls) / len(recalls)
        corpus_recall = total_mentioned_gt_objects / total_gt_objects if total_gt_objects > 0 else 0.0
        
        print("\n--- OMISSION EVALUATION RESULTS ---")
        print(f"Mean Instance Recall: {mean_recall:.4f} (Higher is better)")
        print(f"Corpus Overall Recall: {corpus_recall:.4f}")
        print("-----------------------------------\n")
        
        return {
            "mean_recall": mean_recall,
            "corpus_recall": corpus_recall
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--preds", type=str, required=True, help="Path to predicted captions JSONL")
    args = parser.parse_args()
    
    evaluator = OmissionRecallEvaluator(args.gt)
    evaluator.evaluate(args.preds)
