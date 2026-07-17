"""
DualPath-VLM: Real Data Generator
Generates fabrication and omission training triplets from COCO 2017 and Visual Genome.

Requires:
    - data/coco/annotations/instances_train2017.json
    - data/coco/annotations/captions_train2017.json
    - data/vg/objects.json (optional, for dense omission mapping)

Usage:
    python scripts/generate_real_pairs.py
    python scripts/generate_real_pairs.py --num_samples 10000
"""

import json
import os
import sys
import argparse
import random
import spacy
from tqdm import tqdm
from collections import defaultdict

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class RealDataGenerator:
    """
    Generates training triplets using ONLY real human-annotated data.
    
    Fabrication pairs: Ground truth caption + injected absent COCO object
    Omission pairs: Dense caption + noun-chunk dropout
    """
    
    def __init__(self, 
                 coco_instances_path: str, 
                 coco_captions_path: str,
                 vg_objects_path: str = None):
        self.coco_instances_path = coco_instances_path
        self.coco_captions_path = coco_captions_path
        self.vg_objects_path = vg_objects_path
        
        self.all_coco_objects = set()
        self.image_objects = {}      # image_id -> set of object names
        self.image_captions = {}     # image_id -> list of captions
        self.vg_image_objects = {}   # image_id -> list of object names (from VG)
    
    def load_coco(self):
        """Loads COCO instances (for object labels) and captions (for text) separately."""
        
        # --- Load Instances (object categories per image) ---
        print(f"Loading COCO instances from {self.coco_instances_path}...")
        if not os.path.exists(self.coco_instances_path):
            raise FileNotFoundError(
                f"COCO instances file not found: {self.coco_instances_path}\n"
                f"Run: python scripts/download_cloud_data.py"
            )
        
        with open(self.coco_instances_path, 'r') as f:
            instances = json.load(f)
        
        # Build category mapping
        categories = {cat['id']: cat['name'] for cat in instances['categories']}
        self.all_coco_objects = set(categories.values())
        print(f"  Found {len(self.all_coco_objects)} COCO object categories")
        
        # Build per-image object sets
        self.image_objects = defaultdict(set)
        for ann in instances['annotations']:
            img_id = ann['image_id']
            cat_name = categories[ann['category_id']]
            self.image_objects[img_id].add(cat_name)
        
        print(f"  Found {len(self.image_objects)} images with object annotations")
        
        # --- Load Captions ---
        print(f"Loading COCO captions from {self.coco_captions_path}...")
        if not os.path.exists(self.coco_captions_path):
            raise FileNotFoundError(
                f"COCO captions file not found: {self.coco_captions_path}\n"
                f"Run: python scripts/download_cloud_data.py"
            )
        
        with open(self.coco_captions_path, 'r') as f:
            captions_data = json.load(f)
        
        # Build per-image caption lists
        self.image_captions = defaultdict(list)
        for ann in captions_data['annotations']:
            self.image_captions[ann['image_id']].append(ann['caption'])
        
        print(f"  Found {len(self.image_captions)} images with captions")
        
        # Find images that have BOTH objects and captions
        valid_ids = set(self.image_objects.keys()) & set(self.image_captions.keys())
        print(f"  Images with both objects + captions: {len(valid_ids)}")
    
    def load_visual_genome(self):
        """Loads Visual Genome objects.json for dense object annotations."""
        if not self.vg_objects_path or not os.path.exists(self.vg_objects_path):
            print(f"Visual Genome not found at {self.vg_objects_path}, skipping.")
            return
        
        print(f"Loading Visual Genome from {self.vg_objects_path}...")
        with open(self.vg_objects_path, 'r') as f:
            vg_data = json.load(f)
        
        # VG objects.json format: list of {image_id, objects: [{names: [...], ...}]}
        for item in vg_data:
            img_id = item.get('image_id', item.get('id'))
            obj_names = set()
            for obj in item.get('objects', []):
                # VG objects have 'names' (list) or 'name' (string)
                names = obj.get('names', [])
                if not names and 'name' in obj:
                    names = [obj['name']]
                obj_names.update([n.lower().strip() for n in names if n])
            
            if obj_names:
                self.vg_image_objects[img_id] = obj_names
        
        print(f"  Loaded {len(self.vg_image_objects)} VG images with dense objects")
    
    def _create_fabrication(self, gt_caption: str, present_objects: set) -> tuple:
        """
        Creates a fabricated caption by injecting a real COCO noun 
        that is NOT present in the image.
        """
        available = list(self.all_coco_objects - present_objects)
        if not available:
            return "", []
        
        # Select 1-2 objects to hallucinate
        num_inject = random.choice([1, 2])
        to_inject = random.sample(available, k=min(len(available), num_inject))
        
        # Insert at sentence end
        bad_caption = gt_caption.rstrip('.')
        injection = " and a " + " and a ".join(to_inject)
        bad_caption = bad_caption + injection + "."
        
        return bad_caption, to_inject
    
    def _create_omission(self, gt_caption: str) -> tuple:
        """
        Creates an omission caption by dropping noun chunks from the original.
        """
        doc = nlp(gt_caption)
        chunks = list(doc.noun_chunks)
        
        if len(chunks) < 2:
            return "", []
        
        # Drop 30%-50% of noun chunks
        num_to_drop = max(1, int(len(chunks) * random.uniform(0.3, 0.5)))
        chunks_to_drop = random.sample(chunks, k=num_to_drop)
        drop_spans = [(chunk.start, chunk.end) for chunk in chunks_to_drop]
        
        # Collect dropped nouns for the loss function
        dropped_nouns = []
        for chunk in chunks_to_drop:
            for token in chunk:
                if token.pos_ in ["NOUN", "PROPN"]:
                    dropped_nouns.append(token.lemma_.lower())
        
        # Build new caption excluding dropped spans
        kept_tokens = []
        for token in doc:
            in_drop = any(start <= token.i < end for start, end in drop_spans)
            if not in_drop:
                kept_tokens.append(token.text_with_ws)
        
        bad_caption = "".join(kept_tokens).replace("  ", " ").replace(" ,", ",").strip()
        
        if not bad_caption or not dropped_nouns:
            return "", []
        
        return bad_caption, dropped_nouns
    
    def generate_fabrication_triplets(self, output_file: str, max_samples: int = 50000):
        """Generates fabrication training pairs and saves to .jsonl."""
        print(f"\nGenerating fabrication triplets (max {max_samples})...")
        
        # Use images that have both objects and captions
        valid_ids = list(set(self.image_objects.keys()) & set(self.image_captions.keys()))
        random.shuffle(valid_ids)
        
        triplets = []
        for img_id in tqdm(valid_ids[:max_samples * 2], desc="Fabrication"):
            gt_objs = self.image_objects[img_id]
            captions = self.image_captions[img_id]
            
            # Use first caption
            gt_caption = captions[0]
            img_filename = f"{str(img_id).zfill(12)}.jpg"
            
            bad_caption, fabricated = self._create_fabrication(gt_caption, gt_objs)
            if bad_caption and fabricated:
                triplets.append({
                    "image_id": img_id,
                    "image_path": img_filename,
                    "good_caption": gt_caption,
                    "bad_caption": bad_caption,
                    "fabricated_objects": fabricated,
                    "pathway": "F"
                })
            
            if len(triplets) >= max_samples:
                break
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for t in triplets:
                f.write(json.dumps(t) + '\n')
        
        print(f"  Saved {len(triplets)} fabrication pairs to {output_file}")
        return len(triplets)
    
    def generate_omission_triplets(self, output_file: str, max_samples: int = 50000):
        """Generates omission training pairs and saves to .jsonl."""
        print(f"\nGenerating omission triplets (max {max_samples})...")
        
        # Prefer VG captions (denser), fall back to COCO
        source_ids = list(self.image_captions.keys())
        random.shuffle(source_ids)
        
        triplets = []
        for img_id in tqdm(source_ids[:max_samples * 2], desc="Omission"):
            captions = self.image_captions.get(img_id, [])
            if not captions:
                continue
            
            # Use longest caption (most objects to drop)
            gt_caption = max(captions, key=len)
            img_filename = f"{str(img_id).zfill(12)}.jpg"
            
            bad_caption, dropped = self._create_omission(gt_caption)
            if bad_caption and dropped:
                triplets.append({
                    "image_id": img_id,
                    "image_path": img_filename,
                    "good_caption": gt_caption,
                    "bad_caption": bad_caption,
                    "omitted_objects": dropped,
                    "pathway": "O"
                })
            
            if len(triplets) >= max_samples:
                break
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for t in triplets:
                f.write(json.dumps(t) + '\n')
        
        print(f"  Saved {len(triplets)} omission pairs to {output_file}")
        return len(triplets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualPath-VLM Real Data Generator")
    parser.add_argument("--coco_instances", type=str, 
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--coco_captions", type=str, 
                        default="data/coco/annotations/captions_train2017.json")
    parser.add_argument("--vg_objects", type=str, 
                        default="data/vg/objects.json")
    parser.add_argument("--output_fab", type=str, 
                        default="data/real_fabrication_pairs.jsonl")
    parser.add_argument("--output_om", type=str, 
                        default="data/real_omission_pairs.jsonl")
    parser.add_argument("--num_samples", type=int, default=50000)
    args = parser.parse_args()
    
    generator = RealDataGenerator(
        coco_instances_path=args.coco_instances,
        coco_captions_path=args.coco_captions,
        vg_objects_path=args.vg_objects
    )
    
    generator.load_coco()
    generator.load_visual_genome()
    
    n_fab = generator.generate_fabrication_triplets(args.output_fab, args.num_samples)
    n_om = generator.generate_omission_triplets(args.output_om, args.num_samples)
    
    print(f"\n{'='*60}")
    print(f"Real Data Pipeline Complete.")
    print(f"  Fabrication pairs: {n_fab}")
    print(f"  Omission pairs:    {n_om}")
    print(f"{'='*60}")
