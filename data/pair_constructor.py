import torch
import random
import spacy
import os
from typing import Dict, Any, Tuple, List

class PairConstructor:
    """
    Builds faithful, fabricated, and incomplete caption triplets for contrastive training.
    """
    def __init__(self):
        # Initialize spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Common objects for entity swapping (fabrication synthesis)
        self.common_objects = [
            "cat", "dog", "car", "tree", "person", "bird", "table", "chair", 
            "bicycle", "truck", "boat", "airplane", "bus", "train"
        ]

    def construct_triplet(self, faithful_caption: str, existing_hallucination: str = None) -> Tuple[str, str, str]:
        """
        Constructs (faithful, fabricated, incomplete) triplets.
        
        Args:
            faithful_caption: The ground truth caption.
            existing_hallucination: Optional pre-existing hallucinated caption from dataset.
            
        Returns:
            triplet: (faithful, fabricated, incomplete)
        """
        # 1. Fabricated Caption
        # Use provided negative if available, otherwise synthesize via entity swapping
        if existing_hallucination:
            fabricated = existing_hallucination
        else:
            fabricated = self._synthesize_fabrication(faithful_caption)
            
        # 2. Incomplete Caption
        # Synthesize via object omission (dropping NOUNs/PROPNs)
        incomplete = self._synthesize_omission(faithful_caption)
        
        return faithful_caption, fabricated, incomplete

    def _synthesize_fabrication(self, caption: str) -> str:
        """
        Synthesizes a fabricated caption by swapping an existing noun with a different one.
        """
        doc = self.nlp(caption)
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        
        if not nouns:
            # Fallback: append a hallucinated object
            return caption + f" near a {random.choice(self.common_objects)}."
            
        target = random.choice(nouns)
        replacement = random.choice([obj for obj in self.common_objects if obj != target.lower()])
        
        # Use regex for whole-word replacement to avoid sub-word issues
        import re
        pattern = re.compile(r'\b' + re.escape(target) + r'\b')
        return pattern.sub(replacement, caption)

    def _synthesize_omission(self, caption: str, drop_rate: float = 0.3) -> str:
        """
        Synthesizes an incomplete caption by dropping object mentions.
        """
        doc = self.nlp(caption)
        tokens = [token.text for token in doc]
        object_indices = [i for i, token in enumerate(doc) if token.pos_ in ["NOUN", "PROPN"]]
        
        if not object_indices:
            return caption
            
        num_to_drop = max(1, int(len(object_indices) * drop_rate))
        drops = set(random.sample(object_indices, num_to_drop))
        
        incomplete_tokens = [tokens[i] for i in range(len(tokens)) if i not in drops]
        # Clean up double spaces from joins
        return " ".join(incomplete_tokens).replace("  ", " ").strip()

if __name__ == "__main__":
    constructor = PairConstructor()
    text = "A man is sitting on a wooden bench in the park."
    f, fab, inc = constructor.construct_triplet(text)
    print(f"Faithful:   {f}")
    print(f"Fabricated: {fab}")
    print(f"Incomplete: {inc}")
