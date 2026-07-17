"""
DualPath-LLM: Text Pair Generator
Generates simulated LLM hallucinations (fabrications and omissions) based on textual context.
This acts as a placeholder for larger datasets like HaluEval.
"""

import json
import os
import random

def generate_pairs(num_samples=100):
    os.makedirs("data", exist_ok=True)
    
    contexts = [
        {
            "context": "The capital of France is Paris. The Eiffel Tower was built in 1889 by Gustave Eiffel.",
            "question": "Who built the Eiffel Tower and when?",
            "good": "The Eiffel Tower was built in 1889 by Gustave Eiffel.",
            "fab": "The Eiffel Tower was built in 1889 by Gustave Eiffel, who was secretly an alien from Mars.",
            "fab_words": ["alien", "Mars", "secretly"],
            "om": "It was built in 1889.",
            "om_words": ["Gustave", "Eiffel"]
        },
        {
            "context": "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.",
            "question": "What is photosynthesis?",
            "good": "Photosynthesis is the process where plants, algae, and some bacteria use sunlight to create chemical energy.",
            "fab": "Photosynthesis is the process where plants use sunlight to create chemical energy, which powers their built-in laser beams.",
            "fab_words": ["laser", "beams", "built-in"],
            "om": "It is a process used by plants to make energy.",
            "om_words": ["algae", "bacteria", "sunlight", "chemical"]
        },
        {
            "context": "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics. He won the Nobel Prize in 1921.",
            "question": "What did Albert Einstein do and when did he win the Nobel Prize?",
            "good": "Albert Einstein developed the theory of relativity and won the Nobel Prize in 1921.",
            "fab": "Albert Einstein developed the theory of relativity and won the Nobel Prize in 1921 for his invention of the time machine.",
            "fab_words": ["time", "machine", "invention"],
            "om": "He developed the theory of relativity.",
            "om_words": ["Nobel", "Prize", "1921"]
        }
    ]

    fab_pairs = []
    om_pairs = []
    
    print(f"Generating {num_samples} simulated text hallucination pairs...")
    
    for i in range(num_samples):
        base = random.choice(contexts)
        prompt = f"Context: {base['context']}\nQuestion: {base['question']}"
        
        # Fabrication Pair
        fab_pairs.append({
            "prompt": prompt,
            "good_caption": base["good"],
            "bad_caption": base["fab"],
            "fabricated_objects": base["fab_words"],
            "pathway": "F"
        })
        
        # Omission Pair
        om_pairs.append({
            "prompt": prompt,
            "good_caption": base["good"],
            "bad_caption": base["om"],
            "omitted_objects": base["om_words"],
            "pathway": "O"
        })
        
    with open("data/text_fabrication_pairs.jsonl", "w") as f:
        for p in fab_pairs:
            f.write(json.dumps(p) + "\n")
            
    with open("data/text_omission_pairs.jsonl", "w") as f:
        for p in om_pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"[✓] Saved {len(fab_pairs)} fabrication pairs to data/text_fabrication_pairs.jsonl")
    print(f"[✓] Saved {len(om_pairs)} omission pairs to data/text_omission_pairs.jsonl")

if __name__ == "__main__":
    generate_pairs(100)
