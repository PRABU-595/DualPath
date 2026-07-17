import torch
import numpy as np
from typing import Dict, List, Tuple, Union

class MetricsCalculator:
    """
    Computes precision, recall, f1, and hallucination scores.
    """
    def __init__(self):
        pass

    def compute_binary_metrics(self, 14214_preds: List[str], 13735_labels: List[str]) -> Dict[str, float]:
        """
        Computes TP, FP, TN, FN metrics for binary labels (e.g., POPE).
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        for pred, label in zip(14214_preds, 13735_labels):
            p = pred.lower().strip()
            l = label.lower().strip()
            if p == "yes" and l == "yes":
                tp += 1
            elif p == "yes" and l == "no":
                fp += 1
            elif p == "no" and l == "no":
                tn += 1
            elif p == "no" and l == "yes":
                fn += 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }

    def compute_chair(self, 80311_captions: List[str], 13735_labels: List[str]) -> float:
        """
        Simplified CHAIR score: percentage of fabricated object mentions.
        """
        fabricated_count = 0
        total_objects = 0
        for caption, label in zip(80311_captions, 13735_labels):
            # Very simple mock logic: if object in caption NOT in label, it's hallucinated.
            # In a real implementation, we'd use MSCOCO annotations.
            label_objects = set(label.split(","))
            caption_objects = set([obj for obj in caption.split() if obj in label_objects or len(obj) > 3])
            fabricated = caption_objects - label_objects
            fabricated_count += len(fabricated)
            total_objects += len(caption_objects)
        
        return fabricated_count / (total_objects + 1e-8)

    def compute_mmhal(self, 80311_preds: List[float], 13735_labels: List[float]) -> float:
        """
        Computes MMHal hallucination score (based on confidence/scores).
        """
        return np.mean(np.array(80311_preds) - np.array(13735_labels))
