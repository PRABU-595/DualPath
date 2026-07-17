import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

class FeatureDataset(Dataset):
    """
    Loads pre-extracted hidden states and signals for rapid DualPath training.
    """
    def __init__(self, feature_dir: str, mode: str = "fabrication"):
        """
        Args:
            feature_dir: Directory containing .pt feature chunks.
            mode: "fabrication" or "omission" to load specific path data.
        """
        self.data = []
        
        # Find all relevant chunks
        chunks = [f for f in os.listdir(feature_dir) if f.startswith(mode) and f.endswith(".pt")]
        chunks.sort()
        
        print(f"Loading {mode} features from {len(chunks)} chunks...")
        for chunk in chunks:
            path = os.path.join(feature_dir, chunk)
            chunk_data = torch.load(path, map_location="cpu")
            self.data.extend(chunk_data)
        
        print(f"  [✓] Loaded {len(self.data)} {mode} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "hidden_states": sample["hidden_states"],   # [seq_len, hidden_dim]
            "image_embeddings": sample.get("image_embeddings", sample["hidden_states"]), # Fallback
            "fab_signal": sample["fab_signal"],         # [num_layers, num_heads]
            "drift_signal": sample["drift_signal"],     # [1]
            "labels": sample["labels"],                 # [seq_len]
            "fabricated_tokens": sample["fabricated_tokens"],
            "omitted_tokens": sample["omitted_tokens"]
        }

def feature_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate to handle padded tensors from pre-extracted features.
    """
    hidden_states = [b["hidden_states"] for b in batch]
    image_embeddings = [b["image_embeddings"] for b in batch]
    fab_signals = [b["fab_signal"] for b in batch]
    drift_signals = [b["drift_signal"] for b in batch]
    labels = [b["labels"] for b in batch]
    
    # Pad sequences
    hidden_padded = torch.nn.utils.rnn.pad_sequence(
        [h.squeeze(0) if h.dim() == 3 else h for h in hidden_states], 
        batch_first=True, 
        padding_value=0.0
    )
    
    img_padded = torch.nn.utils.rnn.pad_sequence(
        [i.squeeze(0) if i.dim() == 3 else i for i in image_embeddings],
        batch_first=True,
        padding_value=0.0
    )
    
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        [l.squeeze(0) if l.dim() == 2 else l for l in labels], 
        batch_first=True, 
        padding_value=-100
    )
    
    return {
        "hidden_states": hidden_padded,
        "image_embeddings": img_padded,
        "fab_signal": torch.stack(fab_signals),
        "drift_signal": torch.stack(drift_signals),
        "labels": labels_padded,
        "fabricated_tokens": [b["fabricated_tokens"] for b in batch],
        "omitted_tokens": [b["omitted_tokens"] for b in batch]
    }
