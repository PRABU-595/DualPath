import torch
import torch.nn as nn
from typing import Tuple

class OmissionCorrector(nn.Module):
    """
    Corrects omission hallucinations by injecting a compressed visual refresh signal.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Projection for visual compression/refresh
        self.proj_down = nn.Linear(hidden_dim, hidden_dim // 4)
        self.proj_up = nn.Linear(hidden_dim // 4, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor, injection_weight: torch.Tensor, image_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [batch, seq_len, hidden_dim]
            injection_weight: From VisualDriftTracker [batch, 1]
            image_embeddings: [batch, num_image_tokens, hidden_dim]
        """
        # image_summary: [batch, 1, hidden_dim]
        image_summary = image_embeddings.mean(dim=1, keepdim=True)
        
        # Projection Pipeline
        refresh_signal = self.proj_down(image_summary)
        refresh_signal = self.relu(refresh_signal)
        refresh_signal = self.proj_up(refresh_signal) 
        
        # Combined weight injection
        # refresh_signal: [batch, 1, hidden_dim]
        # injection_weight: [batch, seq_len, 1] or [batch, 1]
        refresh_weighted = refresh_signal * injection_weight
        
        # Add the visual refresh signal to the residual stream
        corrected_hidden_state = hidden_state + refresh_weighted
        
        return corrected_hidden_state, injection_weight
