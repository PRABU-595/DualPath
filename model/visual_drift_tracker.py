import torch
import torch.nn as nn
from typing import Optional

class VisualDriftTracker(nn.Module):
    """
    Tracks visual attention decay over decoding steps.
    Used by Pathway O (Omission Corrector) to trigger visual refresh.
    """
    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Initial drift threshold for triggering refresh.
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.register_buffer("step0_hidden", None)

    def reset(self):
        """Resets the tracker for a new sequence."""
        self.step0_hidden = None

    def update(self, current_visual_attention: torch.Tensor) -> torch.Tensor:
        """Legacy update method for raw drift from attention monitor."""
        return torch.sigmoid((current_visual_attention.unsqueeze(-1) - self.threshold) * 5.0)

    def forward(self, h_current: torch.Tensor) -> torch.Tensor:
        """
        Computes drift and returns soft injection weight.
        
        Args:
            h_current: Current token hidden state [batch, 1, hidden_dim].
            
        Returns:
            injection_weight: Sigmoid(drift - threshold) [batch, 1].
        """
        if self.step0_hidden is None:
            self.step0_hidden = h_current.detach()
            return torch.zeros((h_current.shape[0], 1)).to(h_current.device)
            
        drift = torch.norm(self.step0_hidden - h_current, dim=-1).mean(dim=-1, keepdim=True)
        injection_weight = torch.sigmoid((drift - self.threshold) * 5.0)
        # Squeeze the middle dimension to achieve [batch, 1] instead of [batch, 1, 1]
        if len(injection_weight.shape) == 3:
            injection_weight = injection_weight.squeeze(1)
        return injection_weight
