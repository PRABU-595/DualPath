import torch
import torch.nn as nn
from typing import Tuple

class GatingMLP(nn.Module):
    """
    Arbitrates between Fabrication and Omission pathways per decoding step.
    Produces a single complementary gate alpha_O (and alpha_F = 1 - alpha_O).
    """
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: The dimension of the hidden state.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # MLP: Takes [hidden_state_mean, f_signal_mean, injection_weight]
        # Using raw f_signal and injection_weight as features for gating
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Initialize final linear layer bias so that sigmoid(bias) results in ~0.5
        nn.init.constant_(self.mlp[3].bias, 0.5)

    def forward(self, hidden_state: torch.Tensor, f_signal: torch.Tensor, injection_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: The current token hidden state [batch, seq_len, hidden_dim].
            f_signal: Fabrication signal [batch, 1] or [batch, seq_len, 1].
            injection_weight: Injection weight from OmissionCorrector [batch, 1].

        Returns:
            gate_g: The scalar gate value alpha_O [batch, seq_len, 1].
        """
        # Mean pool hidden state along sequence dimension
        h_mean = hidden_state.mean(dim=1) # [batch, hidden_dim]
        
        # Ensure weights are [batch, 1] for the MLP
        if f_signal.dim() > 2:
            f_signal = f_signal.mean(dim=(1, 2)).unsqueeze(-1)
        elif f_signal.dim() == 1:
            f_signal = f_signal.unsqueeze(-1)
            
        if injection_weight.dim() > 2:
            injection_weight = injection_weight.mean(dim=(1, 2)).unsqueeze(-1)
        elif injection_weight.dim() == 1:
            injection_weight = injection_weight.unsqueeze(-1)
            
        # Concatenate: [h_mean, f_signal, injection_weight]
        mlp_input = torch.cat([h_mean, f_signal, injection_weight], dim=-1) # [batch, hidden_dim + 2]
        
        # Compute gate g (alpha_O) [batch, 1]
        gate_g = self.mlp(mlp_input)
        
        # Broadcast gate across sequence
        seq_len = hidden_state.size(1)
        g_expanded = gate_g.unsqueeze(1).expand(-1, seq_len, -1) # [batch, seq_len, 1]
        
        return g_expanded
