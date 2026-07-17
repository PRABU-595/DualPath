import torch
import torch.nn as nn
from typing import Tuple

class FabricationSuppressor(nn.Module):
    """
    Suppresses fabrication hallucinations by predicting a logit-space penalty based on attention signals.
    Architecture: z_t^F = lambda_F * sigmoid(W_pen(ReLU(W_F([h_t ; rho_t]))))
    """
    def __init__(self, hidden_dim: int, vocab_size: int, d_F: int = 1024):
        """
        Args:
            hidden_dim: The dimension of the hidden state (d).
            vocab_size: Size of the vocabulary (|V|).
            d_F: Intermediate bottleneck dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Linear projection W_F: R^{d+1} -> R^{d_F}
        self.W_F = nn.Linear(hidden_dim + 1, d_F)
        
        # Vocabulary projection W_pen: R^{d_F} -> R^{|V|}
        self.W_pen = nn.Linear(d_F, vocab_size)
        
        self.relu = nn.ReLU()
        
        # Learned scale parameter lambda_F (unconstrained here, constrained via softplus in forward)
        # Initialize slightly positive so softplus(raw_lambda) starts reasonably
        self.raw_lambda = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_state: torch.Tensor, fabrication_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: The current token hidden state [batch, seq_len, hidden_dim].
            fabrication_signal: Fabrication signal averaged across layers/heads [batch, 1] or [batch, seq_len, 1].
            
        Returns:
            penalty: The calculated logit penalty [batch, seq_len, vocab_size].
        """
        # Ensure fabrication_signal is [batch, 1] or [batch, seq_len, 1]
        if fabrication_signal.dim() > 2:
            # Average across layers and heads for a single suppressor signal
            fabrication_signal = fabrication_signal.mean(dim=(1, 2)).unsqueeze(-1)
        elif fabrication_signal.dim() == 1:
            fabrication_signal = fabrication_signal.unsqueeze(-1)
            
        # Match seq_len of hidden_state if necessary (e.g. during generation vs training)
        batch_size, seq_len, _ = hidden_state.shape
        if fabrication_signal.size(1) != seq_len:
            # Assume fabrication_signal is [batch, 1] and we need to expand it
            # (Though typically fabrication_signal is per-token during training)
            fabrication_signal = fabrication_signal.expand(-1, seq_len, -1)
            
        # Concatenate hidden state and fabrication signal: [batch, seq_len, d+1]
        combined_input = torch.cat([hidden_state, fabrication_signal], dim=-1)
        
        # Project through W_F
        x = self.W_F(combined_input)
        x = self.relu(x)
        
        # Project to vocabulary size
        raw_penalty = self.W_pen(x) # [batch, seq_len, vocab_size]
        
        # Constrain lambda_F to be non-negative
        lambda_F = torch.nn.functional.softplus(self.raw_lambda)
        
        # Final penalty bounded by sigmoid and scaled by lambda_F
        penalty = lambda_F * torch.sigmoid(raw_penalty)
        
        return penalty
