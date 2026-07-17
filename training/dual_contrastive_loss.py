import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set

class DualContrastiveLoss(nn.Module):
    """
    Implements the explicit token-level fabrication suppression and omission coverage metrics.
    """
    def __init__(self, 
                 lambda_f: float = 1.0, 
                 lambda_o: float = 1.0, 
                 lambda_g: float = 0.1):
        super().__init__()
        self.lambda_f = lambda_f
        self.lambda_o = lambda_o
        self.lambda_g = lambda_g

    def compute_L_fabrication(self, logits: torch.Tensor, good_labels: torch.Tensor, fabricated_token_ids: List[int]) -> torch.Tensor:
        """
        Suppresses the probability of fabricated tokens across the entire generated sequence.
        """
        # Standard Cross-Entropy on grounded (good) caption
        # shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = good_labels[..., 1:].contiguous()
        
        L_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        
        if not fabricated_token_ids:
            return L_ce

        # fabrication suppression penalty
        probs = F.softmax(shift_logits, dim=-1)
        
        # Gather probabilities of the fabricated tokens across all sequence steps
        fab_probs = []
        for f_token in fabricated_token_ids:
            # probability mass placed on this fabricated token across the sequence
            fab_probs.append(probs[..., f_token].mean())
            
        if fab_probs:
            # We want to minimize the probability of fabricated tokens
            L_suppress = torch.stack(fab_probs).mean()
        else:
            L_suppress = 0.0
            
        return L_ce + self.lambda_f * L_suppress

    def compute_L_omission(self, logits: torch.Tensor, good_labels: torch.Tensor, omitted_token_ids: List[int]) -> torch.Tensor:
        """
        Rewards the model for dedicating probability mass to omitted tokens (soft coverage).
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = good_labels[..., 1:].contiguous()
        
        L_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        
        if not omitted_token_ids:
            return L_ce

        probs = F.softmax(shift_logits, dim=-1)
        
        # Soft coverage: Max probability assigned to the omitted token at any point in the sequence
        coverage_scores = []
        for o_token in omitted_token_ids:
            # Did the model try to output this token somewhere?
            max_prob_in_seq, _ = probs[..., o_token].max(dim=-1)
            coverage_scores.append(max_prob_in_seq.mean()) # mean across batch
            
        if coverage_scores:
            avg_coverage = torch.stack(coverage_scores).mean()
            L_coverage = 1.0 - avg_coverage # Minimize (1 - coverage)
        else:
            L_coverage = 0.0

        return L_ce + self.lambda_o * L_coverage

    def compute_L_gating(self, gate_values: torch.Tensor) -> torch.Tensor:
        """
        Regularizes the Gating MLP so Pathway F dominates early and Pathway O dominates late.
        """
        # Ensure 3D shape: [batch, seq_len, 1]
        if gate_values.dim() == 2:
            gate_values = gate_values.unsqueeze(1)
            
        batch, seq_len, _ = gate_values.shape
        L_gate_total = 0.0
        
        # We expect alpha_F to decay from 1 -> 0 and alpha_O to rise from 0 -> 1 over time t.
        # Since gate_values = alpha_O, expected gate at step t is just t / seq_len.
        
        # Shape: [seq_len]
        t = torch.linspace(0.0, 1.0, steps=seq_len, device=gate_values.device)
        expected_O = t.view(1, seq_len, 1).expand(batch, -1, -1)
        
        # Mean Squared Error between actual alpha_O and expected O
        L_gate = F.mse_loss(gate_values, expected_O)
        
        return self.lambda_g * L_gate

    def forward(self, 
                logits: torch.Tensor, 
                good_labels: torch.Tensor, 
                fabricated_token_ids: List[int],
                omitted_token_ids: List[int],
                gate_values: torch.Tensor,
                pathway: str) -> Dict[str, torch.Tensor]:
        
        # Always add gate regularization if gating values exist
        l_g = torch.tensor(0.0, device=gate_values.device if gate_values is not None else "cpu")
        
        if logits is None:
            # Laptop Mode: No logits available, focus on Gating and Path Weights
            # We assign a small constant loss if gate_values is None to keep optimizer happy
            loss = torch.tensor(0.0, device=l_g.device, requires_grad=True)
            fab_loss = torch.tensor(0.0, device=l_g.device)
            om_loss = torch.tensor(0.0, device=l_g.device)
        else:
            if pathway == "F":
                loss = self.compute_L_fabrication(logits, good_labels, fabricated_token_ids)
                fab_loss = loss.detach()
                om_loss = torch.tensor(0.0, device=loss.device)
            elif pathway == "O":
                loss = self.compute_L_omission(logits, good_labels, omitted_token_ids)
                om_loss = loss.detach()
                fab_loss = torch.tensor(0.0, device=loss.device)
            else:
                # Joint Training
                l_f = self.compute_L_fabrication(logits, good_labels, fabricated_token_ids)
                l_o = self.compute_L_omission(logits, good_labels, omitted_token_ids)
                loss = (l_f + l_o) / 2.0
                fab_loss = l_f.detach()
                om_loss = l_o.detach()

        if gate_values is not None and (pathway == "Joint" or logits is None):
            l_g = self.compute_L_gating(gate_values)
            loss = loss + l_g

        return {
            "loss": loss,
            "fabrication_loss": fab_loss,
            "omission_loss": om_loss,
            "gate_regularization": l_g.detach() if isinstance(l_g, torch.Tensor) else l_g
        }
