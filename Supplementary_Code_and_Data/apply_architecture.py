import os
import re

pathway_f_code = '''import torch
import torch.nn as nn
from typing import Tuple

class FabricationSuppressor(nn.Module):
    """
    Suppresses fabrication hallucinations by predicting a logit-space penalty based on attention signals.
    Architecture: z_t^F = lambda_F * sigmoid(W_pen(ReLU(W_F([h_t ; rho_t]))))
    """
    def __init__(self, hidden_dim: int, vocab_size: int, d_F: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Linear projection W_F: R^{d+1} -> R^{d_F}
        self.W_F = nn.Linear(hidden_dim + 1, d_F)
        
        # Vocabulary projection W_pen: R^{d_F} -> R^{|V|}
        self.W_pen = nn.Linear(d_F, vocab_size)
        
        self.relu = nn.ReLU()
        
        # Learned scale parameter lambda_F (unconstrained here, constrained via softplus in forward)
        self.raw_lambda = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_state: torch.Tensor, fabrication_signal: torch.Tensor) -> torch.Tensor:
        # Ensure fabrication_signal is [batch, 1] or [batch, seq_len, 1]
        if fabrication_signal.dim() > 2:
            fabrication_signal = fabrication_signal.mean(dim=(1, 2)).unsqueeze(-1)
        elif fabrication_signal.dim() == 1:
            fabrication_signal = fabrication_signal.unsqueeze(-1)
            
        # Match seq_len of hidden_state if necessary
        batch_size, seq_len, _ = hidden_state.shape
        if fabrication_signal.size(1) != seq_len:
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
'''

gating_mlp_code = '''import torch
import torch.nn as nn
from typing import Tuple

class GatingMLP(nn.Module):
    """
    Arbitrates between Fabrication and Omission pathways per decoding step.
    Produces a single complementary gate alpha_O (and alpha_F = 1 - alpha_O).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # MLP: Takes [hidden_state_mean, f_signal_mean, injection_weight]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        nn.init.constant_(self.mlp[3].bias, 0.5)

    def forward(self, hidden_state: torch.Tensor, f_signal: torch.Tensor, injection_weight: torch.Tensor) -> torch.Tensor:
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
'''

with open("model/pathway_f.py", "w") as f:
    f.write(pathway_f_code)
print("Updated model/pathway_f.py")

with open("model/gating_mlp.py", "w") as f:
    f.write(gating_mlp_code)
print("Updated model/gating_mlp.py")

# Update dualpath_vlm.py
with open("model/dualpath_vlm.py", "r") as f:
    dv_code = f.read()

# Fix init
dv_code = re.sub(
    r"self\.pathway_f = FabricationSuppressor\(self\.hidden_dim\)",
    r"if hasattr(self.backbone.config, 'vocab_size'):\n            self.vocab_size = self.backbone.config.vocab_size\n        else:\n            self.vocab_size = 152064\n        \n        self.pathway_f = FabricationSuppressor(self.hidden_dim, self.vocab_size)",
    dv_code
)

# Fix forward integration
old_forward = '''        # Pathway F
        f_out, suppression_w = self.pathway_f(h_current, f_signal)
        # Pathway O
        o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
        # Gating
        h_corrected, gate_g = self.gating(h_current, suppression_w, injection_w, f_out, o_out)

        # 4. Final logit computation using the backbone's head
        logits = self._lm_head(h_corrected)'''

new_forward = '''        # Pathway F: Predicts logit-space penalty vector of size |V|
        f_penalty = self.pathway_f(h_current, f_signal)
        
        # Pathway O: Predicts corrected hidden state
        o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
        
        # Gating: Arbitrates between pathways (returns alpha_O)
        gate_g = self.gating(h_current, f_signal, injection_w) # alpha_O

        # 4. Final integration (Option B)
        # Apply Omission correction to hidden state via alpha_O
        h_corrected = (1 - gate_g) * h_current + gate_g * o_out
        
        # Project through backbone head to get raw logits
        logits = self._lm_head(h_corrected)
        
        # Apply Fabrication suppression penalty directly in logit space
        alpha_F = 1.0 - gate_g
        logits = logits - (alpha_F * f_penalty)'''

if old_forward in dv_code:
    dv_code = dv_code.replace(old_forward, new_forward)
    print("Updated dualpath_vlm.py forward()")

# Add f_penalty to return
old_return = '''        return {
            "loss": loss,
            "logits": logits,
            "gate_g": gate_g,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }'''
new_return = '''        return {
            "loss": loss,
            "logits": logits,
            "gate_g": gate_g,
            "f_penalty": f_penalty,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }'''
if old_return in dv_code:
    dv_code = dv_code.replace(old_return, new_return)
    print("Updated dualpath_vlm.py return dict")
    
with open("model/dualpath_vlm.py", "w") as f:
    f.write(dv_code)

# Update retrain_phase1_verified.py to include penalty pre-flight
with open("retrain_phase1_verified.py", "r") as f:
    tr_code = f.read()

old_preflight = '''                sig_val = f_sig.mean().item()
                preflight_signals.append(sig_val)
                print(f"  Batch {i:2d} | f_signal = {sig_val:.6f}")'''

new_preflight = '''                sig_val = f_sig.mean().item()
                preflight_signals.append(sig_val)
                print(f"  Batch {i:2d} | f_signal = {sig_val:.6f}")
                
                if "f_penalty" in outputs:
                    pen = outputs["f_penalty"].float()
                    p_min, p_max, p_mean = pen.min().item(), pen.max().item(), pen.mean().item()
                    print(f"           | penalty  = [{p_min:.6f}, {p_mean:.6f}, {p_max:.6f}]")'''

if old_preflight in tr_code:
    tr_code = tr_code.replace(old_preflight, new_preflight)
    with open("retrain_phase1_verified.py", "w") as f:
        f.write(tr_code)
    print("Updated retrain_phase1_verified.py with penalty pre-flight check")
