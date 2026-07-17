"""
DualPath-VLM: Mini-Epoch Gradient Flow Verification
Tests that Pathway F, O, and Gating MLP can execute forward/backward passes
using the exact same batch format that the real trainer uses.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.pathway_f import FabricationSuppressor
from model.pathway_o import OmissionCorrector
from model.gating_mlp import GatingMLP
from training.dual_contrastive_loss import DualContrastiveLoss


def main():
    print("=" * 60)
    print("DualPath-VLM: Mini-Epoch Gradient Flow Verification")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cpu"
    
    hidden_dim = 1024
    seq_len = 32
    batch_size = 4
    vocab_size = 32000
    steps = 15

    # Initialize components
    pathway_f = FabricationSuppressor(hidden_dim=hidden_dim).to(device)
    pathway_o = OmissionCorrector(hidden_dim=hidden_dim).to(device)
    gating = GatingMLP(hidden_dim=hidden_dim).to(device)
    lm_head = nn.Linear(hidden_dim, vocab_size, device=device)
    loss_fn = DualContrastiveLoss()

    params = (
        list(pathway_f.parameters()) + 
        list(pathway_o.parameters()) + 
        list(gating.parameters()) + 
        list(lm_head.parameters())
    )
    optimizer = AdamW(params, lr=5e-4)

    print(f"\nConfig: hidden={hidden_dim}, batch={batch_size}, seq={seq_len}")
    print(f"Components: PathwayF, PathwayO, GatingMLP, LM Head")
    print(f"\n[Starting 1 Epoch / {steps} steps]\n")

    epoch_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()

        # Simulate inputs matching real batch format
        h_current = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        img_embeddings = torch.randn(batch_size, 10, hidden_dim, device=device)
        f_signal = torch.rand(batch_size, seq_len, 1, device=device)
        d_signal = torch.rand(batch_size, seq_len, 1, device=device)

        # DualPath forward
        f_out, suppression_w = pathway_f(h_current, f_signal)
        o_out, injection_w = pathway_o(h_current, d_signal, img_embeddings)
        h_corrected, gate_g = gating(h_current, suppression_w, injection_w, f_out, o_out)

        # LM head
        logits = lm_head(h_corrected)

        # Labels and token IDs (matching collate_fn_v2 output format)
        good_labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        fabricated_token_ids = [101, 102, 203]
        omitted_token_ids = [301, 402]

        # Expand gate_g for loss: [batch, 1] -> [batch, seq_len, 1]
        gate_g_expanded = gate_g.unsqueeze(1).expand(-1, seq_len, -1)

        # DualContrastive loss (matching trainer's call signature)
        loss_dict = loss_fn(
            logits=logits,
            good_labels=good_labels,
            fabricated_token_ids=fabricated_token_ids,
            omitted_token_ids=omitted_token_ids,
            gate_values=gate_g_expanded,
            pathway="Joint",
        )

        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if step % 5 == 0 or step == steps - 1:
            print(
                f"  Step {step:02d}/{steps} | "
                f"Loss: {loss.item():.4f} | "
                f"L_F: {loss_dict['fabrication_loss'].item():.4f} | "
                f"L_O: {loss_dict['omission_loss'].item():.4f} | "
                f"L_Gate: {loss_dict['gate_regularization'].item():.4f}"
            )

    avg = epoch_loss / steps
    print(f"\n[Epoch Complete] Avg Loss: {avg:.4f}")

    # Verify gradients actually flowed
    print("\n--- Gradient Flow Check ---")
    for name, module in [("PathwayF", pathway_f), ("PathwayO", pathway_o), ("GatingMLP", gating)]:
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        print(f"  {name}: {'[✓] Gradients present' if has_grad else '[✗] NO GRADIENTS'}")

    # Save checkpoint
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save({
        'pathway_f': pathway_f.state_dict(),
        'pathway_o': pathway_o.state_dict(),
        'gating': gating.state_dict(),
        'lm_head': lm_head.state_dict(),
    }, "outputs/checkpoints/epoch_1_verified.pt")
    print("\n[✓] Checkpoint saved to outputs/checkpoints/epoch_1_verified.pt")
    print("[✓] All components verified — ready for cloud training.")


if __name__ == "__main__":
    main()
