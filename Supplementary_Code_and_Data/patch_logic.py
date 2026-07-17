import re

with open('model/dualpath_vlm_original.py', 'r') as f:
    content = f.read()

# 2. Patch Init (to provide vocab_size to FabricationSuppressor)
init_old = "self.pathway_f = FabricationSuppressor(self.hidden_dim)"
init_new = """vocab_size = getattr(self.backbone.config, "vocab_size", 152064)
        self.pathway_f = FabricationSuppressor(self.hidden_dim, vocab_size)"""
content = content.replace(init_old, init_new)

# 3. Patch Forward (Option B integration)
fwd_old = """        # Pathway F
        f_out, suppression_w = self.pathway_f(h_current, f_signal)
        # Pathway O
        o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
        # Gating
        h_corrected, gate_g = self.gating(h_current, suppression_w, injection_w, f_out, o_out)

        # 4. Final logit computation using the backbone's head
        logits = self._lm_head(h_corrected)"""

fwd_new = """        # Pathway F
        f_penalty = self.pathway_f(h_current, f_signal)
        # Pathway O
        o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
        # Gating
        gate_g = self.gating(h_current, f_signal, injection_w)

        h_corrected = (1 - gate_g) * h_current + gate_g * o_out
        
        logits = self._lm_head(h_corrected)
        
        alpha_F = 1.0 - gate_g
        logits = logits - (alpha_F * f_penalty)"""
content = content.replace(fwd_old, fwd_new)

# 4. Patch Generate Ablations (Option B integration)
gen1_old = """            f_out, suppression_w = self.pathway_f(h_current, f_signal)
            o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
            h_corrected, gate_g = self.gating(h_current, suppression_w, injection_w, f_out, o_out)
            
            # --- Ablation Hooks ---
            if self.ablation_mode == "force_f":
                gate_g = torch.zeros_like(gate_g) # 100% Pathway F
                h_corrected = h_current + f_out
            elif self.ablation_mode == "force_o":
                gate_g = torch.ones_like(gate_g) # 100% Pathway O
                h_corrected = h_current + o_out
            elif self.ablation_mode == "force_equal":
                gate_g = torch.ones_like(gate_g) * 0.5
                h_corrected = h_current + 0.5 * f_out + 0.5 * o_out
            elif self.ablation_mode == "baseline":
                h_corrected = h_current"""

gen1_new = """            f_penalty = self.pathway_f(h_current, f_signal)
            o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
            gate_g = self.gating(h_current, f_signal, injection_w)
            
            if self.ablation_mode == "force_f":
                gate_g = torch.zeros_like(gate_g)
            elif self.ablation_mode == "force_o":
                gate_g = torch.ones_like(gate_g)
            elif self.ablation_mode == "force_equal":
                gate_g = torch.ones_like(gate_g) * 0.5
            
            if self.ablation_mode != "baseline":
                h_corrected = (1 - gate_g) * h_current + gate_g * o_out
            else:
                h_corrected = h_current"""
content = content.replace(gen1_old, gen1_new)

# 5. Patch Generate Logits (Apply logit penalty)
gen2_old = """            logits = self._lm_head(h_corrected)
            next_token = torch.argmax(logits, dim=-1)"""

gen2_new = """            logits = self._lm_head(h_corrected)
            
            if self.ablation_mode != "baseline":
                alpha_F = 1.0 - gate_g
                logits = logits - (alpha_F * f_penalty)
                
            next_token = torch.argmax(logits, dim=-1)"""
content = content.replace(gen2_old, gen2_new)

with open('model/dualpath_vlm_fixed.py', 'w') as f:
    f.write(content)
print('Wrote dualpath_vlm_fixed.py')
