import torch
import torch.nn as nn
from typing import List, Dict, Optional

class AttentionMonitor:
    """
    Registers forward hooks on self-attention layers of a transformer LLM (Qwen2.5-VL).
    Extracts visual and text attention to compute fabrication and drift signals.
    """
    def __init__(self, model: nn.Module, context_token_start: int = 0, context_token_end: int = 0):
        """
        Args:
            model: The frozen LLM backbone.
            context_token_start: The starting index of context/image tokens in the sequence.
            context_token_end: The ending index of context/image tokens in the sequence.
        """
        self.model = model
        self.context_token_range = (context_token_start, context_token_end)
        self.hooks = []
        self.attn_weights = {}
        self.step0_context_attn = None
        self.current_step = 0
        
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on all self-attention layers."""
        for name, module in self.model.named_modules():
            # Targets Qwen2.5-VL attention modules
            if any(key in name.lower() for key in ["self_attn", "attn"]):
                if not any(sub in name.lower() for sub in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    hook = module.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)

    def _hook_fn(self, name: str):
        """The hook function to capture attention weights."""
        def hook(module, input, output):
            # Qwen2.5-VL attention output usually includes weights if output_attentions=True
            # However, for efficiency, we might need to intercept the attention computation or
            # ensure the model is called with output_attentions=True.
            # If weights are not in output, we assume they are cached or we'll need to 
            # modify the forward pass of the attention layer itself.
            if isinstance(output, tuple) and len(output) > 1:
                # [batch, num_heads, seq_len, seq_len]
                self.attn_weights[name] = output[1]
        return hook

    def reset(self):
        """Resets the monitor for a new generation sequence."""
        self.attn_weights = {}
        self.step0_context_attn = None
        self.current_step = 0

    def get_fabrication_signal(self) -> torch.Tensor:
        """
        Computes the text/image attention ratio.
        ratio = text_attention / (image_attention + 1e-8)
        Returns: Tensor of shape [batch, num_layers, num_heads]
        """
        if not self.attn_weights:
            return torch.zeros(1)
            
        signals = []
        start, end = self.context_token_range
        
        for name, weights in self.attn_weights.items():
            if weights is None: continue
            try:
                current_token_weights = weights[:, :, -1, :]
                context_attn = current_token_weights[:, :, start:end].sum(dim=-1)
                generation_attn = current_token_weights.sum(dim=-1) - context_attn
                ratio = generation_attn / (context_attn + 1e-8)
                signals.append(ratio)
            except Exception:
                continue
            
        if not signals:
            return torch.zeros(1, 28, 1).to(next(self.model.parameters()).device)
            
        return torch.stack(signals, dim=1)

    def get_drift_signal(self) -> torch.Tensor:
        """
        Computes visual attention drift: drift = step0_visual_attn - current_visual_attn.
        Returns: Tensor of shape [batch, 1] (averaged over heads/layers)
        """
        if not self.attn_weights:
            return torch.zeros(1)
            
        start, end = self.context_token_range
        current_context_attns = []
        
        for name, weights in self.attn_weights.items():
            if weights is None: continue
            try:
                current_token_weights = weights[:, :, -1, :]
                context_attn = current_token_weights[:, :, start:end].sum(dim=-1).mean(dim=-1)
                current_context_attns.append(context_attn)
            except Exception:
                continue
            
        if not current_context_attns:
            avg_current_context_attn = torch.zeros(1).to(next(self.model.parameters()).device)
        else:
            avg_current_context_attn = torch.stack(current_context_attns, dim=1).mean(dim=1)
        
        if self.step0_context_attn is None:
            self.step0_context_attn = avg_current_context_attn
            return torch.zeros_like(avg_current_context_attn).unsqueeze(-1)
            
        drift = self.step0_context_attn - avg_current_context_attn
        return drift.unsqueeze(-1)

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
