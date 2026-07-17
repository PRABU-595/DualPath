import sys, os
# Ensure the project root is in PYTHONPATH when executing this file directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, List, Tuple, Dict, Any, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.attention_monitor import AttentionMonitor
from model.pathway_f import FabricationSuppressor
from model.pathway_o import OmissionCorrector
from model.gating_mlp import GatingMLP
from model.visual_drift_tracker import VisualDriftTracker

class DualPathVLM(nn.Module):
    """
    DualPath-VLM module that wraps a frozen Qwen2.5-VL backbone.
    Intercepts autoregressive decoding to suppress fabrication and correct omission.
    
    Supports two loading modes:
    - Full precision (bf16): For A100/H100 with >=40GB VRAM
    - 4-bit quantized (QLoRA): For T4/L4 with 16-24GB VRAM
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                 device: str = "cuda", 
                 baseline_mode: bool = False,
                 use_4bit: bool = False,
                 gradient_checkpointing: bool = False):
        """
        Args:
            model_id: The HuggingFace model ID for Qwen2.5-VL.
            device: The device to load the model on.
            baseline_mode: If True, bypasses DualPath logic for benchmark comparisons.
            use_4bit: If True, loads backbone in 4-bit (NF4) quantization for low-VRAM GPUs.
            gradient_checkpointing: If True, enables gradient checkpointing to reduce memory.
        """
        super().__init__()
        self.baseline_mode = baseline_mode
        self.use_4bit = use_4bit
        
        # 1. Load backbone
        if use_4bit:
            # QLoRA mode: 4-bit quantized backbone (~4GB instead of ~14GB)
            print("[DualPath] Loading backbone in 4-bit quantization (QLoRA mode for T4/L4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
                local_files_only=(os.environ.get("HF_HUB_OFFLINE") == "1"),
            )
            # Prepare for k-bit training (handles layer norms, etc.)
            self.backbone = prepare_model_for_kbit_training(
                self.backbone, 
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            )
        else:
            # Full precision mode for high-VRAM GPUs
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
                local_files_only=(os.environ.get("HF_HUB_OFFLINE") == "1"),
            )
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if gradient_checkpointing:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            
        # 2. Apply LoRA to the top 4 LLM transformer layers
        # Robust layer count detection (handles different transformer versions)
        if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'num_hidden_layers'):
            num_layers = self.backbone.config.num_hidden_layers
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'layers'):
            num_layers = len(self.backbone.model.layers)
        else:
            num_layers = 28  # Qwen2.5-VL-7B default
            print(f"[WARN] Could not auto-detect layer count, defaulting to {num_layers}")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=list(range(num_layers - 4, num_layers)),
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()
        
        # 3. Instantiate DualPath Modules
        # Robust hidden_dim detection (handles composite VLM configs)
        if hasattr(self.backbone.config, "hidden_size"):
            self.hidden_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "text_config") and hasattr(self.backbone.config.text_config, "hidden_size"):
            self.hidden_dim = self.backbone.config.text_config.hidden_size
        else:
            self.hidden_dim = 3584  # Default for Qwen2.5-VL-7B
            print(f"[WARN] Could not auto-detect hidden_dim, defaulting to {self.hidden_dim}")
            
        if hasattr(self.backbone.config, "vocab_size"):
            self.vocab_size = self.backbone.config.vocab_size
        else:
            self.vocab_size = 152064
            print(f"[WARN] Could not auto-detect vocab_size, defaulting to {self.vocab_size}")
        
        self.pathway_f = FabricationSuppressor(self.hidden_dim, self.vocab_size)
        self.pathway_o = OmissionCorrector(self.hidden_dim)
        self.gating = GatingMLP(self.hidden_dim)
        
        # Resolve lm_head through possible PEFT wrapper paths
        self._lm_head = self._get_lm_head()
        
        # Monitor and Tracker are initialized per-batch
        self.monitor = None
        self.drift_tracker = VisualDriftTracker(threshold=0.3)
        
        # Ensure DualPath modules are on the correct device/dtype
        compute_dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.target_dtype = compute_dtype
        self.pathway_f.to(device).to(compute_dtype)
        self.pathway_o.to(device).to(compute_dtype)
        self.gating.to(device).to(compute_dtype)
        self.drift_tracker.to(device).to(compute_dtype)
        
        print(f"[DualPath] Model loaded. 4-bit={use_4bit}, grad_ckpt={gradient_checkpointing}")

    def _get_lm_head(self):
        """Resolves the correct lm_head path through possible PEFT wrapper layers."""
        # Try direct access first
        if hasattr(self.backbone, 'lm_head'):
            return self.backbone.lm_head
        # PEFT wraps as base_model.model
        if hasattr(self.backbone, 'base_model'):
            base = self.backbone.base_model
            if hasattr(base, 'lm_head'):
                return base.lm_head
            if hasattr(base, 'model') and hasattr(base.model, 'lm_head'):
                return base.model.lm_head
        raise AttributeError("Cannot find lm_head in the model architecture")

    def _reset_pathway_states(self, context_start: int, context_end: int):
        """Resets both internal monitor and drift tracker for a new batch."""
        if self.monitor is None:
            self.monitor = AttentionMonitor(self.backbone, context_start, context_end)
        else:
            self.monitor.reset()
            self.monitor.context_token_range = (context_start, context_end)
        self.drift_tracker.reset()

    def forward(self, 
                input_ids: torch.LongTensor, 
                pixel_values: Optional[torch.FloatTensor] = None, 
                image_grid_thw: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) -> Union[CausalLMOutputWithPast, Dict[str, Any]]:
        """
        Full forward pass including DualPath logic during training.
        Supports Unified Mode (VLM or pure LLM text-only).
        """
        is_text_only = pixel_values is None
        
        # Baseline Mode: Standard backbone call
        if self.baseline_mode:
            if is_text_only:
                return self.backbone(input_ids=input_ids, labels=labels, **kwargs)
            return self.backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs
            )

        # 1. Forward Pass through backbone and capture hidden states/attention
        if is_text_only:
            outputs = self.backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                **kwargs
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=True,
                **kwargs
            )
        
        # 2. Update Signals and Tracking
        context_start, context_end = self._get_context_token_range(input_ids, labels, is_text_only)
        self._reset_pathway_states(context_start, context_end)

        # Capture signals from attention hooks
        f_signal = self.monitor.get_fabrication_signal()
        raw_drift = self.monitor.get_drift_signal()
        
        # Cast signals to model compute dtype (attention hooks return float32,
        # but pathway modules run in bfloat16 when backbone is 4-bit quantized)
        f_signal = f_signal.to(self.target_dtype)
        raw_drift = raw_drift.to(self.target_dtype)
        
        # Formulate formal drift signal via tracker
        d_signal = self.drift_tracker.update(raw_drift)

        # 3. DualPath Pathway Processing
        # h_current: last hidden state [batch, seq_len, hidden_dim]
        h_current = outputs.hidden_states[-1]
        context_embeddings = outputs.hidden_states[0][:, context_start:context_end, :]

        # Pathway F: Predicts logit-space penalty vector of size |V|
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
        # alpha_F = 1 - alpha_O
        alpha_F = 1.0 - gate_g
        # gate_g / alpha_F are [batch, seq_len, 1], so they broadcast correctly over vocab_size
        logits = logits - (alpha_F * f_penalty)
        
        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "gate_g": gate_g,
            "f_penalty": f_penalty,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }

    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.LongTensor, 
                 pixel_values: Optional[torch.FloatTensor] = None, 
                 image_grid_thw: Optional[torch.LongTensor] = None,
                 max_new_tokens: int = 128,
                 **kwargs):
        """
        Custom decoding loop to intercept at every generation step.
        """
        is_text_only = pixel_values is None
        
        # Baseline Mode: Use backbone's generate
        if self.baseline_mode:
            if is_text_only:
                return self.backbone.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)
            return self.backbone.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

        context_start, context_end = self._get_context_token_range(input_ids, None, is_text_only)
        self._reset_pathway_states(context_start, context_end)

        # Initial forward pass to extract context embeddings
        if is_text_only:
            outputs = self.backbone(
                input_ids=input_ids,
                output_hidden_states=True
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True
            )
        context_embeddings = outputs.hidden_states[0][:, context_start:context_end, :]

        self.ablation_mode = kwargs.get("ablation_mode", "learned")
        self._last_gate_history = []
        self._last_drift_history = []  # Added for mechanistic validation
        self._intervention_triggers_f = []
        self._intervention_triggers_o = []
        
        generated_ids = input_ids
        for _ in range(max_new_tokens):
            if is_text_only:
                out = self.backbone(
                    input_ids=generated_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
            else:
                out = self.backbone(
                    input_ids=generated_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_attentions=True,
                    output_hidden_states=True
                )
            
            h_current = out.hidden_states[-1][:, -1:, :].to(self.target_dtype)
            f_signal = self.monitor.get_fabrication_signal().to(self.target_dtype)
            raw_drift = self.monitor.get_drift_signal().to(self.target_dtype)
            d_signal = self.drift_tracker.update(raw_drift).to(self.target_dtype)
            context_embeddings = context_embeddings.to(self.target_dtype)
            
            f_penalty = self.pathway_f(h_current, f_signal)
            o_out, injection_w = self.pathway_o(h_current, d_signal, context_embeddings)
            gate_g = self.gating(h_current, f_signal, injection_w)
            
            # --- Ablation Hooks ---
            if self.ablation_mode == "force_f":
                gate_g = torch.zeros_like(gate_g) # 100% Pathway F
            elif self.ablation_mode == "force_o":
                gate_g = torch.ones_like(gate_g) # 100% Pathway O
            elif self.ablation_mode == "force_equal":
                gate_g = torch.ones_like(gate_g) * 0.5
            
            # Apply Omission correction to hidden state via alpha_O
            h_corrected = (1 - gate_g) * h_current + gate_g * o_out if self.ablation_mode != "baseline" else h_current
            
            # --- Analytics Cache ---
            gate_val = gate_g.detach().cpu().item()
            self._last_gate_history.append(gate_val)
            self._last_drift_history.append(d_signal.detach().cpu().item()) # Capture drift
            
            # Log binary trigger if pathway heavily dominated (alpha > 0.7)
            # gate_g represents alpha_O, so (1 - gate_g) is alpha_F
            fired_f = 1 if (1.0 - gate_val) > 0.7 else 0
            fired_o = 1 if gate_val > 0.7 else 0
            self._intervention_triggers_f.append(fired_f)
            self._intervention_triggers_o.append(fired_o)
            
            logits = self._lm_head(h_corrected)
            
            if self.ablation_mode != "baseline":
                alpha_F = 1.0 - gate_g
                logits = logits - (alpha_F * f_penalty)
                
            next_token = torch.argmax(logits, dim=-1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if (next_token == self.backbone.config.eos_token_id).all():
                break
                
        return generated_ids

    def _get_context_token_range(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor], is_text_only: bool) -> Tuple[int, int]:
        """Properly identifies context/image tokens for dual-path monitoring."""
        if is_text_only:
            # For pure LLM text, prompt/history serves as the ground truth context
            if labels is not None:
                # In causal LM, context/prompt is masked out with -100 in labels
                prompt_len = (labels[0] == -100).sum().item()
                if prompt_len > 0:
                    return (0, prompt_len)
            # Fallback for generation mode
            return (0, input_ids.shape[1])
            
        # VLM Mode: Find the <|image_pad|> sequence
        img_token_id = 151655 # Common for Qwen2
        
        # Find first and last occurance in batch 0
        img_pos = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]
        
        if len(img_pos) == 0:
            return (0, 0)
            
        return (int(img_pos[0]), int(img_pos[-1]) + 1)


if __name__ == "__main__":
    # Dummy Test Pass
    print("Testing DualPathVLM initialization and forward pass...")
    # Mocking Qwen2.5-VL config for initialization
    model = DualPathVLM(model_id="Qwen/Qwen2.5-VL-7B-Instruct") # Mocking local device
    
    batch_size = 2
    seq_len = 10
    img_size = 448
    
    dummy_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    dummy_pixels = torch.randn(batch_size, 3, img_size, img_size).to(torch.bfloat16).cuda()
    dummy_thw = torch.tensor([[1, 28, 28]] * batch_size).cuda()
    
    print("Running forward pass...")
    outputs = model(dummy_ids, dummy_pixels, dummy_thw)
    print(f"Output logits shape: {outputs.logits.shape}")
    print("Test passed.")
