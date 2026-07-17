from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn as nn
from typing import List, Optional

def setup_lora(model: nn.Module, 
               target_modules: List[str] = ["q_proj", "v_proj"], 
               r: int = 16, 
               alpha: int = 32, 
               dropout: float = 0.05) -> PeftModel:
    """
    Applies LoRA to the vision-language model backbone.
    
    Args:
        model: The backbone model.
        target_modules: List of modules to target for LoRA.
        r: Rank of LoRA adapters.
        alpha: Scaling factor.
        dropout: Dropout rate for LoRA.
        
    Returns:
        peft_model: The model with LoRA adapters injected.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 1. Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # 2. Set only LoRA parameters to trainable
    peft_model.print_trainable_parameters()
    
    return peft_model
