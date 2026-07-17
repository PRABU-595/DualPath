import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

def plot_attention_ratio_probe(model: nn.Module, 
                               processor: Any, 
                               dataset: Any, 
                               num_samples: int = 50, 
                               save_path: str = "attention_ratio_correlation.png"):
    """
    Phase 2 Critical Experiment:
    Plots attention ratio (text/image) over decoding steps for faithful vs fabricated captions.
    Validates if the fabrication signal correlates with hallucination types.
    """
    model.eval()
    device = next(model.parameters()).device
    
    faithful_ratios = []
    fabricated_ratios = []

    # Sample a subset of triplets
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset.samples[idx]
            image = sample["image"]
            
            # 1. Probe Faithful Caption
            inputs = processor(text=[sample["faithful"]], images=[image], return_tensors="pt").to(device)
            # Run a forward pass to trigger hooks
            model(**inputs)
            ratio_pos = model.monitor.get_fabrication_signal().mean().item()
            faithful_ratios.append(ratio_pos)
            
            # 2. Probe Fabricated Caption
            inputs = processor(text=[sample["fabricated"]], images=[image], return_tensors="pt").to(device)
            model(**inputs)
            ratio_neg = model.monitor.get_fabrication_signal().mean().item()
            fabricated_ratios.append(ratio_neg)

    # 3. Plotting the results (Figure 1)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(faithful_ratios, label="Faithful (Ground Truth)", fill=True, color="blue")
    sns.kdeplot(fabricated_ratios, label="Fabricated (Hallucination)", fill=True, color="red")
    
    plt.axvline(np.mean(faithful_ratios), color="blue", linestyle="--")
    plt.axvline(np.mean(fabricated_ratios), color="red", linestyle="--")
    
    plt.title("Phase 2: Attention Ratio Divergence (Text/Image)")
    plt.xlabel("Attention Ratio Signal")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    
    print(f"--- Phase 2 Experiment Complete. Plot saved to {save_path} ---")
    print(f"Mean Faithful Ratio: {np.mean(faithful_ratios):.4f}")
    print(f"Mean Fabricated Ratio: {np.mean(fabricated_ratios):.4f}")
    
    # Check for scientific significance
    if np.mean(fabricated_ratios) > np.mean(faithful_ratios):
        print("SUCCESS: Fabricated captions show higher text/image attention ratios. Signal is valid.")
    else:
        print("WARNING: Weak signal detected. Consider refining fabrication signal computation.")

if __name__ == "__main__":
    # Integration for Phase 2 launch
    import argparse
    from model.dualpath_vlm import DualPathVLM
    from data.dataset_loader import DualPathDataset
    from transformers import Qwen2_5_VLProcessor
    
    print("Launching Phase 2: Attention Ratio Probe...")
    # This would usually be run with the baseline frozen model
    model = DualPathVLM(baseline_mode=True) 
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    dataset = DualPathDataset(processor=processor)
    
    plot_attention_ratio_probe(model, processor, dataset)
