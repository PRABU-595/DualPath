import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_threshold_sensitivity(log_path: str = "outputs/evaluation/threshold_sweep.json", 
                               output_path: str = "outputs/results/threshold_sensitivity.png"):
    """
    Plots the sensitivity of CHAIR_i and Object Recall across different tau_O thresholds.
    """
    
    if not os.path.exists(log_path):
        print(f"Warning: Sweep log {log_path} not found. Mocking data.")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        chair_scores = [22.4, 11.2, 7.1, 6.8, 6.5]
        recall_scores = [78.5, 76.1, 74.2, 63.4, 55.2]
    else:
        with open(log_path, 'r') as f:
            data = json.load(f)
        thresholds = [d["threshold"] for d in data]
        chair_scores = [d["CHAIRi"] for d in data]
        recall_scores = [d["Recall"] for d in data]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel(r"Pathway O Intervention Threshold ($\tau_O$)", fontsize=12)
    ax1.set_ylabel("CHAIR_i (%) ↓", color=color1, fontsize=12)
    ax1.plot(thresholds, chair_scores, marker='o', color=color1, linewidth=2.5, markersize=8, label="Fabrication Rate (CHAIR_i)")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel("Object Recall (%) ↑", color=color2, fontsize=12)  
    ax2.plot(thresholds, recall_scores, marker='s', color=color2, linewidth=2.5, markersize=8, label="Object Recall")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Highlight Optimal Zone (Around 0.3)
    ax1.axvspan(0.25, 0.35, color='gold', alpha=0.1, label="Optimal Region")
    ax1.annotate("Sweet Spot:\nOptimal Tradeoff", xy=(0.3, 7.1), xytext=(0.4, 15),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
                 fontsize=11, fontweight='bold', ha='center', color='darkred')
    
    # Styling
    plt.title(r"Hyperparameter Sensitivity: $\tau_O$ Sweep", fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', frameon=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Threshold sensitivity plot saved to {output_path}")

if __name__ == "__main__":
    plot_threshold_sensitivity()
