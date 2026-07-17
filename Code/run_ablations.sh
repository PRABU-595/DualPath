#!/bin/bash

# DualPath-VLM Ablation Study Launch Script
# Usage: ./scripts/run_ablations.sh

export CUDA_VISIBLE_DEVICES=0

echo "===================================================="
echo "Starting DualPath-VLM Ablation Study"
echo "===================================================="

# 1. Pathway F Only
echo "Ablation: Pathway F Only..."
python ablations/pathway_f_only.py

# 2. Pathway O Only
echo "Ablation: Pathway O Only..."
python ablations/pathway_o_only.py

# 3. Fixed Gate (g=0.5)
echo "Ablation: Fixed Gate..."
python ablations/fixed_gate.py

# 4. No Gate (Baseline)
echo "Ablation: No Gate (Vanilla)..."
python ablations/no_gate.py

echo "===================================================="
echo "Ablation Study Finished."
echo "Each script generates results for the POPE benchmark."
echo "===================================================="
