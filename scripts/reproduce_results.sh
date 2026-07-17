#!/bin/bash

# DualPath-VLM Master Reproduction Pipeline
# Automates Phase 2 (Hypothesis Validation) -> Phase 6 (Ablations) -> Phase 8 (Results)

CHECKPOINT=${1:-"./outputs/checkpoints/checkpoint-final"}
QUICK_MODE=${2:-""}

echo "===================================================="
echo "DualPath-VLM: Full Experimental Replication"
echo "Checkpoint: $CHECKPOINT"
echo "===================================================="

# 1. Phase 2: Hypothesis Validation (Visualizing Attention Ratios)
echo "[1/4] Running Attention Ratio Probe..."
python utils/visualize_attention.py
echo "Hypothesis plot saved to attention_drift_analysis.png"

# 2. Phase 3 & 5: Baseline vs DualPath Benchmarking
echo "[2/4] Running Main Benchmarks (POPE)..."
if [ "$QUICK_MODE" == "--quick" ]; then
    echo "Quick mode enabled: Skipping full benchmark."
else
    python evaluation/eval_pope.py --baseline
    python evaluation/eval_pope.py --checkpoint $CHECKPOINT
fi

# 3. Phase 6: Ablation Studies
echo "[3/4] Running Pathway Ablations..."
./scripts/run_ablations.sh

# 4. Phase 8: Final Dashboard Generation
echo "[4/4] Consolidating Results into Dashboard..."
echo "Please check RESULTS_DASHBOARD.md for the finalized metrics."

echo "===================================================="
echo "Replication Complete."
echo "===================================================="
