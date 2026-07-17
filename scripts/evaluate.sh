#!/bin/bash

# DualPath-VLM Evaluation Launch Script
# Usage: ./scripts/evaluate.sh --model_path [CHECKPOINT_OR_MODEL_ID]

MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}
export CUDA_VISIBLE_DEVICES=0

echo "===================================================="
echo "Starting DualPath-VLM Full Evaluation"
echo "Model Path: $MODEL_PATH"
echo "===================================================="

# 1. POPE Evaluation
echo "Running POPE..."
python evaluation/eval_pope.py --model_id $MODEL_PATH

# 2. CHAIR Evaluation
echo "Running CHAIR (COCO)..."
python evaluation/eval_chair.py --model_id $MODEL_PATH

# 3. AMBER Evaluation
echo "Running AMBER..."
python evaluation/eval_amber.py --model_id $MODEL_PATH

# 4. MMHal-Bench Evaluation
echo "Running MMHal-Bench..."
python evaluation/eval_mmhal.py --model_id $MODEL_PATH

echo "===================================================="
echo "Evaluation Pipeline Finished."
echo "Results saved to local .json files."
echo "===================================================="
