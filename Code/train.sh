#!/bin/bash

# DualPath-VLM Training Launch Script
# Usage: ./scripts/train.sh [CONFIG_PATH]

# 1. Environment Setup
# Assuming a virtual environment 'dualpath' is used
# source activate dualpath 
# Or if using a local venv:
# source venv/bin/activate

# 2. CUDA Devices
# Set to use 2x A100 GPUs (e.g., 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# 3. Accelerate Configuration
# If accelerate is not configured, run 'accelerate config' first.
# Here we assume 2 GPUs, multi-gpu training.
NUM_GPUS=2

# 4. Configuration Path
CONFIG_PATH=${1:-"configs/base_config.yaml"}

# 5. Output Management
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/train_$TIMESTAMP.log"

echo "===================================================="
echo "Starting DualPath-VLM Training"
echo "Config: $CONFIG_PATH"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"
echo "===================================================="

# 6. Launch Training
# Redirect both stdout and stderr to the log file for persistence
accelerate launch --num_processes=$NUM_GPUS \
    training/trainer.py \
    --config_path $CONFIG_PATH \
    2>&1 | tee $LOG_FILE

# 7. Post-Training Cleanup
echo "Training finished. Logs saved to $LOG_FILE"
