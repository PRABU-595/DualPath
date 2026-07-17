#!/bin/bash

# DualPath-VLM GCP Training Launcher
# Usage: ./scripts/train_gcp.sh [GCS_BUCKET_URI]

GCS_BUCKET=${1:-"gs://dualpath-checkpoints-default"}
CONFIG_PATH="configs/base_config.yaml"
NUM_GPUS=2

echo "--- Initializing GCP Training Pipeline ---"
echo "Target Bucket: $GCS_BUCKET"

# 1. Environment Verification
if ! command -v nvidia-smi &> /dev/null
then
    echo "[ERROR] NVIDIA GPU not detected. GCP trainer requires A100/H100."
    exit 1
fi

# 2. Start Training with Accelerate
accelerate launch --num_processes=$NUM_GPUS \
    training/trainer.py \
    --config_path $CONFIG_PATH

# 3. Synchronize Checkpoints to GCS
# This handles persistence even if the VM is preempted.
echo "--- Backing up checkpoints to GCS ---"
gcloud storage cp -r ./outputs/checkpoints/* $GCS_BUCKET

echo "--- GCP Training Session Finished ---"
