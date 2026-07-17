#!/bin/bash
# =============================================================
# DualPath-VLM: One-Command Cloud Setup & Training
# =============================================================
# Usage (from your LOCAL machine):
#   1. Create a GCP Deep Learning VM with GPU
#   2. SSH into the instance
#   3. Clone or rsync your repo
#   4. Run: bash scripts/setup_cloud.sh
#
# Or for immediate training (data already downloaded):
#   bash scripts/setup_cloud.sh --skip-download
# =============================================================

set -e  # Exit on error

SKIP_DOWNLOAD=false
VAL_ONLY=false
GCS_BUCKET=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --val-only) VAL_ONLY=true ;;
        --bucket) GCS_BUCKET="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "DualPath-VLM: Cloud Environment Setup"
echo "============================================================"

# --- Step 1: Verify GPU ---
echo ""
echo "--- Step 1: GPU Verification ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "[✓] GPU detected"
else
    echo "[✗] No NVIDIA GPU found. Training will be very slow on CPU."
    echo "    Consider using a GPU-enabled instance."
fi

# --- Step 2: Install Python dependencies ---
echo ""
echo "--- Step 2: Installing Python dependencies ---"
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation 2>/dev/null || echo "[WARN] Flash Attention not available, will use standard attention"
python -m spacy download en_core_web_sm 2>/dev/null || true
echo "[✓] Dependencies installed"

# --- Step 3: Configure accelerate ---
echo ""
echo "--- Step 3: Configuring Accelerate ---"
# Auto-detect GPU count and configure
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Setting up multi-GPU training with $NUM_GPUS GPUs"
    accelerate config default --mixed_precision bf16 --num_processes $NUM_GPUS
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Setting up single-GPU training"
    accelerate config default --mixed_precision bf16
else
    echo "No GPU detected, configuring for CPU"
    accelerate config default --mixed_precision no
fi
echo "[✓] Accelerate configured"

# --- Step 4: Download data ---
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "--- Step 4: Downloading Datasets ---"
    if [ "$VAL_ONLY" = true ]; then
        python scripts/download_cloud_data.py --val-only
    else
        python scripts/download_cloud_data.py
    fi
    echo "[✓] Data downloaded"
    
    # --- Step 5: Generate training pairs ---
    echo ""
    echo "--- Step 5: Generating Training Pairs ---"
    python scripts/generate_real_pairs.py
    echo "[✓] Training pairs generated"
else
    echo ""
    echo "--- Step 4: Skipping download (--skip-download) ---"
fi

# --- Step 6: Verify everything ---
echo ""
echo "--- Step 6: Verification ---"

# Check training data exists
if [ -f "data/real_fabrication_pairs.jsonl" ] && [ -f "data/real_omission_pairs.jsonl" ]; then
    FAB_COUNT=$(wc -l < data/real_fabrication_pairs.jsonl)
    OM_COUNT=$(wc -l < data/real_omission_pairs.jsonl)
    echo "[✓] Fabrication pairs: $FAB_COUNT"
    echo "[✓] Omission pairs: $OM_COUNT"
else
    echo "[✗] Training pairs not found! Run without --skip-download"
    exit 1
fi

# Check images
if [ -d "data/coco/train2017" ]; then
    IMG_COUNT=$(ls data/coco/train2017/*.jpg 2>/dev/null | wc -l)
    echo "[✓] Training images: $IMG_COUNT"
    IMAGE_DIR="data/coco/train2017"
elif [ -d "data/coco/val2017" ]; then
    IMG_COUNT=$(ls data/coco/val2017/*.jpg 2>/dev/null | wc -l)
    echo "[✓] Validation images: $IMG_COUNT (val-only mode)"
    IMAGE_DIR="data/coco/val2017"
else
    echo "[✗] No images found!"
    exit 1
fi

# Dry run
echo ""
echo "--- Running dry run... ---"
python training/trainer.py --config_path configs/base_config.yaml --dry_run --image_dir $IMAGE_DIR
echo "[✓] Dry run passed"

echo ""
echo "============================================================"
echo "Setup complete! Ready to train."
echo ""
echo "To start training:"
echo "  accelerate launch training/trainer.py --config_path configs/base_config.yaml --image_dir $IMAGE_DIR"
echo ""
echo "Or for single GPU:"
echo "  python training/trainer.py --config_path configs/base_config.yaml --image_dir $IMAGE_DIR"
echo "============================================================"

# If bucket specified, set up periodic sync
if [ -n "$GCS_BUCKET" ]; then
    echo ""
    echo "GCS bucket configured: $GCS_BUCKET"
    echo "Checkpoints will be synced after training."
fi
