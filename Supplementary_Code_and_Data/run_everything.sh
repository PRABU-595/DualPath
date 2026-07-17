#!/bin/bash
# =============================================================
# DualPath-VLM: ONE COMMAND TO RULE THEM ALL
# Paste this entire script on your GPU server.
# It will: setup env → download data → train → evaluate → plot
# =============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_FILE="$PROJECT_DIR/run_everything.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "============================================================"
echo "  DualPath-VLM: Full Pipeline Starting"
echo "  Log: $LOG_FILE"
echo "  Time: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# STEP 0: GPU CHECK
# ---------------------------------------------------------------
echo ""
echo "[Step 0/7] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "[✓] GPU: $GPU_NAME"
else
    echo "[✗] ERROR: No NVIDIA GPU detected. Please use a GPU-enabled instance."
    echo "    For GCP: use n1-standard-8 with V100/A100 accelerator"
    exit 1
fi

# ---------------------------------------------------------------
# STEP 1: PYTHON ENVIRONMENT
# ---------------------------------------------------------------
echo ""
echo "[Step 1/7] Creating Python virtual environment..."

if [ ! -d "venv" ]; then
    # On Windows/GitBash, it's often 'python' not 'python3'
    PYTHON_CMD=$(command -v python3 || command -v python)
    $PYTHON_CMD -m venv venv
    echo "[✓] Virtual environment created using $PYTHON_CMD"
else
    echo "[✓] Virtual environment already exists"
fi

if [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d "venv/Scripts" ]; then
    source venv/Scripts/activate
else
    echo "[✗] ERROR: Could not find venv activation script."
    exit 1
fi
echo "[✓] venv activated: $(which python)"
pip install --upgrade pip --quiet

echo "[Step 1/7] Installing dependencies..."
pip install -r requirements.txt --quiet

# Flash Attention 2 — speeds up training 2x on A100/L4
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "A100|A10|L4|H100|V100"; then
    echo "  [✓] Compatible GPU detected — installing Flash Attention 2..."
    pip install flash-attn --no-build-isolation --quiet 2>/dev/null || \
        echo "  [WARN] Flash Attention build failed, falling back to standard attention"
else
    echo "  [WARN] T4 GPU detected — Flash Attention skipped (not compatible)"
fi

# spaCy model for precision evaluation
python -m spacy download en_core_web_sm --quiet 2>/dev/null || \
    python -m spacy download en_core_web_sm

echo "[✓] All dependencies installed"

# ---------------------------------------------------------------
# STEP 2: AUTO-CONFIGURE ACCELERATE
# ---------------------------------------------------------------
echo ""
echo "[Step 2/7] Configuring Accelerate for your GPU..."

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  Detected $NUM_GPUS GPU(s)"

cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "[✓] Accelerate configured (${NUM_GPUS} GPU, bf16 mixed precision)"

# ---------------------------------------------------------------
# STEP 3: DOWNLOAD DATASETS
# ---------------------------------------------------------------
echo ""
echo "[Step 3/7] Downloading datasets (COCO val2017 + Visual Genome)..."
echo "  This downloads ~1.5GB. Please wait..."

mkdir -p data/coco/annotations data/coco/val2017 data/vg

# COCO val2017 annotations (small — just JSON files)
if [ ! -f "data/coco/annotations/instances_val2017.json" ]; then
    echo "  Downloading COCO val2017 annotations..."
    wget -q --show-progress -O /tmp/annotations.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q /tmp/annotations.zip -d data/coco/
    rm /tmp/annotations.zip
    echo "  [✓] COCO annotations downloaded"
else
    echo "  [✓] COCO annotations already present"
fi

# COCO val2017 images (~800MB)
if [ ! "$(ls -A data/coco/val2017/ 2>/dev/null)" ]; then
    echo "  Downloading COCO val2017 images (~800MB)..."
    wget -q --show-progress -O /tmp/val2017.zip \
        http://images.cocodataset.org/zips/val2017.zip
    unzip -q /tmp/val2017.zip -d data/coco/
    rm /tmp/val2017.zip
    echo "  [✓] COCO val2017 images downloaded"
else
    echo "  [✓] COCO val2017 images already present"
fi

# Visual Genome objects (for Object Recall metric)
if [ ! -f "data/vg/objects.json" ]; then
    echo "  Downloading Visual Genome objects (~200MB)..."
    wget -q --show-progress -O /tmp/vg_objects.json.zip \
        https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
    unzip -q /tmp/vg_objects.json.zip -d data/vg/
    rm /tmp/vg_objects.json.zip
    echo "  [✓] Visual Genome downloaded"
else
    echo "  [✓] Visual Genome already present"
fi

echo "[✓] All datasets ready"

# ---------------------------------------------------------------
# STEP 4: GENERATE TRAINING PAIRS
# ---------------------------------------------------------------
echo ""
echo "[Step 4/7] Generating training pairs from real COCO annotations..."
echo "  Using 30,000 pairs per pathway (research-grade sample size)..."

# Use val2017 split since we only downloaded val (training on full train set
# would require an additional 18GB COCO train download).
# For 6-8 hour target, val2017 (5000 images → 30k pairs) is sufficient.
python scripts/generate_real_pairs.py \
    --coco_instances  data/coco/annotations/instances_val2017.json \
    --coco_captions   data/coco/annotations/captions_val2017.json \
    --vg_objects      data/vg/objects.json \
    --output_fab      data/real_fabrication_pairs.jsonl \
    --output_om       data/real_omission_pairs.jsonl \
    --num_samples     30000

FAB_COUNT=$(wc -l < data/real_fabrication_pairs.jsonl)
OM_COUNT=$(wc -l < data/real_omission_pairs.jsonl)
echo "[✓] Generated $FAB_COUNT fabrication pairs + $OM_COUNT omission pairs"

# ---------------------------------------------------------------
# STEP 5: TRAINING
# ---------------------------------------------------------------
echo ""
echo "[Step 5/7] Starting DualPath-VLM 3-Phase Training..."
echo "  Config: configs/base_config.yaml"
echo "  GPU: $GPU_NAME"
echo "  Time: $(date)"
echo ""

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Determine batch size based on VRAM
VRAM_GB=$(python -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(int(gb))
else:
    print(0)
")
echo "  Detected ${VRAM_GB}GB VRAM"

if [ "$VRAM_GB" -ge 40 ]; then
    BATCH=4; echo "  [A100 80GB] batch_size=4"
elif [ "$VRAM_GB" -ge 20 ]; then
    BATCH=2; echo "  [A100 40GB / L4] batch_size=2"
else
    BATCH=1; echo "  [T4 16GB] batch_size=1 — training will be slow"
fi

# Patch batch size for this run
sed -i "s/batch_size: .*/batch_size: $BATCH/" configs/base_config.yaml

# One GPU
python training/trainer.py \
    --config_path configs/base_config.yaml \
    --image_dir   data/coco/val2017

echo ""
echo "[✓] Training complete! Time: $(date)"

# ---------------------------------------------------------------
# STEP 6: EVALUATION SUITE
# ---------------------------------------------------------------
echo ""
echo "[Step 6/7] Running A*-Strong Evaluation Suite..."

mkdir -p outputs/evaluation outputs/results

# 6a. POPE (Adversarial)
echo "  Running POPE adversarial evaluation..."
python evaluation/eval_pope.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --limit 500 \
    --output_file outputs/evaluation/pope_results.json

# 6b. CHAIR
echo "  Running CHAIR hallucination evaluation..."
python evaluation/eval_chair.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --img_dir  data/coco/val2017 \
    --limit    200 \
    --output_file outputs/evaluation/chair_results.json

# 6c. Object Recall (Visual Genome)
echo "  Running Object Recall (Visual Genome)..."
python evaluation/eval_object_recall.py \
    --vg_json    data/vg/objects.json \
    --img_dir    data/coco/val2017 \
    --limit      200 \
    --output_file outputs/evaluation/recall_results.json

# 6d. Mechanistic Inference Log
echo "  Generating mechanistic inference log..."
python evaluation/inference_engine.py \
    --dataset coco \
    --limit   300 \
    --out     outputs/evaluation/inference_log.json

echo "[✓] All evaluations complete"

# ---------------------------------------------------------------
# STEP 7: GENERATE ALL PLOTS
# ---------------------------------------------------------------
echo ""
echo "[Step 7/7] Generating publication-ready figures..."

python scripts/plot_tradeoff_curve.py
python scripts/verify_attention_drift.py --log outputs/evaluation/inference_log.json
python scripts/plot_gating_behavior.py
python scripts/plot_intervention_freq.py
python scripts/plot_threshold_sensitivity.py

echo "[✓] All figures saved to outputs/results/"

# ---------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DualPath-VLM: FULL PIPELINE COMPLETE"
echo "  Time: $(date)"
echo "============================================================"
echo ""
echo "  Training Metrics : outputs/results/training_metrics.json"
echo "  POPE Results     : outputs/evaluation/pope_results.json"
echo "  CHAIR Results    : outputs/evaluation/chair_results.json"
echo "  Object Recall    : outputs/evaluation/recall_results.json"
echo "  Tradeoff Curve   : outputs/results/tradeoff_curve.png"
echo "  Attention Drift  : outputs/results/attention_drift_verification.png"
echo "  Gating Behavior  : outputs/results/gating_temporal_behavior.png"
echo "  Full Log         : $LOG_FILE"
echo "============================================================"
echo ""
echo "  To download all results to your local machine, run:"
echo "  gcloud compute scp GPU_INSTANCE:~/DualPath-VLM/outputs ./outputs --recurse"
echo "============================================================"
