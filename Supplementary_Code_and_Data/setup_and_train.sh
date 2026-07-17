#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║   DualPath-VLM  —  Git Bash / Windows Setup & Training      ║
# ║   Usage (from Git Bash):  bash setup_and_train.sh           ║
# ║   Assumes ALL zip files are in C:/Users/Admin/Desktop/123/  ║
# ╚══════════════════════════════════════════════════════════════╝
set -e

PROJECT_DIR="C:/Users/Admin/Desktop/123/DualPath-VLM_Deployment"
ZIPS_DIR="C:/Users/Admin/Desktop/123"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     DualPath-VLM  ·  Windows/Git Bash Auto Setup            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─── STEP 1: Extract datasets ────────────────────────────────────────────────
echo ""
echo "► [1/5] Extracting datasets into project data/ folder..."

mkdir -p "$PROJECT_DIR/data/coco"
mkdir -p "$PROJECT_DIR/data/DualPath_Text_Data"

# COCO train images
if [ ! -d "$PROJECT_DIR/data/coco/train2017" ] || [ -z "$(ls -A "$PROJECT_DIR/data/coco/train2017" 2>/dev/null)" ]; then
    echo "  Extracting train2017.zip (~18 GB, please wait)..."
    unzip -q "$ZIPS_DIR/train2017.zip" -d "$PROJECT_DIR/data/coco/"
    echo "  ✓ COCO train2017 done."
else
    echo "  ✓ train2017 already extracted."
fi

# COCO val images
if [ ! -d "$PROJECT_DIR/data/coco/val2017" ] || [ -z "$(ls -A "$PROJECT_DIR/data/coco/val2017" 2>/dev/null)" ]; then
    echo "  Extracting val2017.zip..."
    unzip -q "$ZIPS_DIR/val2017.zip" -d "$PROJECT_DIR/data/coco/"
    echo "  ✓ COCO val2017 done."
else
    echo "  ✓ val2017 already extracted."
fi

# COCO annotations
if [ ! -f "$PROJECT_DIR/data/coco/annotations/captions_train2017.json" ]; then
    echo "  Extracting annotations..."
    unzip -q "$ZIPS_DIR/annotations_trainval2017.zip" -d "$PROJECT_DIR/data/coco/"
    echo "  ✓ Annotations done."
else
    echo "  ✓ Annotations already extracted."
fi

# Text datasets
if [ ! -f "$PROJECT_DIR/data/DualPath_Text_Data/text_fabrication_pairs.jsonl" ]; then
    echo "  Extracting DualPath_Text_Data.zip (236K text pairs)..."
    unzip -q "$ZIPS_DIR/DualPath_Text_Data.zip" -d "$PROJECT_DIR/data/"
    echo "  ✓ Text datasets done."
else
    echo "  ✓ Text datasets already extracted."
fi

# ─── STEP 2: Virtual environment ─────────────────────────────────────────────
echo ""
echo "► [2/5] Setting up Python virtual environment..."
cd "$PROJECT_DIR"

if [ ! -d "dualpath_env" ]; then
    python -m venv dualpath_env
    echo "  ✓ Virtual environment created."
else
    echo "  ✓ Virtual environment already exists."
fi

# Windows Git Bash uses Scripts/activate (not bin/activate)
source dualpath_env/Scripts/activate
echo "  ✓ Virtual environment activated."

# ─── STEP 3: Install packages ────────────────────────────────────────────────
echo ""
echo "► [3/5] Installing Python packages (5-10 min)..."
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Packages installed."

# ─── STEP 4: spaCy language model ────────────────────────────────────────────
echo ""
echo "► [4/5] Downloading spaCy model..."
python -m spacy download en_core_web_sm -q
echo "  ✓ spaCy ready."

# ─── STEP 5: Patch data paths to relative (works on any Windows path) ────────
echo ""
echo "► [5/5] Patching data paths to local project paths..."

# Replace the hardcoded D:\Dataset path with the relative path inside the project
python - <<'PYEOF'
import re

with open("gpu_train.py", "r", encoding="utf-8") as f:
    content = f.read()

# Patch all variants of the Windows dataset path
content = re.sub(
    r'[rR]?"?[Dd]:\\\\?Dataset\\\\?DualPath_Text_Data\\\\?"?',
    '"data/DualPath_Text_Data"',
    content
)
# Also patch any remaining raw string variants
content = content.replace(
    r"D:\Dataset\DualPath_Text_Data",
    "data/DualPath_Text_Data"
)

with open("gpu_train.py", "w", encoding="utf-8") as f:
    f.write(content)

print("  ✓ gpu_train.py paths patched successfully.")
PYEOF

# ─── LAUNCH TRAINING ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup complete! Launching Unified Training...             ║"
echo "║   Mode    : UNIFIED (Vision + Text simultaneously)          ║"
echo "║   VLM data: 118,287 COCO image pairs                       ║"
echo "║   LLM data: 236,574 text pairs (HaluEval+XSum+...)         ║"
echo "║   Total   : 473,148 training samples                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python gpu_train.py --mode unified --skip_data
