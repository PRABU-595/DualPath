"""
╔══════════════════════════════════════════════════════════════════╗
║  DualPath-VLM: GPU SMART TRAINER                                ║
║  ONE COMMAND. ZERO MANUAL STEPS.                                 ║
║                                                                  ║
║  This script:                                                    ║
║    1. Auto-detects GPU / CPU                                     ║
║    2. Sets up environment if needed (venv, deps, spacy model)    ║
║    3. Downloads COCO data if missing                             ║
║    4. Generates training pairs                                   ║
║    5. Runs the full 3-phase training pipeline                    ║
║    6. Saves checkpoints, metrics, and training curves            ║  
║                                                                  ║
║  Usage (PowerShell / Git Bash / CMD):                            ║
║    python gpu_train.py                                           ║
║                                                                  ║
║  Options:                                                        ║
║    python gpu_train.py --epochs 300                              ║
║    python gpu_train.py --skip_data   (if data already exists)    ║
║    python gpu_train.py --dry_run     (verify only, no training)  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
import time
import json
import argparse
import shutil

# ── Ensure project root is ALWAYS on sys.path ──────────────────────────────
# This must happen before any local module imports (training/, data/, model/)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ============================================================
# PHASE 0: Environment Bootstrap
# ============================================================

def ensure_environment():
    """Check and install all dependencies."""
    print("\n" + "=" * 60)
    print("  [PHASE 0] Environment Bootstrap")
    print("=" * 60)
    
    # Check Python
    print(f"  Python: {sys.version}")
    print(f"  CWD:    {os.getcwd()}")
    
    # Check torch + GPU
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU:     {gpu_name} ({vram:.1f} GB VRAM)")
            print(f"  CUDA:    {torch.version.cuda}")
        else:
            print("  GPU:     None detected (will use CPU)")
    except ImportError:
        print("  [!] PyTorch not found. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
        import torch
    
    # Check other critical deps
    missing = []
    for pkg in ["transformers", "peft", "accelerate", "yaml", "tqdm", "PIL", "spacy", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            real_name = {"yaml": "pyyaml", "PIL": "Pillow"}.get(pkg, pkg)
            missing.append(real_name)
    
    if missing:
        print(f"  Installing missing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["-q"])
    
    # spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("  spaCy:   en_core_web_sm ✓")
    except (OSError, ImportError):
        print("  spaCy:   Downloading en_core_web_sm...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
    
    print("  [✓] Environment ready.\n")


# ============================================================
# PHASE 1: Data Acquisition
# ============================================================

def ensure_data():
    """Download COCO 2017 annotations and images if not present."""
    import urllib.request
    import zipfile
    
    print("=" * 60)
    print("  [PHASE 1] Data Acquisition")
    print("=" * 60)
    
    os.makedirs("data/coco/annotations", exist_ok=True)
    os.makedirs("data/coco/train2017", exist_ok=True)
    
    # Check annotations
    ann_file = "data/coco/annotations/instances_train2017.json"
    cap_file = "data/coco/annotations/captions_train2017.json"
    
    if not os.path.exists(ann_file) or not os.path.exists(cap_file):
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = "data/coco/annotations_trainval2017.zip"
        
        print(f"  Downloading COCO annotations...")
        urllib.request.urlretrieve(ann_url, ann_zip)
        
        print(f"  Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zf:
            zf.extractall("data/coco/")
        os.remove(ann_zip)
        print(f"  [✓] Annotations ready.")
    else:
        print(f"  [✓] Annotations already exist.")

    # Check for images (we only need a sample for training)
    img_dir = "data/coco/train2017"
    existing_imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    
    if len(existing_imgs) < 100:
        print(f"  Downloading COCO train2017 images (this may take a while)...")
        img_url = "http://images.cocodataset.org/zips/train2017.zip"
        img_zip = "data/coco/train2017.zip"
        
        urllib.request.urlretrieve(img_url, img_zip)
        
        print(f"  Extracting images...")
        with zipfile.ZipFile(img_zip, 'r') as zf:
            zf.extractall("data/coco/")
        os.remove(img_zip)
        print(f"  [✓] Images ready.")
    else:
        print(f"  [✓] {len(existing_imgs)} images already exist.")
    
    print()


# ============================================================
# PHASE 2: Training Pair Generation
# ============================================================

def ensure_training_pairs():
    """Generate fabrication/omission training pairs from COCO data."""
    print("=" * 60)
    print("  [PHASE 2] Training Pair Generation")
    print("=" * 60)
    
    fab_path = "data/real_fabrication_pairs.jsonl"
    om_path = "data/real_omission_pairs.jsonl"
    
    if os.path.exists(fab_path) and os.path.exists(om_path):
        fab_count = sum(1 for _ in open(fab_path))
        om_count = sum(1 for _ in open(om_path))
        print(f"  [✓] Training pairs exist: {fab_count} fabrication, {om_count} omission")
        print()
        return
    
    print("  Generating training pairs from COCO annotations...")
    subprocess.check_call([
        sys.executable, "scripts/generate_real_pairs.py",
        "--num_samples", "118287"  # Full COCO train2017 — all 118,287 images
    ])
    print("  [✓] Training pairs generated.\n")


# ============================================================
# PHASE 3: SMART Training
# ============================================================

def run_training(args):
    """Run the full DualPath-VLM training pipeline with smart hardware detection."""
    import torch
    import yaml
    
    print("=" * 60)
    print("  [PHASE 3] DualPath-VLM SMART Training")
    print("=" * 60)
    
    # Load config
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # ---- SMART HARDWARE ADAPTATION ----
    device = "cpu"
    use_4bit = False
    gradient_checkpointing = False
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"\n  [GPU MODE] {gpu_name} ({vram_gb:.1f} GB)")
        
        if vram_gb < 16:
            # Small GPU: Aggressive quantization
            config["training"]["batch_size"] = 1
            use_4bit = True
            gradient_checkpointing = True
            print(f"    Strategy: 4-bit + GradChkpt + BS=1")
        elif vram_gb < 24:
            # Medium GPU (T4/RTX 3090): Standard quantization
            config["training"]["batch_size"] = 2
            use_4bit = True
            gradient_checkpointing = True
            print(f"    Strategy: 4-bit + GradChkpt + BS=2")
        elif vram_gb < 48:
            # Large GPU (A100-40GB): Light quantization
            config["training"]["batch_size"] = 4
            use_4bit = True
            gradient_checkpointing = False
            print(f"    Strategy: 4-bit + BS=4")
        else:
            # Huge GPU (A100-80GB+): Full precision
            config["training"]["batch_size"] = 8
            use_4bit = False
            gradient_checkpointing = False
            print(f"    Strategy: Full precision + BS=8")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        config["training"]["batch_size"] = 1
        use_4bit = False
        gradient_checkpointing = True
        print(f"\n  [MPS MODE] Apple Silicon detected")
        print(f"    Strategy: Float16 + GradChkpt + BS=1")
    else:
        config["training"]["batch_size"] = 1
        print(f"\n  [CPU MODE] No GPU detected")
        print(f"    Strategy: Float32 + BS=1 (will be slow)")

    # Override epochs if requested
    if args.epochs:
        total = args.epochs
        p1 = max(1, total // 10)
        p2 = max(1, total - p1 * 2)
        p3 = total - p1 - p2
        config["training"]["epochs_phase1"] = p1
        config["training"]["epochs_phase2"] = p2
        config["training"]["epochs_phase3"] = p3
        print(f"    Epochs: {p1} + {p2} + {p3} = {total}")
    
    # Update config
    config["model"]["device"] = device
    config["model"]["use_4bit"] = use_4bit
    config["model"]["gradient_checkpointing"] = gradient_checkpointing
    
    print(f"\n  Loading backbone: {config['model']['backbone_id']}...")
    
    # Add project root to path
    project_root = os.path.abspath(".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from transformers import AutoProcessor
    from model.dualpath_vlm import DualPathVLM
    from training.trainer import Trainer
    
    # Load processor
    model_id = config["model"]["backbone_id"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model
    model = DualPathVLM(
        model_id=model_id,
        device=device,
        use_4bit=use_4bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    # Print stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Ensure output dirs exist
    os.makedirs(config.get("paths", {}).get("output_dir", "./outputs/checkpoints"), exist_ok=True)
    os.makedirs(config.get("paths", {}).get("log_dir", "./outputs/logs"), exist_ok=True)
    os.makedirs(config.get("paths", {}).get("results_dir", "./outputs/results"), exist_ok=True)
    
    if args.dry_run:
        print("\n  [DRY RUN] Model loaded successfully. Exiting without training.")
        return
    
    # Initialize trainer
    trainer = Trainer(model=model, processor=processor, config=config)
    
    # Setup data based on unified modes
    data_cfg = config.get("data", {})
    if args.mode == "llm":
        fab_jsonl = r"D:\Dataset\DualPath_Text_Data\text_fabrication_pairs.jsonl"
        om_jsonl = r"D:\Dataset\DualPath_Text_Data\text_omission_pairs.jsonl"
        image_dir = None
        print("\n  [UNIFIED] Loading pure Text-Only LLM dataset from D:\\Dataset...")
        trainer.setup_data(
            fab_jsonl=fab_jsonl,
            om_jsonl=om_jsonl,
            image_dir=image_dir,
        )
    elif args.mode == "unified":
        vlm_fab   = data_cfg["datasets"]["fabrication_triplets"]
        vlm_om    = data_cfg["datasets"]["omission_triplets"]
        llm_fab   = r"D:\Dataset\DualPath_Text_Data\text_fabrication_pairs.jsonl"
        llm_om    = r"D:\Dataset\DualPath_Text_Data\text_omission_pairs.jsonl"
        image_dir = "data/coco/train2017"
        print("\n  [UNIFIED] Loading BOTH Vision (COCO) + Text (HaluEval) datasets...")
        trainer.setup_unified_data(
            vlm_fab_jsonl=vlm_fab,
            vlm_om_jsonl=vlm_om,
            image_dir=image_dir,
            llm_fab_jsonl=llm_fab,
            llm_om_jsonl=llm_om,
        )
    else:
        fab_jsonl = data_cfg["datasets"]["fabrication_triplets"]
        om_jsonl = data_cfg["datasets"]["omission_triplets"]
        image_dir = "data/coco/train2017"
        print("\n  [UNIFIED] Loading Vision-Language COCO dataset...")
        trainer.setup_data(
            fab_jsonl=fab_jsonl,
            om_jsonl=om_jsonl,
            image_dir=image_dir,
            val_image_dir="data/coco/val2017",
        )
    
    # Launch training
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    
    print(f"\n  Total training time: {elapsed / 3600:.1f} hours")
    print(f"  Checkpoints: {config['paths']['output_dir']}")
    print(f"  Logs:        {config['paths']['log_dir']}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DualPath-VLM GPU Smart Trainer — One Command, Zero Config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_train.py                     # Full auto: download, prepare, train
  python gpu_train.py --skip_data         # Skip data download (already have it)
  python gpu_train.py --epochs 300        # Override total epoch count
  python gpu_train.py --dry_run           # Verify everything without training
        """
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override total epoch count")
    parser.add_argument("--skip_data", action="store_true", help="Skip data download")
    parser.add_argument("--dry_run", action="store_true", help="Verify setup without training")
    parser.add_argument("--mode", type=str, choices=["vlm", "llm", "unified"], default="vlm",
                        help="Training mode: vlm=images only, llm=text only, unified=both together")
    args = parser.parse_args()
    
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       DualPath-VLM: GPU SMART TRAINER                   ║")
    print("║       One Command. Zero Manual Steps.                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 0: Environment
    # ensure_environment()
    
    # Step 1: Data
    if not args.skip_data:
        if args.mode == "vlm":
            ensure_data()
            ensure_training_pairs()
        elif args.mode == "llm":
            print("  [UNIFIED] Text mode selected. Downloading all text datasets...")
            if not os.path.exists(r"D:\Dataset\DualPath_Text_Data\text_fabrication_pairs.jsonl"):
                subprocess.check_call([sys.executable, "scripts/download_all_text_data.py"])
        elif args.mode == "unified":
            ensure_data()
            ensure_training_pairs()
            print("  [UNIFIED] Downloading all text datasets (HaluEval+TruthfulQA+FaithDial+CNN/DM)...")
            subprocess.check_call([sys.executable, "scripts/download_all_text_data.py"])
    else:
        print("  [SKIP] Data acquisition skipped by user.\n")
    
    # Step 2: Train
    run_training(args)
    
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              ALL STEPS COMPLETE ✓                       ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
