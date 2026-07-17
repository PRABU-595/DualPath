# =============================================================
# DualPath-VLM: ONE SCRIPT TO RULE THEM ALL (Windows Version)
# Save this as 'run_everything.ps1' on your Windows GPU server.
# It will: setup venv -> download data -> train -> evaluate -> plot
# =============================================================

$ErrorActionPreference = "Stop"
$ProjectDir = Get-Location
$LogFile = Join-Path $ProjectDir "run_everything.log"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DualPath-VLM: Full Windows Pipeline Starting"
Write-Host "  Log: $LogFile"
Write-Host "  Time: $(Get-Date)"
Write-Host "============================================================" -ForegroundColor Cyan

# Start logging
Start-Transcript -Path $LogFile -Append

try {
    # ---------------------------------------------------------------
    # STEP 0: GPU CHECK
    # ---------------------------------------------------------------
    Write-Host "`n[Step 0/7] Checking GPU..." -ForegroundColor Yellow
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        $GpuName = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
        Write-Host "[✓] GPU: $GpuName" -ForegroundColor Green
    } else {
        Write-Host "[✗] WARNING: No NVIDIA GPU detected. Inference will be slow." -ForegroundColor Red
        # We continue anyway, but VLM models usually fail without CUDA on Windows
    }

    # ---------------------------------------------------------------
    # STEP 1: PYTHON ENVIRONMENT
    # ---------------------------------------------------------------
    Write-Host "`n[Step 1/7] Creating Python virtual environment..." -ForegroundColor Yellow
    if (-not (Test-Path "venv")) {
        python -m venv venv
        Write-Host "[✓] Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "[✓] Virtual environment already exists" -ForegroundColor Green
    }

    # Activate venv
    $VenvActivate = Join-Path $ProjectDir "venv\Scripts\Activate.ps1"
    & $VenvActivate
    Write-Host "[✓] venv activated" -ForegroundColor Green

    Write-Host "[Step 1/7] Installing dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet

    # spaCy model for precision evaluation
    Write-Host "  Installing spaCy model..."
    python -m spacy download en_core_web_sm --quiet

    Write-Host "[✓] All dependencies installed" -ForegroundColor Green

    # ---------------------------------------------------------------
    # STEP 2: DOWNLOAD DATASETS
    # ---------------------------------------------------------------
    Write-Host "`n[Step 2/7] Downloading datasets (COCO val2017 + Visual Genome)..." -ForegroundColor Yellow
    $DataDirs = @("data/coco/annotations", "data/coco/val2017", "data/vg")
    foreach ($dir in $DataDirs) { if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force } }

    # COCO val2017 annotations
    if (-not (Test-Path "data/coco/annotations/instances_val2017.json")) {
        Write-Host "  Downloading COCO val2017 annotations..."
        $url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        $dest = "$env:TEMP\annotations.zip"
        Invoke-WebRequest -Uri $url -OutFile $dest
        Expand-Archive -Path $dest -DestinationPath "data/coco/" -Force
        Remove-Item $dest
    }

    # COCO val2017 images
    if (-not (Test-Path "data/coco/val2017")) {
        Write-Host "  Downloading COCO val2017 images (~800MB)..."
        $url = "http://images.cocodataset.org/zips/val2017.zip"
        $dest = "$env:TEMP\val2017.zip"
        Invoke-WebRequest -Uri $url -OutFile $dest
        Expand-Archive -Path $dest -DestinationPath "data/coco/" -Force
        Remove-Item $dest
    }

    # Visual Genome objects
    if (-not (Test-Path "data/vg/objects.json")) {
        Write-Host "  Downloading Visual Genome objects..."
        $url = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"
        $dest = "$env:TEMP\vg_objects.zip"
        Invoke-WebRequest -Uri $url -OutFile $dest
        Expand-Archive -Path $dest -DestinationPath "data/vg/" -Force
        Remove-Item $dest
    }

    Write-Host "[✓] All datasets ready" -ForegroundColor Green

    # ---------------------------------------------------------------
    # STEP 3: GENERATE TRAINING PAIRS
    # ---------------------------------------------------------------
    Write-Host "`n[Step 3/7] Generating training pairs..." -ForegroundColor Yellow
    python scripts/generate_real_pairs.py `
        --coco_instances  data/coco/annotations/instances_val2017.json `
        --coco_captions   data/coco/annotations/captions_val2017.json `
        --vg_objects      data/vg/objects.json `
        --output_fab      data/real_fabrication_pairs.jsonl `
        --output_om       data/real_omission_pairs.jsonl `
        --num_samples     30000

    Write-Host "[✓] Training pairs generated" -ForegroundColor Green

    # ---------------------------------------------------------------
    # STEP 4: TRAINING
    # ---------------------------------------------------------------
    Write-Host "`n[Step 4/7] Starting DualPath-VLM 3-Phase Training..." -ForegroundColor Yellow
    $env:PYTHONPATH = "$ProjectDir;$env:PYTHONPATH"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

    # One GPU Training (Standard on Windows)
    python training/trainer.py `
        --config_path configs/base_config.yaml `
        --image_dir   data/coco/val2017

    Write-Host "[✓] Training complete!" -ForegroundColor Green

    # ---------------------------------------------------------------
    # STEP 5: EVALUATION SUITE
    # ---------------------------------------------------------------
    Write-Host "`n[Step 5/7] Running A*-Strong Evaluation Suite..." -ForegroundColor Yellow
    if (-not (Test-Path "outputs/evaluation")) { New-Item -ItemType Directory -Path "outputs/evaluation" -Force }
    if (-not (Test-Path "outputs/results")) { New-Item -ItemType Directory -Path "outputs/results" -Force }

    Write-Host "  Running POPE..."
    python evaluation/eval_pope.py --limit 500 --output_file outputs/evaluation/pope_results.json
    
    Write-Host "  Running CHAIR..."
    python evaluation/eval_chair.py --img_dir data/coco/val2017 --limit 200 --output_file outputs/evaluation/chair_results.json
    
    Write-Host "  Running Object Recall..."
    python evaluation/eval_object_recall.py --vg_json data/vg/objects.json --img_dir data/coco/val2017 --limit 200 --output_file outputs/evaluation/recall_results.json
    
    Write-Host "  Generating Mechanistic Log..."
    python evaluation/inference_engine.py --dataset coco --limit 300 --out outputs/evaluation/inference_log.json

    Write-Host "[✓] All evaluations complete" -ForegroundColor Green

    # ---------------------------------------------------------------
    # STEP 6: GENERATE ALL PLOTS
    # ---------------------------------------------------------------
    Write-Host "`n[Step 6/7] Generating publication-ready figures..." -ForegroundColor Yellow
    python scripts/plot_tradeoff_curve.py
    python scripts/verify_attention_drift.py --log outputs/evaluation/inference_log.json
    python scripts/plot_gating_behavior.py
    python scripts/plot_intervention_freq.py
    python scripts/plot_threshold_sensitivity.py

    Write-Host "[✓] All figures saved to outputs/results/" -ForegroundColor Green

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "  DualPath-VLM: FULL PIPELINE COMPLETE"
    Write-Host "  Time: $(Get-Date)"
    Write-Host "============================================================" -ForegroundColor Cyan

} catch {
    Write-Host "`n[✗] ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Consult $LogFile for details."
} finally {
    Stop-Transcript
}
