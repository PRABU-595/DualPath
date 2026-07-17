# GPU Server Evaluation Package for DualPath-VLM

This package contains the script `real_threshold_sweep.py` designed to run on your GPU server. It evaluates the raw Vanilla backbone (Qwen2.5-VL-7B-Instruct) on your exact datasets, extracting softmax logits for "Yes"/"No" and computing a dynamic confidence threshold sweep to calculate the trivial baseline.

## Environment Setup
Reviewers need to know the exact environment. We recommend setting up and logging the following versions:
- `transformers == 4.39.0` (or whichever your cluster runs for Qwen2.5)
- `torch == 2.2.0` (or whichever version you use)
- `qwen-vl-utils` (if required by Qwen2.5 on your server)
- `scikit-learn`

Run:
```bash
pip install -r requirements_server.txt
```

## Running the Evaluation

### Step 1: Dry Run (Highly Recommended)
Before burning GPU hours, verify the token extraction logic and visual model loading works properly on your specific GPU cluster. This tests the logic on just 10 examples.
```bash
python real_threshold_sweep.py \
    --pope_file /path/to/your/exact/Table1/pope_dataset.json \
    --vg_file /path/to/your/exact/Table3/vg_dataset.json \
    --dry_run
```
*Crucial*: You MUST pass the EXACT same `pope_file` and `vg_file` that generated Table I and Table III to guarantee Dataset Alignment. The script logs the Question IDs; please verify they match your previous evaluations.

### Step 2: Full Sweep
Once the dry run successfully outputs expected `Score(Yes)` values without crashing, run the full 500-image sweep.
```bash
python real_threshold_sweep.py \
    --pope_file /path/to/your/exact/Table1/pope_dataset.json \
    --vg_file /path/to/your/exact/Table3/vg_dataset.json
```

## Outputs
The script will output two fully auditable logs:
1. `outputs/raw_pope_confidence_log.csv`
2. `outputs/raw_vg_confidence_log.csv`

It will also print the exact Precision/Recall trade-off at the chosen threshold and the full Confusion Matrix. You can include these CSVs in your supplementary material for total transparency.
