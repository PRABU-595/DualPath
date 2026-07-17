# DualPath-VLM: Supplementary Code and Evaluation Suite

This repository contains the source code, pre-trained weights, and rigorous evaluation scripts for **DualPath-VLM**. 

## Repository Structure
- `model/`: Core architectural implementation of the DualPath modules (Pathway F, Pathway O, and the Gating MLP) built on top of the Qwen2.5-VL-7B backbone.
- `outputs/phase2_retrain/`: Contains the Phase 2 retrained gating weights.
- `outputs/pope_adversarial/`: Contains the raw per-question logs and confidence scores across 6,000 queries.
- `evaluate_chair_vg_verified.py`: The evaluation script for open-ended generation hallucination (CHAIR) and omission (VG Recall).
- `pope_adversarial_eval.py`: The evaluation script for discriminative VQA (POPE Adversarial).

## Reproducing the Results
We provide the exact scripts used to generate the results in the paper. To reproduce the adversarial POPE evaluations (n=3000 images, 6000 queries) with full threshold sweeping and AUC calculation, run:

```bash
python pope_adversarial_eval.py
```

To reproduce the open-ended generative evaluations (CHAIR on MSCOCO and VG Recall on Visual Genome), run:

```bash
python evaluate_chair_vg_verified.py
```

## Hardware Requirements
- **GPU:** 1x NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090 / 4090 / A5000).
- **Environment:** PyTorch >= 2.1.0, Transformers >= 4.38.0.

## Note to Reviewers
To ensure maximum transparency, our evaluation scripts explicitly output the raw integer counts for True Positives, False Positives, True Negatives, and False Negatives prior to percentage aggregation, and feature integrated McNemar's statistical testing for paired significance. The full per-query evaluation logs are included in `outputs/pope_adversarial/` for independent verification.
