# 🚀 DualPath-VLM: Bridging Generative and Discriminative Grounding

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green)

**DualPath-VLM** is a novel architectural intervention built on top of the **Qwen2.5-VL-7B** backbone. It addresses object omission hallucinations in Vision-Language Models by introducing an explicit, learned visual grounding pathway that routes visual features dynamically during generation.

## 📊 Performance at a Glance
Our evaluations on the **Adversarial POPE** benchmark (n=6000 queries) demonstrate that DualPath-VLM pushes the state-of-the-art recall boundary, operating strictly above the baseline model's Precision-Recall curve.

| Metric | Vanilla Qwen2.5-VL | DualPath-VLM (Ours) | Delta |
|---|---|---|---|
| **Precision** | 95.80% | 92.45% | -3.35% |
| **Recall** | 89.00% | 93.83% | **+4.83%** |
| **F1 Score** | 92.27% | 93.13% | **+0.86%** |

*(See the `Figures/` directory for the complete joint fabrication-omission Pareto frontier and ROC curves).*

## 📁 Repository Structure
- **`Figures/`**: Publication-ready evaluation plots, confusion matrices, and precision-recall curves.
- **`Supplementary_Code_and_Data/`**: The core source code, model checkpoint scripts, and raw per-query evaluation logs.
- **`Model_Checkpoints/`**: The verified final Phase 2 gating weights.

## 🛠️ Reproducing the Results
All code to run the model and reproduce the evaluations is located in the `Supplementary_Code_and_Data` directory.

### Discriminative VQA (Adversarial POPE)
```bash
cd Supplementary_Code_and_Data
python pope_adversarial_eval.py
```

### Open-Ended Generative Grounding (CHAIR & VG Recall)
```bash
cd Supplementary_Code_and_Data
python evaluate_chair_vg_verified.py
```

## 📝 Citation
If you use this codebase or model in your research, please cite our upcoming publication.
