<div align="center">

<br/>

```
███╗   ███╗████████╗    ███████╗██╗      ██████╗ ██████╗ ███████╗███████╗
████╗ ████║╚══██╔══╝    ██╔════╝██║     ██╔═══██╗██╔══██╗██╔════╝██╔════╝
██╔████╔██║   ██║       █████╗  ██║     ██║   ██║██████╔╝█████╗  ███████╗
██║╚██╔╝██║   ██║       ██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  ╚════██║
██║ ╚═╝ ██║   ██║       ██║     ███████╗╚██████╔╝██║  ██║███████╗███████║
╚═╝     ╚═╝   ╚═╝       ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
```

# Investigating Machine Translation Quality in African Languages

### *Corrected FLORES Evaluation Dataset — Retraining vs. Incremental Correction*

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF9A00?style=for-the-badge&logo=huggingface&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-f59e0b?style=for-the-badge)
![University](https://img.shields.io/badge/COS_760-Research_Project-6366f1?style=for-the-badge)

<br/>

> **Core Research Question:** Is it more effective and efficient to *retrain* machine translation models from scratch, or to *incrementally correct* models originally trained on erroneous datasets?

<br/>

---

</div>

## Overview

Low-resourced African languages remain severely under-represented in Natural Language Processing (NLP). This research investigates a critical quality bottleneck: the **FLORES evaluation dataset** — one of the most widely used multilingual MT benchmarks — has been found to contain inconsistencies and inaccuracies in its African language subsets ([Abdulmumin et al., 2024](https://aclanthology.org/2024.wmt-1.57/)).

When evaluation data is flawed, the models trained and benchmarked against it inherit those flaws. This project empirically compares two recovery strategies:

| Strategy | Description |
|---|---|
| **Full Retraining** | Re-train the model from scratch using the corrected FLORES dataset |
| **Incremental Correction** | Adapt an existing model trained on erroneous data without full retraining |

We measure **effectiveness** via translation quality metrics (BLEU, chrF, COMET) and **efficiency** via training time and compute resources.

<br/>

---

## Research Objectives

- **Quantify** the impact of dataset errors on MT evaluation metrics (BLEU, chrF, COMET)
- **Evaluate** data-driven error-detection methods (quality estimation, consistency checks)
- **Compare** quality gains and computational costs of full retraining vs. incremental model correction
- **Contribute** actionable findings to the low-resource African language NLP community

<br/>

---

## Methodology

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

  [Erroneous FLORES]──────┐
                          ▼
               ┌─────────────────────┐
               │  Pre-trained Model  │  ◄── Baseline Evaluation
               │  (Multilingual MT)  │      (BLEU / chrF / COMET)
               └──────────┬──────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
  ┌─────────────────┐         ┌──────────────────────┐
  │  Full Retrain   │         │  Incremental Correct  │
  │  on Corrected   │         │  on Existing Model    │
  │  FLORES Data    │         │                       │
  └────────┬────────┘         └───────────┬───────────┘
           ▼                              ▼
  ┌─────────────────┐         ┌──────────────────────┐
  │  Evaluate:      │         │  Evaluate:            │
  │  BLEU/chrF/     │         │  BLEU/chrF/           │
  │  COMET + Time   │         │  COMET + Time         │
  └────────┬────────┘         └───────────┬───────────┘
           └──────────────┬───────────────┘
                          ▼
               ┌─────────────────────┐
               │  Comparative        │
               │  Analysis           │
               └─────────────────────┘
```

<br/>

---

## Results

> *Results will be populated as experiments are completed.*

### Baseline Metrics (Erroneous FLORES)

| Language | BLEU | chrF | COMET |
|----------|------|------|-------|
| *TBD* | — | — | — |
| *TBD* | — | — | — |
| *TBD* | — | — | — |
| *TBD* | — | — | — |

### Post-Correction Comparison

| Strategy | BLEU Δ | chrF Δ | COMET Δ | Training Time | GPU Hours |
|----------|--------|--------|---------|---------------|-----------|
| Full Retraining | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Incremental Correction | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### Key Findings

- *To be added upon completion of experiments.*

<br/>

---

## File Structure [Replace with Actual Structure]

```
flores-african-mt/
│
├── data/
│   ├── flores_original/          # Original (erroneous) FLORES evaluation sets
│   ├── flores_corrected/         # Corrected FLORES sets (Abdulmumin et al., 2024)
│   └── splits/                   # Train/dev/test splits used in experiments
│
├── src/
│   ├── 📁 data_validation/       # Dataset validation and error detection
│   │   ├── quality_estimation.py # QE-based error flagging
│   │   ├── consistency_check.py  # Cross-reference consistency checks
│   │   └── utils.py
│   │
│   ├── 📁 evaluation/            # Metric computation scripts
│   │   ├── compute_bleu.py       # SacreBLEU wrapper
│   │   ├── compute_chrf.py       # chrF scorer
│   │   ├── compute_comet.py      # COMET neural metric
│   │   └── evaluate_all.py       # Run full evaluation pipeline
│   │
│   ├── 📁 retraining/            # Full retraining experiment
│   │   ├── train.py              # Training entry point
│   │   ├── config.yaml           # Hyperparameters and model config
│   │   └── dataset_loader.py
│   │
│   └── 📁 incremental_correction/  # Incremental model correction experiment
│       ├── correct.py            # Correction/fine-tuning entry point
│       ├── config.yaml
│       └── adapter_utils.py
│
├── 📁 models/
│   ├── baseline/                 # Saved baseline model checkpoints
│   ├── retrained/                # Retrained model checkpoints
│   └── corrected/                # Incrementally corrected model checkpoints
│
├── 📁 results/
│   ├── baseline_metrics.json     # Baseline BLEU/chrF/COMET results
│   ├── retrained_metrics.json    # Post-retraining results
│   ├── corrected_metrics.json    # Post-correction results
│   └── comparison_report.md     # Final comparative analysis
│
├── 📁 notebooks/
│   ├── exploratory_analysis.ipynb  # Dataset error exploration
│   ├── results_visualization.ipynb # Plots and charts for results
│   └── ablation_study.ipynb
│
├── 📄 requirements.txt
├── 📄 environment.yml            # Conda environment spec
├── 📄 .env.example               # Environment variable template
└── 📄 README.md
```

<br/>

---

## Setup & Installation [Replace with actual]

```bash
# Clone the repository
git clone https://github.com/<your-org>/flores-african-mt.git
cd flores-african-mt

# Create and activate conda environment
conda env create -f environment.yml
conda activate flores-mt

# Or use pip
pip install -r requirements.txt
```

### Download Datasets

```bash
# Download corrected FLORES datasets
python src/data_validation/utils.py --download --output data/flores_corrected

# Download original FLORES datasets
python src/data_validation/utils.py --download --original --output data/flores_original
```

<br/>

---

## 🚀 Running Experiments [Replace with Actual]

```bash
# 1. Validate datasets and flag errors
python src/data_validation/quality_estimation.py \
  --input data/flores_original \
  --output results/flagged_errors.json

# 2. Run baseline evaluation
python src/evaluation/evaluate_all.py \
  --model models/baseline \
  --data data/flores_original \
  --output results/baseline_metrics.json

# 3. Full retraining experiment
python src/retraining/train.py \
  --config src/retraining/config.yaml \
  --data data/flores_corrected \
  --output models/retrained

# 4. Incremental correction experiment
python src/incremental_correction/correct.py \
  --config src/incremental_correction/config.yaml \
  --base-model models/baseline \
  --data data/flores_corrected \
  --output models/corrected

# 5. Final comparative evaluation
python src/evaluation/evaluate_all.py \
  --models models/retrained models/corrected \
  --data data/flores_corrected \
  --output results/
```

<br/>

---

## Evaluation Metrics

| Metric | Description | Reference |
|--------|-------------|-----------|
| **BLEU** | N-gram precision-based score; widely used but surface-level | Post (2018) |
| **chrF** | Character n-gram F-score; more robust for morphologically rich languages | Popović (2015) |
| **Afri-COMET** | Neural MT evaluation; correlates better with human judgements | Rei et al. (2020) |

<br/>

---

## Team

| Member | Student No. | Responsibility |
|--------|-------------|----------------|
| **Tlhalefo Dikolomela** | u21507792 | Dataset validation & error detection |
| **Boloka Kgopodithate** | u20456213 | Evaluation metrics & benchmarking |
| **Samvit Prakash** | u23525119 | Model adaptation & incremental correction |

<br/>

---

## References

- Abdulmumin, I. et al. (2024). *Correcting FLORES Evaluation Dataset for Four African Languages.* WMT 2024.
- Goyal, N. et al. (2022). *The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation.* TACL.
- Krishnan, S. et al. (2016). *ActiveClean: Interactive Data Cleaning for Statistical Modeling.* VLDB.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT.
- Rei, R. et al. (2020). *COMET: A Neural Framework for MT Evaluation.* EMNLP.
- Specia, L. et al. (2020). *Findings of the WMT 2020 Shared Task on Quality Estimation.* WMT.

<br/>

---

<div align="center">

*COS 760 Research Project · University of Pretoria · April 2026*

</div>