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

We measure **effectiveness** via translation quality metrics (chrF, COMET) and **efficiency** via training time and compute resources.

<br/>

---

## Research Objectives

- **Quantify** the impact of dataset errors on MT evaluation metrics (chrF, COMET)
- **Evaluate** data-driven error-detection methods (quality estimation, consistency checks)
- **Compare** quality gains and computational costs of full retraining vs. incremental model correction
- **Contribute** actionable findings to the low-resource African language NLP community

<br/>

---

## Methodology
```
                                ┌────────────────────────────────────────────────────────────┐
                                │              FULL MODEL FINE-TUNING PIPELINE               │
                                └────────────────────────────────────────────────────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 1. Data Loading: OPUS-100   │
                                               │    (Train) & FLORES (Eval)  │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 2. Initialization: Load     │
                                               │    AfriNLLB Base Model      │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 3. Preprocessing: Tokenize  │
                                               │    Source & Target strings  │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 4. Fine-Tuning: HF Trainer  │
                                               │    (Updates model weights)  │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 5. Generation: Retrained    │
                                               │    Model translates eval set│
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 6. Evaluation: chrF++,      │
                                               │    Latency & Throughput     │
                                               └──────────────┬──────────────┘
                                                              │
                                                              ▼
                                        [ Retrained Metrics & Predictions JSON ]

```


```
                                ┌────────────────────────────────────────────────────────────┐
                                │             SOFT-CONSTRAINT RAG (LOGIT BOOST)              │
                                └────────────────────────────────────────────────────────────┘
                                                              │
                                                      ┌───────▼───────┐
                                                      │  Source Text  │
                                                      └───────┬───────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 1. Encoding: Sentence-      │
                                               │    Transformers             │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 2. Retrieval: FAISS L2/     │
                                               │    Cosine                   │
                                               └──────────────┬──────────────┘
                                                              │
                                               /──────────────▼──────────────\
                                              /  3. Quality Gate: Similarity  \
                                              \         >= Threshold?         /
                                               \──────────────┬──────────────/
                                                              │
                                               ┌──────────────┴──────────────┐
                                         [Fail]│                             │[Pass]
                                               ▼                             ▼
                                    ┌───────────────────┐         ┌────────────────────────┐
                                    │ Skip Constraints  │         │ Retrieve target Zulu   │
                                    │                   │         │ sentence               │
                                    └─────────┬─────────┘         └──────────┬─────────────┘
                                              │                              │
                                              │                   ┌──────────▼─────────────┐
                                              │                   │ 4. Processing: Tokenize│
                                              │                   │    target & Init       │
                                              │                   │    LogitsProcessor     │
                                              │                   └──────────┬─────────────┘
                                              │                              │
                                              └──────────────┬───────────────┘
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ 5. Generation: NMT Model     │
                                              │    translates                │
                                              └──────────────┬───────────────┘
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ 6. Post-Processing: Regex &  │
                                              │    decoding                  │
                                              └──────────────┬───────────────┘
                                                             ▼
                                                   [ Refined RAG Output ]

```

```
                                ┌────────────────────────────────────────────────────────────┐
                                │                 CONSTRAINED BEAM SEARCH                    │
                                └────────────────────────────────────────────────────────────┘
                                                              │
                                                      ┌───────▼───────┐
                                                      │  Source Text  │
                                                      └───────┬───────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 1. Encoding: Sentence-      │
                                               │    Transformers             │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 2. Retrieval: FAISS L2/     │
                                               │    Cosine                   │
                                               └──────────────┬──────────────┘
                                                              │
                                               /──────────────▼──────────────\
                                              /  3. Quality Gate: Similarity  \
                                              \         >= Threshold?         /
                                               \──────────────┬──────────────/
                                                              │
                                               ┌──────────────┴──────────────┐
                                         [Fail]│                             │[Pass]
                                               ▼                             ▼
                                    ┌───────────────────┐         ┌────────────────────────┐
                                    │ force_words_ids   │         │ Retrieve target Zulu   │
                                    │ = None            │         │ sentence               │
                                    └─────────┬─────────┘         └──────────┬─────────────┘
                                              │                              │
                                              │                   ┌──────────▼─────────────┐
                                              │                   │ 4. Extraction: Filter  │
                                              │                   │    len > 3, get longest│
                                              │                   └──────────┬─────────────┘
                                              │                              │
                                              │                   ┌──────────▼─────────────┐
                                              │                   │ 5. Tokenization:       │
                                              │                   │    Convert to token IDs│
                                              │                   └──────────┬─────────────┘
                                              │                              │
                                              └──────────────┬───────────────┘
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ 6. Generation: NMT via Grid  │
                                              │    Beam Search               │
                                              └──────────────┬───────────────┘
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ 7. Post-Processing: Regex &  │
                                              │    decoding                  │
                                              └──────────────┬───────────────┘
                                                             ▼
                                                  [ Constrained Output ]

```

```
                                ┌────────────────────────────────────────────────────────────┐
                                │              PURE POST-PROCESSING (BASELINE)               │
                                └────────────────────────────────────────────────────────────┘
                                                              │
                                                      ┌───────▼───────┐
                                                      │  Source Text  │
                                                      └───────┬───────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 1. Generation: Standard     │
                                               │    baseline NMT             │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 2. Decoding: Tokens to      │
                                               │    raw string               │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 3. Normalization: Unicode   │
                                               │    NFC                      │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 4. BPE Cleanup: Hyphens &   │
                                               │    punctuation artifacts    │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 5. Spacing Cleanup: Padding │
                                               │    & currency separation    │
                                               └──────────────┬──────────────┘
                                                              │
                                               ┌──────────────▼──────────────┐
                                               │ 6. Final Polish: Whitespace │
                                               │    & casing                 │
                                               └──────────────┬──────────────┘
                                                              │
                                                              ▼
                                                  [ Cleaned Baseline Output ]

```


<br/>

---

## Results

> *Results will be populated as experiments are completed.*

### Baseline Metrics (Erroneous FLORES)

| Language | chrF | COMET |
|----------|------|-------|
| *TBD* | — | — |
| *TBD* | — | — |
| *TBD* | — | — |
| *TBD* | — | — |

### Post-Correction Comparison

| Strategy | chrF Δ | COMET Δ | Training Time |  Latency |
|----------|--------|---------|---------------|-----------|
| Full Retraining | *TBD* | *TBD* | *TBD* | *TBD* |
| Incremental Correction | *TBD* | *TBD* | *TBD* | *TBD* |

### Key Findings

- *To be added upon completion of experiments.*

<br/>

---

## File Structure

```
main/
│
├── 📁 Implementation/
│   ├── 📁 Benchmark Outputs/
│   │   ├── 📁 sam
│   │   │   ├── baseline_metrics.json         # Evaluation metrics produced by corrected baseline model
│   │   │   ├── comet_input.json              # Input used for COMET evaluation
│   │   │   └── predictions.json              # Predictions made by corrected baseline model
│   │   │
│   │   ├── baseline_metrics.json             # Evaluation metrics produced by baseline model
│   │   ├── baseline_predicitons.json         # Predictions made by baseline model
│   │   ├── constrained_metrics.json          # Evaluation metrics produced by Constrained Beam Search refinement strategy
│   │   ├── post_processing_metrics.json      # Evaluation metrics produced by Post Processing refinement strategy
│   │   └── rag_rat_metrics.json              # Evaluation metrics produced by RAT/RAG refinement strategy
│   │
│   └── 📁 notebooks/                   
│       ├── AfriCOMET_Eval.ipynb              # AfriCOMET evaluation implementation
│       ├── Fine-Tuned.ipynb                  # Fine-tuned model implem
│       ├── PairedBootstrapPipeline.ipynb
│       ├── baseline.ipynb                    # Baseline model implementation
│       ├── baseline[Fine-tuned].ipynb        # Model using OPUS100 corpus for fine-tuning
│       ├── corrected-baseline.ipynb          # Model trained on FLORES+ dataset
│       └── refinements.ipynb                 # Refinement strategies for improving model performance
└── 📄 README.md
```

<br/>

---

## Setup & Installation 

```bash
  Upload notebooks to Google Colab or Kaggle
  Click "Run All"
```

<br/>

---

## 🚀 Running Experiments

```bash
# 1. Run baseline evaluation
  Open "baseline.ipynb" using Google Colab or Kaggle
  Click Run All
  Results will be stored in a file named "baseline_metrics.json"

# 2. Corrected dataset model evaluation
  Open "baseline.ipynb" using Google Colab or Kaggle

# 3. Refinement strategy experiment
  Open "refinements.ipynb" using Google Colab or Kaggle
  Click Run All
  3.1 RAT/RAG implementation
    Will be found under a cell labeled "Refinement RAT/RAG"
    Results will be stored in a file named "rag_rat_metrics.json"
  3.2 Constrained Beam Search
    Will be found below RAT/RAG implementation
    Results will be stored in a file named "constrained_metrics.json"
  3.3 Post-Processing and Unicode Normalization
    Will be found under a cell labeled "Refinement RAT/RAG"
    Results will be stored in a file named "rag_rat_metrics.json"

# 4. Final comparative evaluation
  Open "AfriCOMET_Eval.ipynb" using Google Colab or Kaggle
  Click "Run All"
  Open "PairedBootstrapPipeline.ipynb" using Google Colab or Kaggle
  Click "Run All"

```

<br/>

---

## Evaluation Metrics

| Metric | Description | Reference |
|--------|-------------|-----------|
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
