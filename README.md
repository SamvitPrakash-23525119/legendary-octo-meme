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

![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF9A00?style=for-the-badge&logo=huggingface&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-f59e0b?style=for-the-badge)
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

We measure **effectiveness** via translation quality metrics (chrF++, AfriCOMET) and **efficiency** via training time and compute resources.

<br/>

---

## Research Objectives

- **Quantify** the impact of dataset errors on MT evaluation metrics (chrF++, AfriCOMET)
- **Evaluate** data-driven error-detection methods (quality estimation, consistency checks)
- **Compare** quality gains and computational costs of full retraining vs. incremental model correction
- **Contribute** actionable findings to the low-resource African language NLP community

<br/>

---

## Methodology
```
                      ┌────────────────────────────────────────────────────────────┐
                      │                  BASELINE NMT PIPELINE                     │
                      └────────────────────────────────────────────────────────────┘
                                                    │
                                            ┌───────▼───────┐
                                            │  Source Text  │
                                            └───────┬───────┘
                                                    │
                                     ┌──────────────▼──────────────┐
                                     │ 1. Initialization: Load     │
                                     │    AfriNLLB Base Model &    │
                                     │    SentencePiece Tokenizer  │
                                     └──────────────┬──────────────┘
                                                    │
                                     ┌──────────────▼──────────────┐
                                     │ 2. Preprocessing: Tokenize  │
                                     │    to input_ids (padding &  │
                                     │    truncation to max_len)   │
                                     └──────────────┬──────────────┘
                                                    │
                                     ┌──────────────▼──────────────┐
                                     │ 3. Generation: Default Beam │
                                     │    Search (forced_bos_token │
                                     │    = zul_Latn)              │
                                     └──────────────┬──────────────┘
                                                    │
                                     ┌──────────────▼──────────────┐
                                     │ 4. Decoding: batch_decode   │
                                     │    to raw text strings      │
                                     └──────────────┬──────────────┘
                                                    │
                                                    ▼
                                      [ Baseline Translation Output ]

```

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

### Translation Quality and Efficiency Results

| Experiment | chrF++ | AfriCOMET | Lat. (s) | Thr. |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | 57.16 | 0.7384 | 0.63 | 1.59 |
| **E2** | 58.02 | 0.7384 | 0.70 | 1.44 |
| **E3** | 55.13 | 0.7171 | 0.20 | 4.91 |
| **E4a** | 56.37 | 0.7261 | 0.88 | 1.14 |
| **E4b** | 43.99 | 0.4936 | 4.17 | 0.24 |
| **E4c** | 49.08 | 0.7152 | 0.17 | 5.96 |

### Paired Bootstrap Resampling Results

| Comparison | Mean Δ | CI Low. | CI Up. | p-value | Sig. (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fine-Tuned vs Corrected** | -0.0382 | -0.0781 | -0.0048 | 0.9904 | Significant (↓) |
| **RAT vs Corrected** | -0.0192 | -0.0379 | 0.0021 | 0.9634 | Not Significant |
| **Constrained vs Corrected** | -0.2439 | -0.3800 | -0.1276 | 1.0000 | Significant (↓) |
| **Post vs Corrected** | -0.0262 | -0.0528 | 0.0024 | 0.9632 | Not Significant |

### Key Findings

- Improving the dataset through correction was the only intervention that consistently enhanced performance over the baseline, underscoring the critical role of benchmark quality in reliable machine translation evaluation.
- Retraining on OPUS100 did not yield performance gains despite increased computational cost, suggesting that larger or additional data alone is insufficient without strong domain alignment, particularly in low-resource settings.
- Retrieval-Augmented Translation proved to be the most effective refinement approach among the tested modifications. While it did not exceed the baseline, it significantly mitigated the performance drop introduced by retraining without requiring model parameter updates. In comparison, constrained decoding and post-processing methods showed limited effectiveness.
- Data quality and retrieval-based augmentation are more cost-effective and reliable improvement strategies than large-scale retraining or post-hoc corrections.

<br/>

---

## File Structure

```
main/
│
├── Implementation/
│   ├── Benchmark Outputs/
│   │   ├── corrected_baseline
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
│   └── notebooks/                   
│       ├── AfriCOMET_Eval.ipynb              # AfriCOMET evaluation implementation
│       ├── Fine-Tuned.ipynb                  # Fine-tuned model implem
│       ├── PairedBootstrapPipeline.ipynb
│       ├── baseline.ipynb                    # Baseline model implementation
│       ├── baseline[Fine-tuned].ipynb        # Model using OPUS100 corpus for fine-tuning
│       ├── corrected-baseline.ipynb          # Model trained on FLORES+ dataset
│       └── refinements.ipynb                 # Refinement strategies for improving model performance
└── README.md
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

## Running Experiments

```bash
# 1. Run baseline evaluation
  - Open "baseline.ipynb" using Google Colab or Kaggle
  - Click "Run All"
  - Results will be stored in a file named "baseline_metrics.json"

# 2. Corrected dataset model evaluation
  - Open "corrected-baseline.ipynb" using Google Colab or Kaggle
  - DO NOT CLICK "Run All"
  - Please see instructions inside the notebook on how to run the notebook.
  -In addtion to this; the references, sources and predictions are saved in .json files to be used for the comparative   evaluations.

# 3. Fine-tuned dataset model evaluation
  - Open "Fine-tuned.ipynb" using Google Colab or Kaggle
  - Click "Run All"
  -In addtion to this; the references, sources and predictions are saved in .json files to be used for the comparative   evaluations.

# 4. Refinement strategy experiment
  - Open "refinements.ipynb" using Google Colab or Kaggle
  - Click "Run All"

  4.1 RAT/RAG implementation
    - Will be found under a cell labeled "Refinement RAT/RAG"
    - Results will be stored in a file named "rag_rat_metrics.json"
    -In addtion to this; the references, sources and predictions are saved in .json files to be used for the comparative   evaluations.

  4.2 Constrained Beam Search
    - Will be found below RAT/RAG implementation
    - Results will be stored in a file named "constrained_metrics.json"
    -In addtion to this; the references, sources and predictions are saved in .json files to be used for the comparative   evaluations.

  4.3 Post-Processing and Unicode Normalization
    - Will be found under a cell labeled "Refinement RAT/RAG"
    - Results will be stored in a file named "rag_rat_metrics.json"
    -In addtion to this; the references, sources and predictions are saved in .json files to be used for the comparative   evaluations.

# 5. Final comparative evaluation
In order to run the comparative evaluations ensure that the session storage contains
-corrected_predictions.json
-retrained_predictions.json
-rat_predictions.json
-constrained_predictions.json
-post_processing_predictions.json

  - Open "AfriCOMET_Eval.ipynb" using Google Colab or Kaggle
  - Click "Run All"
  - Open "PairedBootstrapPipeline.ipynb" using Google Colab or Kaggle
  - In addition to the prediction .json files above, ensure that the following files are also in the session's storage
      -retrained_sources.json
      -retrained_references.json
  - Click "Run All"

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
| **Boloka Kgopodithate** | u20456213 | Refinement strategies |
| **Samvit Prakash** | u23525119 | Model adaptation & incremental correction |

<br/>

---

## References

- Abdulmumin, I. et al. (2024). *Correcting FLORES Evaluation Dataset for Four African Languages.* WMT 2024.
- Ache, M. (2024). FLORES-101 Dataset. Kaggle.
- Tummala, V. A. (2024). FLORES-200 Data. Kaggle.
- Chousa, K. & Morishita, M. (2021). Input Augmentation Improves Constrained Beam Search for Neural Machine Translation: NTT at WAT 2021. WAT.
- Goyal, N. et al. (2022). The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation. TACL.
- Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. EMNLP.
- Moslem, Y. et al. (2026). AfriNLLB: Efficient Translation Models for African Languages. AfricaNLP.
- Nekoto, W. et al. (2020). Participatory Research for Low-Resourced Machine Translation: A Case Study in African Languages. arXiv.
- Park, C. et al. (2021). Should We Find Another Model?: Improving Neural Machine Translation Performance with One-Piece Tokenization Method Without Model Modification. NAACL-HLT.
- NLLB Team et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv.
- Tiedemann, J. (2012). Parallel Data, Tools and Interfaces in OPUS. LREC.
- Wang, R.-C. et al. (2025). Hybrid Dictionary–Retrieval-Augmented Generation–Large Language Model for Low-Resource Translation. Engineering Proceedings.

<br/>

---

<div align="center">

*COS 760 Research Project · University of Pretoria · April 2026*

</div>
