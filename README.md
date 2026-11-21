# Keras Time-Series Benchmark: Transformer Classification & LSTM Forecasting

CMPE 401 — Instructor-defined Project 2

This project reproduces, organizes, and extends two official Keras time-series examples:

| | Task | Dataset | Model | Notebook |
|---|---|---|---|---|
| 1 | Classification | FordA (engine sensor) | Transformer encoder | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/river-kris/Keras-Timeseries/blob/main/notebooks/01_transformer_classification.ipynb) |
| 2 | Forecasting | Jena Climate (weather) | LSTM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/river-kris/Keras-Timeseries/blob/main/notebooks/02_lstm_weather_forecasting.ipynb) |

> **TL;DR of findings:** _TODO: 2–3 sentences once results are in._

---

## Table of contents
1. [Objective](#1-objective)
2. [Datasets](#2-datasets)
3. [Models](#3-models)
4. [Repository structure](#4-repository-structure)
5. [How to run](#5-how-to-run)
6. [Baseline reproduction (Task 1)](#6-baseline-reproduction-task-1)
7. [Improvements (Task 2)](#7-improvements-task-2)
8. [Benchmark summary (Task 3)](#8-benchmark-summary-task-3)
9. [Questions (Task 4)](#9-questions-task-4)
10. [Team contributions](#10-team-contributions)
11. [References](#11-references)

---

## 1. Objective
- Understand the difference between time-series **classification** (one label per sequence) and **forecasting** (predict a future value).
- Understand the difference between a **Transformer** (self-attention over all time steps at once) and an **LSTM** (recurrent, step-by-step hidden state).
- Reproduce both official baselines, make at least three controlled changes to one of them, and summarize the results as a benchmark.

## 2. Datasets

### FordA (classification)
- _TODO: source (UCR archive), what one sample is (500 time steps of engine noise), 2 classes (fault / no fault), train/test sizes, class balance._

### Jena Climate 2009–2016 (forecasting)
- _TODO: 14 features recorded every 10 minutes; 7 features used; target = temperature 12 h ahead; train/val split 71.5% / 28.5%; normalization._

## 3. Models

### Transformer encoder classifier
| | |
|---|---|
| Input | `(500, 1)` univariate sequence |
| Architecture | 4 × [Multi-head attention (4 heads, head size 256) → Dropout → LayerNorm → residual → Conv1D FFN → LayerNorm → residual] → GlobalAveragePooling → Dense(128) → Dropout(0.4) → Dense(2, softmax) |
| Loss / metric | Sparse categorical cross-entropy / accuracy |
| Training | Adam lr = 1e-4, batch 64, up to 150 epochs, early stopping (patience 10) |

### LSTM forecaster
| | |
|---|---|
| Input | `(120, 7)`: 120 hourly steps (past 720 records, sampled every 6) × 7 features |
| Architecture | LSTM(32) → Dense(1) |
| Output | Normalized temperature 72 records (12 h) after the input window |
| Loss / metric | MSE / MAE |
| Training | Adam lr = 1e-3, batch 256, 10 epochs, early stopping (patience 5), best-checkpoint |

_TODO: one paragraph on how the two models differ in the way they process the sequence._

## 4. Repository structure
```
.
├── notebooks/
│   ├── 01_transformer_classification.ipynb   # FordA + Transformer (baseline & experiments)
│   └── 02_lstm_weather_forecasting.ipynb     # Jena Climate + LSTM (baseline & experiments)
├── results/
│   └── <run_name>/                           # one folder per training run
│       ├── config.json                       # hyperparameters used
│       ├── metrics.json                      # final metrics, epochs, params, train time
│       ├── history.csv                       # per-epoch loss/metrics
│       ├── model.weights.h5                  # trained weights
│       └── *.png                             # loss/accuracy curves, predictions
├── figures/                                  # figures used in this README
├── scripts/
│   └── collect_results.py                    # builds the benchmark tables from results/
├── requirements.txt
└── README.md
```

## 5. How to run

### Google Colab (recommended)
1. Click an **Open in Colab** badge above.
2. `Runtime > Change runtime type > T4 GPU`.
3. Leave `CONFIG` at its defaults for the baseline, or change one hyperparameter and set a new `RUN_NAME` for an experiment.
4. `Runtime > Run all`. The final cell downloads `<RUN_NAME>.zip`.
5. Unzip it into `results/` so you get `results/<RUN_NAME>/...`, then commit.

### Build the benchmark tables
```bash
python scripts/collect_results.py --csv
```

### Local (optional)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

### Reproducing trained models
Every run is seeded (`seed = 42`) and its exact hyperparameters are stored in `results/<run>/config.json`. Copy them into the notebook's `CONFIG` to retrain, or load `model.weights.h5` into the model built by the notebook.

## 6. Baseline reproduction (Task 1)

| Model | Metric | Official example | Ours | Epochs | Train time | GPU |
|---|---|---|---|---|---|---|
| Transformer (FordA) | Test accuracy | ~0.95 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| LSTM (Jena) | Val MSE (normalized) | _see example_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

**Observations**
- Transformer: _TODO (early stopping epoch, train/val gap, training time)._
- LSTM: _TODO (was val loss still falling at epoch 10? MAE in °C, prediction plots)._

<!-- ![Transformer baseline loss](figures/transformer_baseline_loss.png) -->
<!-- ![LSTM baseline loss](figures/lstm_baseline_loss.png) -->

## 7. Improvements (Task 2)
**Chosen model:** _TODO (LSTM or Transformer) — why._

Each experiment changes **one** setting from the baseline; everything else (seed, data split, epochs) stays fixed.

| # | Run | Change | Hypothesis | Result vs. baseline |
|---|---|---|---|---|
| 1 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 2 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 3 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

## 8. Benchmark summary (Task 3)
_Paste the output of `python scripts/collect_results.py` here._

**Key findings**
- _TODO_
- _TODO_
- _TODO_

**Limitations**
- _TODO (single seed, short training for LSTM, no separate LSTM test set, Colab GPU variability, ...)._

## 9. Questions (Task 4)

**Which model did you find easier to understand and why?**
_TODO_

**What improvement did you try, and what did you learn from it?**
_TODO_

## 10. Team contributions
| Member | Contributions |
|---|---|
| _TODO_ | _TODO_ |

## 11. References
- T. Ntakouris, *Timeseries classification with a Transformer model*, Keras examples. https://keras.io/examples/timeseries/timeseries_classification_transformer/
- P. Attri, Y. Sharma, K. Takach, F. Shah, *Timeseries forecasting for weather prediction*, Keras examples. https://keras.io/examples/timeseries/timeseries_weather_forecasting/
- A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- S. Hochreiter, J. Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997.
- FordA dataset, UCR Time Series Classification Archive.
- Jena Climate dataset, Max Planck Institute for Biogeochemistry.
