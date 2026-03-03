# Protein Thermostability Prediction

Predicting protein melting temperature (Tm) from amino acid sequences using ESM-2 embeddings and a specialist-ensemble meta-learning architecture.

---

## Results

| Model | Test RMSE (°C) | Test R² | Pearson |
|-------|---------------|---------|---------|
| Extra Features XGBoost (Optuna) | ~6.5–7.0 | ~0.63 | — |
| XGBoost Meta-Learner | 6.33 | 0.789 | 0.889 |
| **MLP Meta-Learner (best)** | **6.07** | **0.806** | **0.898** |

---

## Overview

Protein thermostability — the resistance of a protein to thermal unfolding — is characterised by its melting temperature (Tm). Predicting Tm directly from sequence has applications in protein engineering, enzyme design, and drug development. This project approaches the problem as a supervised regression task on the Kaggle Novozymes Enzyme Stability dataset.

### Core design choices

- **ESM-2 650M** (`esm2_t33_650M_UR50D`) for sequence embeddings. Mean pooling over residue positions yields a 1280-dimensional vector per sequence. Tm is a global property, so per-residue resolution is unnecessary.
- **Handcrafted physicochemical features** via BioPython ProteinAnalysis (composition, charge, instability, aliphatic index, GRAVY score, etc.) augment the embeddings.
- **Temperature-regime segmentation**: the training set is split into psychrophile (20–40°C), mesophile (40–80°C), and thermophile (80–130°C) subsets. Three specialist XGBoost models are trained on these segments, reducing within-model target heterogeneity.
- **Meta-learning**: out-of-fold predictions from the three specialists (mean + std = 6 features) are combined with top-50 SHAP-selected embedding/physicochemical features. Both an XGBoost and a residual MLP are trained as meta-learners.

---

## Repository Structure

```
protein-thermostability-prediction/
│
├── configs/
│   └── config.py                   # All paths, hyperparameters, constants
│
├── src/
│   ├── data/
│   │   ├── clean.py                # Step 1: Data cleaning pipeline
│   │   └── segment.py              # Step 2b: Tm-range segmentation
│   ├── embeddings/
│   │   └── generate_embeddings.py  # Step 2a: ESM-2 mean-pooled embeddings
│   ├── features/
│   │   └── feature_engineering.py  # Step 3: Physicochemical features + scaling
│   ├── models/
│   │   ├── train_extra_features_xgb.py  # Step 4a: XGBoost + Optuna baseline
│   │   ├── train_specialists.py         # Step 4b: 3 specialist XGBoost + OOF preds
│   │   ├── shap_feature_selection.py    # Step 5: SHAP-based top-50 feature selection
│   │   ├── train_xgb_meta.py            # Step 6a: XGBoost meta-learner
│   │   └── train_mlp_meta.py            # Step 6b: Residual MLP meta-learner
│   └── evaluation/
│       └── evaluate.py             # Permutation importance + SHAP validation
│
├── scripts/
│   └── run_pipeline.py             # End-to-end pipeline runner
│
├── docs/
│   ├── 01_data_pipeline.md         # Data cleaning, embeddings, features, segmentation
│   ├── 02_modelling.md             # Model architectures, hyperparameters, training
│   └── 03_evaluation_results.md    # Metrics, SHAP analysis, design decisions
│
├── assets/
│   └── images/                     # Validation plots (SHAP, residuals, learning curves)
│
├── data/                           # Not tracked in git (see Data section below)
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   ├── features/
│   ├── segments/
│   └── meta/
│
├── outputs/                        # Not tracked in git
│   ├── models/
│   ├── artifacts/
│   └── plots/
│
└── requirements.txt
```

---

## Pipeline

The full pipeline runs in 9 sequential stages:

```
train.csv
    │
    ▼ 1. clean.py
clean_filtered.csv
    │
    ▼ 2a. generate_embeddings.py  (GPU)
train_embeddings_meta.parquet
    │
    ▼ 3. feature_engineering.py
train_with_features_scaled.parquet
    │
    ├──▶ 2b. segment.py
    │    psychrophile / mesophile / thermophile .parquet
    │
    ├──▶ 4a. train_extra_features_xgb.py  [baseline model]
    │    xgb_optuna_best.json
    │
    ▼ 4b. train_specialists.py
df_with_preds.parquet  (+ 6 OOF prediction columns)
    │
    ▼ 5. shap_feature_selection.py
top_50_features.json
    │
    ├──▶ 6a. train_xgb_meta.py
    │    xgb_meta_best.json
    │
    └──▶ 6b. train_mlp_meta.py
         mlp_best.pt
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended for embedding generation; optional for training)

### Installation

```bash
git clone https://github.com/your-username/protein-thermostability-prediction.git
cd protein-thermostability-prediction
pip install -r requirements.txt
```

### Data

Download the Kaggle Novozymes Enzyme Stability dataset and place files as follows:

```
data/raw/train.csv
data/raw/test.csv
data/raw/test_labels.csv
```

The data is not tracked in this repository. You can obtain it from the [Kaggle competition page](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction).

---

## Running

### Full pipeline

```bash
python scripts/run_pipeline.py
```

### Skip stages (e.g. if embeddings already generated)

```bash
python scripts/run_pipeline.py --skip-embed --no-optuna
```

### Individual stages

```bash
# Clean data
python src/data/clean.py

# Generate embeddings (both splits)
python src/embeddings/generate_embeddings.py --split both

# Feature engineering
python src/features/feature_engineering.py

# Segment by Tm range
python src/data/segment.py

# Train extra features XGBoost (with Optuna)
python src/models/train_extra_features_xgb.py

# Train specialist models + generate OOF predictions
python src/models/train_specialists.py

# SHAP feature selection
python src/models/shap_feature_selection.py

# Train meta-learners
python src/models/train_xgb_meta.py
python src/models/train_mlp_meta.py
```

### Evaluation

```bash
# Evaluate XGB meta-learner
python src/evaluation/evaluate.py \
    --model-type xgboost \
    --model-path outputs/models/meta/xgb_meta_best.json \
    --data-path data/meta/df_with_preds.parquet \
    --features-path outputs/artifacts/top_50_features.json \
    --output-dir outputs/evaluation/xgb_meta \
    --segment-name xgb_meta

# Evaluate MLP meta-learner
python src/evaluation/evaluate.py \
    --model-type mlp \
    --model-path outputs/models/meta/mlp_best.pt \
    --data-path data/meta/df_with_preds.parquet \
    --features-path outputs/artifacts/top_50_features.json \
    --output-dir outputs/evaluation/mlp_meta \
    --segment-name mlp_meta
```

---

## Configuration

All paths, hyperparameters, and constants are centralised in `configs/config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `TM_MIN / TM_MAX` | 10 / 130°C | Tm filter bounds |
| `ESM_MODEL_NAME` | `esm2_t33_650M_UR50D` | ESM-2 variant |
| `ESM_BATCH_SIZE` | 8 | Sequences per forward pass |
| `SHAP_SAMPLE_SIZE` | 5000 | Samples for SHAP computation |
| `TOP_K_FEATURES` | 50 | Features selected by SHAP |
| `SEED` | 42 | Global random seed |

For GPU-accelerated training on Kaggle, change `tree_method` from `"hist"` to `"gpu_hist"` in the relevant param dicts.

---

## Validation Plots

All generated validation plots are stored in `assets/images/`. Key plots:

| Plot | Description |
|------|-------------|
| `shap_summary.png` | SHAP beeswarm (top 12 features) |
| `feature_importance_combined.png` | XGBoost feature importance (weight/gain/cover) |
| `actual_vs_predicted.png` | Predicted vs actual Tm scatter |
| `residuals.png` | Residual analysis |
| `learning_curve_rmse.png` | Training convergence |
| `train_val_gap.png` | Overfitting analysis |

---

## Detailed Documentation

- [Data Pipeline](docs/01_data_pipeline.md) — cleaning thresholds, embedding strategy, feature engineering details
- [Modelling](docs/02_modelling.md) — architectures, hyperparameters, Optuna results, OOF strategy
- [Evaluation & Results](docs/03_evaluation_results.md) — full metrics table, feature importance, SHAP analysis, design decisions

---

## Dataset

Novozymes Enzyme Stability Prediction (Kaggle, 2022)
- Training set: ~31k protein sequences with experimental Tm measurements
- Test set: held-out sequences for competition evaluation
- Features: amino acid sequence, experimental pH

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Sequence embeddings | ESM-2 (fair-esm) |
| Gradient boosting | XGBoost |
| Deep learning | PyTorch |
| Physicochemical features | BioPython |
| Hyperparameter search | Optuna |
| Explainability | SHAP |
| Data | pandas, numpy, pyarrow |
