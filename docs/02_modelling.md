# Phase 2 — Modelling

## Architecture Overview

Two parallel modelling strategies are implemented:

```
                    Full Training Set (ESM-2 + handcrafted features)
                           │
               ┌───────────┼───────────┐
               │                       │
    ┌──────────▼──────────┐  ┌────────▼────────────────────────────┐
    │  Extra Features XGB  │  │     Specialist XGBoost Ensemble      │
    │  (Optuna-tuned)      │  │   Psychrophile │ Mesophile │ Thermo  │
    │  Test RMSE ~6.5-7.0  │  │   (trained on segmented parquets)   │
    └─────────────────────┘  └────────┬────────────────────────────┘
                                       │
                              OOF predictions on full set
                       6 columns: pred_*_mean, pred_*_std
                                       │
                             SHAP Feature Selection
                       (top-50 from embeddings + physicochemical)
                                       │
                    ┌──────────────────┴──────────────────┐
                    │           56-feature input            │
                    │   (6 specialist + 50 SHAP-top)        │
                    ├──────────────────┬──────────────────┘
             ┌──────▼──────┐    ┌──────▼──────┐
             │  XGB Meta   │    │  MLP Meta   │
             │ RMSE=6.33   │    │ RMSE=6.07   │
             │  R²=0.789   │    │  R²=0.806   │
             └─────────────┘    └─────────────┘
```

---

## Model 1: Extra Features XGBoost (`src/models/train_extra_features_xgb.py`)

A single XGBoost regressor trained on the full combined feature set.

**Feature input:** ESM-2 embeddings (1280) + scaled handcrafted features (~50) + pH_scaled

### Hyperparameter Optimisation (Optuna)

- 30 trials, 5-fold cross-validation, MedianPruner
- Search space: learning_rate, max_depth, subsample, colsample_*, min_child_weight, gamma, reg_lambda, reg_alpha, max_leaves
- Objective: mean CV RMSE

**Best trial (Trial 27):**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.0478 |
| `max_depth` | 6 |
| `subsample` | 0.896 |
| `colsample_bytree` | 0.790 |
| `colsample_bylevel` | 0.947 |
| `colsample_bynode` | 0.913 |
| `min_child_weight` | 14 |
| `gamma` | 1.827 |
| `reg_lambda` | 0.000115 |
| `reg_alpha` | 0.0000836 |
| `max_leaves` | 21 |
| `grow_policy` | lossguide |

**Results:**

| Split | RMSE | R² |
|-------|------|----|
| CV (5-fold) | 8.4316 | — |
| Test | ~6.5–7.0°C | ~0.63 |

---

## Model 2: Specialist + Meta-Learner Ensemble

### 2a. Specialist Models (`src/models/train_specialists.py`)

Three separate XGBoost models, each trained on its Tm-range segment.

**Hyperparameters (same for all three):**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 1000 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `reg_alpha` | 0.001 |
| `reg_lambda` | 1.0 |

**OOF prediction generation:**

To avoid leakage, specialist predictions on the full training set are generated via 5-fold cross-validation out-of-fold (OOF) inference. Each fold retrains a specialist on the segment data and predicts on the held-out portion of the full set.

Six prediction columns are created:
- `pred_psychrophile_mean`, `pred_psychrophile_std`
- `pred_mesophile_mean`, `pred_mesophile_std`
- `pred_thermophile_mean`, `pred_thermophile_std`

These are saved to `data/meta/df_with_preds.parquet`.

---

### 2b. SHAP Feature Selection (`src/models/shap_feature_selection.py`)

A surrogate XGBoost model is trained on `emb_* + *_scaled` features (pred_* columns are excluded from selection). SHAP TreeExplainer values are computed on up to 5,000 test samples, and the top-50 features by mean absolute SHAP value are saved to `outputs/artifacts/top_50_features.json`.

**Feature composition of top-50 (typical):** ~42 embeddings, ~8 physicochemical

---

### 2c. XGBoost Meta-Learner (`src/models/train_xgb_meta.py`)

**Input:** 56 features = 6 specialist prediction stats + 50 SHAP-selected features

| Split | RMSE | MAE | R² | Pearson |
|-------|------|-----|----|---------|
| Train | 4.9963 | 2.9773 | 0.8694 | 0.9335 |
| Validation | 6.3138 | 3.5569 | 0.7963 | 0.8928 |
| Test | 6.3272 | 3.5158 | 0.7889 | 0.8885 |

The feature importance plot shows specialist prediction columns (especially `pred_mesophile_mean`) dominate, contributing a combined ~30–40% of total importance. This validates the segmentation design.

---

### 2d. MLP Meta-Learner (`src/models/train_mlp_meta.py`)

**Architecture: ResMLPEnsemble**

```
Input (56) → Linear(56→256) → BN → ReLU → Dropout(0.2)
           ↓
           ResBlock: Linear(256→128) → BN → ReLU → Dropout  [no residual — dims differ]
           ↓
           ResBlock: Linear(128→64) → BN → ReLU → Dropout   [no residual — dims differ]
           ↓
           Linear(64→1)
```

**Training config:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 512 |
| Max epochs | 200 |
| Early stopping | patience=25 |
| LR schedule | Warmup (10 epochs) + cosine annealing |
| Loss | MSELoss |

**Results:**

| Split | RMSE | MAE | R² | Pearson |
|-------|------|-----|----|---------|
| Train | 5.2399 | 3.1522 | 0.8563 | 0.9269 |
| Validation | 6.0168 | 3.5809 | 0.8150 | 0.9036 |
| Test | 6.0728 | 3.5901 | 0.8055 | 0.8980 |

The MLP outperforms the XGB meta-learner on the test set (RMSE 6.07 vs 6.33), suggesting it better captures non-linear interactions between the specialist predictions and selected features.

---

## Model Comparison Summary

| Model | Test RMSE (°C) | Test R² | Notes |
|-------|---------------|---------|-------|
| Extra Features XGB | ~6.5–7.0 | ~0.63 | Single model, full features, Optuna-tuned |
| XGB Meta-Learner | 6.33 | 0.789 | Ensemble: specialists + SHAP selection |
| **MLP Meta-Learner** | **6.07** | **0.806** | Best; residual MLP on same 56 features |
