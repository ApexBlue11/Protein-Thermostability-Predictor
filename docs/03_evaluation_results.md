# Phase 3 — Evaluation & Results

## Evaluation Strategy

All models are evaluated on a held-out test split (15% of training data, random_state=42). The following metrics are reported:

| Metric | Symbol | Interpretation |
|--------|--------|----------------|
| Root Mean Squared Error | RMSE (°C) | Primary metric; penalises large errors |
| Mean Absolute Error | MAE (°C) | Robust to outliers |
| Coefficient of Determination | R² | Proportion of Tm variance explained |
| Pearson Correlation | r | Linear association between predicted and actual Tm |

A lower RMSE is the primary optimisation target throughout.

---

## Results by Model

### Extra Features XGBoost (Optuna-tuned)

| Split | RMSE | MAE | R² | Pearson |
|-------|------|-----|----|---------|
| Train | — | — | — | — |
| Test | ~6.5–7.0°C | — | ~0.63 | — |

Best CV RMSE: **8.4316°C** (5-fold, 30 Optuna trials)

---

### XGBoost Meta-Learner

| Split | RMSE | MAE | R² | Pearson |
|-------|------|-----|----|---------|
| Train | 4.9963 | 2.9773 | 0.8694 | 0.9335 |
| Validation | 6.3138 | 3.5569 | 0.7963 | 0.8928 |
| Test | 6.3272 | 3.5158 | 0.7889 | 0.8885 |

**Per-Tm-range performance (test set):**

| Range | RMSE | R² | n |
|-------|------|----|---|
| Psychrophile (<40°C) | — | — | — |
| Mesophile (40–70°C) | — | — | — |
| Thermophile (>70°C) | — | — | — |

---

### MLP Meta-Learner (Best Model)

| Split | RMSE | MAE | R² | Pearson |
|-------|------|-----|----|---------|
| Train | 5.2399 | 3.1522 | 0.8563 | 0.9269 |
| Validation | 6.0168 | 3.5809 | 0.8150 | 0.9036 |
| Test | **6.0728** | 3.5901 | **0.8055** | **0.8980** |

---

## Feature Importance Analysis

### XGB Meta-Learner Feature Importance

Specialist prediction columns dominate the feature importance landscape:

- `pred_mesophile_mean` — highest individual importance (~17%)
- `pred_thermophile_std` — second (~12%)
- `pred_psychrophile_mean` — third
- Total specialist contribution: ~30–40% of all feature importance

This confirms the segmentation hypothesis: the meta-learner heavily relies on which regime the specialist models assign the input to, then refines with embedding-derived features.

See: `assets/images/feature_importance_combined.png`

---

## SHAP Analysis

SHAP TreeExplainer values were computed on the surrogate XGB model trained on the full embedding + physicochemical feature space.

Key findings:
- Embedding dimensions (especially mid-range indices) dominate the top-50 selected features
- Physicochemical features in the top-50 include: `frac_polar_scaled`, `aa_frac_Q_scaled`, `aa_frac_N_scaled`, `isoelectric_point_scaled`
- `pH_scaled` consistently appears in the top features, consistent with the known strong pH dependence of protein stability

See: `assets/images/shap_summary.png`, `assets/images/shap_dependence.png`

---

## Validation Plots

All plots are stored in `assets/images/`. Reference:

| Plot | Description |
|------|-------------|
| `shap_summary.png` | SHAP beeswarm: top 12 features, direction of effect |
| `shap_waterfall.png` | Waterfall plot for a single example prediction |
| `shap_dependence.png` | SHAP dependence plots for top features |
| `shap_distribution.png` | Distribution of SHAP values across test set |
| `shap_correlation_heatmap.png` | Correlation of SHAP values between features |
| `feature_importance_combined.png` | Weight/gain/cover/combined importance (extra XGB model) |
| `partial_dependence.png` | Partial dependence plots |
| `actual_vs_predicted.png` | Scatter plot: predicted vs actual Tm |
| `residuals.png` | Residual plot vs predicted Tm |
| `error_distribution.png` | Histogram of prediction errors |
| `learning_curve_rmse.png` | RMSE over training iterations |
| `learning_curve_zoomed.png` | Last 500 rounds zoomed |
| `train_val_gap.png` | Train vs validation RMSE gap |
| `convergence.png` | Convergence analysis |

---

## Design Decisions & Tradeoffs

### Why not per-residue models?
Per-residue approaches (e.g., fine-tuning ESM-2 on the Tm regression task directly) would likely achieve better performance but require substantially more GPU memory, training time, and engineering overhead. The mean-pooling + XGB/MLP approach achieves competitive results at a fraction of the computational cost.

### Why OOF predictions for specialist meta-features?
Training specialists on the full dataset and then predicting on the same data would cause leakage — the meta-learner would see "in-sample" predictions that are unrealistically good. OOF (out-of-fold) generation ensures the meta-learner trains on predictions that approximate test-time distribution.

### Why separate SHAP selection step?
With 1280 embedding dimensions + ~50 physicochemical features, the full feature space is 1330+. Including all of these in the meta-learner alongside the 6 specialist predictions would be very high-dimensional for a relatively small 56-feature meta-learning task. SHAP selection compresses this to the 50 most informative features without arbitrary truncation.
