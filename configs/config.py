"""
Central configuration for the protein thermostability prediction project.
All paths, hyperparameters, and constants are defined here.
Update DATA_DIR and OUTPUTS_DIR to match your local environment.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROOT PATHS  (edit these for your environment)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
MODELS_DIR   = OUTPUTS_DIR / "models"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"
PLOTS_DIR    = OUTPUTS_DIR / "plots"

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
RAW_TRAIN_CSV     = DATA_DIR / "raw" / "train.csv"
RAW_TEST_CSV      = DATA_DIR / "raw" / "test.csv"
RAW_TEST_LABELS   = DATA_DIR / "raw" / "test_labels.csv"

CLEAN_CSV         = DATA_DIR / "processed" / "clean_filtered.csv"

TRAIN_EMBEDDINGS  = DATA_DIR / "embeddings" / "train_embeddings_meta.parquet"
TEST_EMBEDDINGS   = DATA_DIR / "embeddings" / "test_embeddings_meta.parquet"

TRAIN_FEATURES    = DATA_DIR / "features" / "train_with_features_scaled.parquet"
TEST_FEATURES     = DATA_DIR / "features" / "test_with_features_scaled.parquet"
SCALER_PATH       = DATA_DIR / "features" / "handcrafted_scaler.joblib"

PSYCHROPHILE_DATA = DATA_DIR / "segments" / "psychrophile_data.parquet"
MESOPHILE_DATA    = DATA_DIR / "segments" / "mesophile_data.parquet"
THERMOPHILE_DATA  = DATA_DIR / "segments" / "thermophile_data.parquet"

DF_WITH_PREDS     = DATA_DIR / "meta" / "df_with_preds.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARTIFACT PATHS
# ─────────────────────────────────────────────────────────────────────────────
# Extra features XGBoost (Optuna-tuned)
XGB_EXTRA_MODEL   = MODELS_DIR / "extra_features" / "xgb_optuna_best.json"
XGB_EXTRA_FEATS   = MODELS_DIR / "extra_features" / "xgb_feature_columns.json"
XGB_EXTRA_METRICS = MODELS_DIR / "extra_features" / "model_metrics.json"

# Specialist models
PSYCHRO_MODEL     = MODELS_DIR / "specialists" / "xgb_psychrophile.json"
MESO_MODEL        = MODELS_DIR / "specialists" / "xgb_mesophile.json"
THERMO_MODEL      = MODELS_DIR / "specialists" / "xgb_thermophile.json"

# SHAP feature selection artifacts
TOP_FEATURES_JSON   = ARTIFACTS_DIR / "top_50_features.json"
SHAP_VALUES_NPY     = ARTIFACTS_DIR / "shap_values.npy"
SHAP_IMPORTANCE_CSV = ARTIFACTS_DIR / "shap_feature_importance.csv"

# XGB meta-learner
XGB_META_MODEL    = MODELS_DIR / "meta" / "xgb_meta_best.json"
XGB_META_METRICS  = MODELS_DIR / "meta" / "xgb_meta_metrics.json"

# MLP meta-learner
MLP_META_MODEL    = MODELS_DIR / "meta" / "mlp_best.pt"
MLP_META_METRICS  = MODELS_DIR / "meta" / "mlp_metrics.json"

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
TM_MIN      = 10.0     # °C  — absolute lower bound
TM_MAX      = 130.0    # °C  — absolute upper bound
PH_MIN      = 2.0
PH_MAX      = 11.0
SEQ_LEN_MIN = 20
SEQ_LEN_MAX = 2000

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION BOUNDARIES  (Tm ranges for specialist models)
# ─────────────────────────────────────────────────────────────────────────────
PSYCHROPHILE_RANGE = (20.0, 40.0)   # °C
MESOPHILE_RANGE    = (40.0, 80.0)   # °C
THERMOPHILE_RANGE  = (80.0, 130.0)  # °C

# ─────────────────────────────────────────────────────────────────────────────
# ESM-2 EMBEDDING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ESM_MODEL_NAME   = "esm2_t33_650M_UR50D"
ESM_EMBED_DIM    = 1280
ESM_BATCH_SIZE   = 8     # sequences per forward pass (tune for GPU VRAM)
ESM_CHUNK_SIZE   = 500   # checkpoint every N sequences

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING: EXTRA FEATURES XGB (Optuna best — Trial 27)
# ─────────────────────────────────────────────────────────────────────────────
XGB_EXTRA_PARAMS = {
    "learning_rate":      0.0478,
    "max_depth":          6,
    "subsample":          0.896,
    "colsample_bytree":   0.790,
    "colsample_bylevel":  0.947,
    "colsample_bynode":   0.913,
    "min_child_weight":   14,
    "gamma":              1.827,
    "reg_lambda":         0.000115,
    "reg_alpha":          0.0000836,
    "max_leaves":         21,
    "grow_policy":        "lossguide",
    "n_estimators":       2000,
    "tree_method":        "hist",   # use 'gpu_hist' on Kaggle
    "random_state":       42,
    "n_jobs":             -1,
    "verbosity":          0,
}
XGB_EXTRA_CV_FOLDS    = 5
XGB_OPTUNA_TRIALS     = 30
XGB_EARLY_STOPPING    = 50

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING: SPECIALIST XGB MODELS
# ─────────────────────────────────────────────────────────────────────────────
XGB_SPECIALIST_PARAMS = {
    "n_estimators":     1000,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.001,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
XGB_SPECIALIST_EARLY_STOPPING = 50
SPECIALIST_CV_FOLDS           = 5

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING: SHAP FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 5000
TOP_K_FEATURES   = 50

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING: XGB META-LEARNER
# ─────────────────────────────────────────────────────────────────────────────
XGB_META_PARAMS = {
    "n_estimators":     1000,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
XGB_META_TEST_SIZE = 0.15
XGB_META_VAL_SIZE  = 0.15
XGB_META_EARLY_STOPPING = 50

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING: MLP META-LEARNER
# ─────────────────────────────────────────────────────────────────────────────
MLP_HIDDEN_DIMS   = [256, 128, 64]
MLP_DROPOUT       = 0.2
MLP_BATCH_SIZE    = 512
MLP_LEARNING_RATE = 3e-4
MLP_WEIGHT_DECAY  = 1e-4
MLP_NUM_EPOCHS    = 200
MLP_PATIENCE      = 25
MLP_WARMUP_EPOCHS = 10
MLP_TEST_SIZE     = 0.15
MLP_VAL_SIZE      = 0.15
MLP_USE_RESIDUAL  = True   # ResMLPEnsemble vs plain MLPEnsemble

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# CREATE OUTPUT DIRECTORIES ON IMPORT
# ─────────────────────────────────────────────────────────────────────────────
for _d in [
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    DATA_DIR / "embeddings",
    DATA_DIR / "features",
    DATA_DIR / "segments",
    DATA_DIR / "meta",
    MODELS_DIR / "extra_features",
    MODELS_DIR / "specialists",
    MODELS_DIR / "meta",
    ARTIFACTS_DIR,
    PLOTS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)
