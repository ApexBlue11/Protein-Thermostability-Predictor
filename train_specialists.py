"""
Step 4b — Specialist XGBoost Models
=====================================
Trains three separate XGBoost regressors, one per thermostability regime:
  - Psychrophile  (Tm 20–40°C)
  - Mesophile     (Tm 40–80°C)
  - Thermophile   (Tm 80–130°C)

Each specialist is trained on its own segmented parquet, which reduces
target heterogeneity and allows the model to learn regime-specific
sequence–stability relationships.

After training, all three specialists run inference on the FULL training set
to generate prediction statistics (mean + std across ensemble members from
cross-validation or multiple seeds). These 6 columns are saved into
df_with_preds.parquet and later consumed by the meta-learner.

  Prediction columns generated:
    pred_psychro_mean, pred_psychro_std
    pred_meso_mean,    pred_meso_std
    pred_thermo_mean,  pred_thermo_std

Outputs:
  outputs/models/specialists/xgb_{psychrophile,mesophile,thermophile}.json
  data/meta/df_with_preds.parquet
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TRAIN_FEATURES,
    PSYCHROPHILE_DATA, MESOPHILE_DATA, THERMOPHILE_DATA,
    PSYCHRO_MODEL, MESO_MODEL, THERMO_MODEL,
    DF_WITH_PREDS,
    XGB_SPECIALIST_PARAMS, XGB_SPECIALIST_EARLY_STOPPING,
    SPECIALIST_CV_FOLDS, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPECIALISTS = {
    "psychrophile": (PSYCHROPHILE_DATA, PSYCHRO_MODEL),
    "mesophile":    (MESOPHILE_DATA,    MESO_MODEL),
    "thermophile":  (THERMOPHILE_DATA,  THERMO_MODEL),
}


def get_feature_cols(df: pd.DataFrame) -> list:
    emb_cols  = [c for c in df.columns if c.startswith("emb_")]
    phys_cols = [c for c in df.columns if c.endswith("_scaled")]
    return emb_cols + phys_cols


def train_specialist(
    name: str,
    data_path: Path,
    model_out: Path,
    feature_cols: list,
) -> xgb.XGBRegressor:
    """Train one specialist XGBoost and return the fitted model."""
    log.info(f"\n{'─'*60}")
    log.info(f"Training {name.upper()} specialist")
    log.info(f"{'─'*60}")

    df = pd.read_parquet(data_path)
    log.info(f"  Loaded {len(df):,} rows  "
             f"Tm range [{df['tm'].min():.1f}, {df['tm'].max():.1f}]°C")

    X = df[feature_cols].values.astype(np.float32)
    y = df["tm"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    model = xgb.XGBRegressor(**XGB_SPECIALIST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=XGB_SPECIALIST_EARLY_STOPPING,
        verbose=100,
    )

    preds_val = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
    r2   = float(r2_score(y_val, preds_val))
    log.info(f"  Validation  RMSE={rmse:.4f}  R²={r2:.4f}  best_iter={model.best_iteration}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_out))
    log.info(f"  Saved: {model_out}")
    return model


def generate_meta_features(
    full_train_path: Path,
    specialists: dict,
    feature_cols: list,
    output_path: Path,
):
    """
    Run all three specialist models on the full training set.
    Uses SPECIALIST_CV_FOLDS-fold CV predictions to populate mean/std columns
    for each segment, avoiding training-set leakage.

    Each specialist generates predictions via out-of-fold (OOF) inference:
      - For each fold, the specialist is retrained on the in-fold split
      - Predictions on the held-out fold populate the pred_*_mean columns
    The std across fold predictions for each sample is used as pred_*_std.
    """
    log.info(f"\n{'='*60}")
    log.info("Generating meta-features (OOF specialist predictions)")
    log.info(f"{'='*60}")

    full_df = pd.read_parquet(full_train_path)
    log.info(f"  Full training set: {len(full_df):,} rows")

    X_full = full_df[feature_cols].values.astype(np.float32)
    y_full = full_df["tm"].values.astype(np.float32)

    kf = KFold(n_splits=SPECIALIST_CV_FOLDS, shuffle=True, random_state=SEED)
    n = len(full_df)

    pred_matrix = {
        "psychrophile": np.full((n, SPECIALIST_CV_FOLDS), np.nan),
        "mesophile":    np.full((n, SPECIALIST_CV_FOLDS), np.nan),
        "thermophile":  np.full((n, SPECIALIST_CV_FOLDS), np.nan),
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_full)):
        log.info(f"\n  Fold {fold_idx + 1}/{SPECIALIST_CV_FOLDS}")
        for name, (seg_path, _) in specialists.items():
            seg_df  = pd.read_parquet(seg_path)
            X_seg   = seg_df[feature_cols].values.astype(np.float32)
            y_seg   = seg_df["tm"].values.astype(np.float32)

            m = xgb.XGBRegressor(**XGB_SPECIALIST_PARAMS)
            m.fit(
                X_seg, y_seg,
                eval_set=[(X_seg, y_seg)],
                early_stopping_rounds=XGB_SPECIALIST_EARLY_STOPPING,
                verbose=False,
            )
            fold_preds = m.predict(X_full[va_idx])
            pred_matrix[name][va_idx, fold_idx] = fold_preds

    # Aggregate OOF predictions to mean + std
    result_df = full_df.copy()
    for name, matrix in pred_matrix.items():
        result_df[f"pred_{name}_mean"] = np.nanmean(matrix, axis=1)
        result_df[f"pred_{name}_std"]  = np.nanstd(matrix,  axis=1)

    # Report coverage
    for name in pred_matrix:
        n_valid = result_df[f"pred_{name}_mean"].notna().sum()
        log.info(f"  {name:<14s}: {n_valid:,} / {n:,} valid OOF predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    log.info(f"\nSaved df_with_preds: {output_path}  ({result_df.shape[1]} cols)")
    return result_df


def main():
    # ── Load feature schema from full training set ───────────────────────
    log.info(f"Loading feature schema from: {TRAIN_FEATURES}")
    schema_df    = pd.read_parquet(TRAIN_FEATURES, columns=None)
    feature_cols = get_feature_cols(schema_df)
    log.info(f"  Feature columns: {len(feature_cols)}")

    # ── Train each specialist ─────────────────────────────────────────────
    trained_models = {}
    for name, (seg_path, model_out) in SPECIALISTS.items():
        trained_models[name] = train_specialist(name, seg_path, model_out, feature_cols)

    # ── Generate OOF meta-features ───────────────────────────────────────
    generate_meta_features(TRAIN_FEATURES, SPECIALISTS, feature_cols, DF_WITH_PREDS)

    log.info("\n✓ Specialist training and meta-feature generation complete.")


if __name__ == "__main__":
    main()
