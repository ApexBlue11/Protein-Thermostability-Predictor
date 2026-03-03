"""
Step 5 — SHAP Feature Selection
=================================
Trains a surrogate XGBoost model on the full feature set
(embeddings + scaled physicochemical features) and uses TreeExplainer
SHAP values to identify the top-50 most informative features.

Critically, the pred_* columns from the specialist models are EXCLUDED
from this selection. They are used directly as inputs to the meta-learner,
not ranked via SHAP. SHAP selection is only over the 1280+50-ish
embedding + handcrafted feature space.

Input:  data/meta/df_with_preds.parquet
Outputs:
  outputs/artifacts/top_50_features.json
  outputs/artifacts/shap_values.npy
  outputs/artifacts/shap_feature_importance.csv
  outputs/models/extra_features/xgb_metrics.json  (surrogate model metrics)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    DF_WITH_PREDS,
    TOP_FEATURES_JSON, SHAP_VALUES_NPY, SHAP_IMPORTANCE_CSV,
    ARTIFACTS_DIR,
    SHAP_SAMPLE_SIZE, TOP_K_FEATURES,
    SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_shap_selection(input_path: Path = DF_WITH_PREDS):
    log.info(f"\n{'='*60}")
    log.info("SHAP FEATURE SELECTION")
    log.info(f"{'='*60}")

    df = pd.read_parquet(input_path)
    log.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Feature columns (exclude pred_* — those go directly to meta-learner)
    emb_cols  = [c for c in df.columns if c.startswith("emb_")]
    phys_cols = [c for c in df.columns if c.endswith("_scaled")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    feature_cols = emb_cols + phys_cols

    log.info(f"\nFeature breakdown:")
    log.info(f"  Embeddings:              {len(emb_cols)}")
    log.info(f"  Physiological:           {len(phys_cols)}")
    log.info(f"  Total (for selection):   {len(feature_cols)}")
    log.info(f"  Pred cols (excluded):    {len(pred_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["tm"].values.astype(np.float32)

    log.info(f"\nTarget stats: mean={y.mean():.1f}°C  std={y.std():.1f}°C  "
             f"range=[{y.min():.1f}, {y.max():.1f}]°C")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    log.info(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # ── Surrogate XGBoost ────────────────────────────────────────────────
    log.info("\nTraining surrogate XGBoost for SHAP...")
    xgb_params = {
        "n_estimators":     1000,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha":        0.001,
        "reg_lambda":       1.0,
        "random_state":     SEED,
        "n_jobs":           -1,
        "tree_method":      "hist",
        "verbosity":        0,
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False,
    )
    log.info(f"  Best iteration: {model.best_iteration}")

    # Metrics
    def _eval(X_e, y_e, name):
        p = model.predict(X_e)
        rmse = float(np.sqrt(mean_squared_error(y_e, p)))
        r2   = float(r2_score(y_e, p))
        pearson, _ = pearsonr(y_e, p)
        log.info(f"  {name:<8s}  RMSE={rmse:.4f}  R²={r2:.4f}  Pearson={pearson:.4f}")
        return {"rmse": rmse, "r2": r2, "mae": float(mean_absolute_error(y_e, p)), "pearson": float(pearson)}

    log.info("\nSurrogate model evaluation:")
    tr_m   = _eval(X_train, y_train, "Train")
    test_m = _eval(X_test,  y_test,  "Test")

    # ── SHAP ─────────────────────────────────────────────────────────────
    log.info(f"\nComputing SHAP values (sample size={min(SHAP_SAMPLE_SIZE, len(X_test))})...")
    if len(X_test) <= SHAP_SAMPLE_SIZE:
        X_shap = X_test
    else:
        shap_idx = np.random.RandomState(SEED).choice(len(X_test), SHAP_SAMPLE_SIZE, replace=False)
        X_shap = X_test[shap_idx]

    # Wrap in DMatrix-compatible DataFrame for TreeExplainer
    X_shap_df = pd.DataFrame(X_shap, columns=feature_cols)
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_df)
    log.info("  ✓ SHAP values computed")

    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":          feature_cols,
        "shap_importance":  shap_importance,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    log.info(f"\nTop 20 features:")
    log.info(f"  {'Rank':<5} {'Feature':<30} {'Type':<5} {'SHAP Importance'}")
    for i, row in importance_df.head(20).iterrows():
        ftype = "EMB" if row["feature"].startswith("emb_") else "PHYS"
        log.info(f"  {i+1:<5} {row['feature']:<30} [{ftype}]  {row['shap_importance']:.6f}")

    top_features = importance_df.head(TOP_K_FEATURES)["feature"].tolist()
    n_emb  = sum(1 for f in top_features if f.startswith("emb_"))
    n_phys = sum(1 for f in top_features if f.endswith("_scaled"))
    log.info(f"\nTop-{TOP_K_FEATURES} composition:  {n_emb} embeddings  {n_phys} physiological")

    # ── Save ─────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TOP_FEATURES_JSON, "w") as f:
        json.dump(top_features, f, indent=2)
    log.info(f"\nSaved top features:      {TOP_FEATURES_JSON}")

    np.save(SHAP_VALUES_NPY, shap_values)
    log.info(f"Saved SHAP values array: {SHAP_VALUES_NPY}")

    importance_df.to_csv(SHAP_IMPORTANCE_CSV, index=False)
    log.info(f"Saved importance CSV:    {SHAP_IMPORTANCE_CSV}")

    metrics_path = ARTIFACTS_DIR / "surrogate_xgb_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "train":  tr_m, "test": test_m,
            "best_iteration": model.best_iteration,
            "feature_counts": {
                "total": len(feature_cols),
                "embeddings": len(emb_cols),
                "physiological": len(phys_cols),
                "top_k_embeddings": n_emb,
                "top_k_physiological": n_phys,
            },
            "model_params": xgb_params,
        }, f, indent=2)
    log.info(f"Saved surrogate metrics: {metrics_path}")

    log.info("\n✓ SHAP feature selection complete.")
    log.info(f"\nNext step: meta-learner input = {len(pred_cols)} specialist pred cols + {TOP_K_FEATURES} top features = {len(pred_cols) + TOP_K_FEATURES} total features")
    return top_features


if __name__ == "__main__":
    run_shap_selection()
