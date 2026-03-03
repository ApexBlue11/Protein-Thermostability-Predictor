"""
Step 6a — XGBoost Meta-Learner
================================
Trains an XGBoost regressor as the first meta-learner, using:
  - 6 specialist prediction statistics (pred_*_mean, pred_*_std)
  - Top-50 SHAP-selected features (embeddings + physicochemical)

Total input dimensionality: 56 features.

This model achieved:
  Train  RMSE=4.9963  MAE=2.9773  R²=0.8694  Pearson=0.9335
  Val    RMSE=6.3138  MAE=3.5569  R²=0.7963  Pearson=0.8928
  Test   RMSE=6.3272  MAE=3.5158  R²=0.7889  Pearson=0.8885

Inputs:
  data/meta/df_with_preds.parquet
  outputs/artifacts/top_50_features.json

Outputs:
  outputs/models/meta/xgb_meta_best.json
  outputs/models/meta/xgb_meta_metrics.json
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    DF_WITH_PREDS, TOP_FEATURES_JSON,
    XGB_META_MODEL, XGB_META_METRICS,
    XGB_META_PARAMS, XGB_META_TEST_SIZE, XGB_META_VAL_SIZE,
    XGB_META_EARLY_STOPPING, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate_split(model, X_e, y_e, split_name: str) -> dict:
    preds = model.predict(X_e)
    rmse  = float(np.sqrt(mean_squared_error(y_e, preds)))
    mae   = float(mean_absolute_error(y_e, preds))
    r2    = float(r2_score(y_e, preds))
    pearson, pval = pearsonr(y_e, preds)
    log.info(f"  {split_name:<12s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  Pearson={pearson:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": float(pearson), "pearson_p": float(pval)}


def train(
    data_path: Path = DF_WITH_PREDS,
    features_path: Path = TOP_FEATURES_JSON,
    model_out: Path = XGB_META_MODEL,
    metrics_out: Path = XGB_META_METRICS,
):
    log.info(f"\n{'='*60}")
    log.info("XGBoost META-LEARNER")
    log.info(f"{'='*60}")

    df = pd.read_parquet(data_path)
    log.info(f"  Loaded {len(df):,} rows")

    with open(features_path) as f:
        top_features = json.load(f)
    log.info(f"  Loaded {len(top_features)} top SHAP features")

    specialist_mean_cols = ["pred_psychrophile_mean", "pred_mesophile_mean",    "pred_thermophile_mean"]
    specialist_std_cols  = ["pred_psychrophile_std",  "pred_mesophile_std",     "pred_thermophile_std"]
    specialist_cols      = specialist_mean_cols + specialist_std_cols

    # Validate columns exist
    missing = [c for c in specialist_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing specialist prediction columns: {missing}")

    all_features = specialist_cols + top_features
    log.info(f"\nInput features: {len(all_features)}  "
             f"({len(specialist_cols)} specialist + {len(top_features)} SHAP-top)")

    X = df[all_features].values.astype(np.float32)
    y = df["tm"].values.astype(np.float32)
    log.info(f"  Target range: [{y.min():.1f}, {y.max():.1f}]°C")

    # ── Splits ────────────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=XGB_META_TEST_SIZE, random_state=SEED
    )
    val_fraction = XGB_META_VAL_SIZE / (1 - XGB_META_TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=SEED
    )
    log.info(f"\nSplits:  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("\nTraining XGBoost meta-learner...")
    log.info(f"  Params: {XGB_META_PARAMS}")
    model = xgb.XGBRegressor(**XGB_META_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=XGB_META_EARLY_STOPPING,
        verbose=True,
    )
    log.info(f"\n  ✓ Best iteration: {model.best_iteration}  Best RMSE: {model.best_score:.4f}°C")

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("\nEvaluation:")
    tr_m   = evaluate_split(model, X_train, y_train, "Train")
    val_m  = evaluate_split(model, X_val,   y_val,   "Validation")
    test_m = evaluate_split(model, X_test,  y_test,  "Test")

    # ── Per-range performance ─────────────────────────────────────────────
    test_preds = model.predict(X_test)
    per_range  = {}
    for rname, (lo, hi) in [("Psychrophile", (0,40)), ("Mesophile", (40,70)), ("Thermophile", (70,200))]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() < 10:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_test[mask], test_preds[mask])))
        r2   = float(r2_score(y_test[mask], test_preds[mask]))
        log.info(f"  {rname:<14s} (n={mask.sum():,})  RMSE={rmse:.4f}  R²={r2:.4f}")
        per_range[rname] = {"rmse": rmse, "r2": r2, "n": int(mask.sum())}

    # ── Feature importance plot ───────────────────────────────────────────
    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature":    all_features,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    specialist_imp_total = imp_df[imp_df["feature"].isin(specialist_cols)]["importance"].sum()
    log.info(f"\nSpecialist features total importance: {specialist_imp_total:.4f} "
             f"({specialist_imp_total / importance.sum() * 100:.1f}%)")

    top_n = 30
    top_plot = imp_df.head(top_n)
    colors   = ["red" if f in specialist_cols else "steelblue" for f in top_plot["feature"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_plot)), top_plot["importance"], color=colors)
    ax.set_yticks(range(len(top_plot)))
    ax.set_yticklabels(top_plot["feature"], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features (Red = Specialist Predictions)")
    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = model_out.parent / "feature_importance.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved importance plot: {plot_path}")

    # ── Save ─────────────────────────────────────────────────────────────
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_out))
    log.info(f"\nSaved model:   {model_out}")

    all_metrics = {
        "train": tr_m, "val": val_m, "test": test_m,
        "per_range": per_range,
        "feature_importance": {
            "specialist_total": float(specialist_imp_total),
            "specialist_pct":   float(specialist_imp_total / importance.sum() * 100),
            "top_10": imp_df.head(10)[["feature", "importance"]].to_dict("records"),
        },
    }
    with open(metrics_out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Saved metrics: {metrics_out}")

    log.info("\n✓ XGBoost meta-learner training complete.")
    return model


if __name__ == "__main__":
    train()
