"""
Step 4a — Extra Features XGBoost (Optuna-Tuned)
================================================
Trains an XGBoost regressor on the full feature set:
  - ESM-2 embeddings (1280 dims)
  - Scaled handcrafted physicochemical features (~50 dims)
  - pH_scaled

Hyperparameters are optimised with Optuna (30 trials, 5-fold CV,
MedianPruner). The best trial's params are then used to train a final
model on the full training set with early stopping.

Best trial (Trial 27):
  CV RMSE    : 8.4316
  Test RMSE  : ~6.5–7.0°C
  Test R²    : ~0.63

Outputs:
  outputs/models/extra_features/xgb_optuna_best.json
  outputs/models/extra_features/xgb_feature_columns.json
  outputs/models/extra_features/model_metrics.json
"""

import argparse
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
    TRAIN_FEATURES, TEST_FEATURES,
    XGB_EXTRA_MODEL, XGB_EXTRA_FEATS, XGB_EXTRA_METRICS,
    XGB_EXTRA_CV_FOLDS, XGB_OPTUNA_TRIALS, XGB_EARLY_STOPPING,
    SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_feature_cols(df: pd.DataFrame) -> list:
    emb_cols  = [c for c in df.columns if c.startswith("emb_")]
    phys_cols = [c for c in df.columns if c.endswith("_scaled")]
    return emb_cols + phys_cols


def optuna_search(X: np.ndarray, y: np.ndarray, n_trials: int, n_folds: int) -> dict:
    """Run Optuna hyperparameter search. Returns best params dict."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna is required. Install with: pip install optuna")

    from optuna.pruners import MedianPruner

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "learning_rate":      trial.suggest_float("learning_rate",     1e-3, 0.3, log=True),
            "max_depth":          trial.suggest_int("max_depth",            3, 10),
            "subsample":          trial.suggest_float("subsample",          0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree",   0.5, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel",  0.5, 1.0),
            "colsample_bynode":   trial.suggest_float("colsample_bynode",   0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight",     1, 20),
            "gamma":              trial.suggest_float("gamma",              0.0, 5.0),
            "reg_lambda":         trial.suggest_float("reg_lambda",         1e-8, 10.0, log=True),
            "reg_alpha":          trial.suggest_float("reg_alpha",          1e-8, 10.0, log=True),
            "max_leaves":         trial.suggest_int("max_leaves",           10, 50),
            "grow_policy":        "lossguide",
            "n_estimators":       500,
            "tree_method":        "hist",
            "random_state":       SEED,
            "n_jobs":             -1,
            "verbosity":          0,
        }

        rmses = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=30,
                verbose=False,
            )
            preds = model.predict(X_va)
            rmse  = np.sqrt(mean_squared_error(y_va, preds))
            rmses.append(rmse)

            trial.report(np.mean(rmses), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(rmses)

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"\nBest trial: {study.best_trial.number}  CV RMSE: {study.best_value:.5f}")
    log.info(f"Best params: {study.best_params}")
    return study.best_params


def train(
    train_parquet: Path,
    test_parquet: Path,
    model_out: Path,
    feats_out: Path,
    metrics_out: Path,
    run_optuna: bool = True,
):
    log.info(f"\n{'='*60}")
    log.info("EXTRA FEATURES XGBoost MODEL")
    log.info(f"{'='*60}")

    log.info(f"Loading train: {train_parquet}")
    train_df = pd.read_parquet(train_parquet)
    log.info(f"  {len(train_df):,} rows")

    feature_cols = get_feature_cols(train_df)
    log.info(f"  Features: {len(feature_cols)}  "
             f"({sum(c.startswith('emb') for c in feature_cols)} emb + "
             f"{sum(c.endswith('scaled') for c in feature_cols)} phys)")

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["tm"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # ── Optuna or use stored best params ──────────────────────────────────
    if run_optuna:
        log.info(f"\nRunning Optuna ({XGB_OPTUNA_TRIALS} trials, {XGB_EXTRA_CV_FOLDS}-fold CV)...")
        best_params = optuna_search(X_train, y_train, XGB_OPTUNA_TRIALS, XGB_EXTRA_CV_FOLDS)
    else:
        log.info("\nUsing stored best hyperparameters (Trial 27)")
        from configs.config import XGB_EXTRA_PARAMS
        best_params = {k: v for k, v in XGB_EXTRA_PARAMS.items()
                       if k not in ("n_estimators", "tree_method", "random_state", "n_jobs", "verbosity")}

    # ── Final model training ───────────────────────────────────────────────
    log.info("\nTraining final model (2000 rounds, early stopping)...")
    final_params = {
        **best_params,
        "n_estimators":  2000,
        "tree_method":   "hist",
        "random_state":  SEED,
        "n_jobs":        -1,
        "verbosity":     0,
    }
    model = xgb.XGBRegressor(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=XGB_EARLY_STOPPING,
        verbose=100,
    )
    log.info(f"  Best iteration: {model.best_iteration}")

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(X_e, y_e, split_name):
        preds = model.predict(X_e)
        rmse  = float(np.sqrt(mean_squared_error(y_e, preds)))
        mae   = float(mean_absolute_error(y_e, preds))
        r2    = float(r2_score(y_e, preds))
        pearson, _ = pearsonr(y_e, preds)
        log.info(f"  {split_name:<8s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  Pearson={pearson:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": float(pearson)}

    log.info("\nEvaluation:")
    tr_metrics   = evaluate(X_train, y_train, "Train")
    test_metrics = evaluate(X_test,  y_test,  "Test")

    # ── Save ─────────────────────────────────────────────────────────────
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_out))
    log.info(f"\nSaved model:    {model_out}")

    feats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(feats_out, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"Saved features: {feats_out}")

    metrics = {
        "train": tr_metrics,
        "test":  test_metrics,
        "best_iteration": model.best_iteration,
        "n_features": len(feature_cols),
        "hyperparams": final_params,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics:  {metrics_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train extra-features XGBoost model.")
    parser.add_argument("--train",       type=Path, default=TRAIN_FEATURES)
    parser.add_argument("--test",        type=Path, default=TEST_FEATURES)
    parser.add_argument("--model-out",   type=Path, default=XGB_EXTRA_MODEL)
    parser.add_argument("--feats-out",   type=Path, default=XGB_EXTRA_FEATS)
    parser.add_argument("--metrics-out", type=Path, default=XGB_EXTRA_METRICS)
    parser.add_argument("--no-optuna",   action="store_true",
                        help="Skip Optuna; use stored best hyperparams instead")
    args = parser.parse_args()

    train(
        args.train, args.test,
        args.model_out, args.feats_out, args.metrics_out,
        run_optuna=not args.no_optuna,
    )
