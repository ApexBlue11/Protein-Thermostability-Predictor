"""
Evaluation — Permutation Importance + SHAP Validation
=======================================================
Post-training validation utility. Runs for a given trained model and test
split, computing:
  1. Predictions + basic regression metrics (RMSE, MAE, R², Pearson)
  2. Permutation feature importance (sklearn, works for both XGB and MLP)
  3. SHAP values:
       - XGBoost: TreeExplainer (fast, exact)
       - MLP:     KernelExplainer on top-K features only (approximation)

Results (JSON + plots) are saved to the specified output directory.

Usage:
  python evaluation/evaluate.py \
    --model-type xgboost \
    --model-path outputs/models/meta/xgb_meta_best.json \
    --data-path  data/meta/df_with_preds.parquet \
    --features-path outputs/artifacts/top_50_features.json \
    --output-dir outputs/evaluation/xgb_meta \
    --segment-name xgb_meta
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TOP_K         = 40
PERM_REPEATS  = 8
SHAP_BG       = 100
SHAP_NSAMPLES = 200
SHAP_TEST_SAMPLE = 200
N_JOBS_PERM   = 1


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn-compatible wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SklearnXGBWrapper:
    def __init__(self, booster):
        self.bst = booster

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.bst.predict(xgb.DMatrix(X))


class SklearnTorchWrapper:
    def __init__(self, model, device, batch_size=256):
        self.model      = model
        self.device     = device
        self.batch_size = batch_size

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[i:i+self.batch_size].astype(np.float32)).to(self.device)
                preds.append(self.model(xb).cpu().numpy().flatten())
        return np.concatenate(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_type: str,       # "xgboost" or "mlp"
    model_path: Path,
    data_path: Path,
    features_path: Path,
    output_dir: Path,
    segment_name: str,
    test_size: float = 0.15,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"errors": {}}
    np.random.seed(SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(data_path)
    with open(features_path) as f:
        top_features = json.load(f)

    pred_cols    = [c for c in df.columns if c.startswith("pred_")]
    feature_cols = pred_cols + top_features

    from sklearn.model_selection import train_test_split
    X = df[feature_cols].values.astype(np.float32)
    y = df["tm"].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    X_rest = X[:len(X) - len(X_test)]

    # ── Load model ────────────────────────────────────────────────────────
    device = None
    if model_type == "xgboost":
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        estimator = SklearnXGBWrapper(bst)
    elif model_type == "mlp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)

        # Reconstruct model from checkpoint metadata
        from src.models.train_mlp_meta import ResMLPEnsemble, MLPEnsemble
        cls = ResMLPEnsemble if checkpoint.get("model_class") == "ResMLPEnsemble" else MLPEnsemble
        model = cls(
            n_features=len(feature_cols),
            hidden_dims=checkpoint.get("hidden_dims", [256, 128, 64]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Normalise test data
        X_mean = checkpoint["X_mean"]
        X_std  = checkpoint["X_std"]
        X_test = (X_test - X_mean) / X_std
        X_rest = (X_rest - X_mean) / X_std

        estimator = SklearnTorchWrapper(model, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ── 1. Predictions + metrics ──────────────────────────────────────────
    try:
        if model_type == "xgboost":
            preds = bst.predict(xgb.DMatrix(X_test))
        else:
            preds = estimator.predict(X_test)

        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        pearson_r, pearson_p = pearsonr(y_test, preds)
        log.info(f"RMSE={rmse:.4f}  Pearson r={pearson_r:.4f}  (p={pearson_p:.2e})")
        results["metrics"] = {"test_rmse": rmse, "test_mse": mse,
                               "pearson_r": float(pearson_r), "pearson_p": float(pearson_p)}
    except Exception as e:
        results["errors"]["metrics"] = str(e)
        log.error(f"Metrics failed: {e}")
        raise

    # ── 2. Permutation importance ─────────────────────────────────────────
    try:
        log.info("Computing permutation importance...")
        perm_res = permutation_importance(
            estimator, X_test, y_test,
            n_repeats=PERM_REPEATS, random_state=SEED,
            scoring="neg_mean_squared_error", n_jobs=N_JOBS_PERM,
        )
        importances_mean = perm_res.importances_mean
        idx_sorted = np.argsort(importances_mean)[::-1]
        top_idx = idx_sorted[:min(TOP_K, len(importances_mean))]
        top_feat_names = [feature_cols[int(i)] for i in top_idx]
        top_importances = importances_mean[top_idx]

        perm_json = {
            "top_features": [(feature_cols[int(i)], float(importances_mean[int(i)])) for i in top_idx],
        }
        perm_path = output_dir / f"{segment_name}_permutation_importance.json"
        with open(perm_path, "w") as fh:
            json.dump(perm_json, fh, indent=2)

        fig, ax = plt.subplots(figsize=(8, max(3, len(top_feat_names) * 0.15)))
        y_pos = np.arange(len(top_feat_names))
        ax.barh(y_pos[::-1], top_importances[::-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feat_names)
        ax.set_xlabel("Permutation importance (mean neg MSE decrease)")
        ax.set_title(f"{segment_name} Top-{len(top_feat_names)} Permutation Importances")
        plt.tight_layout()
        perm_plot_path = output_dir / f"{segment_name}_perm_importance_top{len(top_feat_names)}.png"
        plt.savefig(perm_plot_path, dpi=150)
        plt.close()
        results["perm_top_features"] = top_feat_names
        log.info("  ✓ Permutation importance done")
    except Exception as e:
        results["errors"]["permutation"] = str(e)
        results["errors"]["permutation_tb"] = traceback.format_exc()
        log.warning(f"Permutation importance failed: {e}")

    # ── 3. SHAP on top-K features ─────────────────────────────────────────
    try:
        log.info(f"Computing SHAP values on top-{len(top_feat_names)} features...")
        shap_test_idx = np.random.choice(len(X_test), min(SHAP_TEST_SAMPLE, len(X_test)), replace=False)

        if model_type == "xgboost":
            explainer = shap.TreeExplainer(bst)
            shap_vals = explainer.shap_values(X_test[shap_test_idx])
            mean_abs  = np.mean(np.abs(shap_vals), axis=0)
            shap_top  = mean_abs[top_idx]

            fig, ax = plt.subplots(figsize=(8, max(3, len(top_feat_names) * 0.15)))
            y_pos = np.arange(len(top_feat_names))
            ax.barh(y_pos[::-1], shap_top[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_feat_names)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{segment_name} SHAP (TreeExplainer) top-{len(top_feat_names)}")
            plt.tight_layout()
            shap_plot = output_dir / f"{segment_name}_shap_top{len(top_feat_names)}.png"
            plt.savefig(shap_plot, dpi=150)
            plt.close()
            results["shap_plot"] = str(shap_plot)
        else:
            bg_idx    = np.random.choice(len(X_rest), min(SHAP_BG, len(X_rest)), replace=False)
            X_bg_top  = X_rest[bg_idx][:, top_idx]
            X_test_top = X_test[shap_test_idx][:, top_idx]

            def predict_top(x_small):
                X_full = np.zeros((len(x_small), X_test.shape[1]), dtype=np.float32)
                X_full[:, top_idx] = x_small
                return estimator.predict(X_full)

            ke = shap.KernelExplainer(predict_top, X_bg_top)
            small_idx = np.random.choice(len(X_test_top), min(100, len(X_test_top)), replace=False)
            shap_vals = ke.shap_values(X_test_top[small_idx], nsamples=SHAP_NSAMPLES)
            mean_abs  = np.mean(np.abs(np.array(shap_vals)), axis=0)

            fig, ax = plt.subplots(figsize=(8, max(3, len(top_feat_names) * 0.15)))
            y_pos = np.arange(len(top_feat_names))
            ax.barh(y_pos[::-1], mean_abs[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_feat_names[::-1])
            ax.set_xlabel("Mean |SHAP value| (KernelExplainer)")
            ax.set_title(f"{segment_name} SHAP (Kernel) top-{len(top_feat_names)}")
            plt.tight_layout()
            shap_plot = output_dir / f"{segment_name}_shap_kernel_top{len(top_feat_names)}.png"
            plt.savefig(shap_plot, dpi=150)
            plt.close()
            results["shap_plot"] = str(shap_plot)

        log.info("  ✓ SHAP done")
    except Exception as e:
        results["errors"]["shap"] = str(e)
        results["errors"]["shap_tb"] = traceback.format_exc()
        log.warning(f"SHAP failed: {e}")

    # ── Save results ───────────────────────────────────────────────────────
    save_path = output_dir / f"{segment_name}_eval_results.json"
    with open(save_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"\nSaved evaluation results: {save_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type",    required=True, choices=["xgboost", "mlp"])
    parser.add_argument("--model-path",    type=Path, required=True)
    parser.add_argument("--data-path",     type=Path, required=True)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--output-dir",    type=Path, required=True)
    parser.add_argument("--segment-name",  default="model")
    parser.add_argument("--test-size",     type=float, default=0.15)
    args = parser.parse_args()

    evaluate(
        model_type=args.model_type,
        model_path=args.model_path,
        data_path=args.data_path,
        features_path=args.features_path,
        output_dir=args.output_dir,
        segment_name=args.segment_name,
        test_size=args.test_size,
    )
