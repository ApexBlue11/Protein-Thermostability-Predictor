"""
run_pipeline.py — End-to-End Pipeline Runner
=============================================
Executes the full protein thermostability prediction pipeline in order.
Individual stages can be skipped with --skip-* flags.

Pipeline stages:
  1. clean          — Data cleaning
  2. embed          — ESM-2 embeddings (GPU recommended)
  3. features       — Handcrafted feature engineering
  4. segment        — Tm-based data segmentation
  5. xgb-extra      — Extra features XGBoost (Optuna)
  6. specialists    — Train 3 specialist XGBoost models + generate df_with_preds
  7. shap           — SHAP feature selection
  8. xgb-meta       — XGBoost meta-learner
  9. mlp-meta       — MLP meta-learner

Usage:
  # Full pipeline
  python scripts/run_pipeline.py

  # Skip embedding (already done) and Optuna (use stored params)
  python scripts/run_pipeline.py --skip-embed --no-optuna

  # Run only meta-learner stages
  python scripts/run_pipeline.py --skip-clean --skip-embed --skip-features \\
      --skip-segment --skip-xgb-extra --skip-specialists --skip-shap
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _stage(name: str, fn, *args, **kwargs):
    log.info(f"\n{'#'*60}")
    log.info(f"# STAGE: {name}")
    log.info(f"{'#'*60}")
    t0 = time.time()
    fn(*args, **kwargs)
    elapsed = time.time() - t0
    log.info(f"  ✓ {name} complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run protein thermostability prediction pipeline.")
    parser.add_argument("--skip-clean",       action="store_true")
    parser.add_argument("--skip-embed",        action="store_true")
    parser.add_argument("--skip-features",     action="store_true")
    parser.add_argument("--skip-segment",      action="store_true")
    parser.add_argument("--skip-xgb-extra",    action="store_true")
    parser.add_argument("--skip-specialists",  action="store_true")
    parser.add_argument("--skip-shap",         action="store_true")
    parser.add_argument("--skip-xgb-meta",     action="store_true")
    parser.add_argument("--skip-mlp-meta",     action="store_true")
    parser.add_argument("--no-optuna",         action="store_true",
                        help="Skip Optuna for XGB extra; use stored best params")
    args = parser.parse_args()

    if not args.skip_clean:
        from src.data.clean import clean
        from configs.config import RAW_TRAIN_CSV, CLEAN_CSV
        _stage("Data Cleaning", clean, RAW_TRAIN_CSV, CLEAN_CSV)

    if not args.skip_embed:
        from src.embeddings.generate_embeddings import generate_embeddings
        from configs.config import CLEAN_CSV, RAW_TEST_CSV, TRAIN_EMBEDDINGS, TEST_EMBEDDINGS
        _stage("Train Embeddings", generate_embeddings, CLEAN_CSV, TRAIN_EMBEDDINGS)
        _stage("Test Embeddings",  generate_embeddings, RAW_TEST_CSV, TEST_EMBEDDINGS)

    if not args.skip_features:
        from src.features.feature_engineering import add_features
        from configs.config import (TRAIN_EMBEDDINGS, TEST_EMBEDDINGS,
                                    TRAIN_FEATURES, TEST_FEATURES, SCALER_PATH)
        _stage("Feature Engineering", add_features,
               TRAIN_EMBEDDINGS, TEST_EMBEDDINGS,
               TRAIN_FEATURES, TEST_FEATURES, SCALER_PATH)

    if not args.skip_segment:
        from src.data.segment import segment
        from configs.config import TRAIN_FEATURES
        _stage("Segmentation", segment, TRAIN_FEATURES)

    if not args.skip_xgb_extra:
        from src.models.train_extra_features_xgb import train as train_xgb_extra
        from configs.config import (TRAIN_FEATURES, TEST_FEATURES,
                                    XGB_EXTRA_MODEL, XGB_EXTRA_FEATS, XGB_EXTRA_METRICS)
        _stage("XGB Extra Features", train_xgb_extra,
               TRAIN_FEATURES, TEST_FEATURES,
               XGB_EXTRA_MODEL, XGB_EXTRA_FEATS, XGB_EXTRA_METRICS,
               run_optuna=not args.no_optuna)

    if not args.skip_specialists:
        from src.models.train_specialists import main as train_specialists
        _stage("Specialist Models", train_specialists)

    if not args.skip_shap:
        from src.models.shap_feature_selection import run_shap_selection
        _stage("SHAP Feature Selection", run_shap_selection)

    if not args.skip_xgb_meta:
        from src.models.train_xgb_meta import train as train_xgb_meta
        _stage("XGB Meta-Learner", train_xgb_meta)

    if not args.skip_mlp_meta:
        from src.models.train_mlp_meta import train as train_mlp_meta
        _stage("MLP Meta-Learner", train_mlp_meta)

    log.info(f"\n{'='*60}")
    log.info("✓ PIPELINE COMPLETE")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
