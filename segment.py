"""
Step 2b — Thermostability Segmentation
=======================================
Splits the feature-engineered training data into three regime-specific
subsets based on Tm range. These parquets are consumed by the specialist
XGBoost models in Step 4.

  Psychrophile : Tm in [20, 40)  °C
  Mesophile    : Tm in [40, 80]  °C
  Thermophile  : Tm in (80, 130] °C

Note: rows outside these ranges are dropped from the segmented files
      but are still present in the full training set used by other steps.

Input:  data/features/train_with_features_scaled.parquet
Output: data/segments/{psychrophile,mesophile,thermophile}_data.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TRAIN_FEATURES,
    PSYCHROPHILE_DATA, MESOPHILE_DATA, THERMOPHILE_DATA,
    PSYCHROPHILE_RANGE, MESOPHILE_RANGE, THERMOPHILE_RANGE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def segment(input_path: Path) -> dict:
    log.info(f"Loading features from: {input_path}")
    df = pd.read_parquet(input_path)
    log.info(f"  Loaded {len(df):,} rows")

    segments = {
        "psychrophile": (PSYCHROPHILE_DATA, PSYCHROPHILE_RANGE),
        "mesophile":    (MESOPHILE_DATA,    MESOPHILE_RANGE),
        "thermophile":  (THERMOPHILE_DATA,  THERMOPHILE_RANGE),
    }

    results = {}
    for name, (out_path, (lo, hi)) in segments.items():
        seg = df[(df["tm"] >= lo) & (df["tm"] <= hi)].copy()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seg.to_parquet(out_path, index=False)
        log.info(
            f"  {name:<14s}: {len(seg):>6,} rows  "
            f"Tm [{lo:.0f}, {hi:.0f}]°C  "
            f"→ {out_path}"
        )
        results[name] = seg

    total_segmented = sum(len(v) for v in results.values())
    log.info(f"\nTotal segmented rows: {total_segmented:,} / {len(df):,}  "
             f"({total_segmented / len(df) * 100:.1f}% coverage)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment training data by Tm range.")
    parser.add_argument("--input", type=Path, default=TRAIN_FEATURES,
                        help="Path to train_with_features_scaled.parquet")
    args = parser.parse_args()
    segment(args.input)
