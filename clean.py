"""
Step 1 — Data Cleaning
======================
Loads the raw Kaggle training CSV and applies a multi-stage cleaning pipeline:
  1. Remove physiologically impossible Tm values (< 0°C or > 130°C)
  2. Filter pH to valid range [2.0, 11.0]
  3. Drop rows with missing pH
  4. Deduplicate exact (sequence, pH, Tm) triplets
  5. Filter sequence length to [20, 2000] amino acids
  6. Final Tm filter [10, 130°C]

Output: data/processed/clean_filtered.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root or from src/data/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    RAW_TRAIN_CSV, CLEAN_CSV,
    TM_MIN, TM_MAX, PH_MIN, PH_MAX,
    SEQ_LEN_MIN, SEQ_LEN_MAX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def clean(input_path: Path, output_path: Path) -> pd.DataFrame:
    log.info(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"  Raw rows: {len(df):,}")

    # ── 1. Remove impossible Tm ──────────────────────────────────────────────
    before = len(df)
    df = df[(df["tm"] >= 0) & (df["tm"] <= 130)]
    log.info(f"  After Tm [0, 130] filter:       {len(df):,}  (removed {before - len(df):,})")

    # ── 2. Filter pH range ───────────────────────────────────────────────────
    before = len(df)
    df = df[(df["pH"] >= PH_MIN) & (df["pH"] <= PH_MAX)]
    log.info(f"  After pH [{PH_MIN}, {PH_MAX}] filter:  {len(df):,}  (removed {before - len(df):,})")

    # ── 3. Drop missing pH ───────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["pH"])
    log.info(f"  After dropping NaN pH:          {len(df):,}  (removed {before - len(df):,})")

    # ── 4. Deduplicate exact (sequence, pH, Tm) triplets ────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["protein_sequence", "pH", "tm"])
    log.info(f"  After deduplication:            {len(df):,}  (removed {before - len(df):,})")

    # ── 5. Sequence length filter ────────────────────────────────────────────
    df["seq_len"] = df["protein_sequence"].str.len()
    before = len(df)
    df = df[(df["seq_len"] >= SEQ_LEN_MIN) & (df["seq_len"] <= SEQ_LEN_MAX)]
    log.info(f"  After length [{SEQ_LEN_MIN}, {SEQ_LEN_MAX}] filter: {len(df):,}  (removed {before - len(df):,})")

    # ── 6. Final Tm filter ───────────────────────────────────────────────────
    before = len(df)
    df = df[(df["tm"] >= TM_MIN) & (df["tm"] <= TM_MAX)]
    log.info(f"  After final Tm [{TM_MIN}, {TM_MAX}] filter:{len(df):,}  (removed {before - len(df):,})")

    log.info(f"\nFinal clean dataset: {len(df):,} rows, {df['protein_sequence'].nunique():,} unique sequences")
    log.info(f"  Tm  — mean: {df['tm'].mean():.1f}°C  std: {df['tm'].std():.1f}°C  "
             f"range: [{df['tm'].min():.1f}, {df['tm'].max():.1f}]°C")
    log.info(f"  pH  — mean: {df['pH'].mean():.2f}  range: [{df['pH'].min():.1f}, {df['pH'].max():.1f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"\nSaved to: {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw protein thermostability dataset.")
    parser.add_argument("--input",  type=Path, default=RAW_TRAIN_CSV, help="Path to raw train.csv")
    parser.add_argument("--output", type=Path, default=CLEAN_CSV,     help="Path for cleaned output CSV")
    args = parser.parse_args()
    clean(args.input, args.output)
