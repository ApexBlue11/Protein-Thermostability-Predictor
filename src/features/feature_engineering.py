"""
Step 3 — Handcrafted Feature Engineering
==========================================
Adds sequence-derived physicochemical features to the embedding parquets
using BioPython's ProteinAnalysis. Features are fit-scaled on train only;
the fitted scaler is saved and reused for test.

Feature groups
--------------
Basic           : length, orig_length, n_ambiguous, frac_ambiguous
Composition     : aa_frac_{X} for each of the 20 standard amino acids
Grouped         : frac_hydrophobic, frac_polar, frac_positive, frac_negative
ProtParam       : molecular_weight, aromaticity, instability_index,
                  aliphatic_index, isoelectric_point, gravy
Charge          : charge_at_ph, charge_density
pH              : pH_scaled (StandardScaler, fit on train)

Ambiguous residues (B, Z, X, U, O, J) are replaced with a default residue
before ProteinAnalysis to avoid computation errors.

Inputs:
  data/embeddings/train_embeddings_meta.parquet
  data/embeddings/test_embeddings_meta.parquet

Outputs:
  data/features/train_with_features_scaled.parquet
  data/features/test_with_features_scaled.parquet
  data/features/handcrafted_scaler.joblib
"""

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TRAIN_EMBEDDINGS, TEST_EMBEDDINGS,
    TRAIN_FEATURES, TEST_FEATURES, SCALER_PATH,
    AMINO_ACIDS,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Amino acid groupings for bulk fractions
_HYDROPHOBIC = set("AILMFWV")
_POLAR       = set("STNQ")
_POSITIVE    = set("KRH")
_NEGATIVE    = set("DE")
_AMBIGUOUS   = set("BZXUOJ")
_DEFAULT_SUB = "A"   # replacement for ambiguous residues


def _clean_sequence(seq: str) -> tuple[str, str]:
    """
    Returns (original_seq, cleaned_seq).
    Cleaned: upper-cased, ambiguous residues replaced with _DEFAULT_SUB.
    """
    seq = seq.upper().strip()
    cleaned = re.sub(f"[{''.join(_AMBIGUOUS)}]", _DEFAULT_SUB, seq)
    return seq, cleaned


def compute_features(seq_orig: str, seq_clean: str, ph: float) -> dict:
    """Compute all handcrafted features for one sequence."""
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        raise ImportError("biopython is required. Install with: pip install biopython")

    n_total   = len(seq_orig)
    n_ambig   = sum(1 for c in seq_orig if c in _AMBIGUOUS)
    frac_ambig = n_ambig / n_total if n_total > 0 else 0.0

    # Per-AA fractions (on original, ambiguous chars counted in denominator)
    aa_fracs = {
        f"aa_frac_{aa}": seq_orig.count(aa) / n_total if n_total > 0 else 0.0
        for aa in AMINO_ACIDS
    }

    # Grouped fractions
    frac_hydrophobic = sum(seq_orig.count(aa) for aa in _HYDROPHOBIC) / n_total if n_total > 0 else 0.0
    frac_polar       = sum(seq_orig.count(aa) for aa in _POLAR)       / n_total if n_total > 0 else 0.0
    frac_positive    = sum(seq_orig.count(aa) for aa in _POSITIVE)    / n_total if n_total > 0 else 0.0
    frac_negative    = sum(seq_orig.count(aa) for aa in _NEGATIVE)    / n_total if n_total > 0 else 0.0

    # BioPython ProtParam on cleaned sequence
    pa = ProteinAnalysis(seq_clean)
    mw              = pa.molecular_weight()
    aromaticity     = pa.aromaticity()
    instability     = pa.instability_index()
    aliphatic       = pa.aliphatic_index()
    isoelectric     = pa.isoelectric_point()
    gravy           = pa.gravy()
    charge_at_ph    = pa.charge_at_pH(ph)
    charge_density  = charge_at_ph / n_total if n_total > 0 else 0.0

    feats = {
        "length":            n_total,
        "orig_length":       len(seq_orig),
        "n_ambiguous":       n_ambig,
        "frac_ambiguous":    frac_ambig,
        **aa_fracs,
        "frac_hydrophobic":  frac_hydrophobic,
        "frac_polar":        frac_polar,
        "frac_positive":     frac_positive,
        "frac_negative":     frac_negative,
        "molecular_weight":  mw,
        "aromaticity":       aromaticity,
        "instability_index": instability,
        "aliphatic_index":   aliphatic,
        "isoelectric_point": isoelectric,
        "gravy":             gravy,
        "charge_at_ph":      charge_at_ph,
        "charge_density":    charge_density,
    }
    return feats


def build_feature_matrix(df: pd.DataFrame, ph_col: str = "pH") -> pd.DataFrame:
    """Compute handcrafted features for all rows in df."""
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        seq_orig = row.protein_sequence
        ph       = getattr(row, ph_col)
        _, seq_clean = _clean_sequence(seq_orig)
        try:
            feats = compute_features(seq_orig, seq_clean, ph)
        except Exception as e:
            log.warning(f"  Row {i} feature error: {e}. Using zeros.")
            feats = {}
        records.append(feats)
        if (i + 1) % 5000 == 0:
            log.info(f"    Processed {i+1:,} / {len(df):,}")

    return pd.DataFrame(records)


def add_features(
    train_emb: Path,
    test_emb: Path,
    train_out: Path,
    test_out: Path,
    scaler_out: Path,
):
    # ── Train ────────────────────────────────────────────────────────────────
    log.info(f"\nLoading train embeddings: {train_emb}")
    train_df = pd.read_parquet(train_emb)
    log.info(f"  {len(train_df):,} rows, {train_df.shape[1]} columns")

    log.info("Computing handcrafted features for TRAIN...")
    train_feats = build_feature_matrix(train_df)

    # Scale pH and all numeric handcrafted features that aren't fractions
    # (fractions are already [0,1], still scale for consistency with XGB)
    feat_cols_to_scale = list(train_feats.columns)  # scale all handcrafted features
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats[feat_cols_to_scale])
    train_feats_scaled_df = pd.DataFrame(
        train_feats_scaled,
        columns=[f"{c}_scaled" for c in feat_cols_to_scale],
    )

    # Also scale pH standalone
    ph_scaler = StandardScaler()
    train_df["pH_scaled"] = ph_scaler.fit_transform(train_df[["pH"]]).ravel()

    train_result = pd.concat(
        [train_df.reset_index(drop=True), train_feats_scaled_df],
        axis=1,
    )
    train_result.to_parquet(train_out, index=False)
    log.info(f"  Saved train features: {train_out}  ({train_result.shape[1]} cols)")

    # Save scalers
    joblib.dump({"feat_scaler": scaler, "ph_scaler": ph_scaler, "feat_cols": feat_cols_to_scale},
                scaler_out)
    log.info(f"  Saved scaler: {scaler_out}")

    # ── Test ─────────────────────────────────────────────────────────────────
    log.info(f"\nLoading test embeddings: {test_emb}")
    test_df = pd.read_parquet(test_emb)
    log.info(f"  {len(test_df):,} rows, {test_df.shape[1]} columns")

    log.info("Computing handcrafted features for TEST...")
    test_feats = build_feature_matrix(test_df)

    # Apply train scaler (no re-fit)
    test_feats_scaled = scaler.transform(test_feats[feat_cols_to_scale])
    test_feats_scaled_df = pd.DataFrame(
        test_feats_scaled,
        columns=[f"{c}_scaled" for c in feat_cols_to_scale],
    )
    test_df["pH_scaled"] = ph_scaler.transform(test_df[["pH"]]).ravel()

    test_result = pd.concat(
        [test_df.reset_index(drop=True), test_feats_scaled_df],
        axis=1,
    )
    test_result.to_parquet(test_out, index=False)
    log.info(f"  Saved test features:  {test_out}  ({test_result.shape[1]} cols)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add handcrafted features to embedding parquets.")
    parser.add_argument("--train-emb",  type=Path, default=TRAIN_EMBEDDINGS)
    parser.add_argument("--test-emb",   type=Path, default=TEST_EMBEDDINGS)
    parser.add_argument("--train-out",  type=Path, default=TRAIN_FEATURES)
    parser.add_argument("--test-out",   type=Path, default=TEST_FEATURES)
    parser.add_argument("--scaler-out", type=Path, default=SCALER_PATH)
    args = parser.parse_args()

    add_features(
        args.train_emb, args.test_emb,
        args.train_out, args.test_out,
        args.scaler_out,
    )
