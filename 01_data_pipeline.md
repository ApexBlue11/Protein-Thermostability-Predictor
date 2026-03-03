# Phase 1 — Data Pipeline

## Overview

The data pipeline transforms raw Kaggle CSV data into fully embedded, feature-engineered, and segmented parquets ready for model training. It runs in four sequential steps.

---

## Step 1: Cleaning (`src/data/clean.py`)

**Input:** `data/raw/train.csv` (raw Kaggle competition data, ~31k rows)
**Output:** `data/processed/clean_filtered.csv` (~30,443 rows)

### Filtering stages (applied in order)

| Stage | Condition | Rationale |
|-------|-----------|-----------|
| Tm range | `0 ≤ Tm ≤ 130°C` | Remove physically impossible values |
| pH range | `2.0 ≤ pH ≤ 11.0` | Restrict to experimentally valid range |
| Missing pH | Drop NaN | Downstream features require pH |
| Deduplication | Drop exact `(sequence, pH, Tm)` triplets | Prevent label leakage |
| Sequence length | `20 ≤ len ≤ 2000` | Filter fragments and excessively long sequences |
| Final Tm | `10 ≤ Tm ≤ 130°C` | Tighten lower bound to remove edge cases |

**Final dataset stats:**
- Rows: ~30,443 training samples
- Unique sequences: ~28,747
- Tm: mean ~52°C, range [10, 130°C]

---

## Step 2a: ESM-2 Embeddings (`src/embeddings/generate_embeddings.py`)

**Input:** `data/processed/clean_filtered.csv`, `data/raw/test.csv`
**Outputs:** `data/embeddings/{train,test}_embeddings_meta.parquet`

### Model choice

ESM-2 650M (`esm2_t33_650M_UR50D`) was selected over the 150M variant for richer sequence representations within the available GPU budget. The final layer (layer 33) produces 1280-dimensional per-residue representations.

### Pooling strategy

**Mean pooling** over residue positions (excluding BOS/EOS tokens). Tm is a global thermodynamic property of the whole protein — no position-specific resolution is needed, and mean pooling is both appropriate and dramatically reduces memory/storage compared to storing full token matrices.

### Checkpointing

Processing is chunked in blocks of 500 sequences with `.npy` checkpoints saved after each chunk. This allows interruption and resumption without re-embedding.

### Output format

Each output parquet contains:
- All original metadata columns (excluding `protein_sequence`)
- `emb_0` through `emb_1279` — 1280 float32 embedding dimensions
- `protein_sequence` — re-attached at the end

---

## Step 2b: Segmentation (`src/data/segment.py`)

**Input:** `data/features/train_with_features_scaled.parquet`
**Outputs:** `data/segments/{psychrophile,mesophile,thermophile}_data.parquet`

Splits the training set into three thermostability regimes:

| Segment | Tm Range | Rows (approx) | Biological context |
|---------|----------|---------------|-------------------|
| Psychrophile | 20–40°C | ~2,800 | Cold-adapted organisms |
| Mesophile | 40–80°C | ~18,500 | Typical organisms |
| Thermophile | 80–130°C | ~9,100 | Thermophilic organisms |

**Rationale:** A single model trained on the full Tm range must learn very different sequence-stability relationships simultaneously. Segmentation reduces target heterogeneity per model, allowing each specialist to focus on regime-specific patterns.

---

## Step 3: Feature Engineering (`src/features/feature_engineering.py`)

**Input:** `data/embeddings/{train,test}_embeddings_meta.parquet`
**Outputs:**
- `data/features/{train,test}_with_features_scaled.parquet`
- `data/features/handcrafted_scaler.joblib`

### Features added (BioPython ProteinAnalysis)

**Basic:**
- `length`, `orig_length` — cleaned vs original sequence length
- `n_ambiguous`, `frac_ambiguous` — ambiguous residue counts (B, Z, X, U, O, J)

**Per-amino-acid fractions (20 features):**
- `aa_frac_A` through `aa_frac_Y` — fractional composition of each standard AA

**Grouped fractions (4 features):**
- `frac_hydrophobic` (AILMFWV), `frac_polar` (STNQ), `frac_positive` (KRH), `frac_negative` (DE)

**ProtParam (6 features):**
- `molecular_weight`, `aromaticity`, `instability_index`, `aliphatic_index`, `isoelectric_point`, `gravy`

**Charge (2 features):**
- `charge_at_ph` — net charge computed at actual experimental pH
- `charge_density` — charge / sequence length

**pH (1 feature):**
- `pH_scaled` — StandardScaler-normalised pH

### Scaling

All handcrafted features are scaled with `StandardScaler` **fit on train only**, then applied to test. The scaler is serialised to `handcrafted_scaler.joblib` for reproducibility and inference.

Ambiguous residues are replaced with `A` (alanine) before ProteinAnalysis to prevent computation errors, but original counts are preserved in `n_ambiguous` / `frac_ambiguous`.

---

## Data lineage summary

```
train.csv (raw, ~31k)
    └─► clean_filtered.csv (~30,443)
            └─► train_embeddings_meta.parquet   (+ emb_0..1279)
                    └─► train_with_features_scaled.parquet  (+ *_scaled features)
                                ├─► psychrophile_data.parquet
                                ├─► mesophile_data.parquet
                                ├─► thermophile_data.parquet
                                └─► [used directly by specialist training]
```
