"""
Step 2a — ESM-2 Embeddings
===========================
Generates mean-pooled per-residue embeddings from ESM-2 650M for all
sequences in the cleaned dataset. Embeddings are 1280-dimensional float32
vectors — one per sequence (mean pooling over residue positions).

Rationale for mean pooling: Tm is a global thermodynamic property of the
entire protein, so collapsing per-residue representations to a sequence-level
vector is both appropriate and computationally tractable.

Model: facebook/esm2_t33_650M_UR50D (650 M parameters, 1280-dim)
Uses chunked processing with checkpointing to avoid OOM on long sequences.

Inputs:
  data/processed/clean_filtered.csv   (train)
  data/raw/test.csv                   (test)

Outputs:
  data/embeddings/train_embeddings_meta.parquet
  data/embeddings/test_embeddings_meta.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    CLEAN_CSV, RAW_TEST_CSV,
    TRAIN_EMBEDDINGS, TEST_EMBEDDINGS,
    ESM_MODEL_NAME, ESM_EMBED_DIM, ESM_BATCH_SIZE, ESM_CHUNK_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_esm_model():
    """Load ESM-2 650M model and batch converter."""
    try:
        import esm
    except ImportError:
        raise ImportError(
            "fair-esm is not installed. Install with: pip install fair-esm"
        )
    log.info(f"Loading ESM-2 model: {ESM_MODEL_NAME}")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"  Model loaded on: {device}")
    return model, batch_converter, device


def embed_sequences(
    sequences: list,
    model,
    batch_converter,
    device,
    batch_size: int = ESM_BATCH_SIZE,
    chunk_size: int = ESM_CHUNK_SIZE,
    checkpoint_path: Path = None,
) -> np.ndarray:
    """
    Embed a list of amino-acid sequences using ESM-2.
    Returns ndarray of shape (N, ESM_EMBED_DIM).
    Supports chunked checkpointing to survive OOM / interruptions.
    """
    n = len(sequences)
    embeddings = np.zeros((n, ESM_EMBED_DIM), dtype=np.float32)

    # Determine starting chunk (resume if checkpoint exists)
    start_chunk = 0
    if checkpoint_path and checkpoint_path.exists():
        cp = np.load(checkpoint_path, allow_pickle=True).item()
        embeddings[: cp["processed"]] = cp["embeddings"][: cp["processed"]]
        start_chunk = cp["processed"] // chunk_size
        log.info(f"  Resuming from checkpoint: {cp['processed']:,} / {n:,} done")

    for chunk_start in range(start_chunk * chunk_size, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        log.info(f"  Processing sequences {chunk_start:,} – {chunk_end:,} / {n:,}")

        for batch_start in range(chunk_start, chunk_end, batch_size):
            batch_end = min(batch_start + batch_size, chunk_end)
            batch_seqs = sequences[batch_start:batch_end]

            # ESM expects list of (label, sequence) tuples
            data = [(str(i), seq) for i, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)

            token_reps = results["representations"][33]  # (B, L+2, D) — includes BOS/EOS

            for j, seq in enumerate(batch_seqs):
                # Mean pool over residue tokens only (exclude BOS at 0 and EOS at -1)
                seq_len = len(seq)
                emb = token_reps[j, 1: seq_len + 1].mean(0).cpu().numpy()
                embeddings[batch_start + j] = emb

        # Save checkpoint after each chunk
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(checkpoint_path, {"processed": chunk_end, "embeddings": embeddings})

    return embeddings


def generate_embeddings(
    input_csv: Path,
    output_parquet: Path,
    sequence_col: str = "protein_sequence",
    checkpoint_path: Path = None,
):
    log.info(f"\n{'='*60}")
    log.info(f"Generating embeddings for: {input_csv.name}")
    log.info(f"{'='*60}")

    df = pd.read_csv(input_csv)
    log.info(f"  Loaded {len(df):,} rows")

    sequences = df[sequence_col].tolist()
    model, batch_converter, device = load_esm_model()

    embeddings = embed_sequences(
        sequences, model, batch_converter, device,
        checkpoint_path=checkpoint_path,
    )
    log.info(f"  Embedding shape: {embeddings.shape}")

    # Build output dataframe: metadata columns + emb_0..emb_1279
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(ESM_EMBED_DIM)],
    )

    # Carry forward metadata columns (everything except sequence itself)
    meta_cols = [c for c in df.columns if c != sequence_col]
    result = pd.concat([df[meta_cols].reset_index(drop=True), emb_df], axis=1)

    # Re-attach sequence for downstream use
    result[sequence_col] = df[sequence_col].values

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_parquet, index=False)
    log.info(f"  Saved to: {output_parquet}  ({result.shape[1]} columns)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to embed")
    args = parser.parse_args()

    if args.split in ("train", "both"):
        generate_embeddings(
            input_csv=CLEAN_CSV,
            output_parquet=TRAIN_EMBEDDINGS,
            checkpoint_path=TRAIN_EMBEDDINGS.parent / "train_embed_checkpoint.npy",
        )

    if args.split in ("test", "both"):
        generate_embeddings(
            input_csv=RAW_TEST_CSV,
            output_parquet=TEST_EMBEDDINGS,
            checkpoint_path=TEST_EMBEDDINGS.parent / "test_embed_checkpoint.npy",
        )
