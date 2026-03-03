"""
Step 6b — MLP Meta-Learner (ResidualMLP)
==========================================
Trains a PyTorch residual MLP as the second meta-learner, using the same
56-feature input as the XGB meta-learner (6 specialist predictions +
top-50 SHAP features). Features are normalised with per-column z-score
(fit on train only; mean/std saved inside the checkpoint).

Architecture: ResMLPEnsemble
  Input → Linear(56→256) → [BN → ReLU → Dropout] × 3 with residual
  connections where dims match → Linear(64→1)

Training:
  Optimizer  : AdamW (lr=3e-4, weight_decay=1e-4)
  Scheduler  : Warmup (10 epochs) + cosine annealing
  Loss       : MSELoss
  Early stop : patience=25 on validation RMSE

Results:
  Train  RMSE=5.2399  R²=0.8563
  Val    RMSE=6.0168  R²=0.8150
  Test   RMSE=6.0728  R²=0.8055  Pearson=0.8980

Inputs:
  data/meta/df_with_preds.parquet
  outputs/artifacts/top_50_features.json

Outputs:
  outputs/models/meta/mlp_best.pt      (state dict + normalisation stats)
  outputs/models/meta/mlp_metrics.json
  outputs/models/meta/training_curves.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    DF_WITH_PREDS, TOP_FEATURES_JSON,
    MLP_META_MODEL, MLP_META_METRICS,
    MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_BATCH_SIZE,
    MLP_LEARNING_RATE, MLP_WEIGHT_DECAY, MLP_NUM_EPOCHS,
    MLP_PATIENCE, MLP_WARMUP_EPOCHS,
    MLP_TEST_SIZE, MLP_VAL_SIZE, MLP_USE_RESIDUAL, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TmDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model architectures
# ─────────────────────────────────────────────────────────────────────────────

class MLPEnsemble(nn.Module):
    """Plain MLP: Linear → BN → ReLU → Dropout, stacked."""

    def __init__(self, n_features: int, hidden_dims=None, dropout: float = 0.2):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResMLPEnsemble(nn.Module):
    """Residual MLP: skip connections where input/output dims match."""

    def __init__(self, n_features: int, hidden_dims=None, dropout: float = 0.2):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        self.input_proj = nn.Linear(n_features, hidden_dims[0])
        self.blocks = nn.ModuleList()
        dims = hidden_dims
        for i in range(len(dims)):
            in_d  = dims[i]
            out_d = dims[i + 1] if i + 1 < len(dims) else dims[i]
            self.blocks.append(
                nn.Sequential(nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d), nn.ReLU(), nn.Dropout(dropout))
            )
        self.output = nn.Linear(dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            identity = x
            x = block(x)
            if x.shape == identity.shape:
                x = x + identity
        return self.output(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        p = model(X_b)
        total_loss += criterion(p, y_b).item() * len(X_b)
        preds_all.append(p.cpu().numpy())
        targets_all.append(y_b.cpu().numpy())
    preds   = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    rmse  = float(np.sqrt(mean_squared_error(targets, preds)))
    mae   = float(mean_absolute_error(targets, preds))
    r2    = float(r2_score(targets, preds))
    pearson, _ = pearsonr(targets, preds)
    return total_loss / len(loader.dataset), rmse, mae, r2, float(pearson)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path: Path = DF_WITH_PREDS,
    features_path: Path = TOP_FEATURES_JSON,
    model_out: Path = MLP_META_MODEL,
    metrics_out: Path = MLP_META_METRICS,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    log.info(f"\n{'='*60}")
    log.info("MLP META-LEARNER")
    log.info(f"{'='*60}")
    log.info(f"  Device: {device}")

    df = pd.read_parquet(data_path)
    with open(features_path) as f:
        top_50 = json.load(f)

    pred_cols    = [c for c in df.columns if c.startswith("pred_")]
    all_features = pred_cols + top_50
    log.info(f"  Input features: {len(all_features)} ({len(pred_cols)} specialist + {len(top_50)} SHAP-top)")

    X = df[all_features].values.astype(np.float32)
    y = df["tm"].values.astype(np.float32)

    # Normalise features (z-score, fit on train later — done after split)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=MLP_TEST_SIZE, random_state=SEED
    )
    val_frac = MLP_VAL_SIZE / (1 - MLP_TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=SEED
    )

    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std

    log.info(f"  Splits: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # DataLoaders
    make_loader = lambda X_, y_, shuffle: DataLoader(
        TmDataset(X_, y_), batch_size=MLP_BATCH_SIZE,
        shuffle=shuffle, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    train_loader = make_loader(X_train_n, y_train, True)
    val_loader   = make_loader(X_val_n,   y_val,   False)
    test_loader  = make_loader(X_test_n,  y_test,  False)

    # Model
    ModelClass = ResMLPEnsemble if MLP_USE_RESIDUAL else MLPEnsemble
    model = ModelClass(n_features=len(all_features), hidden_dims=MLP_HIDDEN_DIMS, dropout=MLP_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model: {ModelClass.__name__}  hidden={MLP_HIDDEN_DIMS}  params={n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _lr_lambda(e, MLP_WARMUP_EPOCHS, MLP_NUM_EPOCHS)
    )

    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_r2": [], "lr": []}
    best_val_rmse = float("inf")
    patience_ctr  = 0

    log.info(f"\n{'─'*60}\nTraining (max {MLP_NUM_EPOCHS} epochs, patience={MLP_PATIENCE})...\n{'─'*60}")

    for epoch in range(MLP_NUM_EPOCHS):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_rmse, val_mae, val_r2, val_pearson = evaluate_loader(model, val_loader, criterion, device)
        val_loss = val_rmse ** 2   # approx
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)
        history["lr"].append(current_lr)

        log.info(f"Epoch {epoch+1:3d}/{MLP_NUM_EPOCHS} | lr={current_lr:.6f} | "
                 f"train_loss={tr_loss:.4f} | val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_ctr  = 0
            model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse":           val_rmse,
                "X_mean":             X_mean,
                "X_std":              X_std,
                "all_features":       all_features,
                "model_class":        ModelClass.__name__,
                "hidden_dims":        MLP_HIDDEN_DIMS,
                "dropout":            MLP_DROPOUT,
            }, model_out)
            log.info(f"  ✓ Best model saved (RMSE={val_rmse:.4f}°C)")
        else:
            patience_ctr += 1
            if patience_ctr >= MLP_PATIENCE:
                log.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    # ── Final evaluation ─────────────────────────────────────────────────
    checkpoint = torch.load(model_out, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.info(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

    log.info("\nFinal Evaluation:")
    _, tr_rmse, tr_mae, tr_r2, tr_p   = evaluate_loader(model, train_loader, criterion, device)
    _, val_rmse, val_mae, val_r2, val_p = evaluate_loader(model, val_loader, criterion, device)
    _, te_rmse, te_mae, te_r2, te_p   = evaluate_loader(model, test_loader, criterion, device)

    for name, (rmse, mae, r2, p) in [
        ("Train",      (tr_rmse,  tr_mae,  tr_r2,  tr_p)),
        ("Validation", (val_rmse, val_mae, val_r2, val_p)),
        ("Test",       (te_rmse,  te_mae,  te_r2,  te_p)),
    ]:
        log.info(f"  {name:<12s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  Pearson={p:.4f}")

    # ── Training curves ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].plot(history["train_loss"], label="Train")
    axes[0,0].plot(history["val_loss"],   label="Val")
    axes[0,0].set(xlabel="Epoch", ylabel="Loss (MSE)", title="Training & Validation Loss")
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(history["val_rmse"], color="orange")
    axes[0,1].set(xlabel="Epoch", ylabel="RMSE (°C)", title="Validation RMSE")
    axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(history["val_r2"], color="green")
    axes[1,0].set(xlabel="Epoch", ylabel="R²", title="Validation R²")
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(history["lr"], color="red")
    axes[1,1].set(xlabel="Epoch", ylabel="Learning Rate", title="LR Schedule", yscale="log")
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = model_out.parent / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"\nSaved training curves: {plot_path}")

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "train": {"rmse": tr_rmse, "mae": tr_mae, "r2": tr_r2, "pearson": tr_p},
        "val":   {"rmse": val_rmse, "mae": val_mae, "r2": val_r2, "pearson": val_p},
        "test":  {"rmse": te_rmse, "mae": te_mae, "r2": te_r2, "pearson": te_p},
        "model_params": {
            "hidden_dims": MLP_HIDDEN_DIMS, "dropout": MLP_DROPOUT,
            "n_params": n_params, "model_class": ModelClass.__name__,
        },
        "training": {
            "epochs_run": epoch + 1, "best_val_rmse": float(best_val_rmse),
        },
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics: {metrics_out}")

    log.info("\n✓ MLP meta-learner training complete.")
    return model


if __name__ == "__main__":
    train()
