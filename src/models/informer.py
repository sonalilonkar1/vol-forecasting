# src/models/informer_rv.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Dict, Tuple, Iterable

import math
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

DEFAULT_TFT_DATASET = Path("data/tft_ready_dataset.csv")
DEFAULT_TFT_PARTS = [
    Path("data/tft_ready_train.csv"),
    Path("data/tft_ready_val.csv"),
    Path("data/tft_ready_test.csv"),
]
DEFAULT_FEATURES = Path("data/processed/features.parquet")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CLI argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Informer-style deep sequence model for RV with YAML splits (HAR-compatible outputs)"
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "CSV/Parquet feature table. Defaults to TFT-ready files or "
            "data/processed/features.parquet"
        ),
    )
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1],
        help="Forecast horizons (e.g., 1 5 22). Default: [1]",
    )
    p.add_argument(
        "--eval-splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["val", "test"],
        help="Which splits to write predictions for. Default: val test",
    )
    p.add_argument(
        "--splits-config",
        type=Path,
        default=Path("configs/splits.yaml"),
        help="YAML file with train/val/test date ranges (same format as har_rv.py)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/preds"),
        help="Directory where prediction CSVs will be written",
    )

    # Sequence / training hyperparameters
    p.add_argument("--seq-len", type=int, default=64,
                   help="Sequence length (lookback window)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Mini-batch size")
    p.add_argument("--max-epochs", type=int, default=50,
                   help="Maximum training epochs")
    p.add_argument("--patience", type=int, default=8,
                   help="Early stopping patience (epochs w/o val RMSE improvement)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay (L2)")
    p.add_argument("--d-ff", type=int, default=256,
                   help="Transformer feed-forward hidden size")
    p.add_argument("--n-layers", type=int, default=3,
                   help="Number of encoder layers")
    p.add_argument("--n-heads", type=int, default=2,
                   help="Number of attention heads")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    return p.parse_args()


def resolve_source(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return [explicit]

    if DEFAULT_TFT_DATASET.exists():
        return [DEFAULT_TFT_DATASET]

    parts = [p for p in DEFAULT_TFT_PARTS if p.exists()]
    if parts:
        return parts

    if DEFAULT_FEATURES.exists():
        return [DEFAULT_FEATURES]

    raise FileNotFoundError(
        "No feature table found; provide --source or create TFT-ready/features.parquet"
    )


def load_feature_table(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            frames.append(pd.read_csv(p, parse_dates=["date"]))
        elif p.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(p))
        else:
            raise ValueError(f"Unsupported file type: {p}")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    if "asset" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "asset"})
        else:
            raise ValueError(
                "Need 'asset' or 'ticker' column in feature table")

    df = (
        df.sort_values(["asset", "date"])
          .drop_duplicates(["asset", "date"])
          .reset_index(drop=True)
    )
    return df


def load_splits_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_ranges(cfg: dict) -> Tuple[pd.Timestamp, ...]:
    if "train" in cfg and isinstance(cfg["train"], (list, tuple)):
        tr_s, tr_e = pd.to_datetime(
            cfg["train"][0]), pd.to_datetime(cfg["train"][1])
        va_s, va_e = pd.to_datetime(
            cfg["val"][0]), pd.to_datetime(cfg["val"][1])
        te_s, te_e = pd.to_datetime(
            cfg["test"][0]), pd.to_datetime(cfg["test"][1])
        emb = int(cfg.get("embargo_days", 0))
        return tr_s, tr_e, va_s, va_e, te_s, te_e, emb

    tr_s = pd.to_datetime(cfg["train_start"])
    tr_e = pd.to_datetime(cfg["train_end"])
    va_s = pd.to_datetime(cfg["val_start"])
    va_e = pd.to_datetime(cfg["val_end"])
    te_s = pd.to_datetime(cfg["test_start"])
    te_e = pd.to_datetime(cfg["test_end"])
    emb = int(cfg.get("embargo_days", 0))
    return tr_s, tr_e, va_s, va_e, te_s, te_e, emb


def compute_masks(dates: pd.Series, cfg: dict) -> Dict[str, pd.Series]:
    tr_s, tr_e, va_s, va_e, te_s, te_e, emb = _parse_ranges(cfg)
    dates = pd.to_datetime(dates)

    tr_raw = (dates >= tr_s) & (dates <= tr_e)
    tr_eff_end = dates[tr_raw].max(
    ) - pd.Timedelta(days=emb) if tr_raw.any() else tr_e
    if tr_eff_end < tr_s:
        tr_eff_end = tr_s

    m = {
        "train": (dates >= tr_s) & (dates <= tr_eff_end),
        "val":   (dates >= va_s) & (dates <= va_e),
        "test":  (dates >= te_s) & (dates <= te_e),
    }
    return {k: v.fillna(False) for k, v in m.items()}


def build_informer_table(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str], str]:
    """
    Prepare a table for sequence modeling at a given horizon H.
    Uses TFT-ready targets: target_logvol_t+H.
    """
    target_col = f"target_logvol_t+{horizon}"
    if target_col not in df.columns:
        raise ValueError(
            f"Missing target column {target_col} in feature table")

    tab = df.copy()

    # ID columns and all targets to exclude from features
    exclude = {"date", "asset"}
    for h in (1, 5, 22):
        col = f"target_logvol_t+{h}"
        if col in tab.columns:
            exclude.add(col)

    feature_cols = [c for c in tab.columns if c not in exclude]

    # Drop rows without target
    tab = tab.dropna(subset=[target_col]).reset_index(drop=True)

    return tab, feature_cols, target_col


def _model_label() -> str:
    return "Informer"


class PanelSequenceDataset(Dataset):
    """
    Creates sequences of length seq_len within each asset, restricted to a given mask.
    Each item yields (X_seq, y_last, idx_last).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        mask: pd.Series,
        seq_len: int = 64,
        id_col: str = "asset",
    ):
        self.df = df
        self.X = X
        self.y = y
        self.seq_len = seq_len

        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != len(df):
            raise ValueError("Mask length does not match DataFrame length")

        self.indices: list[np.ndarray] = []

        # Build sequences asset-by-asset
        for _, g in df.groupby(id_col):
            idx_all = g.index.to_numpy()
            # Keep only indices with mask=True
            idx = idx_all[mask_arr[idx_all]]
            if len(idx) <= seq_len:
                continue
            for i in range(len(idx) - seq_len + 1):
                seq_idx = idx[i: i + seq_len]
                self.indices.append(seq_idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        seq_idx = self.indices[i]
        x_seq = self.X[seq_idx]                       # [seq_len, d_model]
        y_last = self.y[seq_idx[-1]]                  # scalar (log-vol target)
        idx_last = seq_idx[-1]                        # index in df/tab
        x_t = torch.from_numpy(x_seq)                 # float32
        y_t = torch.tensor(y_last, dtype=torch.float32)
        return x_t, y_t, idx_last


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SimpleInformer(nn.Module):
    """
    Simplified Informer-style encoder:
    - Linear input projection
    - Positional encoding
    - Stack of TransformerEncoderLayers
    - MLP head on the last time step
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        n_heads: int = 2,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        last = x[:, -1, :]          # last time step
        out = self.head(last)       # [B, out_dim]
        return out.squeeze(-1)      # [B]

# Metrics (RMSE & QLIKE)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def qlike_from_logvol(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    QLIKE on variance using log-volatility.
    v_true = exp(2 * y_true), v_pred = exp(2 * y_pred)
    L = log(v_pred) + v_true / v_pred
    """
    v_true = np.exp(np.clip(2.0 * y_true_log, -50, 50))
    v_pred = np.exp(np.clip(2.0 * y_pred_log, -50, 50))
    return float(np.mean(np.log(v_pred) + v_true / v_pred))


# Training / evaluation helpers
def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_informer_for_horizon(
    tab: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    masks: Dict[str, pd.Series],
    horizon: int,
    seq_len: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    d_ff: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    seed: int,
):
    idx = tab.index
    base_mask = tab[feature_cols].notna().all(axis=1) & tab[target_col].notna()

    train_mask = masks["train"].reindex(idx).fillna(False) & base_mask
    val_mask = masks["val"].reindex(idx).fillna(False) & base_mask

    if not train_mask.any():
        raise ValueError(f"No training rows found for horizon {horizon}")

    # Standardize features on train only
    scaler = StandardScaler()
    scaler.fit(tab.loc[train_mask, feature_cols].values)
    X_all = scaler.transform(tab[feature_cols].values).astype("float32")
    y_all = tab[target_col].values.astype("float32")

    # Datasets
    train_ds = PanelSequenceDataset(
        tab, X_all, y_all, mask=train_mask, seq_len=seq_len)
    val_ds = PanelSequenceDataset(
        tab, X_all, y_all, mask=val_mask,   seq_len=seq_len)

    if len(train_ds) == 0:
        raise ValueError(
            f"Train dataset empty for horizon {horizon} (seq_len too long?)")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, drop_last=False) if len(val_ds) > 0 else None

    # Model
    d_model = len(feature_cols)
    model = SimpleInformer(
        d_model=d_model,
        out_dim=1,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val_rmse = float("inf")
    patience_left = patience

    def eval_loader(loader: DataLoader) -> tuple[float, float]:
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                all_y.append(yb.cpu().numpy())
                all_p.append(pred.cpu().numpy())
        if not all_y:
            return float("nan"), float("nan")
        y = np.concatenate(all_y, axis=0).astype("float32")
        p = np.concatenate(all_p, axis=0).astype("float32")
        return rmse(y, p), qlike_from_logvol(y, p)

    # Training loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            n_samples += xb.size(0)

        train_loss = running_loss / max(n_samples, 1)

        if val_loader is not None:
            val_rmse, val_qlike = eval_loader(val_loader)
        else:
            # Fallback: use train metrics if no val split
            val_rmse, val_qlike = train_loss**0.5, float("nan")

        print(
            f"[H={horizon}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_RMSE={val_rmse:.4f} | val_QLIKE={val_qlike:.4f}"
        )

        if val_loader is not None:
            improved = val_rmse < best_val_rmse - 1e-5
        else:
            improved = train_loss < best_val_rmse

        if improved:
            best_val_rmse = val_rmse if val_loader is not None else train_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[H={horizon}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler


def collect_predictions_for_split(
    model: nn.Module,
    scaler: StandardScaler,
    tab: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    mask: pd.Series,
    seq_len: int,
    split_name: str,
) -> pd.DataFrame:
    idx = tab.index
    base_mask = tab[feature_cols].notna().all(axis=1) & tab[target_col].notna()
    use_mask = mask.reindex(idx).fillna(False) & base_mask

    ds = PanelSequenceDataset(tab, scaler.transform(tab[feature_cols].values).astype("float32"),
                              tab[target_col].values.astype("float32"),
                              mask=use_mask, seq_len=seq_len)
    if len(ds) == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "asset",
                "y_true_logrv",
                "y_true_rv",
                "yhat_logrv",
                "yhat_rv",
                "model",
                "horizon",
                "split",
            ]
        )

    loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=False)

    model.eval()
    all_idx_last = []
    all_y = []
    all_p = []
    with torch.no_grad():
        for xb, yb, idx_last in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            all_y.append(yb.cpu().numpy())
            all_p.append(pred.cpu().numpy())
            all_idx_last.append(idx_last.numpy())

    y = np.concatenate(all_y, axis=0).astype("float32")
    p = np.concatenate(all_p, axis=0).astype("float32")
    idx_last = np.concatenate(all_idx_last, axis=0).astype("int64")

    y_true_logrv = y
    yhat_logrv = p
    y_true_rv = np.exp(np.clip(y_true_logrv, -50, 50))
    yhat_rv = np.exp(np.clip(yhat_logrv,  -50, 50))

    df_part = pd.DataFrame(
        {
            "date": tab.loc[idx_last, "date"].values,
            "asset": tab.loc[idx_last, "asset"].values,
            "y_true_logrv": y_true_logrv,
            "y_true_rv": y_true_rv,
            "yhat_logrv": yhat_logrv,
            "yhat_rv": yhat_rv,
            "split": split_name,
        }
    ).sort_values(["date", "asset"]).reset_index(drop=True)

    return df_part


def main():
    args = parse_args()
    set_seeds(args.seed)

    df = load_feature_table(resolve_source(args.source))
    cfg = load_splits_config(args.splits_config)

    # Optional asset filtering from YAML (same as har_rv.py)
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Masks on the *full* table; per-horizon we reindex on the subset
    full_masks = compute_masks(df["date"], cfg)

    for H in args.horizons:
        print(f"\n=== Training Informer model for horizon H={H} ===")
        tab, feature_cols, target_col = build_informer_table(df, horizon=H)

        # Reindex masks on the horizon-specific table
        masks = {k: full_masks[k].reindex(tab.index).fillna(
            False) for k in full_masks.keys()}

        model, scaler = train_informer_for_horizon(
            tab=tab,
            feature_cols=feature_cols,
            target_col=target_col,
            masks=masks,
            horizon=H,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            seed=args.seed,
        )

        # Collect predictions for requested splits
        parts = []
        for sp in args.eval_splits:
            print(f"[H={H}] Collecting predictions for split='{sp}'")
            part = collect_predictions_for_split(
                model=model,
                scaler=scaler,
                tab=tab,
                feature_cols=feature_cols,
                target_col=target_col,
                mask=masks[sp],
                seq_len=args.seq_len,
                split_name=sp,
            )
            parts.append(part)

        if not parts:
            raise ValueError(f"No predictions produced for horizon {H}")

        preds = pd.concat(parts, ignore_index=True)
        preds["model"] = _model_label()
        preds["horizon"] = H

        # Reorder columns to exactly match har_h1.csv layout
        preds = preds[
            [
                "date",
                "asset",
                "y_true_logrv",
                "y_true_rv",
                "yhat_logrv",
                "yhat_rv",
                "model",
                "horizon",
                "split",
            ]
        ]

        out = args.out_dir / f"informer_h{H}.csv"
        preds.to_csv(out, index=False)
        counts = preds.groupby("split").size().to_dict()
        print(f"[H={H}] wrote {len(preds):,} rows to {out} {counts}")


if __name__ == "__main__":
    main()
