
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .har_rv import (
    resolve_source,
    load_feature_table,
    load_splits_config,
    compute_masks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("N-BEATS model for volatility forecasting")
    p.add_argument("--source", type=Path, default=None,
                   help="CSV/Parquet feature table. Defaults to TFT-ready files.")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22],
                   help="Forecast horizons (e.g., 1 5 22)")
    p.add_argument("--eval-splits", nargs="+", choices=["train", "val", "test"],
                   default=["val", "test"])
    p.add_argument("--splits-config", type=Path, default=Path("configs/splits.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/preds"),
                   help="Directory for prediction CSVs (default: experiments/preds)")
    p.add_argument("--file-prefix", type=str, default="nbeats",
                   help="Filename prefix so outputs look like <prefix>_h*.csv")
    p.add_argument("--lookback", type=int, default=90,
                   help="Lookback window size L (60, 90, or 180)")
    p.add_argument("--num-stacks", type=int, default=2,
                   help="Number of stacks in N-BEATS")
    p.add_argument("--num-blocks", type=int, default=3,
                   help="Number of blocks per stack")
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="Hidden layer dimension")
    p.add_argument("--theta-dim", type=int, default=8,
                   help="Dimension of theta (basis coefficients)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train", type=int, default=500,
                   help="Minimum training samples required")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NBeatsBlock(nn.Module):
    """
    Basic N-BEATS block with generic basis (fully connected layers).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        theta_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Fully connected stack
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_dim, theta_dim)
        self.theta_f = nn.Linear(hidden_dim, theta_dim)
        
        # Basis layers (generic basis - learned)
        self.backcast_basis = nn.Linear(theta_dim, input_dim)
        self.forecast_basis = nn.Linear(theta_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, input_dim)
        h = self.fc(x)
        
        # Backcast and forecast coefficients
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        
        # Generate backcast and forecast
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        
        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    N-BEATS stack containing multiple blocks.
    """
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        theta_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, hidden_dim, theta_dim, output_dim, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        stack_forecast = torch.zeros(x.size(0), self.blocks[0].output_dim, device=x.device)
        
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast  # Residual connection
            stack_forecast = stack_forecast + forecast
        
        return residual, stack_forecast


class NBeats(nn.Module):
    """
    N-BEATS model with multiple stacks.
    Generic architecture.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        num_stacks: int = 2,
        num_blocks: int = 3,
        hidden_dim: int = 64,
        theta_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stacks = nn.ModuleList([
            NBeatsStack(num_blocks, input_dim, hidden_dim, theta_dim, output_dim, dropout)
            for _ in range(num_stacks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        residual = x
        forecast = torch.zeros(x.size(0), 1, device=x.device)
        
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            forecast = forecast + stack_forecast
        
        return forecast.squeeze(-1)


def build_sequences(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build sliding window sequences for N-BEATS.
    Returns metadata DataFrame, input sequences X, and targets y.
    """
    target_col = f"target_logvol_t+{horizon}"
    
    # Check for required columns
    if "logvol_t" not in df.columns:
        raise ValueError("Need 'logvol_t' column for N-BEATS sequences")
    
    all_X = []
    all_y = []
    all_meta = []
    
    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("date").reset_index(drop=True)
        logvol = grp["logvol_t"].values
        dates = grp["date"].values
        
        # Get targets
        if target_col in grp.columns:
            targets = grp[target_col].values
        else:
            # Shift to create target
            targets = np.roll(logvol, -horizon)
            targets[-horizon:] = np.nan
        
        # Create sequences
        for i in range(lookback, len(grp)):
            if np.isnan(targets[i]):
                continue
            
            # Input: past lookback values
            x_seq = logvol[i - lookback:i]
            
            # Skip if any NaN in sequence
            if np.any(np.isnan(x_seq)):
                continue
            
            all_X.append(x_seq)
            all_y.append(targets[i])
            all_meta.append({
                "date": dates[i],
                "asset": asset,
                "idx": i,
            })
    
    if not all_X:
        raise ValueError("No valid sequences created. Check lookback and data.")
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    meta_df = pd.DataFrame(all_meta)
    
    return meta_df, X, y


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Load data
    df = load_feature_table(resolve_source(args.source))
    cfg = load_splits_config(args.splits_config)
    
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    for horizon in args.horizons:
        print(f"Training N-BEATS for horizon H={horizon}, lookback={args.lookback}...")
        
        # Build sequences
        meta_df, X, y = build_sequences(df, args.lookback, horizon)
        
        # Compute split masks on metadata
        masks = compute_masks(meta_df["date"], cfg)
        
        # Get train indices
        train_mask = masks["train"].values
        if train_mask.sum() < args.min_train:
            raise ValueError(
                f"Not enough training samples for horizon {horizon}. "
                f"Have {train_mask.sum()}, need {args.min_train}."
            )
        
        # Normalize using training statistics
        X_train_raw = X[train_mask]
        train_mean = X_train_raw.mean()
        train_std = X_train_raw.std() + 1e-8
        
        X_norm = (X - train_mean) / train_std
        y_mean = y[train_mask].mean()
        y_std = y[train_mask].std() + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Create tensors
        X_train = torch.from_numpy(X_norm[train_mask]).to(device)
        y_train = torch.from_numpy(y_norm[train_mask]).to(device)
        
        # Create model
        model = NBeats(
            input_dim=args.lookback,
            output_dim=1,
            num_stacks=args.num_stacks,
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            theta_dim=args.theta_dim,
            dropout=args.dropout,
        ).to(device)
        
        # Training
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(X_train)
                print(f"  Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # Prediction
        model.eval()
        outputs = []
        
        for split in args.eval_splits:
            split_mask = masks[split].values
            if not split_mask.any():
                continue
            
            X_split = torch.from_numpy(X_norm[split_mask]).to(device)
            
            with torch.no_grad():
                preds_norm = model(X_split).cpu().numpy()
            
            # Denormalize predictions
            preds = preds_norm * y_std + y_mean
            
            # Get true values
            y_true = y[split_mask]
            
            part = pd.DataFrame({
                "date": meta_df.loc[split_mask, "date"].values,
                "asset": meta_df.loc[split_mask, "asset"].values,
                "y_true_logrv": y_true,
                "y_true_rv": np.exp(np.clip(y_true, -50, 50)),
                "yhat_logrv": preds,
                "yhat_rv": np.exp(np.clip(preds, -50, 50)),
                "model": f"N-BEATS(L={args.lookback})",
                "horizon": horizon,
                "split": split,
            })
            outputs.append(part)
        
        if not outputs:
            raise ValueError(f"No predictions produced for horizon {horizon}.")
        
        pred_df = pd.concat(outputs, ignore_index=True)
        pred_df = pred_df.sort_values(["date", "asset"]).reset_index(drop=True)
        
        out_path = args.out_dir / f"{args.file_prefix}_h{horizon}.csv"
        pred_df.to_csv(out_path, index=False)
        counts = pred_df.groupby("split").size().to_dict()
        print(f"[nbeats] H={horizon}: wrote {len(pred_df):,} rows to {out_path} {counts}")


if __name__ == "__main__":
    main()

