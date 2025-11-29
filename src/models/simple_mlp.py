# src/models/simple_mlp.py
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .har_rv import (
    resolve_source,
    load_feature_table,
    pick_logvol_col,
    load_splits_config,
    compute_masks,
    build_har_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple feed-forward neural net baseline that shares the HAR prediction schema."
    )
    parser.add_argument("--source", type=Path, default=None, help="Feature table (defaults to TFT-ready files).")
    parser.add_argument("--splits-config", type=Path, default=Path("configs/splits.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/preds"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    parser.add_argument("--eval-splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--harx", action="store_true", help="Use HARX-style exogenous features if present.")
    parser.add_argument("--min-train", type=int, default=500, help="Minimum training rows required.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        dims = [input_dim] + [hidden_dim] * max(1, num_layers)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(nn.Linear(in_dim, out_dim))
            blocks.append(nn.ReLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.head(h).squeeze(-1)


def make_tensor(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    arr = frame.to_numpy(dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def fit_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> MLP:
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = MLP(X_train.shape[1], hidden_dim, num_layers, dropout).to(X_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
    return model


def predict(model: MLP, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(X).detach().cpu().numpy()
    return preds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    df = load_feature_table(resolve_source(args.source))
    logcol = pick_logvol_col(df)
    cfg = load_splits_config(args.splits_config)
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        tab, feats = build_har_table(
            df,
            horizon=horizon,
            logcol=logcol,
            target_mode="point",
            use_harx=args.harx,
        )
        masks = compute_masks(tab["date"], cfg)
        train_mask = masks["train"]
        if train_mask.sum() < args.min_train:
            raise ValueError(
                f"Not enough training rows for horizon {horizon}. Have {int(train_mask.sum())}, need {args.min_train}."
            )

        X_train = make_tensor(tab.loc[train_mask, feats], device)
        y_train = torch.from_numpy(tab.loc[train_mask, "y_true_logrv"].to_numpy(dtype=np.float32)).to(device)

        model = fit_model(
            X_train,
            y_train,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        outputs = []
        model_name = f"MLP({'HARX' if args.harx else 'HAR'})"
        for split in args.eval_splits:
            mask = masks[split]
            if not mask.any():
                continue
            X_split = make_tensor(tab.loc[mask, feats], device)
            preds = predict(model, X_split)
            part = pd.DataFrame(
                {
                    "date": tab.loc[mask, "date"].values,
                    "asset": tab.loc[mask, "asset"].values,
                    "y_true_logrv": tab.loc[mask, "y_true_logrv"].values,
                    "y_true_rv": tab.loc[mask, "y_true_rv"].values,
                    "yhat_logrv": preds,
                }
            )
            part["yhat_rv"] = np.exp(np.clip(part["yhat_logrv"], -50, 50))
            part["model"] = model_name
            part["horizon"] = horizon
            part["split"] = split
            outputs.append(part)

        if not outputs:
            raise ValueError(f"No evaluation rows produced for horizon {horizon}.")

        pred_df = pd.concat(outputs, ignore_index=True)
        pred_df = pred_df.sort_values(["date", "asset"]).reset_index(drop=True)
        out_path = args.out_dir / f"mlp_h{horizon}.csv"
        pred_df.to_csv(out_path, index=False)
        counts = pred_df.groupby("split").size().to_dict()
        print(f"[mlp] H={horizon} wrote {len(pred_df):,} rows to {out_path} {counts}")


if __name__ == "__main__":
    main()
