# src/models/simple_mlp.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common.har_dataset import (
    load_base_panel,
    prepare_har_dataset,
    feature_tensor,
    target_tensor,
    meta_from_mask,
)
from .common.train_loop import TrainConfig, predict, set_seed, train_regressor


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
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    df, cfg, logcol = load_base_panel(args.source, args.splits_config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        dataset = prepare_har_dataset(
            df,
            logcol,
            cfg,
            horizon=horizon,
            use_harx=args.harx,
        )
        train_mask = dataset.masks["train"]
        if train_mask.sum() < args.min_train:
            raise ValueError(
                f"Not enough training rows for horizon {horizon}. Have {int(train_mask.sum())}, need {args.min_train}."
            )

        X_train = feature_tensor(dataset, train_mask, device)
        y_train = target_tensor(dataset, train_mask, device)

        model = MLP(X_train.shape[1], args.hidden_dim, args.num_layers, args.dropout).to(device)
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        model = train_regressor(model, X_train, y_train, config)

        outputs = []
        model_name = f"MLP({'HARX' if args.harx else 'HAR'})"
        for split in args.eval_splits:
            mask = dataset.masks[split]
            if not mask.any():
                continue
            X_split = feature_tensor(dataset, mask, device)
            preds = predict(model, X_split)
            meta = meta_from_mask(dataset, mask)
            part = pd.DataFrame(
                {
                    "date": meta["date"].values,
                    "asset": meta["asset"].values,
                    "y_true_logrv": meta["y_true_logrv"].values,
                    "y_true_rv": meta["y_true_rv"].values,
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
