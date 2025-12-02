# src/models/simple_rnn.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common.har_dataset import build_sequence_batch, load_base_panel, prepare_har_dataset
from .common.train_loop import TrainConfig, predict, set_seed, train_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequence model (GRU/LSTM) baseline that consumes HAR tables via sliding windows."
    )
    parser.add_argument("--source", type=Path, default=None, help="Feature table (defaults to TFT-ready files).")
    parser.add_argument("--splits-config", type=Path, default=Path("configs/splits.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/preds"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    parser.add_argument("--eval-splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--harx", action="store_true", help="Use HARX-style exogenous features if present.")
    parser.add_argument("--min-train", type=int, default=500, help="Minimum training sequences required.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0, help="Applied between RNN layers (>=2 layers).")
    parser.add_argument("--lookback", type=int, default=30, help="Rolling window length (in days) per sequence.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--cell-type",
        choices=["gru", "lstm"],
        default="gru",
        help="Recurrent cell type to use for the sequence encoder.",
    )
    return parser.parse_args()


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if cell_type == "gru" else nn.LSTM
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.encoder(x)
        last = seq_out[:, -1, :]
        return self.head(last).squeeze(-1)


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
        train_batch = build_sequence_batch(dataset, dataset.masks["train"], args.lookback, device)
        train_count = len(train_batch.meta)
        if train_count < args.min_train:
            raise ValueError(
                f"Not enough training sequences for horizon {horizon}. Have {train_count}, need {args.min_train}."
            )

        input_dim = train_batch.features.shape[-1]
        model = SequenceRegressor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            cell_type=args.cell_type,
        ).to(device)
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        model = train_regressor(model, train_batch.features, train_batch.targets, config)

        outputs = []
        model_label = f"{args.cell_type.upper()}({'HARX' if args.harx else 'HAR'})"
        for split in args.eval_splits:
            batch = build_sequence_batch(
                dataset,
                dataset.masks[split],
                args.lookback,
                device,
                allow_empty=True,
            )
            if batch is None or batch.meta.empty:
                continue
            preds = predict(model, batch.features)
            part = batch.meta.copy()
            part["yhat_logrv"] = preds
            part["yhat_rv"] = np.exp(np.clip(part["yhat_logrv"], -50, 50))
            part["model"] = model_label
            part["horizon"] = horizon
            part["split"] = split
            outputs.append(part)

        if not outputs:
            raise ValueError(f"No evaluation rows produced for horizon {horizon} (split masks may be empty).")

        pred_df = pd.concat(outputs, ignore_index=True)
        pred_df = pred_df.sort_values(["date", "asset"]).reset_index(drop=True)
        out_path = args.out_dir / f"{args.cell_type}_h{horizon}.csv"
        pred_df.to_csv(out_path, index=False)
        counts = pred_df.groupby("split").size().to_dict()
        print(f"[{args.cell_type}] H={horizon} wrote {len(pred_df):,} rows to {out_path} {counts}")


if __name__ == "__main__":
    main()
