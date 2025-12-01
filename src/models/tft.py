# src/models/tft.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common.har_dataset import build_multi_har_table, load_base_panel
from .common.train_loop import set_seed
from .har_rv import compute_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal Fusion Transformer baseline (PyTorch Lightning) producing HAR-compatible outputs.",
    )
    parser.add_argument("--source", type=Path, default=None, help="Feature table (defaults to TFT-ready files).")
    parser.add_argument("--splits-config", type=Path, default=Path("configs/splits.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/preds"))
    parser.add_argument(
        "--combined-out",
        type=Path,
        default=None,
        help="Optional CSV path to store a single multi-horizon file. Defaults to <out-dir>/tft_all_horizons.csv.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    parser.add_argument("--eval-splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--harx", action="store_true", help="Include HARX-style exogenous features if available.")
    parser.add_argument("--min-train", type=int, default=1000, help="Minimum multi-horizon sequences required for training.")
    parser.add_argument("--lookback", type=int, default=60, help="Encoder sequence length (days).")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--loss", choices=["mse", "qlike"], default="mse")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0 style string")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


class TFTDataset(Dataset):
    """Sliding-window dataset emitting encoder/decoder sequences plus multi-horizon targets."""

    def __init__(
        self,
        table: pd.DataFrame,
        features: List[str],
        horizons: List[int],
        mask: pd.Series,
        lookback: int,
        asset_to_id: Dict[str, int],
    ) -> None:
        self.features = features
        self.horizons = horizons
        self.lookback = lookback
        self.asset_to_id = asset_to_id
        self.max_h = max(horizons)
        self.decoder_template = torch.tensor(
            [[float(h), float(h) / float(self.max_h)] for h in horizons], dtype=torch.float32
        )

        self.encoder_data: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.static_ids: List[int] = []
        self.meta_rows: List[dict] = []

        mask = mask.reindex(table.index).fillna(False)
        grouped = table.groupby("asset", sort=False)
        for asset, group in grouped:
            asset_id = asset_to_id[asset]
            idx = group.index.to_numpy()
            asset_mask = mask.loc[idx].to_numpy(dtype=bool)
            feat_vals = group[self.features].to_numpy(dtype=np.float32)
            target_arrays = {
                h: group[f"y_true_logrv_h{h}"].to_numpy(dtype=np.float32)
                for h in horizons
            }
            target_rv = {
                h: group[f"y_true_rv_h{h}"].to_numpy(dtype=np.float32)
                for h in horizons
            }
            dates = group["date"].to_numpy()
            for pos in range(len(group)):
                if not asset_mask[pos]:
                    continue
                start = pos - lookback + 1
                if start < 0:
                    continue
                window = feat_vals[start : pos + 1]
                if window.shape[0] != lookback:
                    continue
                target_vec = np.array([target_arrays[h][pos] for h in horizons], dtype=np.float32)
                if np.any(np.isnan(target_vec)):
                    continue
                self.encoder_data.append(window)
                self.targets.append(target_vec)
                self.static_ids.append(asset_id)
                meta = {
                    "date": dates[pos],
                    "asset": asset,
                }
                for h in horizons:
                    meta[f"y_true_logrv_h{h}"] = target_arrays[h][pos]
                    meta[f"y_true_rv_h{h}"] = target_rv[h][pos]
                self.meta_rows.append(meta)

    def __len__(self) -> int:
        return len(self.encoder_data)

    def __getitem__(self, idx: int) -> dict:
        return {
            "encoder_cont": torch.from_numpy(self.encoder_data[idx]),
            "decoder_cont": self.decoder_template.clone(),
            "static_id": torch.tensor(self.static_ids[idx], dtype=torch.long),
            "target": torch.from_numpy(self.targets[idx]),
        }

    @property
    def meta(self) -> pd.DataFrame:
        if not self.meta_rows:
            return pd.DataFrame(columns=["date", "asset"])
        return pd.DataFrame(self.meta_rows)


class SimpleTFT(pl.LightningModule):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        static_cardinality: int,
        hidden_dim: int,
        lstm_layers: int,
        dropout: float,
        num_heads: int,
        horizons: List[int],
        learning_rate: float,
        weight_decay: float,
        loss: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.horizons = horizons
        self.hidden_dim = hidden_dim
        self.loss_type = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.static_emb = nn.Embedding(static_cardinality, hidden_dim)
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim)

        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        static_ctx = self.static_emb(batch["static_id"])  # (B, hidden)

        enc = self.encoder_proj(batch["encoder_cont"]) + static_ctx.unsqueeze(1)
        enc_out, _ = self.encoder_lstm(enc)

        dec = self.decoder_proj(batch["decoder_cont"]) + static_ctx.unsqueeze(1)
        dec_out, _ = self.decoder_lstm(dec)

        attn_out, _ = self.attention(dec_out, enc_out, enc_out)
        fused = dec_out + attn_out
        preds = self.output_proj(fused).squeeze(-1)  # (B, len_h)
        return preds

    def _loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return nn.functional.mse_loss(preds, target)
        y_true = torch.exp(target)
        y_hat = torch.exp(preds)
        ratio = y_true / (y_hat + 1e-12)
        return torch.mean(ratio - torch.log(ratio + 1e-12) - 1.0)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self.forward(batch)
        loss = self._loss(preds, batch["target"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self.forward(batch)
        loss = self._loss(preds, batch["target"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        preds = self.forward(batch)
        return preds.detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def make_dataloader(dataset: TFTDataset | None, batch_size: int, shuffle: bool) -> DataLoader | None:
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def infer_device(requested: str) -> tuple[str, int]:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    accelerator, devices = infer_device(args.device)

    df, cfg, logcol = load_base_panel(args.source, args.splits_config)
    table, feats = build_multi_har_table(df, logcol, args.horizons, use_harx=args.harx)
    masks = compute_masks(table["date"], cfg)

    asset_to_id = {asset: idx for idx, asset in enumerate(sorted(table["asset"].unique()))}
    horizons = sorted({int(h) for h in args.horizons})

    datasets: Dict[str, TFTDataset] = {}
    for split in ["train", "val", "test"]:
        mask = masks.get(split)
        if mask is None:
            continue
        ds = TFTDataset(table, feats, horizons, mask, args.lookback, asset_to_id)
        datasets[split] = ds

    train_ds = datasets.get("train")
    if train_ds is None or len(train_ds) < args.min_train:
        have = 0 if train_ds is None else len(train_ds)
        raise ValueError(f"Not enough training sequences ({have}) for TFT. Reduce --lookback or --min-train.")
    val_ds = datasets.get("val")

    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, args.batch_size, shuffle=False)

    module = SimpleTFT(
        encoder_dim=len(feats),
        decoder_dim=train_ds.decoder_template.shape[-1],
        static_cardinality=len(asset_to_id),
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        horizons=horizons,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=args.grad_clip,
        deterministic=True,
    )

    trainer.fit(module, train_loader, val_loader)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for split in args.eval_splits:
        ds = datasets.get(split)
        if ds is None or len(ds) == 0:
            continue
        loader = make_dataloader(ds, args.batch_size, shuffle=False)
        preds_batches = trainer.predict(module, loader)
        if not preds_batches:
            continue
        preds = torch.cat(preds_batches, dim=0).numpy()
        meta = ds.meta.reset_index(drop=True)
        for idx, horizon in enumerate(horizons):
            part = pd.DataFrame(
                {
                    "date": meta["date"],
                    "asset": meta["asset"],
                    "y_true_logrv": meta[f"y_true_logrv_h{horizon}"],
                    "y_true_rv": meta[f"y_true_rv_h{horizon}"],
                    "yhat_logrv": preds[:, idx],
                }
            )
            part["yhat_rv"] = np.exp(np.clip(part["yhat_logrv"], -50, 50))
            part["model"] = f"TFT({'HARX' if args.harx else 'HAR'})"
            part["horizon"] = horizon
            part["split"] = split
            outputs.append(part)

    if not outputs:
        raise ValueError("No evaluation rows produced for requested splits.")

    pred_df = pd.concat(outputs, ignore_index=True)
    pred_df = pred_df.sort_values(["horizon", "date", "asset"]).reset_index(drop=True)
    for horizon in horizons:
        out_path = args.out_dir / f"tft_h{horizon}.csv"
        subset = pred_df[pred_df["horizon"] == horizon]
        if subset.empty:
            print(f"[tft] WARNING: no rows for horizon {horizon}; skipping export to {out_path}")
            continue
        subset.to_csv(out_path, index=False)
        counts = subset.groupby("split").size().to_dict()
        print(f"[tft] H={horizon} wrote {len(subset):,} rows to {out_path} {counts}")

    combined_path = args.combined_out if args.combined_out is not None else args.out_dir / "tft_all_horizons.csv"
    pred_df.to_csv(combined_path, index=False)
    print(f"[tft] Wrote all horizons ({len(pred_df):,} rows) to {combined_path}")


if __name__ == "__main__":
    main()
