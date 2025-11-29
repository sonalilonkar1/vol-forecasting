# src/backtest/run.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .engine import backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cost-aware inverse-vol backtest for a given prediction file."
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=None,
        help="Explicit prediction CSV. Overrides --horizon/--pred-dir if provided.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Horizon used to infer the prediction filename when --pred-path is omitted.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("experiments/preds"),
        help="Directory containing prediction CSVs (default: experiments/preds).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for backtest outputs (default: experiments/results).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/backtest.yaml"),
        help="YAML file with default target_vol, cost_bps, and no_trade_pp values.",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=None,
        help="Override target volatility (annualized). If omitted, uses config value.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="Override transaction cost in basis points per unit turnover.",
    )
    parser.add_argument(
        "--no-trade-pp",
        type=float,
        default=None,
        help="Override no-trade band width in percentage points.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def load_predictions(path: Path) -> pd.DataFrame:
    preds = pd.read_csv(path, parse_dates=["date"])
    if "asset" not in preds.columns and "ticker" in preds.columns:
        preds = preds.rename(columns={"ticker": "asset"})
    if "split" in preds.columns:
        before = len(preds)
        preds = preds[preds["split"] == "test"].copy()
        after = len(preds)
        print(f"Filtered predictions to test split: {before:,} -> {after:,} rows")
        preds = preds.drop(columns=["split"])
    missing = {"date", "asset", "yhat_rv"} - set(preds.columns)
    if missing:
        raise ValueError(f"Prediction file {path} missing columns: {', '.join(sorted(missing))}")
    return preds


def resolve_pred_path(args: argparse.Namespace) -> Path:
    if args.pred_path is not None:
        return args.pred_path
    return args.pred_dir / f"har_h{args.horizon}.csv"


def resolve_out_path(args: argparse.Namespace, pred_path: Path) -> Path:
    name = pred_path.stem + "_bt.csv"
    return args.out_dir / name


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    pred_path = resolve_pred_path(args)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    target_vol = args.target_vol if args.target_vol is not None else cfg.get("target_vol", 0.10)
    cost_bps = args.cost_bps if args.cost_bps is not None else cfg.get("cost_bps", 10)
    no_trade_pp = args.no_trade_pp if args.no_trade_pp is not None else cfg.get("no_trade_pp", 5)

    preds = load_predictions(pred_path)
    out = backtest(preds, target_vol=target_vol, cost_bps=cost_bps, no_trade_pp=no_trade_pp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_out_path(args, pred_path)
    out.to_csv(out_path, index=False)
    print(f"Saved backtest results to {out_path}")
    print(out.tail())
