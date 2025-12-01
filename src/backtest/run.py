# src/backtest/run.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

try:  # Support both `python -m` and direct script execution
    from .engine import backtest
except ImportError:  # pragma: no cover - fallback only when run as script
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.backtest.engine import backtest


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
    parser.add_argument(
        "--weight-cap",
        type=float,
        default=None,
        help="Override absolute per-asset weight cap (default sourced from config).",
    )
    parser.add_argument(
        "--allocator",
        choices=["inverse_vol", "risk_parity"],
        default=None,
        help="Select portfolio construction rule (default from config, inverse_vol if unset).",
    )
    parser.add_argument(
        "--risk-parity-window",
        type=int,
        default=None,
        help="Rolling window (days) for risk parity covariance estimation.",
    )
    parser.add_argument(
        "--risk-parity-min-obs",
        type=int,
        default=None,
        help="Minimum historical observations before enabling risk parity weighting.",
    )
    parser.add_argument(
        "--rebalance-fraction",
        type=float,
        default=None,
        help="Fraction of the gap to close each rebalance (0-1). 1.0 = full move.",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Maximum allowed daily turnover (0-1). Default: unlimited.",
    )
    parser.add_argument(
        "--cov-shrink",
        type=float,
        default=None,
        help="Diagonal shrinkage added to rolling covariances for numerical stability.",
    )
    parser.add_argument(
        "--returns-path",
        type=Path,
        default=None,
        help="CSV with realized returns (default: configs/backtest.yaml or data/processed/returns.csv).",
    )
    parser.add_argument(
        "--return-col",
        type=str,
        default=None,
        help="Column name containing one-day returns (defaults to ret_1d, then ret).",
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


def load_returns(path: Path, preferred_col: str | None = None) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {path}. Generate one via 'python -m src.data.build_returns'"
        )
    returns = pd.read_csv(path, parse_dates=["date"])
    if "asset" not in returns.columns and "ticker" in returns.columns:
        returns = returns.rename(columns={"ticker": "asset"})

    requested = preferred_col or "ret_1d"
    candidates = [c for c in [requested, "ret", "ret_1d"] if c and c in returns.columns]
    if not candidates:
        raise ValueError(
            f"Return column '{requested}' not found in {path}."
            " Available columns: " + ", ".join(sorted(returns.columns))
        )
    actual_col = candidates[0]
    if actual_col != requested:
        print(f"Using return column '{actual_col}' instead of requested '{requested}'.")
    required = {"date", "asset", actual_col}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"Returns file missing columns: {', '.join(sorted(missing))}")
    returns = returns[["date", "asset", actual_col]].copy()
    return returns, actual_col


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
    weight_cap = args.weight_cap if args.weight_cap is not None else cfg.get("weight_cap", 1.0)
    default_returns = cfg.get("returns_path", "data/processed/returns.csv")
    returns_path = args.returns_path if args.returns_path is not None else Path(default_returns)
    return_col = args.return_col if args.return_col is not None else cfg.get("returns_col")
    allocator = args.allocator if args.allocator is not None else cfg.get("allocator", "inverse_vol")
    rp_window = (
        args.risk_parity_window if args.risk_parity_window is not None else cfg.get("risk_parity_window", 60)
    )
    rp_min = (
        args.risk_parity_min_obs
        if args.risk_parity_min_obs is not None
        else cfg.get("risk_parity_min_obs", 20)
    )
    rebalance_fraction = (
        args.rebalance_fraction
        if args.rebalance_fraction is not None
        else cfg.get("rebalance_fraction", 1.0)
    )
    max_turnover = args.max_turnover if args.max_turnover is not None else cfg.get("max_turnover")
    cov_shrink = args.cov_shrink if args.cov_shrink is not None else cfg.get("cov_shrink", 1e-6)

    preds = load_predictions(pred_path)
    returns_df, actual_return_col = load_returns(returns_path, preferred_col=return_col)
    out = backtest(
        preds,
        returns_df,
        target_vol=target_vol,
        cost_bps=cost_bps,
        no_trade_pp=no_trade_pp,
        return_col=actual_return_col,
        weight_cap=weight_cap,
        allocator=allocator,
        risk_parity_window=rp_window,
        risk_parity_min_obs=rp_min,
        rebalance_fraction=rebalance_fraction,
        max_turnover=max_turnover,
        cov_shrink=cov_shrink,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_out_path(args, pred_path)
    out.to_csv(out_path, index=False)
    print(f"Saved backtest results to {out_path}")
    print(out.tail())
