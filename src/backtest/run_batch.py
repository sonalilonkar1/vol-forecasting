# src/backtest/run_batch.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import pandas as pd

from .engine import backtest
from .run import load_returns, load_config

DEFAULT_PRED_DIR = Path("experiments/preds")
DEFAULT_RES_DIR = Path("experiments/results")
CONFIG_PATH = Path("configs/backtest.yaml")
DEFAULT_RETURNS_PATH = Path("data/processed/returns.csv")
DEFAULT_RETURN_COL = "ret_1d"

def parse_pred_metadata(path: Path) -> tuple[str, int | None]:
    stem = path.stem
    if "_h" not in stem:
        return stem, None
    model, rest = stem.split("_h", 1)
    digits = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    horizon = int("".join(digits)) if digits else None
    return model, horizon


def normalize_preds(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist and are named consistently
    out = df.copy()
    if "asset" not in out.columns and "ticker" in out.columns:
        out = out.rename(columns={"ticker": "asset"})
    # keep only necessary columns if extras exist
    needed = {"date", "asset", "yhat_rv", "yhat_logrv", "y_true_rv", "y_true_logrv"}
    # don't fail if some are missing; backtest() only needs date/asset/yhat_rv
    if not {"date", "asset", "yhat_rv"}.issubset(out.columns):
        missing = {"date", "asset", "yhat_rv"} - set(out.columns)
        raise ValueError(f"Prediction file missing required columns: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "asset"]).reset_index(drop=True)

def run_one(
    pred_csv: Path,
    out_dir: Path,
    returns_df: pd.DataFrame,
    return_col: str,
    target_vol=0.10,
    cost_bps=10,
    no_trade_pp=5,
    weight_cap=1.0,
    engine_kwargs: dict | None = None,
    name_suffix: str = "",
):
    preds = pd.read_csv(pred_csv, parse_dates=["date"])
    preds = normalize_preds(preds)

    # default to test split if present; otherwise use all rows
    if "split" in preds.columns:
        use = preds[preds["split"] == "test"].copy()
        if use.empty:
            use = preds.copy()
    else:
        use = preds.copy()

    engine_kwargs = engine_kwargs or {}
    out = backtest(
        use,
        returns_df,
        target_vol=target_vol,
        cost_bps=cost_bps,
        no_trade_pp=no_trade_pp,
        return_col=return_col,
        weight_cap=weight_cap,
        **engine_kwargs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = name_suffix or ""
    out_name = f"{pred_csv.stem}{suffix}_bt.csv"
    out_path = out_dir / out_name
    out.to_csv(out_path, index=False)
    print(
        f"[ok] {pred_csv.name} (cost={cost_bps}bps{', suffix='+suffix if suffix else ''}) → {out_path.name} ({len(out)} rows)"
    )
    return out_path

def filter_prediction_files(
    files: Sequence[Path],
    models: set[str] | None,
    horizons: set[int] | None,
) -> list[Path]:
    if not models and not horizons:
        return list(files)
    keep = []
    for path in files:
        model, horizon = parse_pred_metadata(path)
        model_key = model.lower()
        if models and model_key not in models:
            continue
        if horizons:
            if horizon is None or horizon not in horizons:
                continue
        keep.append(path)
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inverse-volatility backtests across multiple prediction files with optional model/horizon filters."
    )
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR, help="Directory containing prediction CSVs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RES_DIR, help="Directory to store backtest outputs.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern (relative to pred-dir) selecting candidate prediction files (default: *.csv).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model prefixes to include (e.g., har harx simple_rnn). Case-insensitive. Default: all models.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Forecast horizons to include (e.g., 1 5 22). Default: all horizons found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the prediction files that would be backtested without running them.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Portfolio config file (default: configs/backtest.yaml).",
    )
    parser.add_argument(
        "--cost-bps-grid",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of transaction-cost levels (bps). Each level runs a separate backtest per file.",
    )
    parser.add_argument(
        "--allocator",
        choices=["inverse_vol", "risk_parity"],
        default=None,
        help="Override allocator (default from config).",
    )
    parser.add_argument(
        "--allocators",
        nargs="+",
        choices=["inverse_vol", "risk_parity"],
        default=None,
        help="Run multiple allocators sequentially (overrides --allocator when provided).",
    )
    parser.add_argument("--risk-parity-window", type=int, default=None)
    parser.add_argument("--risk-parity-min-obs", type=int, default=None)
    parser.add_argument("--rebalance-fraction", type=float, default=None)
    parser.add_argument("--max-turnover", type=float, default=None)
    parser.add_argument("--cov-shrink", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pred_dir = args.pred_dir
    out_dir = args.out_dir
    candidate_files = sorted(pred_dir.glob(args.pattern))
    if not candidate_files:
        print(f"[warn] no prediction files matched pattern '{args.pattern}' under {pred_dir}")
        return

    model_filter = {m.lower() for m in args.models} if args.models else None
    horizon_filter = set(args.horizons) if args.horizons else None
    files = filter_prediction_files(candidate_files, model_filter, horizon_filter)
    if not files:
        model_msg = f" models={sorted(model_filter)}" if model_filter else ""
        horizon_msg = f" horizons={sorted(horizon_filter)}" if horizon_filter else ""
        print(f"[warn] no prediction files matched filters{model_msg}{horizon_msg}")
        return

    if args.dry_run:
        print("Would run backtests for the following files:")
        for p in files:
            model, horizon = parse_pred_metadata(p)
            print(f" - {p.name} (model={model}, horizon={horizon})")
        return

    returns_path = Path(cfg.get("returns_path", DEFAULT_RETURNS_PATH))
    return_pref = cfg.get("returns_col", DEFAULT_RETURN_COL)
    returns_df, return_col = load_returns(returns_path, preferred_col=return_pref)
    target_vol = cfg.get("target_vol", 0.10)
    default_cost = cfg.get("cost_bps", 10)
    cost_grid = [default_cost]
    if args.cost_bps_grid:
        cost_grid = [float(x) for x in args.cost_bps_grid]
    no_trade_pp = cfg.get("no_trade_pp", 5)
    weight_cap = cfg.get("weight_cap", 1.0)
    default_allocator = (args.allocator if args.allocator is not None else cfg.get("allocator", "inverse_vol")).lower()
    selected_allocators = [a.lower() for a in args.allocators] if args.allocators else [default_allocator]
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

    base_engine_kwargs = dict(
        risk_parity_window=rp_window,
        risk_parity_min_obs=rp_min,
        rebalance_fraction=rebalance_fraction,
        max_turnover=max_turnover,
        cov_shrink=cov_shrink,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    for alloc in selected_allocators:
        alloc_dir = out_dir / alloc if len(selected_allocators) > 1 else out_dir
        alloc_dir.mkdir(parents=True, exist_ok=True)
        engine_kwargs = dict(base_engine_kwargs)
        engine_kwargs["allocator"] = alloc

        for p in files:
            for cost in cost_grid:
                suffix_parts = []
                if len(selected_allocators) > 1:
                    suffix_parts.append(f"_{alloc}")
                if len(cost_grid) > 1:
                    suffix_parts.append(f"_cost{str(cost).replace('.', 'p')}")
                suffix = "".join(suffix_parts)
                out_path = run_one(
                    p,
                    alloc_dir,
                    returns_df,
                    return_col,
                    target_vol=target_vol,
                    cost_bps=cost,
                    no_trade_pp=no_trade_pp,
                    weight_cap=weight_cap,
                    engine_kwargs=engine_kwargs,
                    name_suffix=suffix,
                )
                summary_rows.append(
                    {
                        "allocator": alloc,
                        "cost_bps": str(cost),
                        "pred_file": p.name,
                        "output_csv": str(out_path),
                    }
                )

    if summary_rows:
        summary_path = out_dir / "allocator_summary.csv"
        with summary_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[summary] Wrote allocator summary to {summary_path}")

if __name__ == "__main__":
    main()
