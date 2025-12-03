#!/usr/bin/env python3
"""Aggregate backtest metrics per experiment/run from automation_summary.csv.

Example usage:
    python scripts/summarize_experiments.py --experiments har_baselines seq_rnn \
        --output experiments/results/har_summary.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "experiments" / "results" / "automation_summary.csv"
TRADING_DAYS = 252


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize backtest outputs for each experiment/run listed in "
            "experiments/results/automation_summary.csv"
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to automation_summary.csv (default: experiments/results/automation_summary.csv)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional list of experiment names to include (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the aggregated table as CSV",
    )
    parser.add_argument(
        "--sort-by",
        default="final_cum_net",
        choices=[
            "final_cum_net",
            "final_cum_gross",
            "net_cagr",
            "sharpe_net",
            "avg_turnover",
            "avg_cost",
            "max_drawdown",
        ],
        help="Column to sort the summary by (descending order).",
    )
    return parser.parse_args(argv)


def compute_max_drawdown(cum_curve: pd.Series) -> float:
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1.0
    return drawdown.min()


def summarize_backtest(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Backtest file {path} is empty")
    n_obs = len(df)
    final_cum_net = float(df["cum_net"].iloc[-1])
    final_cum_gross = float(df["cum_gross"].iloc[-1])
    avg_turnover = float(df["turnover"].mean()) if "turnover" in df.columns else math.nan
    avg_cost = float(df["cost"].mean()) if "cost" in df.columns else math.nan
    net_ret = df["net_ret"] if "net_ret" in df.columns else pd.Series(dtype=float)
    mean_net = float(net_ret.mean()) if not net_ret.empty else math.nan
    std_net = float(net_ret.std(ddof=0)) if not net_ret.empty else math.nan
    sharpe = math.sqrt(TRADING_DAYS) * mean_net / std_net if std_net and std_net > 0 else math.nan
    net_cagr = final_cum_net ** (TRADING_DAYS / n_obs) - 1 if final_cum_net > 0 else math.nan
    max_dd = compute_max_drawdown(df["cum_net"]) if "cum_net" in df.columns else math.nan
    return {
        "final_cum_net": final_cum_net,
        "final_cum_gross": final_cum_gross,
        "net_cagr": net_cagr,
        "sharpe_net": sharpe,
        "avg_turnover": avg_turnover,
        "avg_cost": avg_cost,
        "max_drawdown": max_dd,
        "n_days": n_obs,
    }


def load_summary_rows(args: argparse.Namespace) -> pd.DataFrame:
    summary_path = args.summary if args.summary.is_absolute() else (REPO_ROOT / args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if args.experiments:
        df = df[df["experiment"].isin(args.experiments)]
    if df.empty:
        raise ValueError("No rows match the requested experiment filter.")
    return df


def aggregate(args: argparse.Namespace) -> pd.DataFrame:
    summary_df = load_summary_rows(args)
    records: List[dict] = []
    for _, row in summary_df.iterrows():
        bt_path = REPO_ROOT / row["backtest"]
        if not bt_path.exists():
            print(f"[warn] Missing backtest file: {bt_path}", file=sys.stderr)
            continue
        metrics = summarize_backtest(bt_path)
        records.append(
            {
                "experiment": row["experiment"],
                "run": row["run"],
                "horizon": row["horizon"],
                **metrics,
                "backtest": row["backtest"],
            }
        )
    if not records:
        raise RuntimeError("No backtest files were summarized.")
    result = pd.DataFrame(records)
    result = result.sort_values(by=args.sort_by, ascending=False).reset_index(drop=True)
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = aggregate(args)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    if args.output:
        output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\n[summary] wrote {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
