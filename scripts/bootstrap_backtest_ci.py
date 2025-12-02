#!/usr/bin/env python
"""Bootstrap confidence intervals for backtest KPIs.

The script reads one or more backtest result CSVs (the *_bt.csv files emitted by
src.backtest.run or run_batch), computes point estimates for key KPIs, and then
runs a moving-block bootstrap to estimate confidence intervals. Results are
written to a tidy CSV for downstream reporting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

MetricFn = Callable[[pd.DataFrame], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for backtest KPIs."
    )
    parser.add_argument(
        "bt_files",
        nargs="+",
        type=Path,
        help="One or more *_bt.csv files to analyze.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/results/bootstrap"),
        help="Directory to store bootstrap summary CSVs (default: experiments/results/bootstrap).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha level for confidence intervals (default: 0.05).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap draws (default: 1000).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Block size for moving-block bootstrap to retain serial dependence (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: set[str], path: Path) -> None:
    missing = cols - set(df.columns)
    if missing:
        cols_fmt = ", ".join(sorted(missing))
        raise ValueError(f"Backtest file {path} missing required columns: {cols_fmt}")


def _load_backtest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    _require_columns(df, {"net_ret", "gross_ret", "turnover", "cost"}, path)
    return df.sort_values("date").reset_index(drop=True)


def _annualization_factor(freq: int = 252) -> float:
    return np.sqrt(freq)


def _cagr(net_rets: np.ndarray, freq: int = 252) -> float:
    n = len(net_rets)
    if n == 0:
        return float("nan")
    cum = np.cumprod(1.0 + net_rets)[-1]
    years = n / freq
    if cum <= 0 or years <= 0:
        return float("nan")
    return float(cum ** (1.0 / years) - 1.0)


def _sharpe(net_rets: np.ndarray, freq: int = 252) -> float:
    mean = np.mean(net_rets)
    std = np.std(net_rets, ddof=1)
    if std <= 0:
        return float("nan")
    return float((mean / std) * _annualization_factor(freq))


def _max_drawdown(net_rets: np.ndarray) -> float:
    if len(net_rets) == 0:
        return float("nan")
    cum = np.cumprod(1.0 + net_rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum / np.where(running_max == 0, np.nan, running_max)) - 1.0
    return float(np.nanmin(drawdowns))


def _avg(series: pd.Series | np.ndarray) -> float:
    return float(np.mean(series)) if len(series) else float("nan")


def _compute_point_estimates(df: pd.DataFrame) -> Dict[str, float]:
    net = df["net_ret"].to_numpy(float)
    summary = {
        "net_cagr": _cagr(net),
        "net_sharpe": _sharpe(net),
        "max_drawdown": _max_drawdown(net),
        "avg_turnover": _avg(df["turnover"].to_numpy(float)),
        "avg_cost": _avg(df["cost"].to_numpy(float)),
    }
    summary["realized_vol"] = float(np.std(net, ddof=1) * _annualization_factor())
    summary["final_cum_net"] = float(np.cumprod(1.0 + net)[-1]) if len(net) else float("nan")
    return summary


def _sample_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=int)
    block = max(block, 1)
    starts = np.arange(n)
    picks = []
    while len(picks) < n:
        start = int(rng.choice(starts))
        end = min(start + block, n)
        picks.extend(range(start, end))
    return np.array(picks[:n], dtype=int)


def _bootstrap_metrics(
    df: pd.DataFrame,
    metrics: Dict[str, MetricFn],
    n_bootstrap: int,
    block_size: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    idx = np.arange(len(df))
    values = {name: [] for name in metrics}
    for _ in range(n_bootstrap):
        sample_idx = _sample_indices(len(df), block_size, rng)
        if len(sample_idx) == 0:
            continue
        sample = df.iloc[sample_idx].reset_index(drop=True)
        for name, fn in metrics.items():
            values[name].append(fn(sample))
    return values


def _metric_functions() -> Dict[str, MetricFn]:
    return {
        "net_cagr": lambda d: _cagr(d["net_ret"].to_numpy(float)),
        "net_sharpe": lambda d: _sharpe(d["net_ret"].to_numpy(float)),
        "max_drawdown": lambda d: _max_drawdown(d["net_ret"].to_numpy(float)),
        "avg_turnover": lambda d: _avg(d["turnover"].to_numpy(float)),
        "avg_cost": lambda d: _avg(d["cost"].to_numpy(float)),
    }


def _percentile_interval(samples: list[float], alpha: float) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    lower = np.nanpercentile(samples, 100 * (alpha / 2))
    upper = np.nanpercentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def summarize_file(
    path: Path,
    out_dir: Path,
    alpha: float,
    n_bootstrap: int,
    block_size: int,
    rng: np.random.Generator,
) -> Path:
    df = _load_backtest(path)
    metrics = _metric_functions()
    point_estimates = _compute_point_estimates(df)
    boot_samples = _bootstrap_metrics(df, metrics, n_bootstrap, block_size, rng)

    rows = []
    for name, fn in metrics.items():
        point = point_estimates[name]
        ci_low, ci_high = _percentile_interval(boot_samples[name], alpha)
        rows.append(
            {
                "file": path.name,
                "metric": name,
                "value": point,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_bootstrap": len(boot_samples[name]),
                "block_size": block_size,
            }
        )

    result = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_bootstrap_ci.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote bootstrap summary to {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    for bt_path in args.bt_files:
        summarize_file(bt_path, args.out_dir, args.alpha, args.n_bootstrap, args.block_size, rng)


if __name__ == "__main__":
    main()
