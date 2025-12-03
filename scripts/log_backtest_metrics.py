#!/usr/bin/env python
"""Summarize backtest CSVs and append a markdown block to metrics_log.md."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

DEFAULT_LOG = Path("experiments/results/metrics_log.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize *_bt.csv files and append metrics to the experiment log."
    )
    parser.add_argument(
        "bt_files",
        nargs="+",
        type=Path,
        help="Backtest CSVs (outputs of src.backtest.run or run_batch).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Markdown log to append to (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label added as a heading above the metrics table.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: set[str], path: Path) -> None:
    missing = cols - set(df.columns)
    if missing:
        cols_fmt = ", ".join(sorted(missing))
        raise ValueError(f"Backtest file {path} missing required columns: {cols_fmt}")


def _cagr(net: np.ndarray, freq: int = 252) -> float:
    n = len(net)
    if n == 0:
        return float("nan")
    cumulative = np.cumprod(1.0 + net)[-1]
    years = n / freq
    if cumulative <= 0 or years <= 0:
        return float("nan")
    return float(cumulative ** (1.0 / years) - 1.0)


def _sharpe(net: np.ndarray, freq: int = 252) -> float:
    std = np.std(net, ddof=1)
    if std <= 0:
        return float("nan")
    mean = np.mean(net)
    return float((mean / std) * np.sqrt(freq))


def _max_drawdown(net: np.ndarray) -> float:
    if len(net) == 0:
        return float("nan")
    cum = np.cumprod(1.0 + net)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum / np.where(running_max == 0, np.nan, running_max) - 1.0
    return float(np.nanmin(drawdown))


def _summarize(path: Path) -> Dict[str, float | int | str]:
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    _require_columns(df, {"turnover", "cost", "net_ret", "cum_net"}, path)
    net = df["net_ret"].to_numpy(float)
    summary = {
        "file": str(path),
        "days": len(df),
        "start": df["date"].iloc[0].date() if not df.empty else "n/a",
        "end": df["date"].iloc[-1].date() if not df.empty else "n/a",
        "net_cagr": _cagr(net),
        "net_sharpe": _sharpe(net),
        "max_drawdown": _max_drawdown(net),
        "avg_turnover": float(df["turnover"].mean()) if not df.empty else float("nan"),
        "avg_cost": float(df["cost"].mean()) if not df.empty else float("nan"),
        "final_cum_net": float(df["cum_net"].iloc[-1]) if not df.empty else float("nan"),
    }
    return summary


def _format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if abs(value) >= 1:
            return f"{value:0.4f}"
        return f"{value:0.4%}"
    return str(value)


def _append_markdown(log_path: Path, label: str | None, summary: Dict[str, float | int | str]) -> None:
    rows = [
        ("File", summary["file"]),
        ("Window", f"{summary['start']} -> {summary['end']}") ,
        ("Days", summary["days"]),
        ("Net CAGR", summary["net_cagr"]),
        ("Net Sharpe", summary["net_sharpe"]),
        ("Max Drawdown", summary["max_drawdown"]),
        ("Avg Turnover", summary["avg_turnover"]),
        ("Avg Cost", summary["avg_cost"]),
        ("Final Cum Net", summary["final_cum_net"]),
    ]
    heading = label or Path(summary["file"]).name
    lines = [f"### {heading}", "| Metric | Value |", "| --- | --- |"]
    for metric, value in rows:
        lines.append(f"| {metric} | {_format_value(value)} |")
    lines.append("")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    for bt_file in args.bt_files:
        summary = _summarize(bt_file)
        _append_markdown(args.log_path, args.label, summary)
        print(f"Logged metrics for {bt_file} -> {args.log_path}")


if __name__ == "__main__":
    main()
