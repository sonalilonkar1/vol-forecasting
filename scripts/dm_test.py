#!/usr/bin/env python
"""Diebold–Mariano test helper for comparing two prediction files.

This CLI loads paired prediction CSVs (baseline vs challenger), computes the
selected loss for each aligned observation, and reports the DM statistic plus
p-value using a Newey–West variance estimate. Results can optionally be grouped
by split, asset, or any other column present in both files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

LossFn = Callable[[pd.DataFrame], np.ndarray]


LOSS_FUNCTIONS: dict[str, LossFn] = {}


def loss(name: str) -> Callable[[LossFn], LossFn]:
    """Decorator to register loss functions by name."""

    def _wrap(func: LossFn) -> LossFn:
        LOSS_FUNCTIONS[name] = func
        return func

    return _wrap


@loss("mse_log")
def _mse_log(df: pd.DataFrame) -> np.ndarray:
    diff = df["y_true_logrv"] - df["yhat_logrv"]
    return np.square(diff.to_numpy(float))


@loss("mae_log")
def _mae_log(df: pd.DataFrame) -> np.ndarray:
    diff = df["y_true_logrv"] - df["yhat_logrv"]
    return np.abs(diff.to_numpy(float))


@loss("qlike")
def _qlike(df: pd.DataFrame) -> np.ndarray:
    eps = 1e-12
    y = np.clip(df["y_true_rv"].to_numpy(float), eps, None)
    f = np.clip(df["yhat_rv"].to_numpy(float), eps, None)
    return y / f - np.log(y / f) - 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Diebold–Mariano test on two prediction CSVs."
    )
    parser.add_argument("baseline", type=Path, help="Path to the baseline prediction CSV.")
    parser.add_argument(
        "challenger", type=Path, help="Path to the challenger prediction CSV."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate (if present in CSV). Use 'all' to keep every row.",
    )
    parser.add_argument(
        "--loss",
        choices=sorted(LOSS_FUNCTIONS.keys()),
        default="mse_log",
        help="Loss to compare (default: mse_log on log volatility).",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=None,
        help="Optional columns to group by (e.g., split asset).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag for Newey–West variance (default: 5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV sink for the DM summary table.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Friendly label for the comparison (defaults to challenger vs baseline stems).",
    )
    return parser.parse_args()


def _load_predictions(path: Path, split: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "asset", "y_true_logrv", "yhat_logrv", "y_true_rv", "yhat_rv"}
    missing = required - set(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Prediction file {path} missing columns: {cols}")
    if split and split.lower() != "all" and "split" in df.columns:
        before = len(df)
        df = df[df["split"].str.lower() == split.lower()].copy()
        after = len(df)
        if after == 0:
            raise ValueError(f"No rows remain in {path} after filtering split='{split}'.")
        print(f"Filtered {path.name}: {before:,} -> {after:,} rows")
    return df


def _align_frames(baseline: pd.DataFrame, challenger: pd.DataFrame) -> pd.DataFrame:
    merged = baseline.merge(
        challenger,
        on=["date", "asset"],
        suffixes=("_base", "_chall"),
        how="inner",
        copy=False,
    )
    if merged.empty:
        raise ValueError("No overlapping rows between baseline and challenger predictions.")
    expected_cols = {
        "y_true_logrv_base",
        "yhat_logrv_base",
        "y_true_rv_base",
        "yhat_rv_base",
        "y_true_logrv_chall",
        "yhat_logrv_chall",
        "y_true_rv_chall",
        "yhat_rv_chall",
    }
    missing = expected_cols - set(merged.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Merged frame missing expected columns: {cols}")
    # Ensure truths match.
    for col in ["y_true_logrv", "y_true_rv"]:
        diff = merged[f"{col}_base"] - merged[f"{col}_chall"]
        if not np.allclose(diff, 0.0, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Ground-truth column '{col}' differs between baseline and challenger files."
            )
    merged.rename(
        columns={
            "y_true_logrv_base": "y_true_logrv",
            "y_true_rv_base": "y_true_rv",
            "yhat_logrv_base": "yhat_logrv_baseline",
            "yhat_rv_base": "yhat_rv_baseline",
            "yhat_logrv_chall": "yhat_logrv_challenger",
            "yhat_rv_chall": "yhat_rv_challenger",
        },
        inplace=True,
    )
    return merged


def _newey_west_variance(d: np.ndarray, max_lag: int) -> float:
    n = len(d)
    if n <= 1:
        return float("nan")
    centered = d - np.mean(d)
    gamma0 = np.dot(centered, centered) / n
    variance = gamma0
    max_k = min(max_lag, n - 1)
    for lag in range(1, max_k + 1):
        cov = np.dot(centered[lag:], centered[:-lag]) / n
        weight = 1.0 - lag / (max_k + 1)
        variance += 2.0 * weight * cov
    return float(max(variance, 0.0))


def _dm_stat(d: np.ndarray, max_lag: int) -> tuple[float, float]:
    n = len(d)
    mean_diff = float(np.mean(d))
    var = _newey_west_variance(d, max_lag)
    if n <= 1 or var <= 0:
        return mean_diff, float("nan")
    stat = mean_diff / np.sqrt(var / n)
    return mean_diff, float(stat)


def _compute_loss_vector(df: pd.DataFrame, role: str, loss_name: str) -> np.ndarray:
    cols = {
        "yhat_logrv": f"yhat_logrv_{role}",
        "yhat_rv": f"yhat_rv_{role}",
    }
    tmp = df.rename(columns=cols)
    tmp = tmp[["y_true_logrv", "y_true_rv", f"yhat_logrv_{role}", f"yhat_rv_{role}"]].copy()
    tmp = tmp.rename(
        columns={
            f"yhat_logrv_{role}": "yhat_logrv",
            f"yhat_rv_{role}": "yhat_rv",
        }
    )
    func = LOSS_FUNCTIONS[loss_name]
    return func(tmp)


def run_dm_test(df: pd.DataFrame, loss_name: str, max_lag: int) -> pd.DataFrame:
    loss_base = _compute_loss_vector(df, "baseline", loss_name)
    loss_chall = _compute_loss_vector(df, "challenger", loss_name)
    diff = loss_base - loss_chall
    mean_diff, stat = _dm_stat(diff, max_lag)
    n = len(diff)
    p_value = float(2 * (1 - norm.cdf(abs(stat)))) if np.isfinite(stat) else float("nan")
    return pd.DataFrame(
        [
            {
                "n": n,
                "mean_loss_baseline": float(np.mean(loss_base)),
                "mean_loss_challenger": float(np.mean(loss_chall)),
                "mean_diff": mean_diff,
                "dm_stat": stat,
                "p_value": p_value,
            }
        ]
    )


def main() -> None:
    args = parse_args()
    base = _load_predictions(args.baseline, args.split)
    challenger = _load_predictions(args.challenger, args.split)
    merged = _align_frames(base, challenger)

    group_cols = args.group_by or []
    if group_cols:
        missing = [col for col in group_cols if col not in merged.columns]
        if missing:
            cols = ", ".join(missing)
            raise ValueError(f"Requested group-by columns not found in merged frame: {cols}")
        grouped: Iterable[tuple[tuple, pd.DataFrame]] = merged.groupby(group_cols)
    else:
        grouped = [((), merged)]

    rows: list[dict] = []
    for key, frame in grouped:
        summary = run_dm_test(frame, args.loss, args.max_lag)
        record = summary.iloc[0].to_dict()
        record["group"] = key if group_cols else "all"
        rows.append(record)

    result = pd.DataFrame(rows)
    default_label = f"{args.challenger.stem} vs {args.baseline.stem}"
    label = args.label or default_label
    print(f"Diebold–Mariano test ({args.loss}) — {label}")
    print(result.to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.insert(0, "label", label)
        result.to_csv(args.out, index=False)
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
