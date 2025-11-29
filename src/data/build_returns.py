# src/data/build_returns.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-asset one-day returns and store them in data/processed/returns.csv"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/tft_ready_dataset.csv"),
        help="Panel file containing date/asset/return columns (default: data/tft_ready_dataset.csv).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/processed/returns.csv"),
        help="Where to write the standardized returns CSV (default: data/processed/returns.csv).",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="date",
        help="Column containing the timestamp (default: date).",
    )
    parser.add_argument(
        "--asset-col",
        type=str,
        default=None,
        help="Column containing the asset identifier (defaults to asset or ticker if present).",
    )
    parser.add_argument(
        "--return-col",
        type=str,
        default="ret_1d",
        help="Column containing the one-day simple or log return (default: ret_1d).",
    )
    return parser.parse_args()


def infer_asset_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None and requested in df.columns:
        return requested
    for candidate in ("asset", "ticker", "symbol"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not determine asset column. Specify --asset-col explicitly (expected columns like 'asset' or 'ticker')."
    )


def build_returns_panel(
    source: Path,
    out_path: Path,
    date_col: str,
    asset_col: str | None,
    return_col: str,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source panel not found: {source}")

    usecols = None
    if asset_col is not None:
        usecols = [c for c in {date_col, asset_col, return_col}]

    df = pd.read_csv(source, parse_dates=[date_col], usecols=usecols)
    actual_asset_col = infer_asset_column(df, asset_col)
    missing = {date_col, actual_asset_col, return_col} - set(df.columns)
    if missing:
        raise ValueError(
            "Source file missing required columns: " + ", ".join(sorted(missing))
        )

    out = (
        df[[date_col, actual_asset_col, return_col]]
        .rename(columns={date_col: "date", actual_asset_col: "asset", return_col: "ret_1d"})
        .dropna(subset=["date", "asset", "ret_1d"])
    )
    if out.empty:
        raise ValueError("No valid rows found after filtering – check the input file and column names.")

    out["asset"] = out["asset"].astype(str)
    out = out.sort_values(["date", "asset"]).drop_duplicates(subset=["date", "asset"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    asset_col = args.asset_col
    written_path = build_returns_panel(
        source=args.source,
        out_path=args.out_path,
        date_col=args.date_col,
        asset_col=asset_col,
        return_col=args.return_col,
    )
    print(f"[ok] Wrote returns file to {written_path}")


if __name__ == "__main__":
    main()
