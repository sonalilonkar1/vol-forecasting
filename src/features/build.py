# src/features/build.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_features(
    rv_csv: str = "data/raw/rv.csv",
    vix_csv: str = "data/raw/vix.csv",
    out: str = "data/processed/features.parquet",
) -> None:
    rv = pd.read_csv(rv_csv, parse_dates=["date"]).sort_values(["asset", "date"]).copy()
    if rv.empty:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        rv.to_parquet(out, index=False)
        print(f"Wrote {out} with 0 rows (rv.csv empty).")
        return

    rv["log_rv"] = np.log(rv["rv"].clip(lower=1e-12))

    # Pick rolling windows that fit the data; fallback for tiny toy data
    per_asset_counts = rv.groupby("asset")["date"].size()
    n_min = int(per_asset_counts.min())
    default_windows = [5, 22, 66, 132, 252]
    windows = [w for w in default_windows if n_min >= (w + 2)]
    if not windows:  # toy data fallback
        windows = [2, 3]

    # Strictly use t-1 history
    for w in windows:
        rv[f"logrv_ma_{w}"] = (
            rv.groupby("asset")["log_rv"]
            .shift(1)
            .rolling(w, min_periods=max(2, min(w, n_min // 2)))
            .mean()
        )

    # Optional: VIX features (don’t force rows to drop if missing)
    if Path(vix_csv).exists():
        vix = pd.read_csv(vix_csv, parse_dates=["date"]).sort_values("date")[
            ["date", "vix"]
        ]
        vix = vix.rename(columns={"vix": "vix_lvl"})
        rv = rv.merge(vix, on="date", how="left")

        lvl = rv["vix_lvl"].shift(1)
        # smaller window if tiny dataset
        vwin = 22 if n_min >= 30 else 5
        mu = lvl.rolling(vwin, min_periods=max(2, vwin // 2)).mean()
        sd = lvl.rolling(vwin, min_periods=max(2, vwin // 2)).std()
        rv["vix_z"] = (lvl - mu) / sd

    # Only require the core columns (don’t drop just because VIX is NaN)
    required = ["log_rv"] + [f"logrv_ma_{w}" for w in windows]
    rv = rv.dropna(subset=required).reset_index(drop=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rv.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(rv):,} rows. Windows used: {windows}")


if __name__ == "__main__":
    build_features()
