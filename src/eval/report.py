# src/eval/report.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Tuple

import pandas as pd

from .metrics import summarize_predictions

PRED_DIR = Path("experiments/preds")
RES_DIR = Path("experiments/results")
GARCH_PRED_DIR = Path("experiments/garch/preds")
NBEATS_PRED_DIR = Path("experiments/nbeats/preds")


def _parse_horizon(path: Path) -> int | None:
    stem = path.stem  # e.g., har_h1, h5
    candidate = None
    if "_h" in stem:
        candidate = stem.rsplit("_h", 1)[-1]
    elif stem.startswith("h"):
        candidate = stem[1:]
    if candidate:
        try:
            return int(candidate)
        except ValueError:
            return None
    return None


def _iter_prediction_files() -> Iterator[Tuple[str, Path]]:
    if PRED_DIR.exists():
        for path in sorted(PRED_DIR.glob("*_h*.csv")):
            model_name = path.stem.split("_h", 1)[0]
            yield model_name, path
    for model_name, directory in [
        ("garch", GARCH_PRED_DIR),
        ("nbeats", NBEATS_PRED_DIR),
    ]:
        if directory.exists():
            for path in sorted(directory.glob("h*.csv")):
                yield model_name, path


def _build_summary_rows(files: Iterable[Tuple[str, Path]]) -> list[dict]:
    rows: list[dict] = []
    for model_name, path in files:
        df = pd.read_csv(path, parse_dates=["date"])
        if "split" not in df.columns:
            df["split"] = "test"
        horizon = _parse_horizon(path)
        summary = summarize_predictions(df)
        for _, record in summary.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "split": record["split"],
                    "rmse_log": record["rmse_log"],
                    "qlike": record["qlike"],
                    "n": record["n"],
                }
            )
    return rows


def _build_bt_overview(res_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    if not res_dir.exists():
        return pd.DataFrame(rows)
    for bt_path in sorted(res_dir.rglob("*_bt.csv")):
        df = pd.read_csv(bt_path)
        if df.empty:
            continue
        last = df.iloc[-1]
        rows.append(
            {
                "file": str(bt_path.relative_to(res_dir)),
                "final_cum_net": last.get("cum_net"),
                "final_cum_gross": last.get("cum_gross"),
                "avg_turnover": df["turnover"].mean() if "turnover" in df else None,
                "avg_cost": df["cost"].mean() if "cost" in df else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pred_rows = _build_summary_rows(_iter_prediction_files())
    bt_overview = _build_bt_overview(RES_DIR)

    if not pred_rows and bt_overview.empty:
        print("No prediction or backtest files found.")
        return

    RES_DIR.mkdir(parents=True, exist_ok=True)

    if pred_rows:
        summary_df = pd.DataFrame(pred_rows).sort_values(["model", "horizon", "split"])
        summary_path = RES_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path} with {len(summary_df)} rows")
    else:
        print("No prediction files found for summary report.")

    if not bt_overview.empty:
        overview_path = RES_DIR / "bt_overview.csv"
        bt_overview.to_csv(overview_path, index=False)
        print(f"Wrote {overview_path} with {len(bt_overview)} rows")
    else:
        print("No backtest files found for overview report.")


if __name__ == "__main__":
    main()
