# src/backtest/run_batch.py
from pathlib import Path
import pandas as pd

from .engine import backtest

PRED_DIR = Path("experiments/preds")
RES_DIR  = Path("experiments/results")

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

def run_one(pred_csv: Path, target_vol=0.10, cost_bps=10, no_trade_pp=5):
    preds = pd.read_csv(pred_csv, parse_dates=["date"])
    preds = normalize_preds(preds)

    # default to test split if present; otherwise use all rows
    if "split" in preds.columns:
        use = preds[preds["split"] == "test"].copy()
        if use.empty:
            use = preds.copy()
    else:
        use = preds.copy()

    out = backtest(use, target_vol=target_vol, cost_bps=cost_bps, no_trade_pp=no_trade_pp)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RES_DIR / f"{pred_csv.stem}_bt.csv"
    out.to_csv(out_path, index=False)
    print(f"[ok] {pred_csv.name} → {out_path.name} ({len(out)} rows)")

def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        print(f"[warn] no prediction files found under {PRED_DIR}")
        return
    for p in files:
        run_one(p)

if __name__ == "__main__":
    main()
