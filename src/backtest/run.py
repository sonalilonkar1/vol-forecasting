# src/backtest/run.py
from pathlib import Path

import pandas as pd

from .engine import backtest

if __name__ == "__main__":
    preds = pd.read_csv("experiments/preds/har_h1.csv", parse_dates=["date"])
    out = backtest(preds, target_vol=0.10, cost_bps=10, no_trade_pp=5)
    Path("reports/figs").mkdir(parents=True, exist_ok=True)
    out.to_csv("experiments/results/har_h1_bt.csv", index=False)
    print(out.tail())
