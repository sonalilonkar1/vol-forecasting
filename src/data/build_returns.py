# src/data/build_returns.py
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROC= Path("data/processed")

def main():
    px = pd.read_csv(RAW/"etf_prices.csv", parse_dates=["date"])
    px = px.sort_values(["asset","date"])
    px["ret"] = px.groupby("asset")["close"].pct_change().fillna(0.0)
    out = PROC/"returns.csv"
    PROC.mkdir(parents=True, exist_ok=True)
    px[["date","asset","ret"]].to_csv(out, index=False)
    print("Wrote", out)

if __name__=="__main__":
    main()
