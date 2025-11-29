# src/data/loaders.py
import pandas as pd


def load_panel(prices_csv, rv_csv, vix_csv):
    px = pd.read_csv(prices_csv, parse_dates=["date"])
    rv = pd.read_csv(rv_csv, parse_dates=["date"])  # realized variance/volatility
    vix = pd.read_csv(
        vix_csv, parse_dates=["date"]
    )  # VIX and term structure if available
    for df in (px, rv, vix):
        df.sort_values("date", inplace=True)
    return px, rv, vix
