# src/models/har_rv.py
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_predict_har(
    feat_path="data/processed/features.parquet", out="experiments/preds/har_h1.csv"
):
    df = pd.read_parquet(feat_path)
    out_rows = []
    for a, g in df.groupby("asset"):
        g = g.sort_values("date").copy()
        g["d"] = g["log_rv"].shift(1)
        g["w"] = g["log_rv"].shift(1).rolling(5).mean()
        g["m"] = g["log_rv"].shift(1).rolling(22).mean()
        g = g.dropna()
        if g.empty:
            # Skip thin assets that cannot support the HAR windows
            continue
        X = sm.add_constant(g[["d", "w", "m"]])
        y = g["log_rv"]
        model = sm.OLS(y, X).fit()
        g["yhat_logrv"] = model.predict(X)
        g["yhat_rv"] = np.exp(g["yhat_logrv"])
        g["horizon"] = 1
        g["model"] = "HAR-RV"
        g.rename(columns={"log_rv": "y_true_logrv", "rv": "y_true_rv"}, inplace=True)
        out_rows.append(
            g[
                [
                    "date",
                    "asset",
                    "y_true_logrv",
                    "y_true_rv",
                    "horizon",
                    "yhat_logrv",
                    "yhat_rv",
                    "model",
                ]
            ]
        )
    if not out_rows:
        raise ValueError("No assets produced HAR features; check input window lengths before fitting.")
    pred = pd.concat(out_rows).reset_index(drop=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"Wrote {out} with {len(pred):,} rows.")


if __name__ == "__main__":
    fit_predict_har()
