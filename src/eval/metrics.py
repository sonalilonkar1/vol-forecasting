# src/eval/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

def qlike(y_true_rv: pd.Series, yhat_rv: pd.Series, eps: float = 1e-12) -> float:
    y = np.clip(y_true_rv.to_numpy(float), eps, None)
    f = np.clip(yhat_rv.to_numpy(float), eps, None)
    return float(np.mean(y/f - np.log(y/f) - 1.0))

def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    splits = pred["split"].unique().tolist() if "split" in pred.columns else ["all"]
    for sp in splits:
        g = pred if sp=="all" else pred[pred["split"]==sp]
        rmse = root_mean_squared_error(g["y_true_logrv"], g["yhat_logrv"])
        ql   = qlike(g["y_true_rv"], g["yhat_rv"])
        rows.append(dict(split=sp, rmse_log=rmse, qlike=ql, n=len(g)))
    return pd.DataFrame(rows)
