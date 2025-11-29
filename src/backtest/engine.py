# src/backtest/engine.py
import numpy as np
import pandas as pd


def inverse_vol_weights(sigmas):
    inv = 1.0 / np.clip(sigmas, 1e-8, None)
    return inv / np.sum(inv)


def backtest(preds_df, target_vol=0.10, cost_bps=10, no_trade_pp=5):
    preds_df = preds_df.sort_values("date")
    assets = preds_df["asset"].unique().tolist()
    w_prev = np.zeros(len(assets))
    results = []
    for dt, g in preds_df.groupby("date"):
        sig = g.set_index("asset")["yhat_rv"] ** 0.5
        w_raw = inverse_vol_weights(sig.reindex(assets).values)
        # no-trade band
        band = no_trade_pp / 100.0
        delta = w_raw - w_prev
        mask = np.abs(delta) >= band
        w = np.where(mask, w_raw, w_prev)
        # vol targeting (diag approx)
        port_sig = np.sqrt((w**2 * sig.reindex(assets).values ** 2).sum())
        scaler = (target_vol / port_sig) if port_sig > 1e-8 else 1.0
        w = np.clip(w * scaler, -1.0, 1.0)
        # costs: turnover * cost_bps
        turnover = 0.5 * np.abs(w - w_prev).sum()
        cost = turnover * (cost_bps / 10000.0)
        results.append(
            dict(date=dt, turnover=turnover, cost=cost, port_sigma=port_sig * scaler)
        )
        w_prev = w
    return pd.DataFrame(results)
