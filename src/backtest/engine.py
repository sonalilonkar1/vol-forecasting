# src/backtest/engine.py
import warnings

import numpy as np
import pandas as pd


def inverse_vol_weights(sigmas):
    inv = 1.0 / np.clip(sigmas, 1e-8, None)
    return inv / np.sum(inv)


def _prepare_returns(returns_df, assets, return_col):
    ret = returns_df.copy()
    if "asset" not in ret.columns and "ticker" in ret.columns:
        ret = ret.rename(columns={"ticker": "asset"})
    required = {"date", "asset", return_col}
    missing = required - set(ret.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Returns frame missing columns: {cols}")

    ret["date"] = pd.to_datetime(ret["date"])

    subset = ret[ret["asset"].isin(assets)].copy()
    if subset.empty:
        raise ValueError("Return data contains none of the assets referenced in predictions.")

    panel = (
        subset.pivot(index="date", columns="asset", values=return_col)
        .reindex(columns=assets)
        .fillna(0.0)
        .sort_index()
    )
    return panel


def backtest(
    preds_df,
    returns_df,
    target_vol=0.10,
    cost_bps=10,
    no_trade_pp=5,
    return_col="ret_1d",
    weight_cap=1.0,
):
    preds_df = preds_df.sort_values(["date", "asset"]).reset_index(drop=True)
    assets = preds_df["asset"].unique().tolist()
    grouped_preds = {dt: g for dt, g in preds_df.groupby("date")}
    dates = sorted(grouped_preds.keys())

    return_panel = _prepare_returns(returns_df, assets, return_col)
    return_panel = return_panel.reindex(dates)
    missing_dates = [dt for dt in dates if return_panel.loc[dt].isna().all()]
    return_panel = return_panel.fillna(0.0)

    band = no_trade_pp / 100.0
    cap = abs(weight_cap) if weight_cap is not None else None
    w_active = np.zeros(len(assets))
    pending_cost = 0.0
    last_turnover = 0.0
    results = []

    for dt in dates:
        asset_returns = return_panel.loc[dt].reindex(assets).to_numpy(dtype=float)
        gross = float(np.dot(w_active, asset_returns))
        net = gross - pending_cost

        sig = grouped_preds[dt].set_index("asset")["yhat_rv"].reindex(assets)
        if sig.isna().any():
            missing_assets = ", ".join(sig[sig.isna()].index.tolist())
            raise ValueError(
                f"Missing predictions for assets [{missing_assets}] on {dt.date()}"
            )
        sigma_vec = np.sqrt(np.clip(sig.values, 1e-12, None))
        port_sigma = float(np.sqrt(np.sum((w_active**2) * (sigma_vec**2))))

        results.append(
            dict(
                date=dt,
                turnover=last_turnover,
                cost=pending_cost,
                port_sigma=port_sigma,
                gross_ret=gross,
                net_ret=net,
            )
        )

        w_raw = inverse_vol_weights(sigma_vec)
        delta = w_raw - w_active
        w_candidate = np.where(np.abs(delta) >= band, w_raw, w_active)

        port_sig_est = float(np.sqrt(np.sum((w_candidate**2) * (sigma_vec**2))))
        scaler = (target_vol / port_sig_est) if port_sig_est > 1e-8 else 1.0
        w_new = w_candidate * scaler
        if cap is not None:
            w_new = np.clip(w_new, -cap, cap)

        last_turnover = 0.5 * np.abs(w_new - w_active).sum()
        pending_cost = last_turnover * (cost_bps / 10000.0)
        w_active = w_new

    if missing_dates:
        warnings.warn(
            f"No returns data for {len(missing_dates)} rebalance dates; treated as zero.",
            RuntimeWarning,
        )

    out = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    out["cum_gross"] = (1.0 + out["gross_ret"]).cumprod()
    out["cum_net"] = (1.0 + out["net_ret"]).cumprod()
    return out
