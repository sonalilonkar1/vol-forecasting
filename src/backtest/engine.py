# src/backtest/engine.py
import warnings

import numpy as np
import pandas as pd


def inverse_vol_weights(sigmas: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.clip(sigmas, 1e-8, None)
    return inv / np.sum(inv)


def risk_parity_weights(cov: np.ndarray, tol: float = 1e-5, max_iter: int = 1000) -> np.ndarray:
    n = cov.shape[0]
    if n == 0:
        raise ValueError("Covariance matrix must be non-empty for risk parity weights.")
    w = np.ones(n, dtype=float) / n
    target = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        marginal = cov @ w
        risk_contrib = w * marginal
        total = np.sum(risk_contrib)
        if total <= 0:
            break
        frac = risk_contrib / total
        if np.max(np.abs(frac - target)) < tol:
            break
        adjust = target / np.clip(frac, 1e-8, None)
        w *= adjust
        w = np.clip(w, 1e-8, None)
        w /= np.sum(w)
    return w


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


def _rolling_cov(panel: pd.DataFrame, idx: int, window: int, min_obs: int) -> np.ndarray | None:
    if window <= 0 or idx == 0:
        return None
    start = max(0, idx - window)
    hist = panel.iloc[start:idx]
    if len(hist) < max(min_obs, 2):
        return None
    return np.cov(hist.values, rowvar=False)


def backtest(
    preds_df,
    returns_df,
    target_vol=0.10,
    cost_bps=10,
    no_trade_pp=5,
    return_col="ret_1d",
    weight_cap=1.0,
    allocator: str = "inverse_vol",
    risk_parity_window: int = 60,
    risk_parity_min_obs: int = 20,
    rebalance_fraction: float = 1.0,
    max_turnover: float | None = None,
    cov_shrink: float = 1e-6,
):
    preds_df = preds_df.sort_values(["date", "asset"]).reset_index(drop=True)
    assets = preds_df["asset"].unique().tolist()
    grouped_preds = {dt: g for dt, g in preds_df.groupby("date")}
    dates = sorted(grouped_preds.keys())

    return_panel = _prepare_returns(returns_df, assets, return_col)
    return_panel = return_panel.reindex(dates)
    missing_dates = [dt for dt in dates if return_panel.loc[dt].isna().all()]
    return_panel = return_panel.fillna(0.0)

    window = max(int(risk_parity_window), 0)
    min_obs = max(int(risk_parity_min_obs), 2)
    use_rp = allocator.lower() == "risk_parity" and len(assets) > 1
    max_turnover = None if max_turnover is None else float(max_turnover)
    rebalance_fraction = float(np.clip(rebalance_fraction, 0.0, 1.0))

    band = no_trade_pp / 100.0
    cap = abs(weight_cap) if weight_cap is not None else None
    w_active = np.zeros(len(assets))
    pending_cost = 0.0
    last_turnover = 0.0
    results = []

    for idx, dt in enumerate(dates):
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

        if use_rp:
            cov = None
            if window > 0 and idx > 0:
                roll_cov = _rolling_cov(return_panel, idx, window, min_obs)
                if roll_cov is not None and roll_cov.shape == (len(assets), len(assets)):
                    cov = roll_cov + np.eye(len(assets)) * cov_shrink
            if cov is not None and len(assets) >= 2:
                w_raw = risk_parity_weights(cov)
            else:
                w_raw = inverse_vol_weights(sigma_vec)
        else:
            w_raw = inverse_vol_weights(sigma_vec)
        delta = w_raw - w_active
        w_candidate = np.where(np.abs(delta) >= band, w_raw, w_active)

        port_sig_est = float(np.sqrt(np.sum((w_candidate**2) * (sigma_vec**2))))
        scaler = (target_vol / port_sig_est) if port_sig_est > 1e-8 else 1.0
        w_target = w_candidate * scaler
        w_new = w_active + rebalance_fraction * (w_target - w_active)
        if cap is not None:
            w_new = np.clip(w_new, -cap, cap)

        last_turnover = 0.5 * np.abs(w_new - w_active).sum()
        if max_turnover is not None and last_turnover > max_turnover and last_turnover > 0:
            scale = max_turnover / last_turnover
            w_new = w_active + scale * (w_new - w_active)
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
