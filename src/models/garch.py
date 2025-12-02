# src/models/garch.py
"""
GARCH(1,1) baseline for volatility forecasting.
Follows the same CLI pattern and output schema as har_rv.py.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from arch import arch_model

from .har_rv import (
    resolve_source,
    load_feature_table,
    load_splits_config,
    compute_masks,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GARCH(1,1) baseline with YAML splits")
    p.add_argument("--source", type=Path, default=None,
                   help="CSV/Parquet feature table. Defaults to TFT-ready files.")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22],
                   help="Forecast horizons (e.g., 1 5 22)")
    p.add_argument("--eval-splits", nargs="+", choices=["train", "val", "test"],
                   default=["val", "test"])
    p.add_argument("--splits-config", type=Path, default=Path("configs/splits.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/preds"),
                   help="Directory for prediction CSVs (default: experiments/preds)")
    p.add_argument("--file-prefix", type=str, default="garch",
                   help="Filename prefix so outputs look like <prefix>_h*.csv")
    p.add_argument("--min-train", type=int, default=252,
                   help="Minimum training observations per asset")
    p.add_argument("--refit-every", type=int, default=22,
                   help="Refit GARCH every N days (rolling window)")
    return p.parse_args()


def build_returns_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table with returns and realized volatility targets.
    Uses ret_1d if available, otherwise computes from logvol.
    """
    tab = df[["date", "asset"]].copy()
    
    # Get returns
    if "ret_1d" in df.columns:
        tab["ret"] = df["ret_1d"]
    elif "log_ret" in df.columns:
        tab["ret"] = df["log_ret"]
    else:
        raise ValueError("Need 'ret_1d' or 'log_ret' column for GARCH")
    
    # Get target log volatility columns
    for h in [1, 5, 22]:
        tgt_col = f"target_logvol_t+{h}"
        if tgt_col in df.columns:
            tab[tgt_col] = df[tgt_col]
        else:
            # Build target from logvol_t if available
            if "logvol_t" in df.columns:
                tab[tgt_col] = df.groupby("asset")["logvol_t"].shift(-h)
    
    # Current log volatility for reference
    if "logvol_t" in df.columns:
        tab["logvol_t"] = df["logvol_t"]
    
    return tab.dropna(subset=["ret"]).reset_index(drop=True)


def fit_garch_predict(
    returns: np.ndarray,
    horizon: int,
    forecast_start_idx: int,
) -> float:
    """
    Fit GARCH(1,1) on returns up to forecast_start_idx and forecast h-step ahead variance.
    Returns log volatility forecast.
    """
    train_ret = returns[:forecast_start_idx]
    
    # Scale returns to percentage for numerical stability
    scale = 100.0
    scaled_ret = train_ret * scale
    
    try:
        model = arch_model(scaled_ret, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Forecast h-step ahead
        forecasts = result.forecast(horizon=horizon, reindex=False)
        # Get the h-step ahead variance (last column)
        var_forecast = forecasts.variance.iloc[-1, horizon - 1]
        
        # Convert back: unscale and take sqrt for volatility
        vol_forecast = np.sqrt(var_forecast) / scale
        
        # Convert to log volatility
        log_vol = np.log(np.clip(vol_forecast, 1e-10, None))
        return log_vol
        
    except Exception:
        return np.nan


def fit_garch_rolling(
    tab: pd.DataFrame,
    masks: Dict[str, pd.Series],
    eval_splits: Iterable[str],
    horizon: int,
    min_train: int,
    refit_every: int,
) -> pd.DataFrame:
    """
    Fit GARCH(1,1) per asset with rolling refits and produce predictions.
    """
    rows = []
    skipped = []
    target_col = f"target_logvol_t+{horizon}"
    
    for asset, grp in tab.groupby("asset"):
        grp = grp.sort_values("date").reset_index(drop=True)
        returns = grp["ret"].values
        dates = grp["date"].values
        
        # Get train end index
        train_mask = masks["train"].reindex(grp.index).fillna(False)
        if train_mask.sum() < min_train:
            skipped.append(asset)
            continue
        
        # Cache for rolling predictions
        last_fit_idx = None
        cached_model = None
        
        for split in eval_splits:
            split_mask = masks[split].reindex(grp.index).fillna(False)
            if not split_mask.any():
                continue
            
            split_indices = grp.index[split_mask].tolist()
            
            for idx in split_indices:
                # Check if we have target
                if target_col not in grp.columns or pd.isna(grp.loc[idx, target_col]):
                    continue
                
                # Determine training end (all data up to current point)
                train_end = idx
                if train_end < min_train:
                    continue
                
                # Refit check
                need_refit = (last_fit_idx is None or 
                             (train_end - last_fit_idx) >= refit_every)
                
                if need_refit:
                    log_vol_pred = fit_garch_predict(returns, horizon, train_end)
                    last_fit_idx = train_end
                else:
                    # Use cached prediction approach - still need fresh forecast
                    log_vol_pred = fit_garch_predict(returns, horizon, train_end)
                
                if np.isnan(log_vol_pred):
                    continue
                
                y_true_logrv = grp.loc[idx, target_col]
                y_true_rv = np.exp(np.clip(y_true_logrv, -50, 50))
                yhat_rv = np.exp(np.clip(log_vol_pred, -50, 50))
                
                rows.append({
                    "date": dates[idx],
                    "asset": asset,
                    "y_true_logrv": y_true_logrv,
                    "y_true_rv": y_true_rv,
                    "yhat_logrv": log_vol_pred,
                    "yhat_rv": yhat_rv,
                    "model": "GARCH(1,1)",
                    "horizon": horizon,
                    "split": split,
                })
    
    if skipped:
        print(f"Skipped {len(skipped)} assets (<{min_train} train rows): {', '.join(skipped)}")
    
    if not rows:
        raise ValueError("No predictions produced; check min-train or data coverage")
    
    pred = pd.DataFrame(rows)
    return pred.sort_values(["date", "asset"]).reset_index(drop=True)


def main():
    args = parse_args()
    
    # Load data
    df = load_feature_table(resolve_source(args.source))
    cfg = load_splits_config(args.splits_config)
    
    # Filter assets if specified
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build returns table
    tab = build_returns_table(df)
    masks = compute_masks(tab["date"], cfg)
    
    for H in args.horizons:
        print(f"Fitting GARCH(1,1) for horizon H={H}...")
        
        preds = fit_garch_rolling(
            tab, masks, args.eval_splits,
            horizon=H,
            min_train=args.min_train,
            refit_every=args.refit_every,
        )
        
        out_path = args.out_dir / f"{args.file_prefix}_h{H}.csv"
        preds.to_csv(out_path, index=False)
        counts = preds.groupby("split").size().to_dict()
        print(f"[garch] H={H}: wrote {len(preds):,} rows to {out_path} {counts}")


if __name__ == "__main__":
    main()

