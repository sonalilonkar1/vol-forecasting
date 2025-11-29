# src/models/har_rv.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

DEFAULT_TFT_DATASET = Path("data/tft_ready_dataset.csv")
DEFAULT_TFT_PARTS   = [Path("data/tft_ready_train.csv"), Path("data/tft_ready_val.csv"), Path("data/tft_ready_test.csv")]
DEFAULT_FEATURES    = Path("data/processed/features.parquet")

# CLI argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("HAR(-X) baselines with strict t-1 features and YAML splits")
    p.add_argument("--source", type=Path, default=None,
                   help="CSV/Parquet feature table. Defaults to TFT-ready files or data/processed/features.parquet")
    p.add_argument("--horizons", type=int, nargs="+", default=[1], help="Forecast horizons (e.g., 1 5 22)")
    p.add_argument("--eval-splits", nargs="+", choices=["train","val","test"], default=["val","test"])
    p.add_argument("--splits-config", type=Path, default=Path("configs/splits.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/preds"))
    p.add_argument("--min-train", type=int, default=100)
    p.add_argument("--harx", action="store_true", help="Include exogenous: lagged VIX z, calendar dummies if available")
    p.add_argument("--target", choices=["point","agg"], default="point",
                   help="point = log RV at t+h; agg = mean of next h days of log RV")
    return p.parse_args()

# Data loading helpers
def resolve_source(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        if not explicit.exists(): raise FileNotFoundError(explicit)
        return [explicit]
    if DEFAULT_TFT_DATASET.exists(): return [DEFAULT_TFT_DATASET]
    parts = [p for p in DEFAULT_TFT_PARTS if p.exists()]
    if parts: return parts
    if DEFAULT_FEATURES.exists(): return [DEFAULT_FEATURES]
    raise FileNotFoundError("No feature table found; provide --source or create TFT-ready/features.parquet")

def load_feature_table(paths: Sequence[Path]) -> pd.DataFrame:
    frames=[]
    for p in paths:
        if p.suffix.lower()==".csv":
            frames.append(pd.read_csv(p, parse_dates=["date"]))
        elif p.suffix.lower() in {".parquet",".pq"}:
            frames.append(pd.read_parquet(p))
        else:
            raise ValueError(f"Unsupported type: {p}")
    df = pd.concat(frames, ignore_index=True)
    df["date"]=pd.to_datetime(df["date"])
    if "asset" not in df.columns:
        if "ticker" in df.columns: df=df.rename(columns={"ticker":"asset"})
        else: raise ValueError("Need 'asset' or 'ticker'")
    return (df.sort_values(["asset","date"])
              .drop_duplicates(["asset","date"])
              .reset_index(drop=True))

def pick_logvol_col(df: pd.DataFrame) -> str:
    if "logvol_t" in df.columns: return "logvol_t"
    if "log_rv"   in df.columns: return "log_rv"
    raise ValueError("Need 'logvol_t' (TFT-ready) or 'log_rv' (builder)")

# Split handling (supports both YAML styles)
def load_splits_config(path: Path) -> dict:
    with open(path,"r") as f: return yaml.safe_load(f)

def _parse_ranges(cfg: dict) -> Tuple[pd.Timestamp, ...]:
    if "train" in cfg and isinstance(cfg["train"], (list,tuple)):
        tr_s,tr_e = pd.to_datetime(cfg["train"][0]), pd.to_datetime(cfg["train"][1])
        va_s,va_e = pd.to_datetime(cfg["val"][0])  , pd.to_datetime(cfg["val"][1])
        te_s,te_e = pd.to_datetime(cfg["test"][0]) , pd.to_datetime(cfg["test"][1])
        emb       = int(cfg.get("embargo_days",0))
        return tr_s,tr_e,va_s,va_e,te_s,te_e,emb
    tr_s=pd.to_datetime(cfg["train_start"]); tr_e=pd.to_datetime(cfg["train_end"])
    va_s=pd.to_datetime(cfg["val_start"])  ; va_e=pd.to_datetime(cfg["val_end"])
    te_s=pd.to_datetime(cfg["test_start"]) ; te_e=pd.to_datetime(cfg["test_end"])
    emb = int(cfg.get("embargo_days",0))
    return tr_s,tr_e,va_s,va_e,te_s,te_e,emb

def compute_masks(dates: pd.Series, cfg: dict) -> Dict[str,pd.Series]:
    tr_s,tr_e,va_s,va_e,te_s,te_e,emb = _parse_ranges(cfg)
    dates = pd.to_datetime(dates)
    tr_raw = (dates>=tr_s)&(dates<=tr_e)
    tr_eff_end = dates[tr_raw].max()-pd.Timedelta(days=emb) if tr_raw.any() else tr_e
    if tr_eff_end<tr_s: tr_eff_end=tr_s
    m = {
        "train": (dates>=tr_s)&(dates<=tr_eff_end),
        "val"  : (dates>=va_s)&(dates<=va_e),
        "test" : (dates>=te_s)&(dates<=te_e),
    }
    return {k:v.fillna(False) for k,v in m.items()}

# Build the HAR/HARX feature table
def build_har_table(df: pd.DataFrame, horizon: int, logcol: str, target_mode: str, use_harx: bool) -> tuple[pd.DataFrame, list[str]]:
    tab = df[["date","asset",logcol]].copy()

    # Future target (either TFT-provided or built on the fly)
    tgt = f"target_logvol_t+{horizon}"
    if tgt in df.columns:
        tab[tgt]=df[tgt]
    else:
        if target_mode=="point":
            tab[tgt]=tab.groupby("asset")[logcol].shift(-horizon)
        else: # aggregated mean of next h days (start at t+1)
            tab[tgt]=(tab.groupby("asset")[logcol]
                        .shift(-1).rolling(horizon, min_periods=horizon).mean())

    # Core HAR regressors (always anchored on t-1)
    grp = tab.groupby("asset")[logcol]
    tab["har_d"]=grp.shift(1)
    tab["har_w"]=grp.transform(lambda s: s.shift(1).rolling(5,  min_periods=5 ).mean())
    tab["har_m"]=grp.transform(lambda s: s.shift(1).rolling(22, min_periods=22).mean())
    feats = ["har_d","har_w","har_m"]

    # Optional HARX block: lagged VIX signals + calendar flags if present
    if use_harx:
        # VIX z (expected columns: vix_z or vix_lvl)
        if "vix_z" in df.columns:
            vix = (df[["date","asset","vix_z"]]
                    .rename(columns={"vix_z":"vix_z_lag"}))
            vix["vix_z_lag"]=vix.groupby("asset")["vix_z_lag"].shift(1)
            tab = tab.merge(vix, on=["date","asset"], how="left")
            feats.append("vix_z_lag")
        elif "vix_lvl" in df.columns:
            v = df[["date","asset","vix_lvl"]].rename(columns={"vix_lvl":"vix_lvl_lag"})
            v["vix_lvl_lag"]=v.groupby("asset")["vix_lvl_lag"].shift(1)
            tab = tab.merge(v, on=["date","asset"], how="left")
            feats.append("vix_lvl_lag")

        # calendar flags (shift to t-1 to be safe)
        for cal in ["is_month_end","is_opex","weekday"]:
            if cal in df.columns:
                col = f"{cal}_lag"
                tab[col] = df[cal]
                tab[col] = tab.groupby("asset")[col].shift(1)
                feats.append(col)

    req = feats+[tgt]
    tab = tab.dropna(subset=req).reset_index(drop=True)
    tab["y_true_logrv"]=tab[tgt]
    tab["y_true_rv"]=np.exp(tab["y_true_logrv"].clip(-50,50))
    return tab, feats

# Fit/predict loop
def fit_har_ols(tab: pd.DataFrame, feats: list[str], masks: Dict[str,pd.Series],
                eval_splits: Iterable[str], min_train: int, horizon: int) -> pd.DataFrame:
    rows=[]; skipped=[]
    for a,g in tab.groupby("asset"):
        idx = g.index
        mtr = masks["train"].reindex(idx).fillna(False)
        if mtr.sum()<min_train:
            skipped.append(a); continue
        Xtr = sm.add_constant(g.loc[mtr, feats], has_constant="add")
        ytr = g.loc[mtr, "y_true_logrv"]
        model = sm.OLS(ytr, Xtr).fit()

        for sp in eval_splits:
            m = masks[sp].reindex(idx).fillna(False)
            m &= g[feats].notna().all(axis=1) & g["y_true_logrv"].notna()
            if not m.any(): continue
            Xoos= sm.add_constant(g.loc[m, feats], has_constant="add")
            yhat= model.predict(Xoos)
            part=pd.DataFrame({
                "date": g.loc[m,"date"].values,
                "asset": a,
                "y_true_logrv": g.loc[m,"y_true_logrv"].values,
                "y_true_rv": g.loc[m,"y_true_rv"].values,
                "yhat_logrv": yhat.values,
                "yhat_rv": np.exp(np.clip(yhat.values,-50,50)),
                "model": "HARX-OLS" if any(c.startswith("vix_") or c.endswith("_lag") for c in feats if c not in ["har_d","har_w","har_m"]) else "HAR-OLS",
                "horizon": horizon,
                "split": sp,
            })
            rows.append(part)
    if not rows: raise ValueError("No predictions produced; check min-train or data coverage")
    pred=(pd.concat(rows, ignore_index=True)
            .sort_values(["date","asset"]).reset_index(drop=True))
    if skipped:
        print(f"Skipped {len(skipped)} assets (<{min_train} train rows): {', '.join(skipped)}")
    return pred

# Main entry point
def main():
    args = parse_args()
    df   = load_feature_table(resolve_source(args.source))
    logc = pick_logvol_col(df)
    cfg  = load_splits_config(args.splits_config)
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for H in args.horizons:
        tab, feats = build_har_table(df, horizon=H, logcol=logc, target_mode=args.target, use_harx=args.harx)
        masks = compute_masks(tab["date"], cfg)
        preds = fit_har_ols(tab, feats, masks, args.eval_splits, args.min_train, horizon=H)
        out   = args.out_dir / f"{'harx' if args.harx else 'har'}_h{H}.csv"
        preds.to_csv(out, index=False)
        counts = preds.groupby("split").size().to_dict()
        print(f"H={H}: wrote {len(preds):,} rows to {out} {counts}")

if __name__ == "__main__":
    main()
