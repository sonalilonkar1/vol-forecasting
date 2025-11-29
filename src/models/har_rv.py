# src/models/har_rv.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml


# Defaults and source resolution (readable, not robotic)

DEFAULT_TFT_DATASET = Path("data/tft_ready_dataset.csv")
DEFAULT_TFT_PARTS = [
    Path("data/tft_ready_train.csv"),
    Path("data/tft_ready_val.csv"),
    Path("data/tft_ready_test.csv"),
]
DEFAULT_FEATURES = Path("data/processed/features.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HAR-RV baselines (per-asset OLS) with leakage-safe features and YAML splits."
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Explicit feature table (CSV or Parquet). If omitted, tries TFT-ready files, then data/processed/features.parquet.",
    )
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1],
        help="Forecast horizons in trading days (e.g., 1 5 22).",
    )
    p.add_argument(
        "--eval-splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["val", "test"],
        help="Which splits to export predictions for (default: val test).",
    )
    p.add_argument(
        "--splits-config",
        type=Path,
        default=Path("configs/splits.yaml"),
        help="YAML file describing train/val/test windows and optional embargo_days.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/preds"),
        help="Directory for prediction CSVs (one per horizon).",
    )
    p.add_argument(
        "--min-train",
        type=int,
        default=100,
        help="Minimum training rows per asset; assets with fewer are skipped.",
    )
    return p.parse_args()


def resolve_source(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Provided source {explicit} does not exist")
        return [explicit]
    if DEFAULT_TFT_DATASET.exists():
        return [DEFAULT_TFT_DATASET]
    tft_parts = [p for p in DEFAULT_TFT_PARTS if p.exists()]
    if tft_parts:
        return tft_parts
    if DEFAULT_FEATURES.exists():
        return [DEFAULT_FEATURES]
    raise FileNotFoundError(
        "No feature table found. Provide --source or create data/tft_ready_dataset.csv "
        "or data/processed/features.parquet."
    )


# Loading features (keeps whatever format you have handy)

def load_feature_table(source_paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in source_paths:
        if path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(path, parse_dates=["date"]))
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(path))
        else:
            raise ValueError(f"Unsupported file type: {path}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # normalize asset column
    if "asset" in df.columns:
        pass
    elif "ticker" in df.columns:
        df = df.rename(columns={"ticker": "asset"})
    else:
        raise ValueError("Expected a column named 'asset' or 'ticker'.")
    # keep the useful columns but don’t blow up if extra fields sneak in
    return df.sort_values(["asset", "date"]).drop_duplicates(["asset", "date"]).reset_index(drop=True)


def pick_logvol_column(df: pd.DataFrame) -> str:
    if "logvol_t" in df.columns:
        return "logvol_t"
    if "log_rv" in df.columns:
        return "log_rv"
    raise ValueError("Need a column named 'logvol_t' (TFT-ready) or 'log_rv' (builder features).")


# Split parsing (works with either the pair or explicit-date style)

def load_splits_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Split config not found: {path}")
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _parse_ranges(cfg: dict) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, int]:
    """
    Supports either:
      Format A (pairs):
        train: [YYYY-MM-DD, YYYY-MM-DD]
        val:   [YYYY-MM-DD, YYYY-MM-DD]
        test:  [YYYY-MM-DD, YYYY-MM-DD]
        embargo_days: 5
      Format B (explicit keys):
        train_start: YYYY-MM-DD
        train_end:   YYYY-MM-DD
        val_start:   YYYY-MM-DD
        val_end:     YYYY-MM-DD
        test_start:  YYYY-MM-DD
        test_end:    YYYY-MM-DD
        embargo_days: 5
    """
    if "train" in cfg and isinstance(cfg["train"], (list, tuple)):
        tr_s, tr_e = pd.to_datetime(cfg["train"][0]), pd.to_datetime(cfg["train"][1])
        va_s, va_e = pd.to_datetime(cfg["val"][0]),   pd.to_datetime(cfg["val"][1])
        te_s, te_e = pd.to_datetime(cfg["test"][0]),  pd.to_datetime(cfg["test"][1])
        emb = int(cfg.get("embargo_days", 0))
        return tr_s, tr_e, va_s, va_e, te_s, te_e, emb
    # else fall back to explicit keys
    tr_s = pd.to_datetime(cfg["train_start"]); tr_e = pd.to_datetime(cfg["train_end"])
    va_s = pd.to_datetime(cfg["val_start"]);   va_e = pd.to_datetime(cfg["val_end"])
    te_s = pd.to_datetime(cfg["test_start"]);  te_e = pd.to_datetime(cfg["test_end"])
    emb = int(cfg.get("embargo_days", 0))
    return tr_s, tr_e, va_s, va_e, te_s, te_e, emb


def compute_split_masks(dates: pd.Series, cfg: dict) -> Dict[str, pd.Series]:
    tr_s, tr_e, va_s, va_e, te_s, te_e, embargo = _parse_ranges(cfg)
    dates = pd.to_datetime(dates)
    tr_mask_raw = (dates >= tr_s) & (dates <= tr_e)
    if tr_mask_raw.any():
        tr_last = dates[tr_mask_raw].max()
        tr_end_eff = tr_last - pd.Timedelta(days=embargo)
        if tr_end_eff < tr_s:
            tr_end_eff = tr_s
    else:
        tr_end_eff = tr_e
    masks = {
        "train": (dates >= tr_s) & (dates <= tr_end_eff),
        "val":   (dates >= va_s) & (dates <= va_e),
        "test":  (dates >= te_s) & (dates <= te_e),
    }
    return {k: v.fillna(False) for k, v in masks.items()}


# Build the HAR table (strictly causal by construction)

def build_har_table(df: pd.DataFrame, horizon: int, logvol_col: str) -> tuple[pd.DataFrame, str]:
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer")
    tab = df[["date", "asset", logvol_col]].copy()

    tgt_col = f"target_logvol_t+{horizon}"
    # Prefer the TFT-ready future target; otherwise, build it by shifting forward
    if tgt_col in df.columns:
        tab[tgt_col] = df[tgt_col]
    else:
        tab[tgt_col] = tab.groupby("asset")[logvol_col].shift(-horizon)

    grouped = tab.groupby("asset")[logvol_col]
    # Every feature is anchored on t-1 information so we never leak the future
    tab["har_d"] = grouped.shift(1)
    tab["har_w"] = grouped.transform(lambda s: s.shift(1).rolling(5,  min_periods=5).mean())
    tab["har_m"] = grouped.transform(lambda s: s.shift(1).rolling(22, min_periods=22).mean())

    req = ["har_d", "har_w", "har_m", tgt_col]
    tab = tab.dropna(subset=req).reset_index(drop=True)
    tab["y_true_logrv"] = tab[tgt_col]
    tab["y_true_rv"] = np.exp(tab["y_true_logrv"].clip(-50, 50))
    return tab, tgt_col


# Fit a tiny OLS per asset and write the predictions

def fit_har_ols(
    tab: pd.DataFrame,
    horizon: int,
    masks: dict[str, pd.Series],
    eval_splits: Iterable[str],
    min_train: int,
) -> pd.DataFrame:
    feats = ["har_d", "har_w", "har_m"]
    rows = []
    skipped = []
    for asset, g in tab.groupby("asset"):
        idx = g.index
        train_mask = masks["train"].reindex(idx).fillna(False)
        if train_mask.sum() < min_train:
            skipped.append(asset)
            continue

        Xtr = sm.add_constant(g.loc[train_mask, feats], has_constant="add")
        ytr = g.loc[train_mask, "y_true_logrv"]
        model = sm.OLS(ytr, Xtr).fit()

        for split in eval_splits:
            mask = masks[split].reindex(idx).fillna(False)
            mask &= g[feats].notna().all(axis=1) & g["y_true_logrv"].notna()
            if not mask.any():
                continue
            Xoos = sm.add_constant(g.loc[mask, feats], has_constant="add")
            yhat = model.predict(Xoos)
            part = pd.DataFrame(
                {
                    "date": g.loc[mask, "date"].values,
                    "asset": asset,
                    "y_true_logrv": g.loc[mask, "y_true_logrv"].values,
                    "y_true_rv": g.loc[mask, "y_true_rv"].values,
                    "yhat_logrv": yhat.values,
                    "yhat_rv": np.exp(np.clip(yhat.values, -50, 50)),
                    "model": "HAR-RV(OLS)",
                    "horizon": horizon,
                    "split": split,
                }
            )
            rows.append(part)

    if not rows:
        raise ValueError("HAR fit produced no predictions. Check min-train threshold or input spans.")
    preds = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )
    if skipped:
        print(f"Skipped {len(skipped)} assets with <{min_train} train rows: {', '.join(skipped)}")
    return preds


# Main entry point

def main() -> None:
    args = parse_args()
    source_paths = resolve_source(args.source)
    df = load_feature_table(source_paths)
    logvol_col = pick_logvol_column(df)

    cfg = load_splits_config(args.splits_config)

    # Honor any asset allow-list in the config, whichever style it used
    assets_filter = cfg.get("assets")
    if assets_filter:
        df = df[df["asset"].isin(assets_filter)].copy()

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        har_table, _ = build_har_table(df, horizon, logvol_col)
        masks = compute_split_masks(har_table["date"], cfg)
        preds = fit_har_ols(har_table, horizon, masks, args.eval_splits, args.min_train)

        out_path = args.out_dir / f"har_h{horizon}.csv"
        preds.to_csv(out_path, index=False)
        split_counts = preds.groupby("split").size().to_dict()
        print(f"H={horizon}: wrote {len(preds):,} rows to {out_path} ({split_counts}).")


if __name__ == "__main__":
    main()
