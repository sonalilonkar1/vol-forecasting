# src/models/gbt_rv.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.preprocessing import StandardScaler

# Defaults aligned with har_rv.py
DEFAULT_TFT_DATASET = Path("data/tft_ready_dataset.csv")
DEFAULT_TFT_PARTS = [
    Path("data/tft_ready_train.csv"),
    Path("data/tft_ready_val.csv"),
    Path("data/tft_ready_test.csv"),
]
DEFAULT_FEATURES = Path("data/processed/features.parquet")


# CLI argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Gradient-Boosted Trees (XGBoost) baseline with strict t-1 features and YAML splits"
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "CSV/Parquet feature table. Defaults to TFT-ready files or "
            "data/processed/features.parquet"
        ),
    )
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 22],
        help="Forecast horizons (e.g., 1 5 22). Default: 1",
    )
    p.add_argument(
        "--eval-splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["val", "test"],
        help="Which splits to write predictions for. Default: val test",
    )
    p.add_argument(
        "--splits-config",
        type=Path,
        default=Path("configs/splits.yaml"),
        help="YAML file with train/val/test date ranges (same format as har_rv.py)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/preds"),
        help="Directory where prediction CSVs will be written",
    )

    # XGBoost hyperparameters
    p.add_argument("--n-estimators", type=int, default=2000,
                   help="Max number of boosting rounds")
    p.add_argument("--learning-rate", type=float, default=0.03,
                   help="Learning rate (eta)")
    p.add_argument("--max-depth", type=int, default=5,
                   help="Max depth of trees")
    p.add_argument("--subsample", type=float, default=0.9,
                   help="Row subsample ratio")
    p.add_argument("--colsample-bytree", type=float, default=0.9,
                   help="Column subsample ratio")
    p.add_argument("--lambda_", type=float, default=1.0,
                   help="L2 regularization strength (XGBoost lambda)")
    p.add_argument("--early-stopping", type=int, default=50,
                   help="Early stopping rounds on validation loss")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for XGBoost")
    return p.parse_args()


def resolve_source(explicit: Path | None) -> list[Path]:
    """Resolve which feature table(s) to load."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return [explicit]

    if DEFAULT_TFT_DATASET.exists():
        return [DEFAULT_TFT_DATASET]

    parts = [p for p in DEFAULT_TFT_PARTS if p.exists()]
    if parts:
        return parts

    if DEFAULT_FEATURES.exists():
        return [DEFAULT_FEATURES]

    raise FileNotFoundError(
        "No feature table found; provide --source or create TFT-ready/features.parquet"
    )


def load_feature_table(paths: Sequence[Path]) -> pd.DataFrame:
    """Load CSV/Parquet feature table and normalize to have 'asset' column."""
    frames = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            frames.append(pd.read_csv(p, parse_dates=["date"]))
        elif p.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(p))
        else:
            raise ValueError(f"Unsupported file type: {p}")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    if "asset" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "asset"})
        else:
            raise ValueError(
                "Need 'asset' or 'ticker' column in feature table")

    df = (
        df.sort_values(["asset", "date"])
          .drop_duplicates(["asset", "date"])
          .reset_index(drop=True)
    )
    return df


def load_splits_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_ranges(cfg: dict) -> Tuple[pd.Timestamp, ...]:
    if "train" in cfg and isinstance(cfg["train"], (list, tuple)):
        tr_s, tr_e = pd.to_datetime(
            cfg["train"][0]), pd.to_datetime(cfg["train"][1])
        va_s, va_e = pd.to_datetime(
            cfg["val"][0]), pd.to_datetime(cfg["val"][1])
        te_s, te_e = pd.to_datetime(
            cfg["test"][0]), pd.to_datetime(cfg["test"][1])
        emb = int(cfg.get("embargo_days", 0))
        return tr_s, tr_e, va_s, va_e, te_s, te_e, emb

    tr_s = pd.to_datetime(cfg["train_start"])
    tr_e = pd.to_datetime(cfg["train_end"])
    va_s = pd.to_datetime(cfg["val_start"])
    va_e = pd.to_datetime(cfg["val_end"])
    te_s = pd.to_datetime(cfg["test_start"])
    te_e = pd.to_datetime(cfg["test_end"])
    emb = int(cfg.get("embargo_days", 0))
    return tr_s, tr_e, va_s, va_e, te_s, te_e, emb


def compute_masks(dates: pd.Series, cfg: dict) -> Dict[str, pd.Series]:
    tr_s, tr_e, va_s, va_e, te_s, te_e, emb = _parse_ranges(cfg)
    dates = pd.to_datetime(dates)

    tr_raw = (dates >= tr_s) & (dates <= tr_e)
    tr_eff_end = dates[tr_raw].max(
    ) - pd.Timedelta(days=emb) if tr_raw.any() else tr_e
    if tr_eff_end < tr_s:
        tr_eff_end = tr_s

    m = {
        "train": (dates >= tr_s) & (dates <= tr_eff_end),
        "val":   (dates >= va_s) & (dates <= va_e),
        "test":  (dates >= te_s) & (dates <= te_e),
    }
    return {k: v.fillna(False) for k, v in m.items()}


def pick_logvol_col(df: pd.DataFrame) -> str:
    if "logvol_t" in df.columns:
        return "logvol_t"
    if "log_rv" in df.columns:
        return "log_rv"
    raise ValueError(
        "Need 'logvol_t' (TFT-ready) or 'log_rv' (builder) in feature table")


def build_gbt_table(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Prepare a table for GBT regression at a given horizon.

    IMPORTANT: to match HAR's composition (e.g. {'val': 1509, 'test': 3774} for H=1),
    we mirror the HAR RV row filtering:
      - require target_logvol_t+H to be non-NaN
      - require HAR regressors har_d, har_w, har_m to be non-NaN

    We don't *have* to use har_d/har_w/har_m as features, but it's fine (and usually helpful)
    to include them.
    """
    target_col = f"target_logvol_t+{horizon}"
    if target_col not in df.columns:
        raise ValueError(
            f"Missing target column {target_col} in feature table")

    tab = df.copy()

    # Build HAR-style regressors on the same log-vol column used in har_rv.py
    logcol = pick_logvol_col(tab)
    grp = tab.groupby("asset")[logcol]
    tab["har_d"] = grp.shift(1)
    tab["har_w"] = grp.transform(lambda s: s.shift(
        1).rolling(5,  min_periods=5).mean())
    tab["har_m"] = grp.transform(lambda s: s.shift(
        1).rolling(22, min_periods=22).mean())

    # HAR-style row requirement: all of these must be defined
    required = [target_col, "har_d", "har_w", "har_m"]
    tab = tab.dropna(subset=required).reset_index(drop=True)

    # Exclude ID + all targets from feature list
    exclude = {"date", "asset", "ticker"}
    for h in (1, 5, 22):
        col = f"target_logvol_t+{h}"
        if col in tab.columns:
            exclude.add(col)

    # Keep *all* remaining features (including har_d/w/m), but you could
    # drop them here if you wanted a "pure GBT on TFT features" baseline.
    features = [c for c in tab.columns if c not in exclude]

    return tab, features, target_col


def _model_label() -> str:
    # Short, similar style to "HAR-OLS" / "HAR-RIDGE"
    return "GBT-XGB"


def fit_gbt_xgb(
    tab: pd.DataFrame,
    features: list[str],
    target_col: str,
    masks: Dict[str, pd.Series],
    eval_splits: Iterable[str],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    lambda_: float,
    early_stopping: int,
    seed: int,
) -> pd.DataFrame:
    """
    Fit a single pooled XGBoost model and generate predictions for the requested splits.
    Returns a DataFrame with the same schema as har_h1.csv:
    [date, asset, y_true_logrv, y_true_rv, yhat_logrv, yhat_rv, model, horizon, split]
    (horizon will be filled by caller).
    """
    idx = tab.index

    # Base mask: rows with all features and target defined
    base_mask = tab[features].notna().all(axis=1) & tab[target_col].notna()
    train_mask = masks["train"].reindex(idx).fillna(False) & base_mask

    if not train_mask.any():
        raise ValueError(
            "No training rows found for GBT model. Check splits or features.")

    X_train = tab.loc[train_mask, features]
    y_train = tab.loc[train_mask, target_col].to_numpy(float)

    # Standardize features (as in the notebook)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Validation mask for early stopping (if any)
    val_mask = masks.get("val", pd.Series(False, index=idx)
                         ).reindex(idx).fillna(False) & base_mask
    has_val = val_mask.any()
    if has_val:
        X_val = tab.loc[val_mask, features]
        y_val = tab.loc[val_mask, target_col].to_numpy(float)
        X_val_s = scaler.transform(X_val)
    else:
        X_val_s = None
        y_val = None

    dtrain = xgb.DMatrix(X_train_s, label=y_train)
    watchlist = [("train", dtrain)]

    params = dict(
        objective="reg:squarederror",
        eta=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=lambda_,
        seed=seed,
    )

    if has_val:
        dval = xgb.DMatrix(X_val_s, label=y_val)
        evals = [(dtrain, "train"), (dval, "val")]
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
    else:
        # No validation split: train without early stopping
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )

    # Out-of-sample predictions for requested splits
    rows = []
    label = _model_label()

    for sp in eval_splits:
        mask = masks[sp].reindex(idx).fillna(False) & base_mask
        if not mask.any():
            continue

        X_split = tab.loc[mask, features]
        X_split_s = scaler.transform(X_split)
        dsplit = xgb.DMatrix(X_split_s)
        yhat = model.predict(dsplit)

        part = pd.DataFrame(
            {
                "date": tab.loc[mask, "date"].values,
                "asset": tab.loc[mask, "asset"].values,
                "y_true_logrv": tab.loc[mask, target_col].values,
                "y_true_rv": np.exp(
                    np.clip(tab.loc[mask, target_col].values, -50, 50)
                ),
                "yhat_logrv": yhat,
                "yhat_rv": np.exp(np.clip(yhat, -50, 50)),
                "model": label,
                "split": sp,
            }
        )
        rows.append(part)

    if not rows:
        raise ValueError(
            "No predictions produced; check eval-splits or feature coverage")
    return (
        pd.concat(rows, ignore_index=True)
          .sort_values(["date", "asset"])
          .reset_index(drop=True)
    )


def main():
    args = parse_args()

    # Load data
    df = load_feature_table(resolve_source(args.source))
    cfg = load_splits_config(args.splits_config)

    # Optional asset sub-selection if config specifies (mirrors har_rv.py behavior)
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_masks = compute_masks(df["date"], cfg)

    for H in args.horizons:
        tab, feats, target_col = build_gbt_table(df, horizon=H)

        # Reindex masks on the horizon-specific table (same idea as informer_rv)
        masks = {k: full_masks[k].reindex(tab.index).fillna(False)
                 for k in full_masks.keys()}

        preds = fit_gbt_xgb(
            tab=tab,
            features=feats,
            target_col=target_col,
            masks=masks,
            eval_splits=args.eval_splits,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            lambda_=args.lambda_,
            early_stopping=args.early_stopping,
            seed=args.seed,
        )

        preds["horizon"] = H
        preds = preds[
            [
                "date",
                "asset",
                "y_true_logrv",
                "y_true_rv",
                "yhat_logrv",
                "yhat_rv",
                "model",
                "horizon",
                "split",
            ]
        ]

        out = args.out_dir / f"gbt_xgb_h{H}.csv"
        preds.to_csv(out, index=False)

        counts = preds.groupby("split").size().to_dict()
        print(f"H={H}: wrote {len(preds):,} rows to {out} {counts}")


if __name__ == "__main__":
    main()
