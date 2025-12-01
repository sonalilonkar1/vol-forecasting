from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..har_rv import (
    resolve_source,
    load_feature_table,
    pick_logvol_col,
    load_splits_config,
    compute_masks,
    build_har_table,
)


def build_multi_har_table(
    df: pd.DataFrame,
    logcol: str,
    horizons: List[int],
    use_harx: bool,
    target_mode: str = "point",
) -> tuple[pd.DataFrame, List[str]]:
    """Return a single feature table that carries targets for multiple horizons."""
    uniq_h = sorted({int(h) for h in horizons})
    tab = df[["date", "asset", logcol]].copy()

    grp = tab.groupby("asset")[logcol]
    tab["har_d"] = grp.shift(1)
    tab["har_w"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    tab["har_m"] = grp.transform(lambda s: s.shift(1).rolling(22, min_periods=22).mean())
    feats: List[str] = ["har_d", "har_w", "har_m"]

    if use_harx:
        if "vix_z" in df.columns:
            vix = df[["date", "asset", "vix_z"]].rename(columns={"vix_z": "vix_z_lag"})
            vix["vix_z_lag"] = vix.groupby("asset")["vix_z_lag"].shift(1)
            tab = tab.merge(vix, on=["date", "asset"], how="left")
            feats.append("vix_z_lag")
        elif "vix_lvl" in df.columns:
            vix = df[["date", "asset", "vix_lvl"]].rename(columns={"vix_lvl": "vix_lvl_lag"})
            vix["vix_lvl_lag"] = vix.groupby("asset")["vix_lvl_lag"].shift(1)
            tab = tab.merge(vix, on=["date", "asset"], how="left")
            feats.append("vix_lvl_lag")

        for cal in ["is_month_end", "is_opex", "weekday"]:
            if cal in df.columns:
                col = f"{cal}_lag"
                tab[col] = df[cal]
                tab[col] = tab.groupby("asset")[col].shift(1)
                feats.append(col)

    for horizon in uniq_h:
        tgt_col = f"target_logvol_t+{horizon}"
        if tgt_col in df.columns:
            target = df[tgt_col]
        else:
            if target_mode == "point":
                target = tab.groupby("asset")[logcol].shift(-horizon)
            else:
                target = (
                    tab.groupby("asset")[logcol]
                    .shift(-1)
                    .rolling(horizon, min_periods=horizon)
                    .mean()
                )
        y_col = f"y_true_logrv_h{horizon}"
        tab[y_col] = target
        tab[f"y_true_rv_h{horizon}"] = np.exp(np.clip(tab[y_col], -50, 50))

    required = feats + [f"y_true_logrv_h{h}" for h in uniq_h]
    tab = tab.dropna(subset=required).reset_index(drop=True)
    return tab, feats


@dataclass
class HarDataset:
    table: pd.DataFrame
    features: List[str]
    masks: Dict[str, pd.Series]


@dataclass
class SequenceBatch:
    features: torch.Tensor
    targets: torch.Tensor
    meta: pd.DataFrame


def load_base_panel(source: Optional[Path], splits_config: Path) -> tuple[pd.DataFrame, dict, str]:
    """Load the raw panel plus split configuration once per CLI invocation."""
    df = load_feature_table(resolve_source(source))
    cfg = load_splits_config(splits_config)
    if cfg.get("assets"):
        df = df[df["asset"].isin(cfg["assets"])].copy()
    logcol = pick_logvol_col(df)
    return df, cfg, logcol


def prepare_har_dataset(
    df: pd.DataFrame,
    logcol: str,
    cfg: dict,
    horizon: int,
    use_harx: bool,
    target_mode: str = "point",
) -> HarDataset:
    table, feats = build_har_table(
        df,
        horizon=horizon,
        logcol=logcol,
        target_mode=target_mode,
        use_harx=use_harx,
    )
    masks = compute_masks(table["date"], cfg)
    return HarDataset(table=table, features=feats, masks=masks)


def feature_tensor(dataset: HarDataset, mask: pd.Series, device: torch.device) -> torch.Tensor:
    frame = dataset.table.loc[mask, dataset.features]
    data = frame.to_numpy(dtype=np.float32)
    return torch.from_numpy(data).to(device)


def target_tensor(dataset: HarDataset, mask: pd.Series, device: torch.device) -> torch.Tensor:
    arr = dataset.table.loc[mask, "y_true_logrv"].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def meta_from_mask(dataset: HarDataset, mask: pd.Series) -> pd.DataFrame:
    cols = ["date", "asset", "y_true_logrv", "y_true_rv"]
    return dataset.table.loc[mask, cols].copy().reset_index(drop=True)


def build_sequence_batch(
    dataset: HarDataset,
    mask: pd.Series,
    lookback: int,
    device: torch.device,
    allow_empty: bool = False,
) -> Optional[SequenceBatch]:
    mask = mask.reindex(dataset.table.index).fillna(False)
    feats = dataset.features
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    meta_rows: List[dict] = []

    for asset, group in dataset.table.groupby("asset", sort=False):
        g = group.sort_values("date")
        g_mask = mask.loc[g.index].to_numpy(dtype=bool)
        feat_vals = g[feats].to_numpy(dtype=np.float32)
        y_log = g["y_true_logrv"].to_numpy(dtype=np.float32)
        y_rv = g["y_true_rv"].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        for idx in range(len(g)):
            if not g_mask[idx]:
                continue
            start = idx - lookback + 1
            if start < 0:
                continue
            seq = feat_vals[start : idx + 1]
            if seq.shape[0] != lookback:
                continue
            sequences.append(seq)
            targets.append(y_log[idx])
            meta_rows.append(
                {
                    "date": dates[idx],
                    "asset": asset,
                    "y_true_logrv": y_log[idx],
                    "y_true_rv": y_rv[idx],
                }
            )

    if not sequences:
        if allow_empty:
            return None
        raise ValueError(
            "No sequences constructed for the requested mask. Reduce --lookback or ensure enough history per asset."
        )

    X = torch.from_numpy(np.stack(sequences)).to(device)
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    meta = pd.DataFrame(meta_rows)
    return SequenceBatch(features=X, targets=y, meta=meta.reset_index(drop=True))
