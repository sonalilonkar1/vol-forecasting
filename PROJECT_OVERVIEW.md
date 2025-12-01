# Volatility-Aware Portfolio Rebalancing – Project Overview

This document gives new teammates a one-stop, high-resolution map of the project: what we are building, how the pieces fit together, and where to find every major artifact in the repository.

---

## 1. Mission & High-Level Workflow

**Goal:** forecast 1-, 5-, and 22-day realized volatility for a small, liquid asset set and feed those forecasts into cost-aware inverse-volatility (and future risk-parity) allocators so portfolios can stay near a target risk level under realistic trading frictions.

**Pipeline stages**

1. **Data ingestion & curation** – Load Yahoo Finance adjusted prices/returns (our current proxy for realized volatility after Oxford–Man discontinued its feed), VIX proxies, and calendar features into a unified panel (`data/tft_ready_dataset.csv`).
2. **Feature engineering** – Derive leakage-safe HAR inputs (daily/weekly/monthly averages plus optional VIX/calendar lags) shared by all models.
3. **Model training** – Fit classical HAR/HARX/HAR+Ridge models plus neural baselines (MLP, GRU/LSTM, TFT). Every model writes standardized prediction CSVs.
4. **Evaluation** – Compute RMSE/QLIKE per split, run cost-aware inverse-volatility backtests with transaction costs, and summarize diagnostics.
5. **Reporting** – Aggregate prediction metrics/backtests into tables and figures for reports and proposal milestones.

Each stage is scriptable so experiments can be reproduced via CLI, `make`, or automation.

---

## 2. Repository Layout & Responsibilities

| Path | Purpose |
| --- | --- |
| `README.md` | Quickstart commands for environment setup, model/backtest CLIs, and end-to-end workflow summary. |
| `proposal.md` | Original CMPE-258 project plan detailing motivation, methodology, deliverables, and timeline. |
| `requirements.txt` | Python dependencies (PyTorch, Lightning, pandas, YAML, sklearn, etc.). |
| `configs/splits.yaml` | Time splits, embargo, and allowed assets; every modeling script references it for train/val/test masks. |
| `configs/backtest.yaml` | Portfolio defaults (`target_vol`, `cost_bps`, `no_trade_pp`, `weight_cap`, returns file). |
| `data/` | All raw, processed, and TFT-ready datasets (see §3). |
| `experiments/preds/` | Model prediction exports (`{model}_h{H}.csv`) following the standard schema. |
| `experiments/results/` | Evaluation artifacts (per-model RMSE/QLIKE tables, backtest equity curves & diagnostics). |
| `src/` | Source code organized by subpackages (data, features, models, backtest, eval, utils). Detailed map in §4. |
| `tests/` | Smoke tests ensuring CLI wiring works (`pytest`). |
| `scripts/rebuild_all.sh` | Convenience script to rebuild data → models → backtests in one shot. |
| `notebooks/` | Exploratory analysis notebooks referenced in the proposal (not part of the prod pipeline). |
| `reports/` | Placeholder for polished figures/tables; `reports/figs/` ready for generated plots. |

---

## 3. Data Assets & Preparation

| File/Folder | Description & usage |
| --- | --- |
| `data/raw/rv.csv`, `data/raw/vix.csv` | Supplier dumps used to generate the TFT-ready panel. |
| `data/tft_ready_dataset.csv` | Canonical panel storing `date, asset`, returns, realized vol, VIX proxies, calendar flags, etc. Scripts fall back to this file unless split-specific CSVs (`*_train/val/test`) are provided. |
| `data/processed/returns.csv` | Realized one-day returns built with `python -m src.data.build_returns`. Backtests rely on this file for P&L calculations. |
| `data/processed/toy_rv.csv`, `processed/features.parquet` | Toy/legacy artifacts retained for experimentation. |
| `data/code/*.ipynb` | Historical data-building notebooks (Dataset Builder, YFinance Proxies). |
| `configs/splits.yaml` | Defines train `[2010–2017]`, val `[2018–2019]`, test `[2020–2024]`, `embargo_days=5`, `assets=[SPY,TLT,GLD]`. All scripts reference these masks through `compute_masks`. |

**Key script** – `src/data/build_returns.py`
- Reads the TFT-ready panel, selects `date`, `asset`, `ret_1d`, and writes standardized returns.
- CLI flags let you override column names and output location.

---

## 4. Source Code Tour

### 4.1 Data & Features (`src/data`, `src/features`)
- `src/data/build_returns.py` – Generates the `ret_1d` panel for backtests (see above).
- `src/data/loaders.py` – Helpers for reading TFT-ready files and applying split filters.
- `src/features/build.py` – Placeholder for richer feature engineering (calendar z-scores, PCA, etc.).

### 4.2 Modeling Layer (`src/models`)
- `common/har_dataset.py` – Shared utilities for:
  - Loading base panels + splits (`load_base_panel`).
  - Building multi-horizon HAR feature tables (`build_multi_har_table`).
  - Constructing torch-ready tensors or sliding-window batches across assets.
- `common/train_loop.py` – Deterministic seeding and small helpers reused by neural models.
- `har_rv.py` – CLI for HAR/HARX baselines (OLS per asset or pooled Ridge). Handles feature creation, target shifts, split-specific training, and writes `har_h{H}.csv` files.
- `simple_mlp.py` – Lightweight PyTorch MLP baseline sharing HAR/HARX features with configurable depth/dropout. Outputs `mlp_h{H}.csv`.
- `simple_rnn.py` – GRU/LSTM sequence baseline over fixed-length windows (lookback configurable). Produces `gru_h{H}.csv` / `lstm_h{H}.csv` depending on `--cell-type`.
- `tft.py` – PyTorch Lightning implementation of a simplified Temporal Fusion Transformer:
  - Builds sliding-window datasets covering all requested horizons simultaneously.
  - Supports HAR vs HARX features, QLIKE vs MSE loss, GPU acceleration, and exports per-horizon CSVs (`tft_h{H}.csv`).

Every model writes the same schema: `date, asset, y_true_logrv, y_true_rv, yhat_logrv, yhat_rv, model, horizon, split` so downstream tooling stays uniform.

### 4.3 Backtesting (`src/backtest`)
- `engine.py` – Portfolio allocator module:
   - Supports inverse-volatility and risk-parity sizing, configurable rolling covariance windows, diagonal shrinkage, and automatic fallback when histories are short.
   - Implements no-trade bands, partial rebalances (`rebalance_fraction`), turnover caps, transaction costs, per-asset weight caps, and volatility targeting before emitting daily P&L diagnostics.
- `run.py` – Single backtest CLI that loads predictions (direct path or inferred by horizon), pulls defaults from `configs/backtest.yaml`, and saves `{model}_h{H}_bt.csv` to `experiments/results/` with the selected allocator settings.
- `run_batch.py` – Batch launcher (see README) to iterate over multiple model prefixes/horizons, mirror all allocator knobs, and optionally sweep transaction-cost assumptions via `--cost-bps-grid`.

### 4.4 Evaluation (`src/eval`)
- `metrics.py` – Implements RMSE on log-vol and QLIKE on variance targets, returning split-level summaries.
- `report.py` – Scans `experiments/preds/` to produce `experiments/results/summary.csv`, then aggregates every backtest CSV into `bt_overview.csv` (final cumulative returns, average turnover/costs, etc.).

### 4.5 Tests & Utilities
- `tests/test_har_cli_smoke.py`, `tests/test_report_smoke.py` – Basic pytest smoke tests confirming CLIs execute and produce expected columns.
- `run_bootstrap.sh`, `Makefile`, `.github/workflows/ci.yml` – Provide reproducible CI hooks and multi-step rebuilds for continuous testing.

---

## 5. Experiments & Artifacts

### Predictions (`experiments/preds/`)
- `har_*` – Baseline HAR regressions (`har`, `har_ridge`, `harx`, `harx_ridge`).
- `mlp_*` – Simple MLP baseline outputs.
- `gru_*`, `lstm_*` – Sequence model baselines.
- `tft_h1.csv` – Pilot TFT run (extend to other horizons as we scale experiments).

All files can be combined with `src/eval/report.py` or fed into backtests directly.

### Backtests (`experiments/results/`)
- `{model}_h{H}_bt.csv` – Time-series diagnostics per model/horizon combination (turnover, costs, gross/net returns, sigma estimates). Cost sweep runs append `_costXXp_bt.csv` suffixes.
- `summary.csv` – Table of RMSE/QLIKE per model, horizon, split.
- `bt_overview.csv` – Compressed view of final cumulative returns and average trading frictions across every model/horizon/cost scenario.
- `inv_vol/`, `risk_parity/` – Sub-folders that capture allocator-specific cost sweeps (e.g., HAR H=1 at 5/10/20 bps). Use these as templates for future allocator studies, and summarize them quickly with a short pandas snippet:

   ```python
   from pathlib import Path
   import pandas as pd
   rows = []
   for allocator in ["inv_vol", "risk_parity"]:
         for csv in (Path("experiments/results") / allocator).glob("*.csv"):
               df = pd.read_csv(csv)
               rows.append({
                     "allocator": allocator,
                     "file": csv.name,
                     "cost_bps": csv.stem.split("_cost")[1].split('p')[0],
                     "final_cum_net": df["cum_net"].iloc[-1],
                     "avg_turnover": df["turnover"].mean(),
                     "avg_cost": df["cost"].mean(),
               })
   print(pd.DataFrame(rows))
   ```

---

## 6. Running the Project End-to-End

1. **Install dependencies**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
2. **Build returns panel** (run once per data refresh)
   ```bash
   python -m src.data.build_returns --source data/tft_ready_dataset.csv
   ```
3. **Train models** (examples)
   ```bash
   python -m src.models.har_rv --horizons 1 5 22 --eval-splits val test
   python -m src.models.simple_mlp --horizons 1 5 22 --eval-splits val test --hidden-dim 64
   python -m src.models.tft --device cuda:0 --loss qlike --horizons 1 5 22
   ```
4. **Evaluate & summarize**
   ```bash
   python -m src.eval.report
   ```
5. **Backtest**
   ```bash
   python -m src.backtest.run --horizon 1
   python -m src.backtest.run_batch --models har mlp --horizons 1 5 22

   # Allocator + cost sensitivity example (writes into sub-folders below experiments/results/)
   python -m src.backtest.run_batch --models har --horizons 1 \
      --allocator inverse_vol --cost-bps-grid 5 10 20 \
      --out-dir experiments/results/inv_vol

   python -m src.backtest.run_batch --models har --horizons 1 \
      --allocator risk_parity --risk-parity-window 90 --risk-parity-min-obs 30 \
      --rebalance-fraction 0.5 --cost-bps-grid 5 10 20 \
      --out-dir experiments/results/risk_parity
   ```
6. **Inspect results** – Explore `experiments/preds/`, `experiments/results/summary.csv`, and the *_bt.csv files for performance diagnostics.

---

## 7. Roadmap & Open Items

- Add GARCH, N-BEATS, and Informer baselines referenced in `proposal.md`.
- Extend TFT exports to all horizons and document interpretability tooling (attention/variable importance plots).
- Stress-test the new risk-parity allocator with longer lookbacks, leverage caps, and decision-focused metrics from the proposal (extend beyond the current HAR H=1 experiment summarized in `experiments/results/{inv_vol,risk_parity}`).
- Integrate Diebold–Mariano significance tests, regime-wise breakdowns, and walk-forward re-fit scripts.
- Populate `reports/` with publication-ready tables/figures for final deliverables.

With this overview and the README quickstart, a new teammate should be able to reproduce existing experiments, extend the modeling set, and contribute to the remaining roadmap items confidently. Good luck, and reach out to the repo maintainers on Slack for environment credentials or data refresh procedures.
