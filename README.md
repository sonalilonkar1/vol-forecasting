# Volatility-Aware Portfolio Rebalancing

Predict future volatility, feed the signals into a cost-aware inverse-volatility allocator, and inspect the resulting portfolio diagnostics. The repo is organized around three building blocks:

1. **Features** – leakage-safe HAR inputs derived from TFT-ready CSVs or the builder parquet.
2. **Models** – HAR/HARX regressions plus a simple neural-net baseline that all emit the same prediction schema.
3. **Backtests** – inverse-vol targeting with transaction costs, no-trade bands, and vol caps.

## Environment setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer conda:

```bash
conda create -n vol-forecasting python=3.10 -y
conda activate vol-forecasting
python -m pip install -r requirements.txt
```

## Data/layout recap

- `data/tft_ready_dataset.csv` (or *_train/val/test CSVs) – preferred input. The scripts fall back to `data/processed/features.parquet` if TFT artifacts are missing. **Note:** Oxford–Man realized-volatility feeds were discontinued, so these panels now come from Yahoo Finance adjusted prices/returns via `src/data/fetch_etf_prices.py` + `src/data/make_rv_from_daily.py` (we square daily returns as the realized-vol proxy).
- `experiments/preds/` – model predictions, one CSV per horizon.
- `experiments/results/` – backtest outputs.
- `configs/splits.yaml` – time windows plus optional `assets` filter and `embargo_days`.
- `configs/backtest.yaml` – portfolio knobs (`target_vol`, `cost_bps`, `no_trade_pp`, etc.).
- `data/processed/returns.csv` – realized one-day returns (`date,asset,ret_1d`) feeding the P&L calculations. Build it directly from the TFT panel with `python -m src.data.build_returns` (see workflow below).

## HAR baseline (Phase B)

The CLI mirrors the refined notebook logic: strict t-1 features, split-aware training, and multi-horizon exports.

```bash
# From repo root
python -m src.models.har_rv --horizons 1 5 22 --eval-splits val test

# pooled Ridge variant
python -m src.models.har_rv --estimator ridge --ridge-alpha 0.5 \
	--horizons 1 5 22 --eval-splits val test

# or use the refined variant with identical flags
python -m src.models.har_rv_refined --horizons 1 5 22 \
	--eval-splits val test \
	--min-train 120 \
	--source data/tft_ready_dataset.csv
```

Key options:

- `--source` – explicit CSV/parquet; otherwise auto-detects TFT-ready files, then `features.parquet`.
- `--horizons` – space-separated list (e.g., `1 5 22`).
- `--eval-splits` – which splits to export (`train`, `val`, `test`).
- `--splits-config` – YAML describing time windows.
- `--out-dir` – destination directory (default `experiments/preds`).
- `--min-train` – minimum train rows per asset before fitting HAR.
- `--estimator` – choose `ols` (per-asset regressions) or `ridge` (pooled/sklearn Ridge).
- `--ridge-alpha` – regularization strength when the Ridge estimator is active.
- `--harx` – extend the feature set with VIX/calendar lags when those columns exist.

Each horizon writes `experiments/preds/har_h{H}.csv` (or `har_ridge_h{H}.csv` / `harx_ridge_h{H}.csv` when Ridge is selected) that respects the standard contract:

```
date,asset,y_true_logrv,y_true_rv,yhat_logrv,yhat_rv,model,horizon,split
```

## GARCH(1,1) baseline

Drop-in ARCH-style benchmark powered by the `arch` package. Set the output directory/prefix so files land next to the other models.

```bash
python -m src.models.garch \
	--source data/tft_ready_dataset.csv \
	--horizons 1 5 22 \
	--eval-splits val test \
	--splits-config configs/splits.yaml \
	--min-train 252 \
	--refit-every 22 \
	--out-dir experiments/preds \
	--file-prefix garch
```

Key knobs:

- `--min-train` – minimum per-asset observations before the first fit (keep ≥ 1 trading year for stability).
- `--refit-every` – rolling refit cadence in trading days (e.g., 22 for monthly, 66 for quarterly).
- `--out-dir` / `--file-prefix` – control the final filepath (e.g., `experiments/preds/garch_h1.csv`).

Each horizon emits `<prefix>_h{H}.csv` with the shared column contract, so you can immediately run `python src/backtest/run.py --pred-path experiments/preds/garch_h1.csv ...`.

## N-BEATS baseline

Multi-stack fully connected forecaster that consumes lagged log-volatility windows. Use CPU (`--device cpu`) unless your PyTorch install has CUDA support.

```bash
python -m src.models.nbeats \
	--source data/tft_ready_dataset.csv \
	--horizons 1 5 22 \
	--lookback 90 \
	--hidden-dim 128 \
	--num-stacks 3 \
	--num-blocks 2 \
	--theta-dim 8 \
	--dropout 0.1 \
	--epochs 50 \
	--batch-size 512 \
	--lr 1e-3 \
	--weight-decay 1e-4 \
	--eval-splits val test \
	--min-train 500 \
	--device cpu \
	--out-dir experiments/preds \
	--file-prefix nbeats
```

Key knobs:

- `--lookback` – sliding window length for each training sample.
- `--hidden-dim`, `--num-stacks`, `--num-blocks`, `--theta-dim`, `--dropout` – capacity/regularization.
- `--device` – `cpu` or `cuda:<id>` depending on your torch build.
- `--out-dir` / `--file-prefix` – match the shared predictions folder just like other models.

Outputs drop into `experiments/preds/nbeats_h{H}.csv` (or whatever prefix you chose) with the standard schema.

## Simple MLP baseline

When you want a quick non-linear reference without leaving pandas/torch, the lightweight MLP leverages the same HAR feature table and produces identical CSVs (`mlp_h{H}.csv`).

```bash
python -m src.models.simple_mlp --horizons 1 5 22 --eval-splits val test \
	--hidden-dim 64 --num-layers 2 --epochs 50

# HARX-style exogenous inputs + GPU device
python -m src.models.simple_mlp --harx --device cuda:0 --horizons 1 5 22
```

Key knobs:

- `--hidden-dim`, `--num-layers`, `--dropout` – architecture choices.
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay` – training loop settings.
- `--harx` – include lagged VIX/calendar features if present (mirrors HARX inputs).
- `--min-train` – guardrail to ensure enough in-sample rows.
- `--device` – torch device string (`cpu`, `cuda:0`, etc.).

Outputs follow the same column contract, so downstream backtests/reports work without modification.

## Gradient-Boosted Trees (XGBoost)

Tree-based baseline that mirrors the HAR CLI contract but fits a pooled XGBoost regressor with strict t-1 features. Uses TFT-ready CSVs (or `features.parquet`) and writes `gbt_xgb_h{H}.csv` per horizon.

```bash
python src/models/gbt.py \
	--horizons 1 5 22 \
	--eval-splits val test \
	--splits-config configs/splits.yaml \
	--out-dir experiments/preds \
	--n-estimators 2000 \
	--learning-rate 0.03 \
	--max-depth 5 \
	--subsample 0.9 \
	--colsample-bytree 0.9 \
	--lambda_ 1.0 \
	--early-stopping 50
```

Key knobs:

- `--source` – explicit feature table (otherwise auto-detect TFT files/parquet).
- `--n-estimators`, `--learning-rate`, `--max-depth` – standard XGBoost capacity controls.
- `--subsample`, `--colsample-bytree`, `--lambda_` – regularization/variance trade-offs.
- `--early-stopping` – patience on validation loss when a `val` split exists.
- `--seed` – ensures reproducible trees across dry runs.

Each export uses the shared schema so backtests/reporting scripts consume it with no changes. For quick smoke tests, target `--eval-splits val` and a scratch `--out-dir experiments/preds/tmp_gbt` to keep artifacts separate.

## GRU / LSTM baselines

Need a sequence model without jumping straight to TFT? The shared RNN CLI slides a fixed lookback window over the HAR table, encodes it with either a GRU or LSTM cell, and writes the same prediction schema (`gru_h{H}.csv`, `lstm_h{H}.csv`).

```bash
# GRU with 45-day window
python -m src.models.simple_rnn --cell-type gru --lookback 45 --horizons 1 5 22 --eval-splits val test

# LSTM variant with HARX features
python -m src.models.simple_rnn --cell-type lstm --harx --lookback 60 \
	--hidden-dim 128 --num-layers 2 --dropout 0.1
```

Key knobs:

- `--cell-type` – `gru` (default) or `lstm`.
- `--lookback` – number of trading days per sequence window (must cover the history you need).
- `--hidden-dim`, `--num-layers`, `--dropout` – encoder depth/regularization.
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--device` – training loop controls.
- `--harx` – include VIX/calendar features when available, just like the MLP/HARX flag.

Outputs mirror the HAR/MLP schema, so no downstream changes are required.

## Informer sequence model

Transformer-style encoder that operates on per-asset sliding windows (default 64 trading days) and trains a pooled model per horizon. Handles the same CLI arguments as the HAR stack plus sequence/training hyperparameters.

```bash
python src/models/informer.py \
	--horizons 1 5 22 \
	--seq-len 64 \
	--batch-size 64 \
	--max-epochs 50 \
	--patience 8 \
	--lr 1e-3 \
	--weight-decay 1e-4 \
	--d-ff 256 \
	--n-layers 3 \
	--n-heads 2 \
	--dropout 0.1 \
	--eval-splits val test \
	--out-dir experiments/preds
```

Highlights:

- `--seq-len` – lookback window; shorten (e.g., 32) for dry runs.
- `--max-epochs` / `--patience` – early-stopping behavior on validation RMSE.
- `--d-ff`, `--n-layers`, `--n-heads` – Informer depth/width controls; scale carefully to keep training tractable.
- `--seed` – routes through deterministic seeding for reproducible experiments.

Outputs land in `experiments/preds/informer_h{H}.csv` with the canonical column order, so downstream allocators treat it just like HAR/TFT results.

## Temporal Fusion Transformer

The Lightning-based TFT baseline trains all requested horizons jointly and now emits:

- Per-horizon CSVs (`tft_h1.csv`, `tft_h5.csv`, `tft_h22.csv`, …) for drop-in backtests.
- A multi-horizon file (`tft_all_horizons.csv` by default) packed with every split/horizon in one place for rapid analysis or custom slicing.

```bash
python -m src.models.tft --horizons 1 5 22 --eval-splits val test --device cuda:0 \
	--loss qlike --lookback 90 --hidden-dim 128 --combined-out experiments/preds/tft_full.csv
```

Useful flags mirror the other neural baselines (`--harx`, `--min-train`, `--splits-config`, etc.), with a few TFT-specific knobs:

- `--lookback`, `--hidden-dim`, `--lstm-layers`, `--dropout`, `--num-heads` – architectural capacity.
- `--loss` – `mse` or `qlike` objective.
- `--grad-clip`, `--batch-size`, `--epochs` – training stability controls.
- `--combined-out` – custom path for the multi-horizon export (omit to use the default under `out-dir`).

Every export respects the standard column contract so evaluation scripts pick them up automatically.

## Backtesting

The runner now consumes any prediction CSV and applies consistent portfolio settings.

```bash
python -m src.backtest.run --horizon 1

# custom prediction file and overrides
python -m src.backtest.run \
	--pred-path experiments/preds/har_h5.csv \
	--target-vol 0.12 \
	--cost-bps 5 \
	--no-trade-pp 3 \
	--returns-path data/processed/returns.csv
```

Options:

- `--pred-path` – explicit CSV (bypasses horizon inference).
- `--horizon` + `--pred-dir` – automatically load `pred_dir/har_h{H}.csv`.
- `--out-dir` – write results to a custom folder (default `experiments/results`).
- `--config` – reads defaults from `configs/backtest.yaml`.
- `--target-vol`, `--cost-bps`, `--no-trade-pp` – CLI overrides.
- `--returns-path` – realized return panel (`date,asset,ret_1d` or `ret`). Defaults to the config value then `data/processed/returns.csv`.
- `--return-col` – custom column name when the file doesn’t use `ret_1d`.
- `--weight-cap` – absolute per-asset cap applied after vol targeting (defaults to config value or 1.0).
- `--allocator` – choose `inverse_vol` or `risk_parity` sizing.
- `--risk-parity-window`, `--risk-parity-min-obs`, `--cov-shrink` – control the rolling covariance that powers risk parity (falls back to inverse-vol if history is short).
- `--rebalance-fraction`, `--max-turnover` – smooth towards the target weights and impose daily turnover caps to mimic no-trade/partial rebalance rules from the proposal.

Runner behavior:

- Renames `ticker → asset` if needed.
- Filters predictions to the `test` split when a `split` column exists.
- Validates the presence of `date`, `asset`, `yhat_rv`.
- Loads realized returns, aligns them to the prediction dates, and emits `gross_ret`, `net_ret`, `cum_gross`, `cum_net` alongside cost/turnover diagnostics.
- Applies costs on a one-day lag (weights chosen at the close of day *t* drive day *t+1* P&L), and reports the turnover/cost that actually hits that day’s net return.
- Saves outputs as `{pred_stem}_bt.csv` inside `out-dir` (e.g., `experiments/results/har_h1_bt.csv`).

### Batch mode (multiple models/horizons)

Use the helper CLI to run backtests for many prediction files at once, optionally filtering by model prefix (`har`, `harx`, `simple_rnn`, …) and/or horizon list:

```bash
# All prediction CSVs under experiments/preds
python -m src.backtest.run_batch

# Only HAR files with H in {1,5}
python -m src.backtest.run_batch --models har --horizons 1 5

# Compare HAR vs HARX at 22 days (dry run to preview)
python -m src.backtest.run_batch --models har harx --horizons 22 --dry-run

# Custom glob/pred directory
python -m src.backtest.run_batch --pred-dir other_preds --pattern "*_best.csv"

# Cost-sensitivity sweep (5/10/20 bps)
python -m src.backtest.run_batch --models har --horizons 1 5 \
	--cost-bps-grid 5 10 20 --allocator risk_parity
```

- `--models` – space-separated prefixes (case-insensitive). Default: all.
- `--horizons` – limit to certain `_h{H}` suffixes (integers).
- `--pattern` – glob applied inside `--pred-dir` before filtering (`*.csv` by default).
- `--dry-run` – print which files match without executing backtests.
- `--allocators` – run several allocators (e.g., `inverse_vol risk_parity`) in one pass. When multiple allocators are supplied, results are routed into per-allocator subfolders and file names get `_allocator_*` suffixes.
- `--cost-bps-grid` – run the selected files at multiple transaction-cost assumptions, automatically suffixing the output filenames (e.g., `_cost5p_bt.csv`).
- All allocator/turnover knobs available in the single-run CLI are mirrored here (`--allocator`, `--risk-parity-window`, `--rebalance-fraction`, etc.).

`make bt` now respects environment overrides, e.g. `make bt MODELS="har harx" H="1 5"`.

## Typical workflow

```bash
# 0. Materialize realized returns (once per dataset refresh)
python -m src.data.build_returns --source data/tft_ready_dataset.csv

# 1. Ensure data artifacts exist (TFT-ready CSVs or features.parquet)

# 2. Fit HAR baselines
python -m src.models.har_rv --horizons 1 5 22 --eval-splits val test

# optional: fit the neural baseline (shares schema with HAR)
python -m src.models.simple_mlp --horizons 1 5 22 --eval-splits val test

# 3. Backtest the desired horizon(s)
python -m src.backtest.run --horizon 1
python -m src.backtest.run --horizon 5

# 4. Inspect outputs under experiments/preds and experiments/results
```

From here you can plug in additional models (HARX, GARCH, TFT, etc.) as long as they emit the same prediction schema, keeping the downstream evaluation stack unchanged.

## Allocator & cost-sweep studies

The new allocator knobs make it easy to benchmark inverse-volatility vs. risk-parity sizing under different trading-cost assumptions. A typical experiment:

```bash
# Inverse-vol, 1-day horizon, three cost levels
python -m src.backtest.run_batch --models har --horizons 1 \
	--allocator inverse_vol --cost-bps-grid 5 10 20 \
	--out-dir experiments/results/inv_vol

# Risk parity (90-day rolling cov, partial rebalances), same costs
python -m src.backtest.run_batch --models har --horizons 1 \
	--allocator risk_parity --risk-parity-window 90 --risk-parity-min-obs 30 \
	--rebalance-fraction 0.5 --cost-bps-grid 5 10 20 \
	--out-dir experiments/results/risk_parity
```

Each run writes `{model}_h{H}_cost{X}p_bt.csv` inside the chosen folder. You can summarize them with a quick pandas script (or drop this into `python - <<'PY' ... PY`):

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
summary = pd.DataFrame(rows).sort_values(["allocator", "cost_bps"])
print(summary)
```

Latest HAR@H=1 results (test split):

| Allocator | Cost (bps) | Final Cum Net | Avg Turnover | Avg Cost |
|-----------|------------|---------------|--------------|----------|
| inverse_vol | 5 | 1.2989 | 0.0009 | 4.5e-7 |
| inverse_vol | 10 | 1.2982 | 0.0009 | 9.1e-7 |
| inverse_vol | 20 | 1.2967 | 0.0009 | 1.8e-6 |
| risk_parity | 5 | **1.4175** | 0.0180 | 9.0e-6 |
| risk_parity | 10 | 1.4016 | 0.0180 | 1.8e-5 |
| risk_parity | 20 | 1.3702 | 0.0180 | 3.6e-5 |

Risk parity trades more (≈20× turnover) but still delivers higher after-cost performance across all three cost levels and lowers the average predicted portfolio volatility (~5.4% vs. 6.6%). Use these folders as a template when running the same comparison for other models/horizons.

### Automation helpers

- **GBT sweeps:** `scripts/run_gbt_sweep.py` fans out hyperparameter grids for `src/models/gbt.py`, dropping each run into `experiments/preds/gbt_sweeps/<run_id>` and writing `sweep_summary.csv` with the hyperparameters, runtime, and produced files. Example:

	```bash
	PYTHONPATH=$PWD python scripts/run_gbt_sweep.py \
			--n-estimators 1000 2000 \
			--learning-rates 0.02 0.03 \
			--max-depths 4 5 \
			--horizons 1 5 22 \
			--out-dir experiments/preds/gbt_sweeps
	```

- **Allocator studies:** `src/backtest/run_batch.py` now accepts `--allocators inverse_vol risk_parity` plus a shared `--cost-bps-grid`. Results are organized by allocator (subdirectories + summary CSV), so you can compare allocator/cost combinations from a single command.
