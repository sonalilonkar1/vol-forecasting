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

- `data/tft_ready_dataset.csv` (or *_train/val/test CSVs) – preferred input. The scripts fall back to `data/processed/features.parquet` if TFT artifacts are missing.
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

Each horizon writes `experiments/preds/har_h{H}.csv` that respects the standard contract:

```
date,asset,y_true_logrv,y_true_rv,yhat_logrv,yhat_rv,model,horizon,split
```

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
```

- `--models` – space-separated prefixes (case-insensitive). Default: all.
- `--horizons` – limit to certain `_h{H}` suffixes (integers).
- `--pattern` – glob applied inside `--pred-dir` before filtering (`*.csv` by default).
- `--dry-run` – print which files match without executing backtests.

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
