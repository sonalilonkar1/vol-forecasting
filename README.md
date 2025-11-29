# Volatility-Aware Portfolio Rebalancing

Predict future volatility, feed the signals into a cost-aware inverse-volatility allocator, and inspect the resulting portfolio diagnostics. The repo is organized around three building blocks:

1. **Features** – leakage-safe HAR inputs derived from TFT-ready CSVs or the builder parquet.
2. **Models** – currently a per-asset HAR-RV baseline with configurable horizons.
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

## Backtesting

The runner now consumes any prediction CSV and applies consistent portfolio settings.

```bash
python -m src.backtest.run --horizon 1

# custom prediction file and overrides
python -m src.backtest.run \
	--pred-path experiments/preds/har_h5.csv \
	--target-vol 0.12 \
	--cost-bps 5 \
	--no-trade-pp 3
```

Options:

- `--pred-path` – explicit CSV (bypasses horizon inference).
- `--horizon` + `--pred-dir` – automatically load `pred_dir/har_h{H}.csv`.
- `--out-dir` – write results to a custom folder (default `experiments/results`).
- `--config` – reads defaults from `configs/backtest.yaml`.
- `--target-vol`, `--cost-bps`, `--no-trade-pp` – CLI overrides.

Runner behavior:

- Renames `ticker → asset` if needed.
- Filters predictions to the `test` split when a `split` column exists.
- Validates the presence of `date`, `asset`, `yhat_rv`.
- Saves outputs as `{pred_stem}_bt.csv` inside `out-dir` (e.g., `experiments/results/har_h1_bt.csv`).

## Typical workflow

```bash
# 1. Ensure data artifacts exist (TFT-ready CSVs or features.parquet)

# 2. Fit HAR baselines
python -m src.models.har_rv --horizons 1 5 22 --eval-splits val test

# 3. Backtest the desired horizon(s)
python -m src.backtest.run --horizon 1
python -m src.backtest.run --horizon 5

# 4. Inspect outputs under experiments/preds and experiments/results
```

From here you can plug in additional models (HARX, GARCH, TFT, etc.) as long as they emit the same prediction schema, keeping the downstream evaluation stack unchanged.
