# Model Testing & Evaluation Guide

This playbook spells out how to exercise every forecasting model, what knobs to sweep, the artifacts to capture, and how to interpret the resulting portfolio diagnostics. It mirrors the evaluation promises in `proposal.md`:

1. **Common data/contract** – every model pulls from the same TFT-ready panel and writes the standard prediction CSV.
2. **Comparable baselines** – HAR/HARX, MLP, GRU/LSTM, TFT run across the same horizons (1, 5, 22).
3. **Portfolio-level evaluation** – net performance after trading costs is the yardstick, not just regression error.
4. **Statistical confidence** – Diebold-Mariano (DM) test on forecast errors plus bootstrap confidence intervals for the backtest KPIs.

---

## 1. Pre-run checklist

- Environment: activate `vol-forecasting` conda env and install `requirements.txt` (Lightning, torch, statsmodels, etc.).
- Data: ensure `data/tft_ready_dataset.csv` (or *_train/val/test) and `data/processed/returns.csv` exist. These now come from Yahoo Finance adjusted prices/returns because the Oxford–Man realized-vol library was discontinued (see `src/data/fetch_etf_prices.py` + `src/data/make_rv_from_daily.py`). Regenerate features via `python -m src.features.build` if the raw panel changed.
- Configs: confirm `configs/splits.yaml` and `configs/backtest.yaml` reflect the proposal’s train/val/test windows, embargo days, target vol, and cost assumptions.

---

## 2. Experiment matrix

| Model        | Key Flags                                  | Parameter Sweep                                                                 |
|--------------|--------------------------------------------|----------------------------------------------------------------------------------|
| HAR / HARX   | `--harx`, `--estimator`, `--ridge-alpha`   | Ridge α ∈ {0.1, 0.5, 1.0}; HARX on/off                                           |
| GARCH(1,1)   | `--refit-every`, `--min-train`, `--file-prefix` | Refit cadence ∈ {5, 22, 66}; min-train ∈ {150, 252}; horizons ∈ {1, 5, 22}  |
| Simple MLP   | `--hidden-dim`, `--num-layers`, `--dropout`| Hidden dim ∈ {32, 64, 128}; depth ∈ {1, 2}; dropout ∈ {0.0, 0.1}                 |
| GRU / LSTM   | `--cell-type`, `--lookback`, `--hidden-dim`| Lookback ∈ {30, 45, 60}; hidden dim ∈ {64, 128}; cell ∈ {gru, lstm}              |
| N-BEATS      | `--lookback`, `--hidden-dim`, `--num-stacks`, `--device` | Lookback ∈ {60, 90}; hidden dim ∈ {64, 128}; stacks ∈ {2, 3}; device ∈ {cpu, cuda:0} |
| TFT          | `--lookback`, `--hidden-dim`, `--num-heads`| Lookback ∈ {60, 90}; hidden dim ∈ {128, 256}; heads ∈ {4, 8}; dropout ∈ {0.1,0.2}|
| GBT (XGBoost)| `--n-estimators`, `--learning-rate`, `--max-depth`, `--subsample` | Trees ∈ {500, 1000, 2000}; learning rate ∈ {0.02, 0.03, 0.05}; depth ∈ {4, 5, 6}; subsample/colsample ∈ {0.7, 0.9} |
| Informer     | `--seq-len`, `--d-ff`, `--n-layers`, `--n-heads`, `--max-epochs` | Seq-len ∈ {32, 64}; FF dim ∈ {256, 384}; layers ∈ {2, 3, 4}; heads ∈ {2, 4}; patience ∈ {6, 10} |

> **Tip:** keep one hyperparameter sweep active at a time to isolate effects and limit GPU time.

---

## 3. Execution template (per sweep value)

1. **Train model**
   ```bash
   python -m src.models.<model_cli> --horizons 1 5 22 ...extra flags...
   ```
2. **Verify artifacts**
   ```bash
   ls experiments/preds/<model_prefix>_h*.csv
   ```
   Ensure file size > 0 and `split` column includes `val`/`test`.
3. **Run backtests**
   ```bash
   python -m src.backtest.run --pred-path experiments/preds/<stem>_h1.csv
   python -m src.backtest.run --pred-path experiments/preds/<stem>_h5.csv
   python -m src.backtest.run --pred-path experiments/preds/<stem>_h22.csv
   ```
4. **Capture metrics**: run `python scripts/log_backtest_metrics.py <path_to_bt_csv>` to append the summary block to `experiments/results/metrics_log.md` (created on first use) and archive each CSV (`*_bt.csv`).

```bash
python scripts/log_backtest_metrics.py experiments/results/har_h1_bt.csv \
   --log-path experiments/results/metrics_log.md
```

---

## 4. Logging results

Create/maintain `experiments/results/metrics_log.md` with one table per horizon (the file is generated automatically the first time you run `scripts/log_backtest_metrics.py`):

```
### Horizon 1
| Model              | MAE (val) | RMSE (val) | DM p-val vs HAR | Net CAGR | Net Sharpe | Max DD | Turnover | Costs (bps/day) |
|--------------------|-----------|------------|-----------------|----------|------------|--------|----------|-----------------|
| HAR OLS            | 0.132     | 0.201      | –               | 8.2%     | 1.05       | -12%   | 0.18     | 3.1             |
| HAR Ridge α=0.5    | 0.129     | 0.198      | 0.42            | 8.9%     | 1.11       | -11%   | 0.20     | 3.6             |
| TFT hidden=256     | 0.118     | 0.185      | **0.02**        | 11.7%    | 1.34       | -10%   | 0.21     | 3.9             |
```

Populate the error columns with the validation split of the prediction CSV. Use `python scripts/dm_test.py <har_file> <candidate_file> --loss mse_log --split val` (or your notebook) to compute DM statistics between each candidate and the HAR baseline for the same horizon. Store the resulting CSV under `experiments/results/dm_tests/` for traceability.

### Statistical confidence helpers

1. **Diebold–Mariano CLI** (`scripts/dm_test.py`)

   ```bash
   python scripts/dm_test.py \
      experiments/preds/har_h1.csv \
      experiments/preds/tft_h1.csv \
      --loss mse_log --split val \
      --out experiments/results/dm_tests/tft_vs_har_h1.csv
   ```

   - Supports `mse_log`, `mae_log`, and `qlike` losses.
   - Use `--group-by split asset` to surface per-split/asset p-values.
   - Adjust the Newey–West lag via `--max-lag` (default 5) if your horizon spacing changes.

2. **Backtest KPI bootstrap** (`scripts/bootstrap_backtest_ci.py`)

   ```bash
   python scripts/bootstrap_backtest_ci.py experiments/results/har_h1_bt.csv \
      --n-bootstrap 2000 --block-size 5 \
      --out-dir experiments/results/bootstrap
   ```

   - Emits `{stem}_bootstrap_ci.csv` with percentile intervals for Sharpe, CAGR, drawdown, turnover, and cost.
   - Increase `--block-size` for heavier serial correlation; bump `--n-bootstrap` for tighter intervals.
   - All outputs live under `experiments/results/bootstrap/` for direct inclusion in the report appendix.

---

## 5. What to inspect in the backtest output

Each `{stem}_bt.csv` includes daily `gross_ret`, `net_ret`, `cost`, `turnover`, and cumulative curves. Focus on:

1. **Net Sharpe & CAGR** – primary performance KPIs from the proposal.
2. **Max Drawdown** – ensure improvements are not just higher leverage.
3. **Turnover & Costs** – better forecasts should not require excessive trading; note when turnover spikes relative to HAR.
4. **Target-vol tracking** – verify realized vol stays near the `target_vol` from `configs/backtest.yaml`. Large deviations indicate unstable forecasts.
5. **Regime sensitivity** – overlay recession/COVID windows; the proposal expects robustness across regimes.

Plot helpers (`notebooks/analysis.ipynb` or a custom script) should chart cumulative returns and drawdowns for all candidates on the same axes.

---

## 6. Comparing models

1. **Forecast accuracy** (validation set): lower MAE/RMSE and statistically significant DM p-values (<0.05) indicate better raw predictions.
2. **Portfolio metrics**: prioritize higher net Sharpe subject to similar or lower drawdowns. If Sharpe gains come with unacceptable churn, call that out.
3. **Stability**: prefer models whose performance advantage persists across horizons and across multiple random seeds.
4. **Complexity vs. gain**: if a deep model delivers marginal improvement with 10× cost, document that trade-off per proposal guidelines.

---

## 7. Allocator comparisons & cost sweeps

Every material model sweep should include at least one allocator study so we can connect forecast skill to economic value under realistic trading frictions. Use the batch CLI to generate matched runs for inverse-volatility and risk-parity sizing (single command when passing `--allocators inverse_vol risk_parity`):

```bash
# Inverse-vol baseline (writes to experiments/results/inv_vol)
python -m src.backtest.run_batch --models <prefix> --horizons 1 5 22 \
   --allocator inverse_vol --cost-bps-grid 5 10 20 \
   --out-dir experiments/results/inv_vol

# Risk parity variant (90-day rolling covariance, partial rebalances)
python -m src.backtest.run_batch --models <prefix> --horizons 1 5 22 \
   --allocator risk_parity --risk-parity-window 90 --risk-parity-min-obs 30 \
   --rebalance-fraction 0.5 --cost-bps-grid 5 10 20 \
   --out-dir experiments/results/risk_parity
```

Guidelines:

1. Export at least the test split for each horizon before launching these runs.
2. Prefer the multi-allocator option: `python -m src.backtest.run_batch --models har --horizons 1 5 --allocators inverse_vol risk_parity --cost-bps-grid 5 10 20 --out-dir experiments/results/allocators --risk-parity-window 90 --risk-parity-min-obs 30 --rebalance-fraction 0.5`. The command creates allocator subfolders automatically and appends `_allocator_<name>` suffixes when needed.
3. Review the generated `experiments/results/allocators/allocator_summary.csv` (written automatically whenever multiple allocators/costs are run) for a quick table of `pred_file`, allocator, cost, and output path.
4. When you need additional diagnostics, summarize the per-run CSVs with a quick pandas script to capture final cumulative net, turnover, cost, and realized volatility for every cost bucket:

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
         "avg_port_sigma": df["port_sigma"].mean(),
      })
summary = pd.DataFrame(rows).sort_values(["allocator", "cost_bps"])
print(summary)
```

5. Record the summary table inside `experiments/results/metrics_log.md` (or a dedicated allocator section). Note especially how risk parity’s lower portfolio sigma and higher turnover interact with costs.
6. When writing the report, include both the gross and net cumulative curves plus turnover/cost diagnostics so readers see the trade-off clearly.

---

## 8. Checklist for reporting back to the proposal

- [ ] HAR vs HARX comparison across all horizons.
- [ ] Neural baselines (MLP, GRU/LSTM) with at least two lookback lengths.
- [ ] TFT runs with both 60-day and 90-day encoders.
- [ ] Backtests for every prediction file (net performance + cost metrics logged).
- [ ] DM tests versus HAR per horizon (via `scripts/dm_test.py`, saved under `experiments/results/dm_tests/`).
- [ ] Bootstrap confidence intervals for net Sharpe (>= 1,000 draws) saved under `experiments/results/bootstrap/` using `scripts/bootstrap_backtest_ci.py`.
- [ ] Summary narrative: which model is preferred, why, and under what conditions (aligns with proposal deliverables section).

Following this document ensures that every model evaluation is reproducible, comparable, and ready for inclusion in the final report.

---

## 9. Model-specific notes (new baselines)

### GARCH(1,1)

Run the classical volatility baseline with the shared output contract so downstream tooling sees `experiments/preds/garch_h*.csv`:

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

Checklist:

1. Confirm each `<prefix>_h{H}.csv` includes the `split` column before launching backtests.
2. Use the batch runner (`--models garch`) to sweep allocator/cost assumptions alongside HAR/HARX.
3. Note any “Skipped <asset>” warnings—reduce `--min-train` or widen the history if necessary.

### N-BEATS

CPU-friendly command (switch `--device cuda:0` only if your PyTorch build includes CUDA):

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

Guidance:

1. Verify the torch device before launching runs to avoid the “Torch not compiled with CUDA enabled” assertion.
2. Because training is stochastic, log the random seed (`--seed`) in your experiment tracker when comparing sweeps.
3. Backtests use the same `<prefix>_h{H}.csv` contract, so no special handling is required once the files land in `experiments/preds/`.

### Gradient-Boosted Trees (XGBoost)

XGBoost regression baseline that shares the HAR CLI contract. Fits one pooled tree ensemble per horizon with strict t-1 features and validation-driven early stopping.

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
   --early-stopping 50 \
   --seed 42
```

Checklist:

1. Confirm TFT-ready CSVs exist; the CLI auto-detects `*_train/val/test` parts or `features.parquet` when `--source` is omitted.
2. Capture validation RMSE/QLIKE from the output CSVs and log them alongside the tree hyperparameters.
3. Run per-horizon backtests with `python -m src.backtest.run --pred-path experiments/preds/gbt_xgb_h1.csv` (repeat for H=5,22) plus allocator/cost sweeps so comparisons stay consistent.
4. For grid searches, adjust `--n-estimators`, `--learning-rate`, and `--max-depth` before widening `--subsample`/`--colsample-bytree`; deeper/more numerous trees can overfit without early stopping.
5. For larger sweeps, prefer `PYTHONPATH=$PWD python scripts/run_gbt_sweep.py --n-estimators 1000 2000 --learning-rates 0.02 0.03 --max-depths 4 5 --out-dir experiments/preds/gbt_sweeps` to keep artifacts separated and capture a `sweep_summary.csv` manifest of every run.

### Informer

Informer-style transformer baseline that builds per-asset sliding windows (default 64 days) and trains a pooled model per horizon. Runs on CPU or CUDA and emits `informer_h{H}.csv` with the canonical column order.

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

Guidelines:

1. Shorten `--seq-len` (e.g., 32) and `--max-epochs`/`--patience` for quick smoke tests; restore the full settings for production sweeps.
2. Monitor the training log for `val_RMSE`/`val_QLIKE` to verify early stopping when progress stalls.
3. Backtest each horizon with the same allocator/cost assumptions used for other models so economic value comparisons remain apples-to-apples.
4. When running on GPU, keep an eye on memory; reduce `--batch-size` or `--d-ff` if you hit OOM errors.
