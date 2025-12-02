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
4. **Capture metrics**: append the summary block printed by `src.backtest.run` to `experiments/results/metrics_log.md` (see next section) and store each CSV (`*_bt.csv`).

---

## 4. Logging results

Create/maintain `experiments/results/metrics_log.md` with one table per horizon:

```
### Horizon 1
| Model              | MAE (val) | RMSE (val) | DM p-val vs HAR | Net CAGR | Net Sharpe | Max DD | Turnover | Costs (bps/day) |
|--------------------|-----------|------------|-----------------|----------|------------|--------|----------|-----------------|
| HAR OLS            | 0.132     | 0.201      | –               | 8.2%     | 1.05       | -12%   | 0.18     | 3.1             |
| HAR Ridge α=0.5    | 0.129     | 0.198      | 0.42            | 8.9%     | 1.11       | -11%   | 0.20     | 3.6             |
| TFT hidden=256     | 0.118     | 0.185      | **0.02**        | 11.7%    | 1.34       | -10%   | 0.21     | 3.9             |
```

Populate the error columns with the validation split of the prediction CSV. Use `scripts/dm_test.py` (or your notebook) to compute DM statistics between each candidate and the HAR baseline for the same horizon.

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

Every material model sweep should include at least one allocator study so we can connect forecast skill to economic value under realistic trading frictions. Use the batch CLI to generate matched runs for inverse-volatility and risk-parity sizing:

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
2. Use separate folders (`inv_vol/`, `risk_parity/`, or more specific names) so summaries stay organized.
3. After the runs finish, summarize the outputs with a quick pandas script to capture final cumulative net, turnover, cost, and realized volatility for every cost bucket:

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

4. Record the summary table inside `experiments/results/metrics_log.md` (or a dedicated allocator section). Note especially how risk parity’s lower portfolio sigma and higher turnover interact with costs.
5. When writing the report, include both the gross and net cumulative curves plus turnover/cost diagnostics so readers see the trade-off clearly.

---

## 8. Checklist for reporting back to the proposal

- [ ] HAR vs HARX comparison across all horizons.
- [ ] Neural baselines (MLP, GRU/LSTM) with at least two lookback lengths.
- [ ] TFT runs with both 60-day and 90-day encoders.
- [ ] Backtests for every prediction file (net performance + cost metrics logged).
- [ ] DM tests versus HAR per horizon.
- [ ] Bootstrap confidence intervals for net Sharpe (>= 1,000 draws) saved under `experiments/results/bootstrap/`.
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
