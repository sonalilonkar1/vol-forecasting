# Forecasting & Backtest Runbook

This living document captures the exact commands, key parameters, and interpretation notes for every model and backtest we support. Extend each section whenever we add a new architecture or allocator experiment.

---

## 0. Prerequisites

1. **Environment**: `conda activate vol-forecasting` and install requirements if not already done.
   ```bash
   pip install -r requirements.txt
   ```
2. **Data assets**: Ensure the TFT-ready panels exist (produced via `python -m src.data.build_returns` + feature builders). Required files:
   - `data/tft_ready_dataset.csv`
   - `data/tft_ready_train.csv`, `..._val.csv`, `..._test.csv`
   - `configs/splits.yaml` describing train/val/test windows
3. **Folder hygiene**: Model predictions land in `experiments/preds/` (optionally subfolders like `dryrun/`), and cost-aware backtests land in `experiments/results/<allocator>/`.

---

## 1. HAR Family (HAR, HARX, HAR Ridge)

**Command template**
```bash
python -m src.models.har_rv \
  --horizon 1 \
  --model har      # har | harx | har_ridge \
  --alpha 0.0      # ridge penalty (only used when model=har_ridge) \
  --pred-path experiments/preds/har_h1.csv
```

**Key parameters**
- `--model`: switches between vanilla HAR (baseline), HARX (adds exogenous regressors like VIX/calendar), and HAR ridge (ℓ2-regularized). Comparing these shows how exogenous info or shrinkage affects RMSE/QLIKE.
- `--alpha`: ridge penalty strength; higher values trade bias for lower variance—watch net RMSE and stability in small samples.

**Outputs & metrics**
- CSV: `experiments/preds/har*_h{H}.csv` (columns `date, asset, yhat_rv, yhat_logrv, split`).
- Backtest with the generic command (section 6) to capture Sharpe, turnover, and cost drag.

---

## 2. Feed-Forward Baseline (Simple MLP)

**Command (dry-run example)**
```bash
python -m src.models.simple_mlp \
  --source data/tft_ready_dataset.csv \
  --splits-config configs/splits.yaml \
  --horizons 1 5 22 \
  --hidden-dim 64 \
  --num-layers 2 \
  --dropout 0.1 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --harx \
  --out-dir experiments/preds/
```

**Parameter meaning**
- `--hidden-dim`, `--num-layers`, `--dropout`: govern capacity vs. overfitting. Track validation QLIKE when scaling up.
- `--harx`: toggles exogenous features. Compare HAR vs. HARX variants to quantify incremental signal.
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`: training knobs—longer epochs improve fit but risk overfitting; monitor validation RMSE/QLIKE.

**Result interpretation**
- Inspect `split` column counts per horizon; ensure `min-train` threshold is satisfied.
- In backtests, MLP serves as the lightweight neural baseline—expect moderate lift over HAR but lower than sequence/TFT models.

---

## 3. Sequence Model (GRU/LSTM)

**Command (GRU dry-run)**
```bash
python -m src.models.simple_rnn \
  --source data/tft_ready_dataset.csv \
  --horizons 1 \
  --cell-type gru      # gru | lstm \
  --lookback 30 \
  --hidden-dim 64 \
  --num-layers 1 \
  --dropout 0.0 \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --out-dir experiments/preds/
```

**Parameter meaning**
- `--cell-type`: choose GRU for faster runs or LSTM when longer memory may help.
- `--lookback`: sequence window length in trading days. Longer lookbacks capture slower regimes but shrink training set; verify `min-train` warnings.
- `--dropout`: only effective with `num-layers > 1`; stabilizes deeper stacks.

**Metrics**
- Compare validation/test RMSE vs. MLP to isolate benefit of temporal dynamics.
- In backtests, GRU predictions typically reduce turnover volatility due to smoother signals.

---

## 4. Temporal Fusion Transformer (TFT)

**Command (CPU-friendly dry-run)**
```bash
python -m src.models.tft \
  --source data/tft_ready_dataset.csv \
  --horizons 1 5 \
  --lookback 30 \
  --hidden-dim 128 \
  --lstm-layers 2 \
  --dropout 0.1 \
  --num-heads 4 \
  --loss mse        # mse | qlike \
  --epochs 3 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cpu \
  --out-dir experiments/preds/dryrun
```

**Scaling for full runs**
- Increase `--epochs` (e.g., 30–50) and move to `--device cuda:0` when GPU is available.
- Switch `--loss qlike` to emphasize variance accuracy; watch QLIKE vs. RMSE trade-offs.

**Interpretation**
- TFT emits per-horizon CSVs plus an optional combined file. Check `trainer.fit` logs for `train_loss`/`val_loss` trends.
- Use feature importances/attention (not yet scripted) for regime analysis once primary metrics look good.

---

## 5. N-BEATS

**Command (baseline run)**
```bash
python -m src.models.nbeats \
  --source data/tft_ready_dataset.csv \
  --horizons 1 5 22 \
  --width 256 \
  --depth 4 \
  --num-blocks 3 \
  --epochs 50 \
  --batch-size 512 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --target-mode point \
  --out-dir experiments/preds/
```

**Parameter meaning**
- `--width`, `--depth`: control block capacity; higher widths fit complex patterns but may need stronger weight decay.
- `--num-blocks`: more residual stacks improve accuracy but extend training time.
- `--target-mode`: `point` forecasts log-RV at t+H; `avg` would use rolling averages if we experiment with smoother targets.

**Result notes**
- Compare against TFT to see how fully connected residual stacks fare without attention; in practice N-BEATS offers a competitive mid-tier baseline.

---

## 6. Cost-Aware Backtest

**Command template**
```bash
python src/backtest/run.py \
  --config configs/backtest.yaml \
  --pred-path experiments/preds/<model>_h1.csv \
  --allocator risk_parity   # or inverse_vol \
  --cost-bps 15 \
  --target-vol 0.10 \
  --no-trade-pp 5 \
  --weight-cap 1.0 \
  --returns-path data/processed/returns.csv
```

**Key parameters & effects**
- `--allocator`: `inverse_vol` (simpler) vs. `risk_parity` (needs covariance estimates). Use the same allocator when comparing models to isolate signal quality.
- `--cost-bps`: raises turnover penalty; evaluate sensitivity at 5/10/15/20 bps. Higher costs punish noisy forecasts more.
- `--no-trade-pp`: no-trade band in percentage points; larger bands lower turnover but can lag fast volatility swings.
- `--rebalance-fraction`, `--max-turnover`, `--weight-cap`: advanced knobs for smoother application; document any deviations when experimenting.

**Outputs**
- CSV saved to `experiments/results/<model>_h*.csv` (or allocator subfolders). Columns include `cum_net`, `turnover`, `cost`, `port_sigma`, enabling Sharpe/Sortino computation.

---

## 7. Reporting Metrics & Analysis

For every run:
1. **Forecast metrics**: Compute RMSE, RMSPE, QLIKE on validation/test splits. Scripts forthcoming; meanwhile, load the CSVs into notebooks for quick aggregation.
2. **Economic metrics**: From backtest CSVs, capture:
   - Final `cum_net` vs. `cum_gross`
   - Average turnover and cost
   - Realized volatility (`port_sigma`) vs. target
   - Sharpe/Sortino if computed externally
3. **Comparisons**: Always compare models under identical allocators, horizons, and cost assumptions to attribute changes to the forecaster, not execution settings.

---

## 8. Adding a New Model (Template)

When introducing another architecture:
1. **Clone the pattern** from existing models (e.g., `simple_mlp.py`): import `har_dataset.py` helpers (`load_base_panel`, `prepare_har_dataset`, `feature_tensor`, `target_tensor`) plus `TrainConfig`/`train_regressor`.
2. **CLI contract:** expose `--source`, `--splits-config`, `--out-dir`, `--horizons`, `--eval-splits`, `--harx`, training hyperparameters, and `--device`. Emit HAR-compatible columns (`date`, `asset`, `y_true_*`, `yhat_*`, `model`, `horizon`, `split`).
3. **Document here:** add a subsection with
  - Complete command example (dry-run + full)
  - Parameter explanations (what to tweak, trade-offs)
  - Expected behavior vs. existing baselines
4. **Integrate:**
  - Save predictions to `experiments/preds/<model>_h{H}.csv` (or subfolders).
  - Run `python src/backtest/run.py ...` for both `inverse_vol` and `risk_parity` with multiple `--cost-bps` values (5/10/15/20).
  - Update `experiments/results/summary.csv`, batch scripts, and Make targets so sweeps include the new model.
5. **Verify:**
  - `pytest tests/test_har_cli_smoke.py tests/test_report_smoke.py`
  - Execute one dry-run training command and a backtest to confirm end-to-end compatibility.

Keep this runbook updated immediately after adding the new section so the team has a single source of truth for experimentation.
