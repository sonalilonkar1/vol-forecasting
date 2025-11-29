.PHONY: features har harx preds bt report all

PY := python

features:
	$(PY) -m src.features.build

# HAR-RV baseline for H=1/5/22 (point-ahead targets)
har:
	$(PY) -m src.models.har_rv --horizons 1 5 22

# HARX = HAR with exogenous (lagged VIX + calendar) if present
harx:
	$(PY) -m src.models.har_rv --horizons 1 5 22 --harx --target point

# Run backtests for any prediction CSVs (HAR/HARX/simple_rnn)
bt:
	$(PY) -m src.backtest.run_batch

# Produce summary CSV of RMSE/QLIKE + final cum_net/turnover
report:
	$(PY) -m src.eval.report

# End-to-end
all: features har bt report
