.PHONY: check

check:
	python -c "print('OK')"

PY := python
MODELS ?=
H ?=
BTPATTERN ?=

features:
	$(PY) -m src.features.build

# HAR-RV baseline for H=1/5/22 (point-ahead targets)
har:
	$(PY) -m src.models.har_rv --horizons 1 5 22

# HARX = HAR with exogenous (lagged VIX + calendar) if present
harx:
	$(PY) -m src.models.har_rv --horizons 1 5 22 --harx --target point

# Simple MLP baseline (shares schema with HAR outputs)
mlp:
	$(PY) -m src.models.simple_mlp --horizons 1 5 22 --eval-splits val test

# Run backtests for any prediction CSVs (HAR/HARX/simple_rnn)
bt:
	$(PY) -m src.backtest.run_batch \
		$(if $(MODELS),--models $(MODELS),) \
		$(if $(H),--horizons $(H),) \
		$(if $(BTPATTERN),--pattern $(BTPATTERN),)

# Produce summary CSV of RMSE/QLIKE + final cum_net/turnover
report:
	$(PY) -m src.eval.report

# PnL backtests
pnl:
	$(PY) -m src.backtest.run_batch_pnl

# End-to-end
all: features har bt report
