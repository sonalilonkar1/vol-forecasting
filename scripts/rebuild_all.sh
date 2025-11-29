#!/usr/bin/env bash
set -euo pipefail

python -m src.models.har_rv --horizons 1 5 22 --eval-splits val test
python -m src.models.har_rv --estimator ridge --horizons 1 5 22 --eval-splits val test
python -m src.models.har_rv --harx --horizons 1 5 22 --eval-splits val test
python -m src.models.har_rv --harx --estimator ridge --horizons 1 5 22 --eval-splits val test
python -m src.models.simple_mlp --horizons 1 5 22 --eval-splits val test
python -m src.backtest.run_batch --models har har_ridge harx harx_ridge mlp --horizons 1 5 22
python -m src.eval.report