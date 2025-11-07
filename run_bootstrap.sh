#!/usr/bin/env bash
set -euo pipefail

# 2) Project skeleton
mkdir -p data/{raw,processed,meta} configs \
         src/{data,features,models,train,backtest,utils} \
         experiments reports/figs .github/workflows

# 3) Python venv
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3 first."; exit 1
fi
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip

# 4) Requirements
cat > requirements.txt <<'REQ'
pandas
numpy
scipy
scikit-learn
statsmodels
arch
torch
torchvision
torchaudio
pytorch-lightning
matplotlib
pyyaml
tqdm
REQ

# 5) .gitignore
cat >> .gitignore <<'IGN'
.venv/
__pycache__/
*.pyc
.DS_Store
# data & models (keep large/binary out of git)
data/raw/
data/processed/
checkpoints/
experiments/
IGN

# 6) Makefile (note: the recipe line must start with a TAB)
cat > Makefile <<'MK'
.PHONY: check
check:
	python -c "print('OK')"
MK

# 7) README
cat > README.md <<'MD'
# Volatility-Aware Portfolio Rebalancing
Two tracks run in parallel:
- A: Forecasting (features, HAR-RV/GARCH, TFT/N-BEATS)
- B: Portfolio & Backtesting (inverse-vol, risk parity, costs)

## Quick start
python -m pip install -r requirements.txt
make check
MD

# 8) Commit & push
git add .
git commit -m "bootstrap structure, env, ignores" || true
git push origin main || echo "Note: push may be blocked by branch protection or missing auth."

# 9) Create branches for both members
git checkout -b feature/forecasting-skeleton
git push -u origin feature/forecasting-skeleton || true

git checkout main
git checkout -b feature/portfolio-engine
git push -u origin feature/portfolio-engine || true

echo "✅ Done. Branches created:"
echo " - feature/forecasting-skeleton"
echo " - feature/portfolio-engine"
