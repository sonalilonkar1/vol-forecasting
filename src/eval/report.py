# src/eval/report.py
from pathlib import Path
import pandas as pd
from .metrics import summarize_predictions

# Model prediction directories
GARCH_PRED_DIR = Path("experiments/garch/preds")
NBEATS_PRED_DIR = Path("experiments/nbeats/preds")
RES_DIR = Path("experiments/results")


def parse_horizon(p: Path):
    """Extract horizon from filename like h1.csv, h5.csv, h22.csv"""
    name = p.stem  # e.g., h1, h5, h22
    if name.startswith("h"):
        try:
            return int(name[1:])
        except ValueError:
            return None
    return None


def main():
    rows = []
    
    # Process GARCH predictions
    if GARCH_PRED_DIR.exists():
        for p in sorted(GARCH_PRED_DIR.glob("h*.csv")):
            df = pd.read_csv(p, parse_dates=["date"])
            H = parse_horizon(p)
            summ = summarize_predictions(df)
            for _, r in summ.iterrows():
                rows.append(dict(
                    model="garch",
                    horizon=H,
                    split=r["split"],
                    rmse_log=r["rmse_log"],
                    qlike=r["qlike"],
                    n=r["n"],
                ))
    
    # Process N-BEATS predictions
    if NBEATS_PRED_DIR.exists():
        for p in sorted(NBEATS_PRED_DIR.glob("h*.csv")):
            df = pd.read_csv(p, parse_dates=["date"])
            H = parse_horizon(p)
            summ = summarize_predictions(df)
            for _, r in summ.iterrows():
                rows.append(dict(
                    model="nbeats",
                    horizon=H,
                    split=r["split"],
                    rmse_log=r["rmse_log"],
                    qlike=r["qlike"],
                    n=r["n"],
                ))
    
    if not rows:
        print("No prediction files found!")
        return
    
    rep = pd.DataFrame(rows).sort_values(["model", "horizon", "split"])
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / "summary.csv"
    rep.to_csv(out, index=False)
    print(f"Wrote {out} with {len(rep)} rows")


if __name__ == "__main__":
    main()
