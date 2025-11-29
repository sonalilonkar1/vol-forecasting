# src/eval/report.py
from pathlib import Path
import pandas as pd
from .metrics import summarize_predictions

PRED_DIR = Path("experiments/preds")
RES_DIR  = Path("experiments/results")

def parse_model_name(p: Path):
    name = p.stem  # e.g., har_h1, harx_h22, simple_rnn_h1
    if "_h" in name:
        model, horizon = name.split("_h", 1)
        try:
            H = int(horizon)
        except ValueError:
            H = None
    else:
        model, H = name, None
    return model, H

def safe_bt_summary(bt_path: Path) -> dict:
    df = pd.read_csv(bt_path, parse_dates=["date"])
    cols = set(df.columns)

    out = {
        "file": bt_path.name,
        "final_cum_net": None,
        "final_cum_gross": None,
        "avg_turnover": None,
        "avg_cost": None,
        "last_port_sigma": None,
    }

    if "turnover" in cols:
        out["avg_turnover"] = float(df["turnover"].mean())
    if "cost" in cols:
        out["avg_cost"] = float(df["cost"].mean())
    if "port_sigma" in cols:
        out["last_port_sigma"] = float(df["port_sigma"].iloc[-1])

    # Only set these if present (newer backtest-with-returns versions)
    if "cum_net" in cols:
        out["final_cum_net"] = float(df["cum_net"].iloc[-1])
    if "cum_gross" in cols:
        out["final_cum_gross"] = float(df["cum_gross"].iloc[-1])

    return out

def main():
    # prediction metrics summary
    rows = []
    for p in sorted(PRED_DIR.glob("*.csv")):
        df = pd.read_csv(p, parse_dates=["date"])
        model, H = parse_model_name(p)
        summ = summarize_predictions(df)
        for _, r in summ.iterrows():
            rows.append(
                dict(
                    model=model,
                    horizon=H,
                    split=r["split"],
                    rmse_log=r["rmse_log"],
                    qlike=r["qlike"],
                    n=r["n"],
                )
            )
    rep = pd.DataFrame(rows).sort_values(["model", "horizon", "split"])
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / "summary.csv"
    rep.to_csv(out, index=False)
    print(f"Wrote {out} with {len(rep)} rows")

    # backtest overview (robust to schema differences)
    bt_rows = []
    for bt in sorted(RES_DIR.glob("*_bt.csv")):
        bt_rows.append(safe_bt_summary(bt))
    if bt_rows:
        bt_out = RES_DIR / "bt_overview.csv"
        pd.DataFrame(bt_rows).to_csv(bt_out, index=False)
        print(f"Wrote {bt_out} with {len(bt_rows)} rows")
    else:
        print("[info] No backtest CSVs found under experiments/results")

if __name__ == "__main__":
    main()
