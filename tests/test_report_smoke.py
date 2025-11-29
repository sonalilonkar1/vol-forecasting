from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import report


def _make_pred_dir(tmp_path: Path) -> Path:
    pred_dir = tmp_path / "experiments" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    preds = pd.DataFrame(
        {
            "date": dates,
            "asset": ["AAA", "AAA", "BBB", "BBB"],
            "y_true_logrv": np.log(np.linspace(0.05, 0.08, len(dates))),
            "y_true_rv": np.linspace(0.05, 0.08, len(dates)),
            "yhat_logrv": np.log(np.linspace(0.055, 0.082, len(dates))),
            "yhat_rv": np.linspace(0.055, 0.082, len(dates)),
            "split": ["val", "val", "test", "test"],
        }
    )
    preds.to_csv(pred_dir / "har_h1.csv", index=False)
    return pred_dir


def _make_results_dir(tmp_path: Path) -> Path:
    res_dir = tmp_path / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    bt = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-10", periods=3, freq="B"),
            "turnover": [0.1, 0.2, 0.15],
            "cost": [0.0005, 0.0007, 0.0006],
            "port_sigma": [0.1, 0.11, 0.12],
            "gross_ret": [0.001, -0.0005, 0.0008],
            "net_ret": [0.0005, -0.001, 0.0002],
        }
    )
    bt["cum_gross"] = (1.0 + bt["gross_ret"]).cumprod()
    bt["cum_net"] = (1.0 + bt["net_ret"]).cumprod()
    bt.to_csv(res_dir / "har_h1_bt.csv", index=False)
    return res_dir


def test_report_smoke(tmp_path, monkeypatch):
    pred_dir = _make_pred_dir(tmp_path)
    res_dir = _make_results_dir(tmp_path)

    monkeypatch.setattr(report, "PRED_DIR", pred_dir)
    monkeypatch.setattr(report, "RES_DIR", res_dir)

    report.main()

    summary_path = res_dir / "summary.csv"
    bt_overview_path = res_dir / "bt_overview.csv"

    assert summary_path.exists(), "report.main() did not emit summary.csv"
    summary = pd.read_csv(summary_path)
    assert not summary.empty
    assert {"model", "horizon", "split", "rmse_log", "qlike"}.issubset(summary.columns)

    assert bt_overview_path.exists(), "report.main() did not emit bt_overview.csv"
    overview = pd.read_csv(bt_overview_path)
    assert not overview.empty
    assert {"file", "final_cum_net", "final_cum_gross"}.issubset(overview.columns)
