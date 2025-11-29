import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_toy_dataset(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=90, freq="B")
    rows = []
    for asset_idx, asset in enumerate(["AAA", "BBB"]):
        base = 0.05 + 0.01 * asset_idx
        noise = np.sin(np.linspace(0, 6, len(dates))) * 0.01
        for d, n in zip(dates, noise):
            rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "logvol_t": np.log(base + max(n + 0.02, 1e-4)),
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "toy_features.csv"
    df.to_csv(path, index=False)
    return path


def _write_splits(tmp_path: Path) -> Path:
    cfg = tmp_path / "splits.yaml"
    cfg.write_text(
        """
train_start: 2020-01-01
train_end: 2020-02-20
val_start: 2020-02-21
val_end: 2020-02-29
test_start: 2020-03-02
test_end: 2020-04-30
embargo_days: 0
""".strip()
    )
    return cfg


def test_har_cli_produces_predictions(tmp_path):
    dataset = _make_toy_dataset(tmp_path)
    splits = _write_splits(tmp_path)
    out_dir = tmp_path / "preds"

    cmd = [
        sys.executable,
        "-m",
        "src.models.har_rv",
        "--source",
        str(dataset),
        "--horizons",
        "1",
        "--eval-splits",
        "test",
        "--splits-config",
        str(splits),
        "--out-dir",
        str(out_dir),
        "--min-train",
        "5",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    output = out_dir / "har_h1.csv"
    assert output.exists(), "HAR CLI did not produce the expected CSV"
    preds = pd.read_csv(output, parse_dates=["date"])
    assert not preds.empty, "Prediction file is empty"
    required_cols = {"date", "asset", "yhat_rv", "split"}
    assert required_cols.issubset(preds.columns), f"Missing columns: {required_cols - set(preds.columns)}"
    assert set(preds["split"]) == {"test"}