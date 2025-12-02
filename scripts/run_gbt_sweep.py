#!/usr/bin/env python3
"""Grid runner for src/models/gbt.py.

Example:
    PYTHONPATH=$PWD python scripts/run_gbt_sweep.py \
        --n-estimators 1000 2000 \
        --learning-rates 0.02 0.03 \
        --max-depths 4 5 \
        --horizons 1 5 22
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
GBT_SCRIPT = REPO_ROOT / "src" / "models" / "gbt.py"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "preds" / "gbt_sweeps"
DEFAULT_SPLITS = REPO_ROOT / "configs" / "splits.yaml"


def _parse_float_list(values: Sequence[str] | None, fallback: float) -> list[float]:
    if not values:
        return [fallback]
    return [float(v) for v in values]


def _parse_int_list(values: Sequence[str] | None, fallback: int) -> list[int]:
    if not values:
        return [fallback]
    return [int(v) for v in values]


def _format_float(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace(".", "p")


def _ensure_pythonpath(env: dict[str, str]) -> None:
    existing = env.get("PYTHONPATH", "")
    root = str(REPO_ROOT)
    if not existing:
        env["PYTHONPATH"] = root
    elif root not in existing.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([root, existing])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GBT baseline across a hyperparameter grid.")
    parser.add_argument("--source", type=Path, default=None, help="Optional explicit feature table for gbt.py")
    parser.add_argument("--splits-config", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], help="Which splits to export per run")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Base directory for sweep outputs")
    parser.add_argument("--n-estimators", nargs="*", default=None, help="Values for --n-estimators (default: 2000)")
    parser.add_argument("--learning-rates", nargs="*", default=None, help="Values for --learning-rate (default: 0.03)")
    parser.add_argument("--max-depths", nargs="*", default=None, help="Values for --max-depth (default: 5)")
    parser.add_argument("--subsamples", nargs="*", default=None, help="Values for --subsample (default: 0.9)")
    parser.add_argument("--colsample-bytree", nargs="*", default=None, help="Values for --colsample-bytree (default: 0.9)")
    parser.add_argument("--lambdas", nargs="*", default=None, help="Values for --lambda_ (default: 1.0)")
    parser.add_argument("--seeds", nargs="*", default=None, help="Seeds to try (default: 42)")
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them")
    return parser.parse_args()


def build_combos(args: argparse.Namespace) -> list[dict[str, float | int]]:
    n_estimators = _parse_int_list(args.n_estimators, 2000)
    max_depths = _parse_int_list(args.max_depths, 5)
    seeds = _parse_int_list(args.seeds, 42)
    learning_rates = _parse_float_list(args.learning_rates, 0.03)
    subsamples = _parse_float_list(args.subsamples, 0.9)
    colsample = _parse_float_list(getattr(args, "colsample_bytree"), 0.9)
    lambdas = _parse_float_list(args.lambdas, 1.0)
    combos = []
    for ne, depth, seed, eta, sub, col, lam in itertools.product(
        n_estimators, max_depths, seeds, learning_rates, subsamples, colsample, lambdas
    ):
        combos.append(
            {
                "n_estimators": ne,
                "max_depth": depth,
                "seed": seed,
                "learning_rate": eta,
                "subsample": sub,
                "colsample_bytree": col,
                "lambda_": lam,
            }
        )
    return combos


def combo_name(params: dict[str, float | int]) -> str:
    parts = [
        f"ne{params['n_estimators']}",
        f"eta{_format_float(params['learning_rate'])}",
        f"depth{params['max_depth']}",
        f"sub{_format_float(params['subsample'])}",
        f"col{_format_float(params['colsample_bytree'])}",
        f"lam{_format_float(params['lambda_'])}",
        f"seed{params['seed']}",
    ]
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    combos = build_combos(args)
    if not combos:
        print("[warn] no hyperparameter combos computed; nothing to do")
        return

    if args.dry_run:
        print("Planned GBT runs:")
        for cfg in combos:
            print(f" - {combo_name(cfg)}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    env = os.environ.copy()
    _ensure_pythonpath(env)

    for cfg in combos:
        run_id = combo_name(cfg)
        run_dir = args.out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd: list[str] = [
            sys.executable,
            str(GBT_SCRIPT),
            "--splits-config",
            str(args.splits_config),
            "--out-dir",
            str(run_dir),
            "--early-stopping",
            str(args.early_stopping),
        ]
        if args.source:
            cmd += ["--source", str(args.source)]
        cmd += ["--horizons", *map(str, args.horizons)]
        cmd += ["--eval-splits", *args.eval_splits]
        for flag, value in cfg.items():
            cmd += [f"--{flag.replace('_', '-')}", str(value)]

        print(f"[gbt-sweep] {run_id} → {run_dir}")
        start = time.time()
        try:
            subprocess.run(cmd, check=True, env=env)
            status = "ok"
        except subprocess.CalledProcessError as exc:
            status = f"error:{exc.returncode}"
        duration = time.time() - start

        produced = sorted(p.name for p in run_dir.glob("*.csv"))
        summary_rows.append(
            {
                "run_id": run_id,
                "status": status,
                "duration_sec": f"{duration:.2f}",
                "out_dir": str(run_dir.relative_to(REPO_ROOT)),
                "files": ",".join(produced),
                **{k: str(v) for k, v in cfg.items()},
            }
        )

    summary_path = args.out_dir / "sweep_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    if fieldnames:
        with summary_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[gbt-sweep] wrote summary to {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
