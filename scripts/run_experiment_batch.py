#!/usr/bin/env python3
"""Batch runner that trains models and launches backtests from a YAML plan.

Example:
    PYTHONPATH=$PWD python scripts/run_experiment_batch.py \
        --config configs/experiment_plan.yaml \
        --experiments har gbt --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_plan.yaml"
TIME_TOLERANCE_SEC = 2.0  # cushion for filesystem timestamp granularity


@dataclass
class GridParam:
    name: str
    flag: str
    kind: str
    values: Sequence[Any]
    only_when: Dict[str, Any]
    format_spec: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of experiments and backtests.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML plan describing experiments")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment names to run (default: all from config)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Experiment plan must be a mapping")
    return cfg


def ensure_pythonpath(env: dict[str, str]) -> None:
    root = str(REPO_ROOT)
    current = env.get("PYTHONPATH", "")
    if not current:
        env["PYTHONPATH"] = root
    elif root not in current.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([root, current])


def normalize_command(cmd: Any) -> List[str]:
    if isinstance(cmd, list):
        return [str(token) for token in cmd]
    if isinstance(cmd, str):
        return cmd.split()
    raise TypeError(f"Command must be a list or string, got {type(cmd)!r}")


def flatten_args(cmd: List[str], arg_map: Dict[str, Any] | None) -> None:
    if not arg_map:
        return
    for flag, value in arg_map.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(str(flag))
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            cmd.append(str(flag))
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([str(flag), str(value)])


def parse_grid(specs: Sequence[dict]) -> tuple[List[GridParam], List[dict]]:
    params: List[GridParam] = []
    for raw in specs:
        if "name" not in raw:
            raise ValueError("Each grid entry needs a 'name' field")
        name = str(raw["name"])
        flag = str(raw.get("flag", f"--{name.replace('_', '-')}")).strip()
        kind = str(raw.get("kind", "value")).lower()
        if kind not in {"value", "flag", "multi"}:
            raise ValueError(f"Unsupported param kind '{kind}' for {name}")
        values = raw.get("values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Grid param '{name}' needs a sequence of values")
        only_when = raw.get("only_when", {})
        if not isinstance(only_when, dict):
            raise ValueError(f"only_when for '{name}' must be a mapping")
        params.append(
            GridParam(
                name=name,
                flag=flag,
                kind=kind,
                values=values,
                only_when=only_when,
                format_spec=raw.get("format"),
            )
        )
    return params, specs  # second item preserves original ordering for slugging


def condition_satisfied(requirements: Dict[str, Any], current: Dict[str, Any]) -> bool:
    for key, expected in requirements.items():
        if key not in current:
            return False
        actual = current[key]
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def expand_combos(params: Sequence[GridParam], idx: int = 0, seed: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    current = dict(seed or {})
    if idx >= len(params):
        return [current]
    spec = params[idx]
    if spec.only_when and not condition_satisfied(spec.only_when, current):
        return expand_combos(params, idx + 1, current)
    combos: List[Dict[str, Any]] = []
    for value in spec.values:
        next_combo = current.copy()
        next_combo[spec.name] = value
        combos.extend(expand_combos(params, idx + 1, next_combo))
    return combos


def slugify(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return str(value).replace(".", "p")
    text = str(value)
    text = text.strip().replace("/", "-")
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", text)
    return text or "val"


def combo_slug(grid_specs: Sequence[dict], combo: Dict[str, Any]) -> str:
    if not combo:
        return "default"
    parts = []
    for raw in grid_specs:
        name = raw["name"]
        if name not in combo:
            continue
        label = raw.get("label", name)
        parts.append(f"{label}-{slugify(combo[name])}")
    return "_".join(parts) or "default"


def build_prediction_command(exp: dict, combo: Dict[str, Any]) -> List[str]:
    cmd = normalize_command(exp["command"])
    horizons = exp.get("horizons", [1])
    cmd.extend(["--horizons", *map(str, horizons)])
    flatten_args(cmd, exp.get("args"))
    for param in exp.get("grid", []):
        name = param["name"]
        if name not in combo:
            continue
        value = combo[name]
        flag = str(param.get("flag", f"--{name.replace('_', '-')}")).strip()
        kind = str(param.get("kind", "value")).lower()
        fmt = param.get("format")
        if fmt is not None and isinstance(value, (int, float)):
            value = format(value, fmt)
        if kind == "flag":
            if value:
                cmd.append(flag)
            continue
        if kind == "multi":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Param '{name}' expected a list for kind=multi")
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
            continue
        # default: value
        cmd.extend([flag, str(value)])
    return cmd


def collect_new_predictions(pred_dir: Path, horizons: Sequence[int], start_time: float) -> Dict[int, Path]:
    matches: Dict[int, Path] = {}
    pattern = re.compile(r"_h(\d+)\.csv$")
    for path in pred_dir.glob("*_h*.csv"):
        m = pattern.search(path.name)
        if not m:
            continue
        horizon = int(m.group(1))
        if horizon not in horizons or horizon in matches:
            continue
        mtime = path.stat().st_mtime
        if mtime + TIME_TOLERANCE_SEC < start_time:
            continue
        matches[horizon] = path
    return matches


def move_predictions(files: Dict[int, Path], archive_dir: Path, prefix: str) -> Dict[int, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    renamed: Dict[int, Path] = {}
    for horizon, src in files.items():
        dest = archive_dir / f"{prefix}_h{horizon}.csv"
        if dest.exists():
            dest.unlink()
        src.rename(dest)
        renamed[horizon] = dest
    return renamed


def build_backtest_command(base_cfg: dict, pred_path: Path) -> List[str]:
    cmd = normalize_command(base_cfg.get("command", [sys.executable, "-m", "src.backtest.run"]))
    flatten_args(cmd, base_cfg.get("args"))
    cmd.extend(["--pred-path", str(pred_path)])
    return cmd


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pred_dir = REPO_ROOT / cfg.get("pred_dir", "experiments/preds")
    archive_dir = REPO_ROOT / cfg.get("archive_dir", pred_dir)
    backtest_cfg = cfg.get("backtest", {})
    experiments = cfg.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")
    selected = set(args.experiments or [])

    env = os.environ.copy()
    ensure_pythonpath(env)

    summary_rows = []

    for exp in experiments:
        name = exp.get("name")
        if not name:
            raise ValueError("Each experiment needs a 'name'")
        if selected and name not in selected:
            continue
        horizons = exp.get("horizons", [1])
        grid_specs = exp.get("grid", [])
        grid_params, ordering = parse_grid(grid_specs)
        combos = expand_combos(grid_params) if grid_params else [{}]
        if not combos:
            combos = [{}]
        print(f"[experiment] {name}: {len(combos)} combos")

        for combo in combos:
            slug = combo_slug(ordering, combo)
            run_label = f"{name}__{slug}"
            prefix_template = exp.get("run_prefix_template", "{name}_{slug}")
            run_prefix = prefix_template.format(name=name, slug=slug)
            pred_cmd = build_prediction_command(exp, combo)
            print(f"  → {run_label}: {' '.join(pred_cmd)}")
            if args.dry_run:
                continue
            start_time = time.time()
            subprocess.run(pred_cmd, check=True, cwd=REPO_ROOT, env=env)
            produced = collect_new_predictions(pred_dir, horizons, start_time)
            if len(produced) != len(horizons):
                missing = sorted(set(horizons) - set(produced))
                raise RuntimeError(f"{run_label}: missing prediction files for horizons {missing}")
            renamed = move_predictions(produced, archive_dir, run_prefix)

            for horizon, pred_path in sorted(renamed.items()):
                bt_config = exp.get("backtest", backtest_cfg)
                bt_cmd = build_backtest_command(bt_config, pred_path)
                print(f"     backtest H={horizon}: {' '.join(bt_cmd)}")
                subprocess.run(bt_cmd, check=True, cwd=REPO_ROOT, env=env)
                out_dir = Path(bt_config.get("args", {}).get("--out-dir", cfg.get("results_dir", "experiments/results")))
                out_dir = REPO_ROOT / out_dir
                bt_path = out_dir / f"{pred_path.stem}_bt.csv"
                summary_rows.append(
                    {
                        "experiment": name,
                        "run": run_label,
                        "horizon": horizon,
                        "prediction": str(pred_path.relative_to(REPO_ROOT)),
                        "backtest": str(bt_path.relative_to(REPO_ROOT)),
                    }
                )

    if args.dry_run:
        print("[dry-run] No commands executed.")
        return

    if summary_rows:
        summary_path = REPO_ROOT / "experiments" / "results" / "automation_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as fh:
            fh.write("experiment,run,horizon,prediction,backtest\n")
            for row in summary_rows:
                fh.write(
                    f"{row['experiment']},{row['run']},{row['horizon']},{row['prediction']},{row['backtest']}\n"
                )
        print(f"[summary] wrote {summary_path.relative_to(REPO_ROOT)}")
    else:
        print("No experiments were executed.")


if __name__ == "__main__":
    main()
