#!/usr/bin/env python
"""Generate backtest summary statistics and equity curve plots."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

RES_DIR = Path("experiments/results")
FIG_DIR = Path("experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Model folders
GARCH_DIR = Path("experiments/garch/results")
NBEATS_DIR = Path("experiments/nbeats/results")


def compute_metrics(df):
    """Compute Sharpe, returns, drawdown from backtest results."""
    ann_factor = np.sqrt(252)
    net_rets = df['net_ret'].dropna()
    
    sharpe = (net_rets.mean() / net_rets.std()) * ann_factor if net_rets.std() > 0 else 0
    total_return = df['cum_net'].iloc[-1] - 1
    n_years = len(df) / 252
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = net_rets.std() * ann_factor
    
    cum_max = df['cum_net'].cummax()
    drawdown = (df['cum_net'] - cum_max) / cum_max
    max_dd = drawdown.min()
    
    return {
        'sharpe': sharpe,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'max_drawdown': max_dd,
        'total_return': total_return,
        'avg_turnover': df['turnover'].mean(),
        'total_cost': df['cost'].sum(),
        'final_cum_net': df['cum_net'].iloc[-1],
    }


def main():
    results = []
    equity_curves = {}
    
    # Load GARCH backtest results
    for bt_file in sorted(GARCH_DIR.glob("*_bt.csv")):
        df = pd.read_csv(bt_file, parse_dates=['date'])
        name = f"garch_{bt_file.stem}"  # e.g., garch_h1
        metrics = compute_metrics(df)
        metrics['model'] = name
        results.append(metrics)
        equity_curves[name] = df[['date', 'cum_net']].copy()
    
    # Load N-BEATS backtest results
    for bt_file in sorted(NBEATS_DIR.glob("*_bt.csv")):
        df = pd.read_csv(bt_file, parse_dates=['date'])
        name = f"nbeats_{bt_file.stem}"  # e.g., nbeats_h1
        metrics = compute_metrics(df)
        metrics['model'] = name
        results.append(metrics)
        equity_curves[name] = df[['date', 'cum_net']].copy()
    
    if not results:
        print("No backtest files found!")
        return
    
    summary = pd.DataFrame(results)
    summary = summary[['model', 'sharpe', 'ann_return', 'ann_vol', 'max_drawdown', 
                       'total_return', 'avg_turnover', 'final_cum_net']]
    summary['Model'] = summary['model'].apply(lambda x: x.replace('_', ' ').upper())
    
    out_path = RES_DIR / "backtest_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    
    # Print results
    print("\n### Backtest Performance Summary\n")
    print("| Model | Sharpe | Ann. Return | Ann. Vol | Max DD | Total Return |")
    print("|-------|--------|-------------|----------|--------|--------------|")
    for _, row in summary.iterrows():
        print(f"| {row['Model']} | {row['sharpe']:.3f} | {row['ann_return']*100:.2f}% | {row['ann_vol']*100:.2f}% | {row['max_drawdown']*100:.2f}% | {row['total_return']*100:.2f}% |")
    
    # Equity curves plot
    colors = {
        'garch_h1': '#E63946', 'garch_h5': '#F4A261', 'garch_h22': '#E9C46A',
        'nbeats_h1': '#2A9D8F', 'nbeats_h5': '#264653', 'nbeats_h22': '#457B9D',
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in equity_curves.items():
        ax.plot(data['date'], data['cum_net'], label=name.replace('_', ' ').upper(), 
                color=colors.get(name, 'gray'), linewidth=1.5)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Portfolio Equity Curves (Test Period: 2020-2024)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "equity_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'equity_curves.png'}")
    
    # Drawdown plot
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, data in equity_curves.items():
        cum_max = data['cum_net'].cummax()
        dd = (data['cum_net'] - cum_max) / cum_max * 100
        ax.fill_between(data['date'], dd, 0, alpha=0.3, color=colors.get(name, 'gray'), 
                       label=name.replace('_', ' ').upper())
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Portfolio Drawdowns')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "drawdowns.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'drawdowns.png'}")


if __name__ == "__main__":
    main()
