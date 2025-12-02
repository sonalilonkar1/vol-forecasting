#!/usr/bin/env python
"""Generate performance plots and tables for model evaluation."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("experiments/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_nbeats_loss():
    """Plot N-BEATS training loss curves from log file."""
    log_path = Path("experiments/nbeats/training_log.txt")
    if not log_path.exists():
        print("N-BEATS training log not found")
        return
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    data = {1: [], 5: [], 22: []}
    current_horizon = None
    
    for line in lines:
        if "horizon H=" in line:
            current_horizon = int(line.split("H=")[1].split(",")[0])
        elif "Epoch" in line and "Loss:" in line:
            epoch = int(line.split("Epoch")[1].split("/")[0].strip())
            loss = float(line.split("Loss:")[1].strip())
            if current_horizon:
                data[current_horizon].append((epoch, loss))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1: '#2E86AB', 5: '#A23B72', 22: '#F18F01'}
    
    for h, vals in data.items():
        if vals:
            epochs, losses = zip(*vals)
            ax.plot(epochs, losses, 'o-', label=f'H={h}', color=colors[h], linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('N-BEATS Training Loss Curves')
    ax.legend(title='Forecast Horizon')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 20, 30, 40, 50])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "nbeats_loss_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'nbeats_loss_curves.png'}")


def plot_performance_comparison():
    """Create bar charts comparing GARCH vs N-BEATS performance."""
    summary_path = Path("experiments/results/summary.csv")
    if not summary_path.exists():
        print("Summary CSV not found")
        return
    
    df = pd.read_csv(summary_path)
    test_df = df[df['split'] == 'test'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    horizons = [1, 5, 22]
    x = np.arange(len(horizons))
    width = 0.35
    
    # RMSE plot
    garch_rmse = [test_df[(test_df['model'] == 'garch') & (test_df['horizon'] == h)]['rmse_log'].values[0] for h in horizons]
    nbeats_rmse = [test_df[(test_df['model'] == 'nbeats') & (test_df['horizon'] == h)]['rmse_log'].values[0] for h in horizons]
    
    bars1 = axes[0].bar(x - width/2, garch_rmse, width, label='GARCH(1,1)', color='#E63946')
    bars2 = axes[0].bar(x + width/2, nbeats_rmse, width, label='N-BEATS', color='#457B9D')
    axes[0].set_xlabel('Forecast Horizon (days)')
    axes[0].set_ylabel('RMSE (log volatility)')
    axes[0].set_title('RMSE Comparison (Test Set)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['H=1', 'H=5', 'H=22'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar in bars1 + bars2:
        h = bar.get_height()
        axes[0].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # QLIKE plot
    garch_qlike = [test_df[(test_df['model'] == 'garch') & (test_df['horizon'] == h)]['qlike'].values[0] for h in horizons]
    nbeats_qlike = [test_df[(test_df['model'] == 'nbeats') & (test_df['horizon'] == h)]['qlike'].values[0] for h in horizons]
    
    bars3 = axes[1].bar(x - width/2, garch_qlike, width, label='GARCH(1,1)', color='#E63946')
    bars4 = axes[1].bar(x + width/2, nbeats_qlike, width, label='N-BEATS', color='#457B9D')
    axes[1].set_xlabel('Forecast Horizon (days)')
    axes[1].set_ylabel('QLIKE')
    axes[1].set_title('QLIKE Comparison (Test Set)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['H=1', 'H=5', 'H=22'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars3 + bars4:
        h = bar.get_height()
        axes[1].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "performance_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'performance_comparison.png'}")


def create_performance_table():
    """Create formatted performance metrics table."""
    summary_path = Path("experiments/results/summary.csv")
    if not summary_path.exists():
        print("Summary CSV not found")
        return
    
    df = pd.read_csv(summary_path)
    df['Model'] = df['model'].str.upper().replace({'GARCH': 'GARCH(1,1)', 'NBEATS': 'N-BEATS(L=90)'})
    df['Horizon'] = df['horizon'].apply(lambda x: f"H={x}")
    df['Split'] = df['split'].str.title()
    df['RMSE'] = df['rmse_log'].round(4)
    df['QLIKE'] = df['qlike'].round(4)
    df['N'] = df['n']
    
    out_df = df[['Model', 'Horizon', 'Split', 'RMSE', 'QLIKE', 'N']].sort_values(['Model', 'Horizon', 'Split'])
    out_df.to_csv(Path("experiments/results/performance_table.csv"), index=False)
    print(f"Saved: experiments/results/performance_table.csv")
    
    print("\n| Model | Horizon | Split | RMSE | QLIKE | N |")
    print("|-------|---------|-------|------|-------|---|")
    for _, row in out_df.iterrows():
        print(f"| {row['Model']} | {row['Horizon']} | {row['Split']} | {row['RMSE']:.4f} | {row['QLIKE']:.4f} | {row['N']} |")


if __name__ == "__main__":
    print("Generating plots and tables...\n")
    plot_nbeats_loss()
    plot_performance_comparison()
    create_performance_table()
    print("\nDone!")
