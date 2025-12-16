#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")

# Paths
RESULTS_DIR = Path("experiments/results")
FIGS_DIR = Path("experiments/figs")

# 1. Forecast Metrics Plot (RMSE/QLIKE by Model/Horizon)
summary_df = pd.read_csv(RESULTS_DIR / "summary.csv")
# Filter to test split
test_df = summary_df[summary_df["split"] == "test"].copy()
test_df["horizon"] = test_df["horizon"].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# RMSE
sns.barplot(data=test_df, x="horizon", y="rmse_log", hue="model", ax=axes[0])
axes[0].set_title("RMSE (Log Volatility) by Model & Horizon (Test Split)")
axes[0].set_ylabel("RMSE")
axes[0].set_xlabel("Horizon (Days)")
# QLIKE
sns.barplot(data=test_df, x="horizon", y="qlike", hue="model", ax=axes[1])
axes[1].set_title("QLIKE by Model & Horizon (Test Split)")
axes[1].set_ylabel("QLIKE")
axes[1].set_xlabel("Horizon (Days)")
plt.tight_layout()
plt.savefig(FIGS_DIR / "forecast_metrics.png", dpi=150)
plt.close()

# 2. Backtest KPIs Plot (Sharpe/CAGR for TFT and key models)
# Load from tft_performance_summary.csv
tft_df = pd.read_csv(RESULTS_DIR / "tft_performance_summary.csv")
# Group by file (horizon)
groups = tft_df.groupby((tft_df["Metric"] == "File").cumsum())
sharpe_data = []
cagr_data = []
for name, group in groups:
    if len(group) > 1:
        file_row = group[group["Metric"] == "File"]
        if not file_row.empty:
            file_name = file_row["Value"].iloc[0]
            horizon = file_name.split("_")[1]  # h1, h5, h22
            sharpe_row = group[group["Metric"] == "Net Sharpe"]
            cagr_row = group[group["Metric"] == "Net CAGR"]
            if not sharpe_row.empty and not cagr_row.empty:
                sharpe = float(sharpe_row["Value"].iloc[0].replace("%", ""))
                cagr = float(cagr_row["Value"].iloc[0].replace("%", ""))
                sharpe_data.append({"Horizon": horizon, "Sharpe": sharpe})
                cagr_data.append({"Horizon": horizon, "CAGR": cagr})

sharpe_df = pd.DataFrame(sharpe_data)
cagr_df = pd.DataFrame(cagr_data)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(data=sharpe_df, x="Horizon", y="Sharpe", ax=axes[0])
axes[0].set_title("TFT Net Sharpe by Horizon")
sns.barplot(data=cagr_df, x="Horizon", y="CAGR", ax=axes[1])
axes[1].set_title("TFT Net CAGR by Horizon")
plt.tight_layout()
plt.savefig(FIGS_DIR / "tft_backtest_kpis.png", dpi=150)
plt.close()

# 3. Cumulative Returns Plot (for TFT H=1, H=5, H=22)
bt_files = ["tft_h1_bt.csv", "tft_h5_bt.csv", "tft_h22_bt.csv"]
plt.figure(figsize=(10, 6))
for file in bt_files:
    df = pd.read_csv(RESULTS_DIR / file)
    df["date"] = pd.to_datetime(df["date"])
    horizon = file.split("_")[1]  # h1, h5, h22
    plt.plot(df["date"], df["cum_net"], label=f"TFT H={horizon}")
plt.title("Cumulative Net Returns: TFT by Horizon")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
# Reduce x-ticks to yearly
plt.xticks(pd.date_range(start="2020-01-01", end="2024-12-31", freq="YS"), rotation=45)
plt.tight_layout()
plt.savefig(FIGS_DIR / "cumulative_returns.png", dpi=150)
plt.close()

print("Plots generated in experiments/figs/: forecast_metrics.png, tft_backtest_kpis.png, cumulative_returns.png")


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
    plt.savefig(FIGS_DIR / "nbeats_loss_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGS_DIR / 'nbeats_loss_curves.png'}")


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
    plt.savefig(FIGS_DIR / "performance_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGS_DIR / 'performance_comparison.png'}")


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
    print("Generating additional plots and tables...\n")
    plot_nbeats_loss()
    plot_performance_comparison()
    create_performance_table()
    print("\nDone!")