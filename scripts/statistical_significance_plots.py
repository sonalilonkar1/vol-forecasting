#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set style
sns.set_style("whitegrid")

# Paths
RESULTS_DIR = Path("experiments/results")
FIGS_DIR = Path("experiments/figs")

# Ensure figs dir exists
FIGS_DIR.mkdir(exist_ok=True)

# 1. DM Test P-Value Heatmap
dm_files = glob.glob(str(RESULTS_DIR / "dm_tests" / "*.csv"))
dm_data = []
for file in dm_files:
    df = pd.read_csv(file)
    if not df.empty:
        row = df.iloc[0]
        label = row['label']
        p_value = row['p_value']
        # Parse label, e.g., "tft_h1 vs har_baselines_harx-off_estimator-ols_h1"
        parts = label.split(' vs ')
        if len(parts) == 2:
            model_a = parts[0].split('_')[0]  # tft
            model_b = 'har' if 'har' in parts[1] else parts[1].split('_')[0]  # har or gbt
            horizon = parts[0].split('_')[1]  # h1
            dm_data.append({'model_A': model_a, 'model_B': model_b, 'horizon': horizon, 'p_value': p_value})

dm_df = pd.DataFrame(dm_data)
if not dm_df.empty:
    # Pivot for heatmap (simplified, assuming few models)
    pivot_df = dm_df.pivot_table(index='model_A', columns=['model_B', 'horizon'], values='p_value', aggfunc='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap="RdYlGn_r", cbar_kws={'label': 'P-Value'}, fmt=".3f")
    plt.title("DM Test P-Values: Model Superiority (Lower = Significant)")
    plt.savefig(FIGS_DIR / "dm_pvalues_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("DM p-values heatmap saved.")

# 2. Bootstrap CIs for Sharpe and CAGR
bootstrap_files = glob.glob(str(RESULTS_DIR / "bootstrap" / "*_bt_bootstrap_ci.csv"))
sharpe_data = []
cagr_data = []
for file in bootstrap_files:
    df = pd.read_csv(file)
    file_name = Path(file).stem.replace('_bt_bootstrap_ci', '')
    model = file_name.split('_')[0]  # tft or har
    horizon = file_name.split('_')[1]  # h1
    sharpe_row = df[df['metric'] == 'net_sharpe']
    cagr_row = df[df['metric'] == 'net_cagr']
    if not sharpe_row.empty:
        sharpe_data.append({
            'model': model,
            'horizon': horizon,
            'mean': sharpe_row['value'].iloc[0],
            'ci_lower': sharpe_row['ci_lower'].iloc[0],
            'ci_upper': sharpe_row['ci_upper'].iloc[0]
        })
    if not cagr_row.empty:
        cagr_data.append({
            'model': model,
            'horizon': horizon,
            'mean': cagr_row['value'].iloc[0],
            'ci_lower': cagr_row['ci_lower'].iloc[0],
            'ci_upper': cagr_row['ci_upper'].iloc[0]
        })

sharpe_df = pd.DataFrame(sharpe_data)
cagr_df = pd.DataFrame(cagr_data)

if not sharpe_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    for horizon in sharpe_df['horizon'].unique():
        sub_df = sharpe_df[sharpe_df['horizon'] == horizon]
        ax.errorbar(sub_df['model'], sub_df['mean'], 
                    yerr=[sub_df['mean'] - sub_df['ci_lower'], sub_df['ci_upper'] - sub_df['mean']], 
                    fmt='o-', capsize=5, label=f"H={horizon}")
    ax.set_title("Bootstrap CIs for Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend()
    plt.savefig(FIGS_DIR / "bootstrap_sharpe_cis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Bootstrap Sharpe CIs plot saved.")

if not cagr_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    for horizon in cagr_df['horizon'].unique():
        sub_df = cagr_df[cagr_df['horizon'] == horizon]
        ax.errorbar(sub_df['model'], sub_df['mean'], 
                    yerr=[sub_df['mean'] - sub_df['ci_lower'], sub_df['ci_upper'] - sub_df['mean']], 
                    fmt='o-', capsize=5, label=f"H={horizon}")
    ax.set_title("Bootstrap CIs for CAGR")
    ax.set_ylabel("CAGR")
    ax.legend()
    plt.savefig(FIGS_DIR / "bootstrap_cagr_cis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Bootstrap CAGR CIs plot saved.")

print("Statistical significance plots generated.")