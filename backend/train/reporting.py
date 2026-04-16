import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
from backend.train.ppo_config import REPORT_DIR, EVAL_DIR

# Set Plotting Style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def plot_regime_dashboard(regime_data, output_dir):
    """
    Creates a multi-chart dashboard for a specific evaluation regime.
    """
    name = regime_data['name']
    equity_curve = np.array(regime_data['equity_curve'])
    bh_curve = np.array(regime_data['bh_curve'])
    ss_curve = np.array(regime_data['ss_curve'])
    actions = regime_data['actions']
    symbol = regime_data.get('symbol', 'Unknown')
    
    os.makedirs(output_dir, exist_ok=True)
        
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    
    # Create a 2x2 multi-plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f"Trading Performance Dashboard: {name} ({symbol})", fontsize=24)
    
    ax1 = axes[0, 0]
    steps = np.arange(len(equity_curve))
    ax1.plot(steps, equity_curve, label="PPO Agent", linewidth=2.5, color="coral")
    
    min_len_bh = min(len(steps), len(bh_curve))
    ax1.plot(steps[:min_len_bh], bh_curve[:min_len_bh], label="Buy & Hold", linestyle="--", alpha=0.7)
    
    min_len_ma = min(len(steps), len(regime_data.get('ma_curve', [])))
    if min_len_ma > 0:
        ax1.plot(steps[:min_len_ma], regime_data['ma_curve'][:min_len_ma], label="EMA Crossover", linestyle="-.", alpha=0.7)
        
    min_len_sl = min(len(steps), len(regime_data.get('straddle_long_curve', [])))
    if min_len_sl > 0:
        ax1.plot(steps[:min_len_sl], regime_data['straddle_long_curve'][:min_len_sl], label="Long Straddle", linestyle=":", alpha=0.7)
    
    ax1.set_title("Equity Curve Comparison")
    ax1.set_ylabel("Total Capital")
    ax1.legend()
    
    # 2. Drawdown Profile
    ax2 = axes[0, 1]
    cum_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cum_max) / np.maximum(cum_max, 1e-6)
    ax2.fill_between(steps, drawdown * 100, 0, color="red", alpha=0.3)
    ax2.set_title("Drawdown Profile (%)")
    ax2.set_ylabel("Drawdown %")
    
    # 3. Action Distribution
    ax3 = axes[1, 0]
    flat_actions = [int(a) for a in np.array(actions).flatten()]
    action_counts = pd.Series(flat_actions).value_counts().sort_index()
    
    action_names = {0: "Hold", 1: "Buy Call", 2: "Buy Put", 3: "Exit Call", 4: "Exit Put"}
    action_labels = [action_names.get(i, f"Action {i}") for i in action_counts.index]
    
    if not action_counts.empty:
        sns.barplot(x=action_labels, y=action_counts.values, ax=ax3, hue=action_labels, palette="viridis", legend=False)
    ax3.set_title("Action Distribution")
    ax3.set_ylabel("Count")
    
    # 4. Returns Histogram
    ax4 = axes[1, 1]
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-6)
    if len(returns) > 0:
        sns.histplot(returns, bins=50, kde=True, ax=ax4, color="teal")
    ax4.set_title("Distribution of Step Returns")
    ax4.set_xlabel("Return %")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"dashboard_{safe_name}.png")
    plt.savefig(save_path)
    print(f"Report saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.results_path if args.results_path else os.path.join(EVAL_DIR, "eval_latest.pkl")
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    # Create plot subdirectory
    plot_dir = os.path.join(REPORT_DIR, f"plots_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)
    
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)
        
    for regime in all_results:
        plot_regime_dashboard(regime, plot_dir)
        
    print(f"\nAll reports generated successfully in {plot_dir}")

if __name__ == "__main__":
    main()
