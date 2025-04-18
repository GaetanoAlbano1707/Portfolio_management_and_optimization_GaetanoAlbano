import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def get_quarter_label(step_index):
    trimestre = (step_index % 4) + 1
    anno = (step_index // 4) + 1
    return f"Q{trimestre} Anno {anno}"

def load_data(performance_file, actions_file, rebalance_file):
    df_perf = pd.read_csv(performance_file)
    with open(actions_file, "rb") as f:
        actions = pickle.load(f)
    rebalance_steps = pd.read_csv(rebalance_file)["Rebalance Step"].values
    return df_perf, actions, rebalance_steps

def plot_weights_at_rebalances(tickers, actions, rebalance_steps, label_set):
    for i, step in enumerate(rebalance_steps):
        weights = actions[step]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(tickers, weights, color='skyblue')
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{weight:.2f}",
                     ha='center', va='bottom', fontsize=8)
        plt.ylim(0, 1)
        plt.ylabel("Peso (%)")
        plt.xlabel("Asset")
        label = get_quarter_label(i)
        plt.title(f"{label_set} - {label}")
        plt.xticks(rotation=45)
        filename = f"results_ppo/step_{i}_{label_set.lower()}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# === CONFIG ===
tickers = ['AAPL', 'AMZN', 'ITA', 'NFLX', 'NKE', 'NVDA', 'WMT',
           'XLE', 'XLF', 'XLI', 'XLK', 'XLV', 'XLY', 'XOM']

os.makedirs("results_ppo", exist_ok=True)

# Validation
_, actions_val, rebalance_val = load_data("validation_performance.csv", "actions_val.pkl", "ribilanciamenti_val.csv")
plot_weights_at_rebalances(tickers, actions_val, rebalance_val, "Validation")

# Test
_, actions_test, rebalance_test = load_data("test_performance.csv", "actions_test.pkl", "ribilanciamenti_test.csv")
plot_weights_at_rebalances(tickers, actions_test, rebalance_test, "Test")
