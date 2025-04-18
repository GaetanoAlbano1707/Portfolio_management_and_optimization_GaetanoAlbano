import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from data_loader import load_financial_data
from analysis_top_assets import analyze_dominant_weights
df, tickers = load_financial_data("merged_data_total_cleaned_wide_multifeatures.csv")

# === LOAD PERFORMANCE METRICS ===
train = pd.read_csv("training_performance.csv")
val = pd.read_csv("validation_performance.csv")
try:
    test = pd.read_csv("test_performance.csv")
except FileNotFoundError:
    test = pd.DataFrame([{"Total Reward": np.nan, "Sharpe Ratio": np.nan}])

summary = pd.DataFrame({
    "Set": ["Train", "Validation", "Test"],
    "Total Reward": [
        train["Total Reward"].iloc[-1],
        val["Reward"].sum(),
        test["Reward"].sum() if not test.isnull().values.any() else np.nan
    ],
    "Sharpe Ratio": [
        train["Sharpe Ratio"].iloc[-1],
        val["Reward"].mean() / (val["Reward"].std() + 1e-6),
        test["Reward"].mean() / (test["Reward"].std() + 1e-6) if not test.isnull().values.any() else np.nan
    ]
})

print("\n=== Performance Summary ===")
print(summary)
summary.to_csv("final_summary.csv", index=False)

# === PLOT CUMULATIVE RETURNS WITH REBALANCE MARKERS ===
def plot_returns_with_rebalances(file_returns, label, rebalance_file=None):
    df = pd.read_csv(file_returns)
    returns = df["Cumulative Return"].values
    plt.plot(returns, label=label)
    if rebalance_file and os.path.exists(rebalance_file):
        steps = pd.read_csv(rebalance_file)["Rebalance Step"].values
        for s in steps:
            plt.axvline(x=s, color='red', linestyle='--', alpha=0.2)

plt.figure(figsize=(10, 6))
plot_returns_with_rebalances("test_performance.csv", "Test", "ribilanciamenti_test.csv")
plot_returns_with_rebalances("validation_performance.csv", "Validation", "ribilanciamenti_val.csv")
plt.title("Confronto Rendimento Cumulato con Ribilanciamenti")
plt.legend()
plt.grid(True)
plt.savefig("confronto_cumulato.png")
plt.show()

# === ANALISI DOMINANT ASSETS ===
print("\n📈 Dominant Assets - Validation:")
with open("actions_val.pkl", "rb") as f:
    val_actions = pickle.load(f)
    analyze_dominant_weights(val_actions, tickers)

print("\n📈 Dominant Assets - Test:")
with open("actions_test.pkl", "rb") as f:
    test_actions = pickle.load(f)
    analyze_dominant_weights(test_actions, tickers)
