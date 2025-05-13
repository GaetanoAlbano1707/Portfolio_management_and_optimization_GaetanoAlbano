#vesione ribilanciamento mensile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import pickle

from data_loader import load_financial_data
from environment_prova import PortfolioEnvProva as PortfolioEnv
from ppo_agent import PPOAgent
from utils_data_split import train_test_split_df
from visualizations import plot_portfolio_weights, plot_cumulative_returns, plot_grouped_bar_weights
from analysis_top_assets import analyze_dominant_weights
from save_weights import export_rebalance_weights_csv
from simulate_efficient_frontier import (
    simulate_cost_frontier,
    plot_efficient_frontier,
    generate_multiple_frontiers
)
from compute_metrics import compute_quarterly_metrics
from compare_strategies_monthly import (
    run_equal_weights_strategy,
    run_minimum_variance_strategy,
    run_markowitz_strategy,
    run_random_strategy,
    run_buy_and_hold_strategy
)

from evaluate_strategy_metrics_monthly import evaluate_strategies

# === CONFIG ===
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
WINDOW_SIZE = 5
RESULTS_DIR = "results_ppo_monthly"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === LOAD DATA ===
df, tickers = load_financial_data(CSV_PATH)

# === TEST SPLIT ===
_, df_test = train_test_split_df(df, train_ratio=0.8)

with open("cov_matrices_monthly.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

# === SETUP ENVIRONMENT ===
env = PortfolioEnv(df_test, tickers, window_size=WINDOW_SIZE, cov_matrices=cov_matrices, rebalance_period=21)
input_dim = np.prod(env.observation_space.shape)

# === LOAD TRAINED MODEL ===
agent = PPOAgent(input_dim=input_dim, num_assets=len(tickers))
agent.policy.load_state_dict(torch.load("ppo_model.pth"))

# === TEST LOOP ===
state = env.reset()
done = False
rewards = []
actions = []

while not done:
    action, _, _ = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)

    rewards.append(reward)
    actions.append(action)
    state = next_state

# === SAVE PERFORMANCE ===
ppo_log_returns = env.portfolio_returns
ppo_cum_returns = np.exp(np.cumsum(ppo_log_returns))
pd.DataFrame({"Log Return": ppo_log_returns, "Cumulative Return": ppo_cum_returns}).to_csv(
    f"{RESULTS_DIR}/test_performance.csv", index=False)

with open("actions_test.pkl", "wb") as f:
    pickle.dump(actions, f)

# === PLOT CUMULATIVE RETURNS ===
plt.figure()
plot_cumulative_returns(ppo_cum_returns, label="Test PPO")
for step in env.rebalance_steps:
    plt.axvline(x=step, color='red', linestyle='--', alpha=0.2)
plt.title("Rendimento Cumulato (Test Set)")
plt.savefig(f"{RESULTS_DIR}/test_reward_cumulativo.png")
plt.close()

# === PLOT WEIGHTS ===
plt.figure(figsize=(12, 6))
plot_portfolio_weights(actions, tickers)
plt.savefig(f"{RESULTS_DIR}/test_pesi.png")
plt.close()

# === SALVA RIBILANCIAMENTI ===
pd.DataFrame({"Rebalance Step": env.rebalance_steps}).to_csv(f"{RESULTS_DIR}/test_ribilanciamenti.csv", index=False)

# === DOMINANT ASSETS ===
analyze_dominant_weights(actions, tickers)

# === SALVA PESI ===
export_rebalance_weights_csv(env, tickers, f"{RESULTS_DIR}/rebalance_weights_test.csv")

plot_grouped_bar_weights(
    filepath=f"{RESULTS_DIR}/rebalance_weights_test.csv",
    output_path=f"{RESULTS_DIR}/grouped_bar_weights_by_quarter.png"
)

# === STRATEGIE COMPARATIVE ===
equal_vals = run_equal_weights_strategy(df_test, tickers, rebalance_period=21)
ppo_dates = equal_vals.index[:len(ppo_cum_returns)]
ppo_series = pd.Series(ppo_cum_returns, index=ppo_dates)
plt.plot(equal_vals, label="Equal Weights")
plt.plot(ppo_series, label="PPO")
plt.legend()
plt.title("Strategie a Confronto (Test)")
plt.savefig(f"{RESULTS_DIR}/test_compare_strategies.png")
plt.close()

# === FRONTIERA EFFICIENTE ===
daily_returns = df_test["adj_close"].unstack().pct_change().dropna()
mean_returns = daily_returns.mean().values * 252
cov_matrix = daily_returns.cov().values * 252

# Singolo esempio con c+ = 0.01, c- = 0.02
c_plus = 0.01
c_minus = 0.02
results = simulate_cost_frontier(cov_matrix, mean_returns, c_plus=c_plus, c_minus=c_minus)
plot_efficient_frontier(
    results,
    c_plus=c_plus,
    c_minus=c_minus,
    filename=f"{RESULTS_DIR}/test_efficient_frontier_cp{int(c_plus*100)}_cm{int(c_minus*100)}.png"
)

# === MULTIPLE CONFIGS ===
configs = [
    (0.001, 0.002),
    (0.005, 0.01),
    (0.01, 0.02)
]

generate_multiple_frontiers(
    cov_matrix=cov_matrix,
    mean_returns=mean_returns,
    configs=configs,
    base_filename=f"{RESULTS_DIR}/test_efficient_frontier"
)

# === METRICHE TRIMESTRALI ===
dates = df_test.index.get_level_values("date").unique().sort_values()
dates = dates[WINDOW_SIZE + 1 : WINDOW_SIZE + 1 + len(ppo_log_returns)]

if len(dates) > len(ppo_log_returns):
    dates = dates[:len(ppo_log_returns)]
elif len(ppo_log_returns) > len(dates):
    ppo_log_returns = ppo_log_returns[:len(dates)]

compute_quarterly_metrics(ppo_log_returns, dates, save_path=f"{RESULTS_DIR}/test_quarterly_metrics.csv")

# === STRATEGIE COMPARATIVE COMPLETE ===
ppo_dates = dates
ppo_series = pd.Series(np.exp(np.cumsum(ppo_log_returns)), index=ppo_dates)

strategies = {
    "Equal Weight": run_equal_weights_strategy(df_test, tickers, rebalance_period=21),
    "Minimum Variance": run_minimum_variance_strategy(df_test, tickers, rebalance_period=21),
    "Markowitz": run_markowitz_strategy(df_test, tickers, rebalance_period=21),
    "Random": run_random_strategy(df_test, tickers, rebalance_period=21),
    "Buy & Hold": run_buy_and_hold_strategy(df_test, tickers, asset_index=0),
    "PPO": ppo_series
}

comparison_df = pd.DataFrame(strategies)
comparison_df.to_csv(f"{RESULTS_DIR}/model_comparison_returns.csv", index=True)

#vedo metriche per i vari modelli
evaluate_strategies(
    comparison_df,
    output_path=f"{RESULTS_DIR}/strategy_risk_metrics.csv"
)

plt.figure(figsize=(12, 6))
for name, series in strategies.items():
    plt.plot(series.index, series.values, label=name)
plt.legend()
plt.title("Confronto Strategie di Portafoglio")
plt.xlabel("Data")
plt.ylabel("Rendimento Cumulativo")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/compare_models_test.png")
plt.close()


