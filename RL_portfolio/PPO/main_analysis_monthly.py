# main_analysis_monthly.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from data_loader import load_financial_data
from ppo_agent import PPOAgent
from visualizations import plot_portfolio_weights, plot_cumulative_returns
from environment_prova import PortfolioEnvProva as PortfolioEnv
from save_weights import export_rebalance_weights_csv
from utils_data_split import train_test_split_df

# === CONFIG ===
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
EPISODES = 70
WINDOW_SIZE = 5
REBALANCE_PERIOD = 21  # mensile
RESULTS_DIR = "results_ppo_monthly"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === LOAD DATA ===
df, tickers = load_financial_data(CSV_PATH)
df_train, _ = train_test_split_df(df, train_ratio=0.8)

with open("cov_matrices_monthly.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

# === SETUP ENVIRONMENT ===
env = PortfolioEnv(df_train, tickers, window_size=WINDOW_SIZE,
                   cov_matrices=cov_matrices, rebalance_period=REBALANCE_PERIOD)
input_dim = np.prod(env.observation_space.shape)

# === INIT PPO AGENT ===
agent = PPOAgent(input_dim=input_dim, num_assets=len(tickers),
                 lr=1e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01)

all_rewards = []
all_sharpes = []
final_weights = []

# === TRAINING LOOP ===
for episode in range(EPISODES):
    state = env.reset()
    done = False

    states, actions, rewards, dones = [], [], [], []

    while not done:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        state = next_state

    agent.update(states, actions, rewards, dones)

    total_reward = sum(rewards)
    sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-6)
    all_rewards.append(total_reward)
    all_sharpes.append(sharpe_ratio)
    final_weights.append(actions[-1])

    print(f"Episode {episode + 1}/{EPISODES} - Reward: {total_reward:.2f} - Sharpe: {sharpe_ratio:.2f}")

# === SAVE MODEL ===
torch.save(agent.policy.state_dict(), "ppo_model.pth")

# === SAVE METRICS ===
perf_df = pd.DataFrame({
    "Episode": list(range(1, EPISODES + 1)),
    "Total Reward": all_rewards,
    "Sharpe Ratio": all_sharpes
})
perf_df.to_csv(f"{RESULTS_DIR}/train_performance.csv", index=False)

# === PLOT REWARD CURVE ===
plt.figure()
plt.plot(all_rewards)
plt.title("Episodic Rewards (Mensile)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig(f"{RESULTS_DIR}/train_reward.png")
plt.close()

# === PLOT FINAL WEIGHTS ===
plt.figure(figsize=(12, 6))
plot_portfolio_weights(actions, tickers)
plt.savefig(f"{RESULTS_DIR}/train_pesi.png")
plt.close()

# === SALVA RIBILANCIAMENTI ===
pd.DataFrame({"Rebalance Step": env.rebalance_steps}).to_csv(f"{RESULTS_DIR}/train_ribilanciamenti.csv", index=False)
returns = np.cumsum(rewards)
plt.figure()
plot_cumulative_returns(returns, label="PPO")
for step in env.rebalance_steps:
    plt.axvline(x=step, color='red', linestyle='--', alpha=0.2)
plt.title("Rendimento Cumulato con Ribilanciamenti (Mensile)")
plt.savefig(f"{RESULTS_DIR}/train_reward_cumulativo.png")
plt.close()

# === SALVA PESI ===
export_rebalance_weights_csv(env, tickers, f"{RESULTS_DIR}/rebalance_weights_train.csv")
