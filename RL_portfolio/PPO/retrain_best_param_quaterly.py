import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from data_loader import load_financial_data
from ppo_agent import PPOAgent
from visualizations import plot_portfolio_weights, plot_cumulative_returns
from environment_prova import PortfolioEnvProva as PortfolioEnv
from save_weights import export_rebalance_weights_csv
from compare_strategies import run_equal_weights_strategy
from compute_metrics import compute_quarterly_metrics
from utils_data_split import train_test_split_df


# === CONFIG ===
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
TOP_CONFIGS_JSON = "top_configs_by_lambda.json"
BASE_MODEL_DIR = "models/quarterly_best_configs_retrain"
EPISODES = 100
PATIENCE = 10
WINDOW_SIZE = 5

# === LOAD DATA ===
df, tickers = load_financial_data(CSV_PATH)
df_train, _ = train_test_split_df(df, train_ratio=0.8)

with open("cov_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

with open(TOP_CONFIGS_JSON, "r") as f:
    top_configs = json.load(f)

# === RETRAINING LOOP ===
for lambda_str, configs in top_configs.items():
    lambda_risk = float(lambda_str)
    for config_name in configs:
        print(f"\n🚀 Retraining: λ={lambda_risk} | {config_name}")

        # === PARSE CONFIG ===
        parts = config_name.split("_")
        config = {parts[i]: float(parts[i + 1]) for i in range(0, len(parts), 2)}
        lr = config["lr"]
        gamma = config["gamma"]
        clip_epsilon = config["clip"]
        entropy_coef = config["entropy"]
        c_plus = config["cplus"]
        c_minus = config["cminus"]
        delta_plus = config["deltaplus"]
        delta_minus = config["deltaminus"]

        # === PATH SETUP ===
        MODEL_DIR = os.path.join(BASE_MODEL_DIR, f"lambda_{lambda_risk}", config_name)
        RESULTS_DIR = os.path.join(MODEL_DIR, "results")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # === ENV & AGENT ===
        env = PortfolioEnv(df_train, tickers, window_size=WINDOW_SIZE, cov_matrices=cov_matrices,
                           lambda_risk=lambda_risk, c_plus=c_plus, c_minus=c_minus,
                           delta_plus=delta_plus, delta_minus=delta_minus)
        input_dim = np.prod(env.observation_space.shape)
        agent = PPOAgent(input_dim=input_dim, num_assets=len(tickers),
                         lr=lr, gamma=gamma, clip_epsilon=clip_epsilon, entropy_coef=entropy_coef)

        all_rewards, all_sharpes, final_weights = [], [], []
        best_sharpe = -np.inf
        wait = 0

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
            sharpe = np.mean(rewards) / (np.std(rewards) + 1e-6)
            all_rewards.append(total_reward)
            all_sharpes.append(sharpe)
            final_weights.append(actions[-1])

            print(f"[λ={lambda_risk}] Ep {episode + 1}/{EPISODES} - Reward: {total_reward:.2f} - Sharpe: {sharpe:.3f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                print("🛑 Early stopping attivato")
                break

        # === SAVE MODEL & METRICS ===
        torch.save(agent.policy.state_dict(), os.path.join(MODEL_DIR, "ppo_model.pth"))
        pd.DataFrame({
            "Episode": range(1, len(all_rewards) + 1),
            "Total Reward": all_rewards,
            "Sharpe Ratio": all_sharpes
        }).to_csv(os.path.join(RESULTS_DIR, "train_performance.csv"), index=False)

        # === PLOT CURVE ===
        plt.figure()
        plt.plot(all_rewards)
        plt.title(f"Episodic Rewards (λ={lambda_risk})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "train_reward.png"))
        plt.close()

        # === SAVE PORTFOLIO & WEIGHTS ===
        plt.figure(figsize=(12, 6))
        plot_portfolio_weights(actions, tickers)
        plt.savefig(os.path.join(RESULTS_DIR, "train_pesi.png"))
        plt.close()

        pd.DataFrame({"Rebalance Step": env.rebalance_steps}).to_csv(
            os.path.join(RESULTS_DIR, "train_ribilanciamenti.csv"), index=False)
        returns = np.cumsum(rewards)

        plt.figure()
        plot_cumulative_returns(returns, label="PPO")
        for step in env.rebalance_steps:
            plt.axvline(x=step, color='red', linestyle='--', alpha=0.2)
        plt.title(f"Rendimento Cumulato (λ={lambda_risk})")
        plt.savefig(os.path.join(RESULTS_DIR, "train_reward_cumulativo.png"))
        plt.close()

        export_rebalance_weights_csv(env, tickers, os.path.join(RESULTS_DIR, "rebalance_weights_train.csv"))

        # === METRICHE QUARTERLY ===
        dates = df_train.index.get_level_values("date").unique().sort_values()
        dates = dates[WINDOW_SIZE + 1: WINDOW_SIZE + 1 + len(returns)]
        if len(dates) > len(returns):
            dates = dates[:len(returns)]
        elif len(returns) > len(dates):
            returns = returns[:len(dates)]

        compute_quarterly_metrics(returns, dates, save_path=os.path.join(RESULTS_DIR, "train_quarterly_metrics.csv"))
