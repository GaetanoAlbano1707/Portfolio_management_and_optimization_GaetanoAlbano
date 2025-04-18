import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

from data_loader import load_financial_data
from ppo_agent import PPOAgent
from visualizations import plot_portfolio_weights, plot_cumulative_returns
from environment_prova import PortfolioEnvProva as PortfolioEnv
from save_weights import export_rebalance_weights_csv
from compare_strategies_monthly import run_equal_weights_strategy
from compute_metrics import compute_monthly_metrics
from utils_data_split import train_test_split_df
import itertools

# === CONFIG ===
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
EPISODES = 40 #40 PER LA GRID SEARCH, POI USIAMO 100
PATIENCE = 5  # Early stopping patience #5 PER LA GRID SEARCH, POI USIAMO 10
WINDOW_SIZE = 5
BASE_MODEL_DIR = "models/monthly"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# === LOAD DATA ===
df, tickers = load_financial_data(CSV_PATH)
df_train, _ = train_test_split_df(df, train_ratio=0.8)

with open("cov_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

# === Definizione spazio degli iperparametri ===
lr_list = [1e-4, 5e-4]
gamma_list = [0.95, 0.99]
clip_epsilon_list = [0.1, 0.2]
entropy_coef_list = [0.0, 0.01]
c_plus_list = [0.0005, 0.001, 0.002]
c_minus_list = [0.0005, 0.001, 0.002]
delta_plus_list = [0.0005, 0.001]
delta_minus_list = [0.0005, 0.001]

# === Generazione combinazioni ===
hyperparameter_combinations = list(itertools.product(
    lr_list, gamma_list, clip_epsilon_list, entropy_coef_list,
    c_plus_list, c_minus_list, delta_plus_list, delta_minus_list
))

# === Sweep per ogni combinazione e livello di rischio ===
for lambda_risk in [0.0, 0.05, 0.1, 0.2]:
    for (lr, gamma, clip_epsilon, entropy_coef,
         c_plus, c_minus, delta_plus, delta_minus) in hyperparameter_combinations:

        config_name = (f"lr_{lr}_gamma_{gamma}_clip_{clip_epsilon}_entropy_{entropy_coef}_"
                       f"cplus_{c_plus}_cminus_{c_minus}_deltaplus_{delta_plus}_deltaminus_{delta_minus}")
        RESULTS_DIR = os.path.join(BASE_MODEL_DIR, f"lambda_{lambda_risk}", config_name, "results")
        MODEL_DIR = os.path.join(BASE_MODEL_DIR, f"lambda_{lambda_risk}", config_name)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        #Evito configurazioni con stessi iperparametri
        if os.path.exists(os.path.join(RESULTS_DIR, "train_performance.csv")):
            print(f"🔁 Già eseguito: {config_name} - Lambda {lambda_risk}")
            continue

        # Setup dell'ambiente
        env = PortfolioEnv(df_train, tickers, window_size=WINDOW_SIZE, cov_matrices=cov_matrices,
                           lambda_risk=lambda_risk, c_plus=c_plus, c_minus=c_minus,
                           delta_plus=delta_plus, delta_minus=delta_minus)
        input_dim = np.prod(env.observation_space.shape)

        # Inizializzazione dell'agente PPO con i parametri correnti
        agent = PPOAgent(input_dim=input_dim, num_assets=len(tickers),
                         lr=lr, gamma=gamma, clip_epsilon=clip_epsilon, entropy_coef=entropy_coef)

        all_rewards = []
        all_sharpes = []
        final_weights = []

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
            sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-6)
            all_rewards.append(total_reward)
            all_sharpes.append(sharpe_ratio)
            final_weights.append(actions[-1])

            print(f"Lambda {lambda_risk} - Episode {episode + 1}/{EPISODES} - "
                  f"Reward: {total_reward:.2f} - Sharpe: {sharpe_ratio:.2f}")

            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                wait = 0
            else:
                wait += 1

            if wait >= PATIENCE:
                print("🛑 Early stopping attivato: performance stabile")
                break

        # === SAVE MODEL ===
        torch.save(agent.policy.state_dict(), os.path.join(MODEL_DIR, "ppo_model.pth"))

        # === SAVE METRICS ===
        perf_df = pd.DataFrame({
            "Episode": list(range(1, len(all_rewards) + 1)),
            "Total Reward": all_rewards,
            "Sharpe Ratio": all_sharpes
        })
        perf_df.to_csv(os.path.join(RESULTS_DIR, "train_performance.csv"), index=False)


        # === PLOT REWARD CURVE ===
        plt.figure()
        plt.plot(all_rewards)
        plt.title(f"Episodic Rewards (Lambda {lambda_risk})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "train_reward.png"))
        plt.close()

        # === PLOT FINAL WEIGHTS ===
        plt.figure(figsize=(12, 6))
        plot_portfolio_weights(actions, tickers)
        plt.savefig(os.path.join(RESULTS_DIR, "train_pesi.png"))
        plt.close()

        # === SALVA RIBILANCIAMENTI ===
        pd.DataFrame({"Rebalance Step": env.rebalance_steps}).to_csv(os.path.join(RESULTS_DIR, "train_ribilanciamenti.csv"), index=False)
        returns = np.cumsum(rewards)
        plt.figure()
        plot_cumulative_returns(returns, label="PPO")
        for step in env.rebalance_steps:
            plt.axvline(x=step, color='red', linestyle='--', alpha=0.2)
        plt.title(f"Rendimento Cumulato con Ribilanciamenti (Lambda {lambda_risk})")
        plt.savefig(os.path.join(RESULTS_DIR, "train_reward_cumulativo.png"))
        plt.close()

        # === SALVA PESI ===
        export_rebalance_weights_csv(env, tickers, os.path.join(RESULTS_DIR, "rebalance_weights_train.csv"))

        # === BENCHMARK EQUAL WEIGHTS ===
        equal_vals = run_equal_weights_strategy(df_train, tickers, rebalance_period=63)
        ppo_vals = np.cumsum(rewards)
        plt.plot(equal_vals, label="Equal Weights")
        plt.plot(ppo_vals, label="PPO")
        plt.legend()
        plt.title(f"Strategie a Confronto (Train) - Lambda {lambda_risk}")
        plt.savefig(os.path.join(RESULTS_DIR, "train_compare_strategies.png"))
        plt.close()


        # === METRICHE TRIMESTRALI ===
        dates = df_train.index.get_level_values("date").unique().sort_values()
        dates = dates[WINDOW_SIZE + 1 : WINDOW_SIZE + 1 + len(returns)]
        if len(dates) > len(returns):
            dates = dates[:len(returns)]
        elif len(returns) > len(dates):
            returns = returns[:len(dates)]
        compute_monthly_metrics(returns, dates, save_path=os.path.join(RESULTS_DIR, "monthly_metrics.csv"))
