import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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
from compare_strategies import (
    run_equal_weights_strategy,
    run_minimum_variance_strategy,
    run_markowitz_strategy,
    run_random_strategy,
    run_buy_and_hold_strategy,
    run_buy_and_hold_portfolio
)
from evaluate_strategy_metrics import evaluate_strategies
from risk_performance_comparison import summarize_cumulative_returns

# === CONFIG ===
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
WINDOW_SIZE = 5
BASE_RESULTS_DIR = "results_ppo_risk_sweep_refined"
MODEL_BASE_PATH = "models/quarterly_best_configs"
TOP_CONFIGS_JSON = "top_configs_by_lambda.json"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# === LOAD DATA ===
df, tickers = load_financial_data(CSV_PATH)
_, df_test = train_test_split_df(df, train_ratio=0.8)

with open("cov_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

# === LOAD TOP CONFIGURATIONS ===
with open(TOP_CONFIGS_JSON, "r") as f:
    best_hyperparameters = json.load(f)

cumulative_returns_by_lambda = {}

for lambda_risk_str, config_list in best_hyperparameters.items():
    lambda_risk = float(lambda_risk_str)
    for config_name in config_list:
        result_dir = os.path.join(MODEL_BASE_PATH, f"lambda_{lambda_risk}", config_name, "results")
        os.makedirs(result_dir, exist_ok=True)

        env = PortfolioEnv(df_test, tickers, window_size=WINDOW_SIZE, cov_matrices=cov_matrices, lambda_risk=lambda_risk)
        input_dim = np.prod(env.observation_space.shape)

        agent = PPOAgent(input_dim=input_dim, num_assets=len(tickers))
        model_path = os.path.join(MODEL_BASE_PATH, f"lambda_{lambda_risk}", config_name, "ppo_model.pth")
        agent.policy.load_state_dict(torch.load(model_path))

        state = env.reset()
        done = False
        rewards, actions = [], []

        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            state = next_state

        ppo_log_returns = env.portfolio_returns
        ppo_cum_returns = np.exp(np.cumsum(ppo_log_returns))
        key = f"{lambda_risk}_{config_name}"
        cumulative_returns_by_lambda[key] = ppo_cum_returns

        pd.DataFrame({"Log Return": ppo_log_returns, "Cumulative Return": ppo_cum_returns}).to_csv(
            f"{result_dir}/test_performance.csv", index=False)

        with open(f"{result_dir}/actions_test.pkl", "wb") as f:
            pickle.dump(actions, f)

        plt.figure()
        plot_cumulative_returns(ppo_cum_returns, label=f"Reinforcement_Learning λ={lambda_risk}")
        for step in env.rebalance_steps:
            plt.axvline(x=step, color='red', linestyle='--', alpha=0.2)
        plt.title(f"Rendimento Cumulato - λ={lambda_risk}")
        plt.savefig(f"{result_dir}/test_reward_cumulativo.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plot_portfolio_weights(actions, tickers)
        plt.savefig(f"{result_dir}/test_pesi.png")
        plt.close()

        pd.DataFrame({"Rebalance Step": env.rebalance_steps}).to_csv(f"{result_dir}/test_ribilanciamenti.csv", index=False)
        analyze_dominant_weights(actions, tickers)
        export_rebalance_weights_csv(env, tickers, f"{result_dir}/rebalance_weights_test.csv")
        plot_grouped_bar_weights(f"{result_dir}/rebalance_weights_test.csv", f"{result_dir}/grouped_bar_weights_by_quarter.png")

        equal_vals = run_equal_weights_strategy(df_test, tickers, rebalance_period=63)
        ppo_dates = equal_vals.index[:len(ppo_cum_returns)]
        ppo_series = pd.Series(ppo_cum_returns, index=ppo_dates)

        plt.figure()
        plt.plot(equal_vals, label="Equal Weights")
        plt.plot(ppo_series, label="Reinforcement Learning")
        plt.legend()
        plt.title("Strategie a Confronto")
        plt.savefig(f"{result_dir}/test_compare_strategies.png")
        plt.close()

        daily_returns = df_test["adj_close"].unstack().pct_change().dropna()
        mean_returns = daily_returns.mean().values * 252
        cov_matrix = daily_returns.cov().values * 252

        c_plus, c_minus = 0.01, 0.02
        results = simulate_cost_frontier(cov_matrix, mean_returns, c_plus=c_plus, c_minus=c_minus)
        plot_efficient_frontier(results, c_plus=c_plus, c_minus=c_minus, filename=f"{result_dir}/test_efficient_frontier.png")

        configs = [(0.001, 0.002), (0.005, 0.01), (0.01, 0.02)]
        generate_multiple_frontiers(cov_matrix, mean_returns, configs, base_filename=f"{result_dir}/test_efficient_frontier")

        dates = df_test.index.get_level_values("date").unique().sort_values()
        dates = dates[WINDOW_SIZE + 1 : WINDOW_SIZE + 1 + len(ppo_log_returns)]
        if len(dates) > len(ppo_log_returns):
            dates = dates[:len(ppo_log_returns)]
        elif len(ppo_log_returns) > len(dates):
            ppo_log_returns = ppo_log_returns[:len(dates)]

        compute_quarterly_metrics(ppo_log_returns, dates, save_path=f"{result_dir}/test_quarterly_metrics.csv")

        strategies = {
            "Equal Weight": run_equal_weights_strategy(df_test, tickers, rebalance_period=63),
            "Minimum Variance": run_minimum_variance_strategy(df_test, tickers, rebalance_period=63),
            "Markowitz": run_markowitz_strategy(df_test, tickers, rebalance_period=63),
            "Random": run_random_strategy(df_test, tickers, rebalance_period=63),
            "Buy & Hold (Asset)": run_buy_and_hold_strategy(df_test, tickers, asset_index=0),
            "Buy & Hold (Portfolio)": run_buy_and_hold_portfolio(df_test, tickers),
            "Reinforcement Learning": pd.Series(ppo_cum_returns, index=ppo_dates)
        }

        comparison_df = pd.DataFrame(strategies)
        comparison_df.to_csv(f"{result_dir}/model_comparison_returns.csv", index=True)
        evaluate_strategies(comparison_df, output_path=f"{result_dir}/strategy_risk_metrics.csv")

        plt.figure(figsize=(12, 6))
        for name, series in strategies.items():
            plt.plot(series.index, series.values, label=name)
        plt.legend()
        plt.title("Confronto Strategie")
        plt.xlabel("Data")
        plt.ylabel("Rendimento Cumulativo")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{result_dir}/compare_models_test.png")
        plt.close()

# === PLOT COMPARATIVO ===
plt.figure(figsize=(10, 6))
for label, cum_returns in cumulative_returns_by_lambda.items():
    plt.plot(cum_returns, label=f"λ {label}")
plt.legend()
plt.title("Confronto Rendimento Cumulato per Diversi λ e Configurazioni Ottimali")
plt.xlabel("Step temporali")
plt.ylabel("Rendimento Cumulativo")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{BASE_RESULTS_DIR}/compare_lambda_risk_trimestrale_refined.png")
plt.close()

# === CSV RIEPILOGO ===
summarize_cumulative_returns(BASE_RESULTS_DIR)
