# main_evaluation.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from data_loader import load_financial_data
from environment import PortfolioEnv
#from environment_prova import PortfolioEnvRiskAdjusted as PortfolioEnv
from ppo_agent import PPOAgent
from portfolio_optimizer import efficient_frontier
from visualizations import plot_cumulative_returns, plot_portfolio_weights

# === CONFIG ===
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
OUTPUT_DIR = "grafici_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRANSACTION_COSTS = (0.001, 0.001)
DELTAS = (0.0, 0.0)
COST_TYPE = 'linear'

# === LOAD DATA & ENV ===
df, tickers = load_financial_data(CSV_PATH)
env = PortfolioEnv(df, tickers,
                   transaction_cost_type=COST_TYPE,
                   transaction_costs=TRANSACTION_COSTS,
                   deltas=DELTAS,
                   penalty_lambda=0.0)

obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
agent = PPOAgent(input_dim=obs_dim, num_assets=len(tickers))

# === RUN PPO AGENT (senza aggiornamento) ===
obs = env.reset()
done = False
ppo_rewards = []
ppo_weights = []

while not done:
    action, _, _ = agent.select_action(obs)
    obs, reward, done, _ = env.step(action)
    ppo_rewards.append(reward)
    ppo_weights.append(action)

# === BASELINE 1: Equal-weight Portfolio ===
equal_weights = np.ones(len(tickers)) / len(tickers)
obs = env.reset()
done = False
equal_rewards = []

while not done:
    _, reward, done, _ = env.step(equal_weights)
    equal_rewards.append(reward)

# === BASELINE 2 & 3: Portafogli ottimali (mean-variance) ===
mu = df.groupby('ticker')['predicted_t'].mean().values
adj_close = df['adj_close'].unstack()
returns_df = adj_close.pct_change()
cov = returns_df.cov().values
w0 = np.ones(len(mu)) / len(mu)

# Senza costi
frontier_no_cost = efficient_frontier(mu, cov, gamma_range=[1.0])
opt_no_cost = frontier_no_cost[0][2]

# Con costi
frontier_with_cost = efficient_frontier(
    mu, cov, w_prev=w0,
    cost_params={'c_minus': np.full(len(mu), TRANSACTION_COSTS[0]),
                 'c_plus': np.full(len(mu), TRANSACTION_COSTS[1]),
                 'delta_minus': np.full(len(mu), DELTAS[0]),
                 'delta_plus': np.full(len(mu), DELTAS[1])},
    cost_type=COST_TYPE,
    gamma_range=[1.0]
)
opt_with_cost = frontier_with_cost[0][2]

# === Simula strategie ottimali (statiche) ===
def simulate_static_strategy(weights):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        _, reward, done, _ = env.step(weights)
        rewards.append(reward)
    return rewards

mv_rewards = simulate_static_strategy(opt_no_cost)
mvcost_rewards = simulate_static_strategy(opt_with_cost)

# === PLOT CUMULATIVE RETURNS ===
plt.figure(figsize=(10, 6))
plot_cumulative_returns(np.cumsum(ppo_rewards), label="PPO", color='green')
plot_cumulative_returns(np.cumsum(equal_rewards), label="Equal Weight", color='blue')
plot_cumulative_returns(np.cumsum(mv_rewards), label="Mean-Variance", color='orange')
plot_cumulative_returns(np.cumsum(mvcost_rewards), label="MV + Costs", color='red')
plt.title("Confronto Rendimento Cumulato")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "confronto_rendimenti.png"))
plt.show()

# === PLOT PESI PPO ===
plt.figure(figsize=(12, 6))
plot_portfolio_weights(ppo_weights, tickers)
plt.savefig(os.path.join(OUTPUT_DIR, "pesi_ppo_valutazione.png"))
plt.show()
