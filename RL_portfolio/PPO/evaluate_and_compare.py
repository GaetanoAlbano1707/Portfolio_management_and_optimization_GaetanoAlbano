import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio_optimization_env import PortfolioOptimizationEnv

# === Percorsi
data_path = "./PPO/main_data_real.csv"
expected_return_path = "./PPO/expected_returns_real.csv"
forecasting_path = "./PPO/forecasting_data_combined.csv"
model_path = "./PPO/models/ppo_portfolio_optimization.zip"
results_dir = "./PPO/results/"
os.makedirs(results_dir, exist_ok=True)

df = pd.read_csv(data_path, parse_dates=['date'])
tickers = df['tic'].unique().tolist()

def make_env():
    return PortfolioOptimizationEnv(
        data_path=data_path,
        expected_return_path=expected_return_path,
        forecasting_path=forecasting_path,
        initial_amount=100000,
        reward_type='log_return'
    )

env = DummyVecEnv([make_env])
model = PPO.load(model_path, env=env)

obs = env.reset()
done = False
portfolio_values = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    portfolio_values.append(env.get_attr('portfolio_value')[0])

# === Serie temporale portafoglio RL
dates = sorted(set(df['date']))[-len(portfolio_values):]
policy_portfolio = pd.Series(portfolio_values, index=dates)

# ✅ Salvataggio curva RL standalone
policy_portfolio.to_frame(name="value").to_csv(os.path.join(results_dir, "policy_values.csv"))

# === Strategie passive
def compute_equal_weight(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    weights = np.ones(len(tickers)) / len(tickers)
    shares = (initial_amount * weights) / prices.iloc[0]
    return prices.dot(shares)

def compute_risk_parity(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    returns = prices.pct_change().dropna()
    vol = returns.std()
    inv_vol_weights = 1 / vol
    weights = inv_vol_weights / inv_vol_weights.sum()
    shares = (initial_amount * weights) / prices.iloc[0]
    return prices.dot(shares)

equal_weight = compute_equal_weight(df, tickers)
risk_parity = compute_risk_parity(df, tickers)

equal_weight = equal_weight.loc[policy_portfolio.index]
risk_parity = risk_parity.loc[policy_portfolio.index]

comparison_df = pd.DataFrame({
    "RL Policy": policy_portfolio,
    "Equal Weight": equal_weight,
    "Risk Parity": risk_parity,
})
comparison_df.to_csv(os.path.join(results_dir, "strategy_comparison.csv"))

# === Grafico confronto
plt.figure(figsize=(12, 6))
plt.plot(policy_portfolio, label="RL Policy")
plt.plot(equal_weight, label="Equal Weight")
plt.plot(risk_parity, label="Risk Parity")
plt.title("Confronto Strategie di Allocazione")
plt.xlabel("Data")
plt.ylabel("Valore Portafoglio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "strategy_comparison.png"))
plt.close()

print("✅ Confronto strategie completato e salvato.")
