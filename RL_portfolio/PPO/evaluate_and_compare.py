import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio_optimization_env import PortfolioOptimizationEnv

# === Percorsi
data_path = "./main_data_real.csv"
expected_return_path = "./expected_returns_real.csv"
forecasting_path = "./forecasting_data_combined.csv"
model_path = "./PPO/models/ppo_portfolio_optimization.zip"
results_dir = "./PPO/results/"
os.makedirs(results_dir, exist_ok=True)

# === Validazione file
for path in [data_path, expected_return_path, forecasting_path, model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Il file {path} non esiste.")

# === Caricamento dati
df = pd.read_csv(data_path, parse_dates=['date']).dropna()
df = df[~df['date'].duplicated(keep='first')]

# === Ambiente RL
def make_env():
    return PortfolioOptimizationEnv(
        data_path=data_path,
        expected_return_path=expected_return_path,
        forecasting_path=forecasting_path,
        initial_amount=100000,
        reward_type='log_return'
    )

env = DummyVecEnv([make_env])
model = PPO.load(model_path)
model.set_env(env)

# === Usa i tickers dall'ambiente (per coerenza!)
tickers = env.get_attr("tickers")[0]

# === Simulazione policy RL
obs = env.reset()
obs = np.nan_to_num(obs)
done = False
portfolio_values = []
weights_log = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    obs = np.nan_to_num(obs)

    portfolio_values.append(env.get_attr("portfolio_value")[0])
    weights_array = np.asarray(env.get_attr("weights")[0]).flatten()
    weights_log.append(weights_array.tolist())

# === Date
dates = sorted(df['date'].unique())[-len(portfolio_values):]

# === Salvataggio output RL
pd.DataFrame({"date": dates, "value": portfolio_values}).to_csv(
    os.path.join(results_dir, "policy_values.csv"), index=False
)

weights_df = pd.DataFrame(weights_log, columns=tickers)
weights_df.insert(0, "date", dates)
weights_df.to_csv(os.path.join(results_dir, "weights_log.csv"), index=False)

# === Strategie passive
def compute_equal_weight(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    prices = prices.dropna(axis=1, how='any')  # rimuove tickers con NaN
    tickers_cleaned = prices.columns.tolist()
    weights = np.ones(len(tickers_cleaned)) / len(tickers_cleaned)
    shares = (initial_amount * weights) / prices.iloc[0]
    return prices.dot(shares)

def compute_risk_parity(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    prices = prices.dropna(axis=1, how='any')  # rimuove tickers con NaN
    returns = prices.pct_change().dropna()
    vol = returns.std()
    inv_vol_weights = 1 / vol
    weights = inv_vol_weights / inv_vol_weights.sum()
    shares = (initial_amount * weights) / prices.iloc[0]
    return prices.dot(shares)


equal_weight = compute_equal_weight(df, tickers)
risk_parity = compute_risk_parity(df, tickers)

# === Allineamento date
policy_portfolio = pd.Series(portfolio_values, index=dates)
common_dates = policy_portfolio.index.intersection(equal_weight.index).intersection(risk_parity.index)

policy_portfolio = policy_portfolio.loc[common_dates]
equal_weight = equal_weight.loc[common_dates]
risk_parity = risk_parity.loc[common_dates]

# === Salvataggio confronto
comparison_df = pd.DataFrame({
    "RL Policy": policy_portfolio,
    "Equal Weight": equal_weight,
    "Risk Parity": risk_parity,
})
comparison_df.to_csv(os.path.join(results_dir, "strategy_comparison.csv"))

# === Plot confronto
plt.figure(figsize=(12, 6))
for col in comparison_df.columns:
    plt.plot(comparison_df[col], label=col)
plt.title("Confronto Strategie di Allocazione")
plt.xlabel("Data")
plt.ylabel("Valore Portafoglio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "strategy_comparison.png"))
plt.close()

print("✅ Confronto strategie completato e salvato.")
