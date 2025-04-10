from data_loader import load_data, get_rebalancing_dates
from simulate import simulate_portfolio
from logger_weights import PortfolioLogger
from compare_strategies import compare_strategies
from validation_utils import validate_final_allocation

import pandas as pd
import numpy as np
import os

# === Percorsi
data_path = "./main_data_real.csv"
mu_path = "./expected_returns_real.csv"
sigma_path = "./forecasting_data_combined.csv"
results_dir = "./PPO/results/"
os.makedirs(results_dir, exist_ok=True)

# === Dati
df_prices, df_mu, df_sigma = load_data(data_path, mu_path, sigma_path)
rebal_dates = get_rebalancing_dates(sorted(df_prices['date'].unique()), freq='Q')
tickers = sorted(df_prices['tic'].unique())

# === Logger & simulazione
logger = PortfolioLogger(save_dir=results_dir)
portfolio_series = simulate_portfolio(
    df_prices, df_mu, df_sigma, rebal_dates,
    gamma=1.0, cost_rate=0.01, logger=logger
)
logger.save()

# === Strategie Passive
def compute_equal_weight(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    prices = prices[~prices.index.duplicated()]
    weights = np.ones(len(tickers)) / len(tickers)
    shares = (initial_amount * weights) / prices.iloc[0]
    return prices.dot(shares).rename("value").reset_index()

def compute_risk_parity(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    prices = prices[~prices.index.duplicated()]
    returns = prices.pct_change().dropna()
    vol = returns.std()
    inv_vol_weights = 1 / vol
    weights = inv_vol_weights / inv_vol_weights.sum()
    shares = (initial_amount * weights) / prices.iloc[0]
    return prices.dot(shares).rename("value").reset_index()

equal_weight_df = compute_equal_weight(df_prices, tickers)
equal_weight_df.to_csv(os.path.join(results_dir, "equal_weight.csv"), index=False)

risk_parity_df = compute_risk_parity(df_prices, tickers)
risk_parity_df.to_csv(os.path.join(results_dir, "risk_parity.csv"), index=False)

# === RL policy
rl_path = os.path.join(results_dir, "policy_values.csv")
if os.path.exists(rl_path):
    compare_strategies(
        rl_path=rl_path,
        opt_path=os.path.join(results_dir, "portfolio_values.csv"),
        passive_equal=os.path.join(results_dir, "equal_weight.csv"),
        passive_risk=os.path.join(results_dir, "risk_parity.csv"),
        save_path=os.path.join(results_dir, "strategy_comparison_all.png")
    )
    print("✅ Grafico confronto strategie salvato.")
else:
    print("⚠️ RL policy non trovata. Salto confronto.")

# === Validazione finale
validate_final_allocation(
    weights_df_path=os.path.join(results_dir, "weights_log.csv"),
    expected_return_path=mu_path,
    price_data_path=data_path,
    result_dir=results_dir
)
