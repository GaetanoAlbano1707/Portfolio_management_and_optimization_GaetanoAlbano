import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

# === Percorsi
data_path = "./main_data_real.csv"
expected_return_path = "./expected_returns_real.csv"
results_dir = "./PPO/results/"
os.makedirs(results_dir, exist_ok=True)

# === Caricamento dati
df = pd.read_csv(data_path, parse_dates=['date'])
returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()

mu_df = pd.read_csv(expected_return_path)
mu_series = mu_df.groupby('tic')['predicted_t'].mean().sort_index()
tickers = mu_series.index.tolist()

S = risk_models.sample_cov(returns[tickers])

# === Funzione utility
def compute_portfolio(mu, cov, gamma, transaction_cost=0.0, w_prev=None):
    ef = EfficientFrontier(mu, cov)
    ef.max_quadratic_utility(risk_aversion=gamma)
    weights_dict = ef.clean_weights()
    w_vec = np.array([weights_dict[t] for t in mu.index])

    cost = transaction_cost * np.sum(np.abs(w_vec - w_prev)) if w_prev is not None else 0.0
    ret, vol, _ = ef.portfolio_performance(verbose=False)
    return (ret - cost) * 100, vol * 100, w_vec

# === Generazione frontiere
gammas = np.linspace(1, 100, 50)
rets_no_tc, vols_no_tc = [], []
rets_tc, vols_tc = [], []
w_prev = np.ones(len(tickers)) / len(tickers)
tc_rate = 0.01  # 1% costi transazionali

for gamma in gammas:
    r1, v1, _ = compute_portfolio(mu_series, S, gamma)
    r2, v2, w2 = compute_portfolio(mu_series, S, gamma, transaction_cost=tc_rate, w_prev=w_prev)

    rets_no_tc.append(r1)
    vols_no_tc.append(v1)
    rets_tc.append(r2)
    vols_tc.append(v2)
    w_prev = w2

# === Plot finale
plt.figure(figsize=(10, 6))
plt.plot(vols_no_tc, rets_no_tc, label="No TC", color="blue", linewidth=2)
plt.plot(vols_tc, rets_tc, label="Linear TC (1%)", color="red", linestyle="--", linewidth=2)
plt.xlabel("Volatility (in %)", fontsize=12)
plt.ylabel("Net expected return (in %)", fontsize=12)
plt.title("Efficient frontier (with and without transaction costs)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "efficient_frontier_comparison.png"))
plt.close()

print("✅ Frontiera efficiente generata con e senza costi di transazione.")
