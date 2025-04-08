import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# === Percorsi
data_path = "./PPO/main_data_real.csv"
expected_return_path = "./PPO/expected_returns_real.csv"
results_dir = "./PPO/results/"
os.makedirs(results_dir, exist_ok=True)

# === Caricamento dati
df = pd.read_csv(data_path, parse_dates=['date'])
returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()

mu_df = pd.read_csv(expected_return_path, parse_dates=['date'])
mu_series = mu_df.groupby('tic')['predicted_t'].mean()
tickers = mu_series.index.tolist()

S = risk_models.sample_cov(returns)

# === Generazione punti sulla frontiera
risk_free_rate = 0.0
gammas = np.linspace(1, 100, 50)  # più gamma, meno rischio

def compute_portfolio(mu, cov, gamma, transaction_cost=0.0, w_prev=None):
    ef = EfficientFrontier(mu, cov)
    ef.max_quadratic_utility(risk_aversion=gamma)

    w = ef.clean_weights()
    w_vec = np.array([w[t] for t in tickers])

    if w_prev is not None and transaction_cost > 0.0:
        cost = transaction_cost * np.sum(np.abs(w_vec - w_prev))
    else:
        cost = 0.0

    ret = ef.portfolio_performance(verbose=False)[0] - cost
    vol = ef.portfolio_performance(verbose=False)[1]

    return ret * 100, vol * 100, w_vec  # ritorna in %

# === Frontiere
rets_no_tc, vols_no_tc = [], []
rets_tc, vols_tc = [], []

w_prev = np.ones(len(tickers)) / len(tickers)
transaction_cost = 0.01  # 1% lineare (es: 2% vendita + 1% acquisto ≈ media)

for gamma in gammas:
    r1, v1, w1 = compute_portfolio(mu_series, S, gamma, transaction_cost=0.0)
    r2, v2, w2 = compute_portfolio(mu_series, S, gamma, transaction_cost=transaction_cost, w_prev=w_prev)
    rets_no_tc.append(r1)
    vols_no_tc.append(v1)
    rets_tc.append(r2)
    vols_tc.append(v2)
    w_prev = w2

# === Plot
plt.figure(figsize=(10, 6))
plt.plot(vols_no_tc, rets_no_tc, label="No TC", color="blue")
plt.plot(vols_tc, rets_tc, label="Linear TC (1%)", color="red", linestyle="--")

plt.xlabel("Volatility (in %)")
plt.ylabel("Net expected return (in %)")
plt.title("Efficient frontier (with and without transaction costs)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "efficient_frontier_comparison.png"))
plt.close()
print("✅ Frontiera efficiente generata con e senza costi di transazione.")
