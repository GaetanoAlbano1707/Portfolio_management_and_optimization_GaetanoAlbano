import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns, EfficientFrontier

results_dir = "./PPO/results/"
data_path = "./PPO/main_data_real.csv"
expected_return_path = "./PPO/expected_returns_real.csv"
weights_log_path = os.path.join(results_dir, "weights_log.csv")

# === Dati
df = pd.read_csv(data_path, parse_dates=['date'])
returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()
mu_df = pd.read_csv(expected_return_path)
mu = mu_df.groupby('tic')['predicted_t'].mean()
tickers = mu.index.tolist()
S = risk_models.sample_cov(returns)

# === Frontiera
gammas = np.linspace(1, 100, 50)
rets, vols = [], []
for g in gammas:
    ef = EfficientFrontier(mu, S)
    ef.max_quadratic_utility(risk_aversion=g)
    r, v, _ = ef.portfolio_performance()
    rets.append(r * 100)
    vols.append(v * 100)

# === Punti del portafoglio finale
weights_df = pd.read_csv(weights_log_path)
final_weights = weights_df.iloc[-1][tickers].values
final_return = np.dot(final_weights, mu.values) * 100
final_vol = np.sqrt(np.dot(final_weights, np.dot(S.values, final_weights))) * 100

# === Plot
plt.figure(figsize=(10, 6))
plt.plot(vols, rets, label="Efficient Frontier", color="blue")
plt.scatter(final_vol, final_return, color="red", label="Final Portfolio")
plt.xlabel("Volatility (%)")
plt.ylabel("Expected Return (%)")
plt.title("Efficient Frontier + Final Portfolio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "efficient_frontier_final_point.png"))
plt.close()
print(f"✅ Punto finale tracciato sulla frontiera: Return={final_return:.2f}%, Vol={final_vol:.2f}%")
