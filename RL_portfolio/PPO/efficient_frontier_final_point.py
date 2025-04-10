import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, EfficientFrontier

# === Percorsi
results_dir = "./PPO/results/"
data_path = "./main_data_real.csv"
expected_return_path = "./expected_returns_real.csv"
weights_log_path = os.path.join(results_dir, "weights_log.csv")

# === Dati di mercato
df = pd.read_csv(data_path, parse_dates=['date'])
print(f"📅 Dati dal {df['date'].min().date()} al {df['date'].max().date()}")

# === Calcolo rendimenti
returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()
if returns.isna().sum().sum() > 0:
    print("⚠️ Attenzione: ci sono NaN nei rendimenti.")
mu_df = pd.read_csv(expected_return_path)
mu_series = mu_df.groupby('tic')['predicted_t'].mean().sort_index()
tickers = mu_series.index.tolist()

returns = returns[tickers]
S = risk_models.sample_cov(returns)
S *= 252  # 👈 Annualizzazione
mu_series = mu_series.reindex(tickers)

# === Frontiera efficiente
gammas = np.linspace(1, 100, 50)
rets, vols = [], []
for g in gammas:
    ef = EfficientFrontier(mu_series, S)
    ef.max_quadratic_utility(risk_aversion=g)
    r, v, _ = ef.portfolio_performance()
    rets.append(r * 100)
    vols.append(v * 100)

# === Punto finale
# === Punto finale
weights_df = pd.read_csv(weights_log_path)
final_weights = weights_df.iloc[-1][tickers].values.astype(float)
final_weights /= final_weights.sum()  # Normalizza

print(f"▶️ Somma pesi finali: {np.sum(final_weights):.4f}")
print(f"▶️ Mu shape: {mu_series.shape}, Cov shape: {S.shape}, Pesi: {final_weights.shape}")

final_return = np.dot(final_weights, mu_series.values) * 100
final_vol = np.sqrt(np.dot(final_weights, np.dot(S.values, final_weights)))

# === Plot
plt.figure(figsize=(10, 6))
plt.plot(vols, rets, label="Efficient Frontier", color="blue")
plt.scatter(final_vol * 100, final_return, color="red", label="Final Portfolio", zorder=5)
plt.xlabel("Volatility (%)")
plt.ylabel("Expected Return (%)")
plt.title("Efficient Frontier + Final Portfolio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "efficient_frontier_final_point.png"))
plt.close()

print(f"✅ Punto finale tracciato: Return={final_return:.2f}%, Vol={final_vol * 100:.2f}%")
