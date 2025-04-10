import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns, EfficientFrontier

# === Percorsi
results_dir = "./PPO/results/"
weights_path = os.path.join(results_dir, "weights_log.csv")
mu_path = "./expected_returns_real.csv"
price_path = "./main_data_real.csv"

# === Caricamento pesi finali
df_weights = pd.read_csv(weights_path)
final_row = df_weights.iloc[-1]
tickers = final_row.index[1:]
weights = final_row[tickers].values.astype(float)
weights /= weights.sum()  # Normalizza

# === Caricamento mu e rendimenti storici
mu_df = pd.read_csv(mu_path)
mu = mu_df.groupby('tic')['predicted_t'].mean().reindex(tickers).values

df = pd.read_csv(price_path, parse_dates=['date'])
print(f"📅 Dati disponibili da {df['date'].min().date()} a {df['date'].max().date()}")
returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()
returns = returns[tickers]

if returns.isna().sum().sum() > 0:
    print("⚠️ NaN rilevati nei rendimenti.")

cov_matrix = risk_models.sample_cov(returns)
cov_matrix *= 252  # 👈 Annualizzazione

# === Calcolo metrica portafoglio finale
ret = np.dot(weights, mu)
vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))

# === Calcolo costi di transazione e rendimento netto
initial_weights = np.ones_like(weights) / len(weights)
transaction_cost = 0.001
cost_final = transaction_cost * np.sum(np.abs(weights - initial_weights))
ret_net = ret - cost_final

# === Salvataggio CSV con metrica estesa
pd.Series({
    "expected_return_%": round(ret * 100, 4),
    "volatility_%": round(vol * 100, 4),
    "transaction_cost_total_%": round(cost_final * 100, 4),
    "net_expected_return_%": round(ret_net * 100, 4)
}).to_csv(os.path.join(results_dir, "final_allocation_metrics.csv"))

# === Frontiera efficiente
gammas = np.linspace(1, 100, 50)
rets, vols = [], []

for g in gammas:
    ef = EfficientFrontier(mu, cov_matrix)
    ef.max_quadratic_utility(risk_aversion=g)
    r, v, _ = ef.portfolio_performance()
    rets.append(r * 100)
    vols.append(v * 100)

# === Plot
plt.figure(figsize=(10, 6))
plt.plot(vols, rets, label="Efficient Frontier", color="blue")
plt.scatter(vol * 100, ret * 100, color="red", label="Final Portfolio", zorder=5)
plt.xlabel("Volatility (%)")
plt.ylabel("Expected Return (%)")
plt.title("Efficient Frontier + Final Portfolio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_point_on_frontier.png"))
plt.close()

# ✅ Log corretto
print(f"✅ Validazione completata: Rendimento atteso = {ret*100:.2f}%, Netto = {ret_net*100:.2f}%, Volatilità = {vol*100:.2f}%")
