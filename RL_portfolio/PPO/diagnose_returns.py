import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === File di input
file_path = "./main_data_real.csv"

# === Caricamento dati
df = pd.read_csv(file_path, parse_dates=["date"])
prices = df.pivot(index="date", columns="tic", values="adj_close")
returns = prices.pct_change().dropna()

# === 1. Statistiche per asset
stats = pd.DataFrame({
    "mean": returns.mean(),
    "std": returns.std(),
    "min": returns.min(),
    "max": returns.max()
}).sort_values("std", ascending=False)

print("\n🔍 Top 5 asset per volatilità:")
print(stats.head(5).round(4))

# === 2. Giorni con variazioni estreme
abs_returns = returns.abs()
max_by_day = abs_returns.max(axis=1)
top_days = max_by_day.sort_values(ascending=False).head(5)

print("\n📆 Top 5 giorni con variazioni estreme:")
for date in top_days.index:
    print(f"{date.date()} → max move: {max_by_day[date]:.4f}")
    print(returns.loc[date].sort_values(ascending=False).to_string())
    print()

# === 3. Heatmap visiva (solo se non troppi asset)
if returns.shape[1] <= 30:
    plt.figure(figsize=(12, 6))
    sns.heatmap(returns.T, cmap="RdBu", center=0, cbar_kws={"label": "Return"})
    plt.title("📊 Rendimenti giornalieri per asset")
    plt.xlabel("Data")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.show()
else:
    print("🎨 Troppi asset per generare heatmap efficace.")

