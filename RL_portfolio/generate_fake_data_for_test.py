import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# === Parametri ===
n_days = 1260  # Giorni di trading (~5 anni)
tickers = ['FAKE1', 'FAKE2', 'FAKE3', 'FAKE4', 'FAKE5', 'FAKE6']
start_date = datetime(2020, 1, 7)
dates = [start_date + timedelta(days=i) for i in range(n_days) if (start_date + timedelta(days=i)).weekday() < 5]

np.random.seed(42)
os.makedirs("TEST", exist_ok=True)

market_data = []
returns_summary = {}

# Generiamo un asset base (FAKE1) e poi altri con correlazione
base_returns = np.random.normal(0, 0.01, len(dates))

returns_dict = {
    'FAKE1': base_returns,
    'FAKE2': base_returns + np.random.normal(0, 0.005, len(dates)),
    'FAKE3': -base_returns + np.random.normal(0, 0.008, len(dates)),  # inversamente correlato
    'FAKE4': np.random.normal(0.001, 0.015, len(dates)),  # trend positivo
    'FAKE5': np.random.normal(-0.001, 0.02, len(dates)),  # trend negativo
    'FAKE6': np.random.normal(0, 0.03, len(dates)),  # volatile
}

for ticker in tickers:
    prices = [100]
    for r in returns_dict[ticker][1:]:
        prices.append(prices[-1] * (1 + r))

    ret_list = []
    for i, date in enumerate(dates):
        adj_close = prices[i]
        close = adj_close * (1 + np.random.normal(0, 0.001))
        open_ = close * (1 + np.random.normal(0, 0.002))
        high = max(close, open_) * (1 + np.random.uniform(0.001, 0.01))
        low = min(close, open_) * (1 - np.random.uniform(0.001, 0.01))
        volume = np.random.randint(1e6, 5e6)
        ret = 0 if i == 0 else (prices[i] - prices[i - 1]) / prices[i - 1]
        log_ret = 0 if i == 0 else np.log(1 + ret)

        ret_list.append(ret)

        market_data.append([
            date.strftime("%Y-%m-%d"), adj_close, close, high, low, open_, volume, ret, log_ret, ticker
        ])

    returns_summary[ticker] = {
        "mean": float(np.mean(ret_list)),
        "std": float(np.std(ret_list))
    }

# === Salvataggio dati mercato
columns = ["date", "adj_close", "close", "high", "low", "open", "volume", "return", "log_return", "tic"]
market_df = pd.DataFrame(market_data, columns=columns)
market_df.to_csv("./TEST/main_data_fake.csv", index=False)
print("✅ Creato: main_data_fake.csv")

# === Expected returns
returns_df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in dates],
    "actual_return": np.random.normal(0, 0.001, len(dates)),
    "predicted_t": np.random.normal(0, 0.001, len(dates))
})
returns_df.to_csv("./TEST/expected_returns_FAKE.csv", index=False)
print("✅ Creato: expected_returns_FAKE.csv")

# === Volatilità previste
def create_vol_file(ticker_name):
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Actual_vol": np.random.uniform(0.005, 0.02, len(dates)),
        "LSTM_Vol": np.random.uniform(0.005, 0.02, len(dates)),
        "GARCH_Vol": np.random.uniform(0.005, 0.02, len(dates))
    })
    df.to_csv(f"./TEST/test_results_{ticker_name}.csv", index=False)
    print(f"✅ Creato: test_results_{ticker_name}.csv")

for t in tickers:
    create_vol_file(t)

# === Logging distribuzioni di ritorni
with open("./TEST/returns_summary.json", "w") as f:
    json.dump(returns_summary, f, indent=2)
print("📈 Riepilogo statistiche rendimento salvato: returns_summary.json")
