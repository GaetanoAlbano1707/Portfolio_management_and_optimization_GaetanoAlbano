import pandas as pd
import glob
import os

# === Parametri ===
start_date = "2020-01-07"
end_date = "2024-12-20"
data_dir = "Tickers_file"
output_file = "TEST/main_data_real.csv"

all_data = []

for file in glob.glob(os.path.join(data_dir, "*_data.csv")):
    ticker = os.path.basename(file).split("_")[0]
    df = pd.read_csv(file, parse_dates=["Date"])

    # Filtra il periodo desiderato
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    # Aggiungi colonna del ticker
    df["tic"] = ticker

    # Rinomina le colonne per consistenza
    df.rename(columns={
        "Date": "date",
        "Adj Close": "adj_close",
        "Close": "close",
        "High": "high",
        "Low": "low",
        "Open": "open",
        "Volume": "volume",
        "return": "return",
        "log_return": "log_return"
    }, inplace=True)

    all_data.append(df)

# Concatena tutto
combined_df = pd.concat(all_data, ignore_index=True)

# Ordina per ticker e data
combined_df.sort_values(by=["tic", "date"], inplace=True)

# Salva
os.makedirs("TEST", exist_ok=True)
combined_df.to_csv(output_file, index=False)
print(f"✅ Creato: {output_file}")
