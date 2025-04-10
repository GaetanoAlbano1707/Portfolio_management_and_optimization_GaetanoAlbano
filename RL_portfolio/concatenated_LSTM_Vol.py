import pandas as pd
import os
from glob import glob

# === Path input/output
path = "./Volatility_Forecasting/Risultati_Forecasting"
files = glob(os.path.join(path, "test_results_*.csv"))
output_path = "./TEST/forecasting_data_combined.csv"
output_path2 = "./PPO/forecasting_data_combined.csv"

all_dfs = []

for file in files:
    # === Estrai ticker dal nome file
    ticker = os.path.basename(file).replace("test_results_", "").replace(".csv", "")

    # === Carica CSV
    df = pd.read_csv(file, parse_dates=["Date"])

    # === Pulizia: sostituisci separatori numerici (es. 12.435.004 -> 12435.004)
    df["LSTM_Vol"] = (
        df["LSTM_Vol"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # === Aggiungi colonna ticker
    df["tic"] = ticker
    df.rename(columns={'Date': 'date'}, inplace=True)
    all_dfs.append(df[["date", "LSTM_Vol", "tic"]])

# === Concatena tutti i DataFrame
combined_df = pd.concat(all_dfs)

# === Ordina e salva
combined_df = combined_df.sort_values(["date", "tic"])
combined_df.to_csv(output_path, index=False)

print(f"✅ File combinato salvato in: {output_path}")
combined_df.to_csv(output_path2, index=False)

print(f"✅ File combinato salvato in: {output_path2}")