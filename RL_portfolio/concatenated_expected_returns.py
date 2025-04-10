import pandas as pd

# === Carica il file originale
df = pd.read_csv("./Rendimenti_attesi/concatenated_returns_MC.csv", parse_dates=["Date"])

# === Rinomina colonne per coerenza con il sistema esistente
df = df.rename(columns={
    "Date": "date",
    "predicted_return": "predicted_t",
    "ticker": "tic"
})

# === Seleziona solo le colonne utili
df = df[["date", "predicted_t", "tic", "predicted_t_std"]]

# === Salva in formato utilizzabile dal codice
df.to_csv("TEST/expected_returns_real.csv", index=False)
df.to_csv("PPO/expected_returns_real.csv", index=False)
print("✅ File expected_returns_real.csv creato.")
