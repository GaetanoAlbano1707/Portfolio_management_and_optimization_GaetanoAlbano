import pandas as pd

# Carica il file con i log return
df = pd.read_csv("./TEST/main_data_real.csv", parse_dates=["date"])

# Filtra il periodo
df = df[(df["date"] >= "2020-01-07") & (df["date"] <= "2024-12-20")]

# Pivota in formato wide: ogni colonna è un ticker
df_wide = df.pivot(index="date", columns="tic", values="log_return")

# Salva il risultato
df_wide.to_csv("./TEST/log_returns_for_covariance.csv")

print("✅ File salvato in formato wide per la covarianza.")
