import pandas as pd
import pickle

# Percorso del tuo file CSV
CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"

# Carica il CSV
df = pd.read_csv(CSV_PATH, index_col=0)

# Seleziona le colonne che terminano con 'adj_close'
adj_close_columns = [col for col in df.columns if col.endswith('adj_close')]

# Crea un nuovo DataFrame con solo le colonne 'adj_close'
adj_close_df = df[adj_close_columns]

# Calcola i rendimenti percentuali giornalieri
returns = adj_close_df.pct_change().dropna()

# Calcola le matrici di covarianza con una finestra mobile di 63 giorni
cov_matrices = {}
for i in range(63, len(returns)):
    window_returns = returns.iloc[i - 63:i]
    cov_date = returns.index[i]
    cov_matrices[cov_date] = window_returns.cov()

# Salva le matrici di covarianza in un file pickle
with open("cov_matrices.pkl", "wb") as f:
    pickle.dump(cov_matrices, f)

print(f"✅ Generato {len(cov_matrices)} matrici di covarianza e salvato in 'cov_matrices.pkl'")
