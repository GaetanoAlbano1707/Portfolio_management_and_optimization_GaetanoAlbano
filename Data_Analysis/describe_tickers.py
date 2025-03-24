import pandas as pd

# 1) Carica il CSV con i log_return di vari ticker
df = pd.read_csv("merged_log_returns.csv", parse_dates=['Date'])

# 2) Rimuovi la colonna 'Date'
df_no_date = df.drop(columns=['Date'])

# 3) Esegui la descrizione statistica e trasponi per avere i log_return come righe
desc = df_no_date.describe(percentiles=[0.25, 0.5, 0.75]).T

# 4) Calcola skewness e kurtosis
skew_vals = df_no_date.skew()
kurt_vals = df_no_date.kurt()

# 5) Aggiungi skewness e kurtosis alla tabella
desc['Skewness'] = skew_vals
desc['Kurtosis'] = kurt_vals

# 6) Rinomina alcune colonne se vuoi
desc = desc.rename(columns={
    'std': 'Std. Dev.',
    '25%': '25%',
    '50%': '50%',
    '75%': '75%'
})

# 7) Rimuovi il prefisso "log_return_" dall'indice
#    (Assumiamo che tutte le colonne abbiano quel prefisso)
desc.index = desc.index.str.replace("log_return_", "")

# 8) Trasforma l'indice in una colonna "tickers"
desc['tickers'] = desc.index

# 9) Reimposta l'indice (diventa un range index di default)
desc.reset_index(drop=True, inplace=True)

# 10) Se vuoi che la colonna "tickers" sia proprio la prima,
#     puoi ridefinire l'ordine delle colonne:
cols_order = [
    'tickers', 'count', 'mean', 'Std. Dev.', 'min',
    '25%', '50%', '75%', 'max', 'Skewness', 'Kurtosis'
]
desc = desc[cols_order]

# 11) Salva o stampa il risultato
desc.to_csv(
    "Descrizione_statistiche_assets.csv",
    float_format="%.4f",
    sep=';',         # Usa ; come separatore di colonna
    decimal=',',     # Usa la virgola come separatore decimale
    index=False
)
print(desc)
