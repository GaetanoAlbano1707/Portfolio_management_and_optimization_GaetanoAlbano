import pandas as pd
import matplotlib.pyplot as plt

# === 1. Carica il CSV ===
df = pd.read_csv("./Risultati_Forecasting/lstm_forecasting_metrics_leaveoneout.csv")

# === 2. Elimina le colonne non richieste ===
df = df.drop(columns=["Spearman_lstm", "Spearman_only_garch"], errors='ignore')

# === 3. Calcola la riga media ===
average_row = df.iloc[:, 1:].mean()
average_row['ticker'] = 'Average'

# === 4. Aggiungi la riga 'Average' ===
df_with_average = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

# === 5. Arrotonda a 3 cifre significative ===
for col in df_with_average.columns:
    if col != 'ticker':
        df_with_average[col] = pd.to_numeric(df_with_average[col], errors='coerce').apply(
            lambda x: f"{x:.3g}" if pd.notnull(x) else '')

# === 6. Salva il nuovo CSV ===
df_with_average.to_csv("lstm_forecasting_metrics_with_average.csv", index=False)

# === 7. Crea tabella PNG ===
fig, ax = plt.subplots(figsize=(12, 0.6 * len(df_with_average)))
ax.axis('off')

table = ax.table(
    cellText=df_with_average.values,
    colLabels=df_with_average.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# === 8. Aggiungi descrizione sotto ===
plt.figtext(0.5, 0.02, "Tabella 2: Risultati MAE e MSE per i vari tickers", ha='center', fontsize=12)

# === 9. Salva immagine PNG ===
plt.savefig("lstm_forecasting_metrics_with_average.png", bbox_inches='tight', dpi=300)
plt.close()