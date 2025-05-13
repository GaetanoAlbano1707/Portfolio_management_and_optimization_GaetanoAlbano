import pandas as pd
import matplotlib.pyplot as plt

# === 1. Carica e pulisci i dati ===
df = pd.read_csv("./Risultati_Forecasting/lstm_forecasting_metrics_leaveoneout.csv")
df = df.drop(columns=["Spearman_lstm", "Spearman_only_garch"], errors='ignore')

# Aggiungi riga Average
avg = df.iloc[:, 1:].mean()
avg['ticker'] = 'Average'
df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

# === 2. Formatta con 3 decimali fissi ===
for col in df.columns:
    if col != 'ticker':
        df[col] = pd.to_numeric(df[col], errors='coerce')\
                    .apply(lambda x: f"{x:.3f}" if pd.notnull(x) else '')

# === 3. Crea figura e tabella ===
fig, ax = plt.subplots(figsize=(12, 0.5 * len(df)))
ax.axis('off')
tbl = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)

# === 4. Forza il rendering per misurare la tabella ===
fig.canvas.draw()

# === 5. Calcola il bordo inferiore della tabella ===
renderer = fig.canvas.get_renderer()
bb = tbl.get_window_extent(renderer)
bb_ax = bb.transformed(ax.transAxes.inverted())
y_bottom = bb_ax.y0

# === 6. Aggiungi la caption subito sotto ===
caption = "Tabella 2: Risultati MAE e MSE per i vari tickers"
ax.text(
    0.5,
    y_bottom - 0.01,
    caption,
    ha='center',
    va='top',
    fontsize=12,
    transform=ax.transAxes,
    clip_on=False
)

# === 7. Salva l’immagine ===
plt.savefig(
    "lstm_forecasting_metrics_with_average.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)
plt.close()

# === 8. Salva il CSV con i 3 decimali ===
df.to_csv("lstm_forecasting_metrics_with_average.csv", index=False)
