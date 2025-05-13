import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

def salva_tabella_png(df, title, filename):
    fig, ax = plt.subplots(figsize=(min(20, df.shape[1] * 1.2), max(2.5, df.shape[0] * 0.5)))
    ax.axis('off')
    tab = table(ax, df, loc='center', cellLoc='center', colWidths=[0.1]*len(df.columns))
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.2, 1.2)
    plt.title(title, fontsize=10, pad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def processa_rebalance_weights(base_path, label_prefix):
    for lambda_dir in sorted(os.listdir(base_path)):
        lambda_path = os.path.join(base_path, lambda_dir)
        if not os.path.isdir(lambda_path) or not lambda_dir.startswith("lambda_"):
            continue

        config_dirs = [d for d in os.listdir(lambda_path) if os.path.isdir(os.path.join(lambda_path, d))]
        if not config_dirs:
            print(f"Nessuna configurazione trovata in {lambda_path}")
            continue

        config_path = os.path.join(lambda_path, config_dirs[0], "results", "rebalance_weights_test.csv")
        if not os.path.exists(config_path):
            print(f"⚠️ File mancante: {config_path}")
            continue

        try:
            df = pd.read_csv(config_path, index_col=0)

            # Rimuove eventuali colonne come 'step'
            df = df.loc[:, ~df.columns.str.lower().str.contains("step")]

            # Converti in percentuali leggibili
            df_percentuali = (df * 100).round(3)

            csv_name = f"allocazioni_percentuali_{label_prefix}_{lambda_dir}.csv"
            png_name = f"allocazioni_percentuali_{label_prefix}_{lambda_dir}.png"
            df_percentuali.to_csv(csv_name)
            salva_tabella_png(df_percentuali, f"{label_prefix.upper()} {lambda_dir}", png_name)

            print(f"✅ Salvati: {csv_name}, {png_name}")

        except Exception as e:
            print(f"Errore leggendo {config_path}: {e}")

# === Esegui per mensile e trimestrale
processa_rebalance_weights("models/monthly_best_configs", "monthly")
processa_rebalance_weights("models/quarterly_best_configs", "quarterly")
