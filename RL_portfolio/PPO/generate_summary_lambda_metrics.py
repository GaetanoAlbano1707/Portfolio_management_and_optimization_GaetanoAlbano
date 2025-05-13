import os
import pandas as pd
import matplotlib.pyplot as plt
def estrai_metriche_rl(base_dir, label_prefix):
    risultati = []
    for lambda_dir in sorted(os.listdir(base_dir)):
        lambda_path = os.path.join(base_dir, lambda_dir)
        if not os.path.isdir(lambda_path) or not lambda_dir.startswith("lambda_"):
            continue

        config_dirs = [d for d in os.listdir(lambda_path) if os.path.isdir(os.path.join(lambda_path, d))]
        if not config_dirs:
            print(f"Nessuna configurazione trovata in {lambda_path}")
            continue

        config_path = os.path.join(lambda_path, config_dirs[0], "results", "strategy_risk_metrics.csv")
        if not os.path.exists(config_path):
            print(f"File mancante: {config_path}")
            continue

        try:
            df = pd.read_csv(config_path, index_col=0)
            rl_metrics = df.loc["Reinforcement Learning"]
            risultati.append({
                "Risk": f"{label_prefix}_lambda_{lambda_dir.split('_')[-1]}",
                "Sharpe Ratio": rl_metrics["Sharpe Ratio"],
                "Sortino Ratio": rl_metrics["Sortino Ratio"],
                "Volatility": rl_metrics["Volatility"],
                "Max Drawdown": rl_metrics["Max Drawdown"],
                "Calmar Ratio": rl_metrics["Calmar Ratio"]
            })
        except Exception as e:
            print(f"Errore leggendo {config_path}: {e}")

    return risultati

# === Percorsi
monthly_dir = "models/monthly_best_configs"
quarterly_dir = "models/quarterly_best_configs"

# === Estrai e combinali
metriche_mensili = estrai_metriche_rl(monthly_dir, "monthly")
metriche_trimestrali = estrai_metriche_rl(quarterly_dir, "quarterly")

summary_df = pd.DataFrame(metriche_mensili + metriche_trimestrali)
summary_df.to_csv("summary_lambda_results.csv", index=False)
print("✅ Tabella riassuntiva salvata in: summary_lambda_results.csv")

# Salva anche come immagine PNG
def salva_tabella_png(df, filename, title="Metriche Reinforcement Learning"):
    fig, ax = plt.subplots(figsize=(12, 0.6 + len(df)*0.5))
    ax.axis('off')

    # Rimuove l'indice, mantiene solo il contenuto e intestazioni
    tabella = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    tabella.auto_set_font_size(False)
    tabella.set_fontsize(10)
    tabella.scale(1.2, 1.2)

    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"📷 Tabella PNG salvata in: {filename}")


# Chiamata alla funzione
salva_tabella_png(summary_df, "summary_lambda_results.png")
