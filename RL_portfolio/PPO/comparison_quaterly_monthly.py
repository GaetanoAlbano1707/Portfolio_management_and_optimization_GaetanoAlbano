import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_quarterly_monthly_results(quarterly_path, monthly_path, output_dir):
    df_q = pd.read_csv(quarterly_path)
    df_q["Periodo"] = "Trimestrale"

    df_m = pd.read_csv(monthly_path)
    df_m["Periodo"] = "Mensile"

    df = pd.concat([df_q, df_m], ignore_index=True)

    # Ordina per chiarezza
    df.sort_values(by=["Lambda Risk", "Periodo", "Config"], inplace=True)

    # Crea chiave combinata per gruppi λ + config
    df["Gruppo"] = df["Lambda Risk"].astype(str) + " | " + df["Config"]

    plt.figure(figsize=(14, 7))
    width = 0.35
    x = range(len(df["Gruppo"].unique()))
    labels = df["Gruppo"].unique()

    trimestrale = df[df["Periodo"] == "Trimestrale"]["Final Cumulative Return"].values
    mensile = df[df["Periodo"] == "Mensile"]["Final Cumulative Return"].values

    plt.bar([i - width/2 for i in x], trimestrale, width=width, label="Trimestrale", color="skyblue")
    plt.bar([i + width/2 for i in x], mensile, width=width, label="Mensile", color="orange")

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Rendimento Cumulativo Finale")
    plt.title("Confronto Mensile vs Trimestrale per λ e Configurazione")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "barplot_confronto_rendimenti.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Grafico migliorato salvato in: {plot_path}")


compare_quarterly_monthly_results(
    quarterly_path="results_ppo_risk_sweep/summary_cumulative_returns.csv",
    monthly_path="results_ppo_risk_sweep_monthly/summary_cumulative_returns.csv",
    output_dir="confronto_risultati"
)
