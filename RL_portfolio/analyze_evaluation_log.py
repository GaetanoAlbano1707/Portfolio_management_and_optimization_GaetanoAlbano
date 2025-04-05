import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_evaluation_log(results_path="./results/test"):
    # Percorsi dei file
    results_dir = Path(results_path)
    log_file = results_dir / "evaluation_log_policy.csv"
    summary_file = results_dir / "quarterly_policy_summary.csv"
    plot_file = results_dir / "quarterly_policy_returns.png"

    # Caricamento dei dati
    df = pd.read_csv(log_file, parse_dates=["date"])
    df.set_index("date", inplace=True)

    # Assicurati che i dati siano ordinati temporalmente
    df.sort_index(inplace=True)

    # Raggruppamento per trimestre e calcolo delle metriche
    quarterly_summary = df.resample("Q").agg(
        start_value=("portfolio_value", "first"),
        end_value=("portfolio_value", "last"),
        avg_reward=("reward", "mean"),
        std_reward=("reward", "std")
    )

    # Calcolo del rendimento percentuale per trimestre
    quarterly_summary["return_%"] = (
        (quarterly_summary["end_value"] / quarterly_summary["start_value"]) - 1
    ) * 100

    # Salvataggio del riepilogo trimestrale
    quarterly_summary.to_csv(summary_file)

    # Creazione del grafico dei rendimenti percentuali trimestrali
    plt.figure(figsize=(10, 6))
    quarterly_summary["return_%"].plot(kind="bar", color="skyblue")
    plt.title("Rendimenti Percentuali Trimestrali del Portafoglio")
    plt.xlabel("Trimestre")
    plt.ylabel("Rendimento (%)")
    plt.grid(axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"📊 Riepilogo trimestrale salvato in: {summary_file}")
    print(f"📈 Grafico dei rendimenti trimestrali salvato in: {plot_file}")

if __name__ == "__main__":
    analyze_evaluation_log()
