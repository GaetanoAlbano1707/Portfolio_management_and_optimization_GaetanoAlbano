import os
import pandas as pd
import matplotlib.pyplot as plt
import json

def summarize_cumulative_returns(base_dir, top_configs_file="top_configs_by_lambda.json", model_type="quarterly"):
    if not os.path.exists(top_configs_file):
        print(f"❌ File non trovato: {top_configs_file}")
        return

    with open(top_configs_file, "r") as f:
        best_configs = json.load(f)

    results = []

    for lambda_str, configs in best_configs.items():
        lambda_risk = float(lambda_str)
        for config in configs:
            result_dir = os.path.join("models", f"{model_type}_best_configs", f"lambda_{lambda_risk}", config,
                                      "results")
            performance_file = os.path.join(result_dir, "test_performance.csv")

            if os.path.exists(performance_file):
                df = pd.read_csv(performance_file)
                final_cum_return = df["Cumulative Return"].iloc[-1]
                results.append({
                    "Lambda Risk": lambda_risk,
                    "Config": config,
                    "Final Cumulative Return": final_cum_return
                })
            else:
                print(f"❌ File non trovato: {performance_file}")

    summary_df = pd.DataFrame(results)
    summary_csv = os.path.join(base_dir, "summary_cumulative_returns.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Riepilogo salvato in: {summary_csv}")

    if not summary_df.empty:
        plt.figure(figsize=(10, 6))
        for lambda_risk in summary_df["Lambda Risk"].unique():
            subset = summary_df[summary_df["Lambda Risk"] == lambda_risk]
            plt.bar([f"{lambda_risk}-{i}" for i in range(len(subset))], subset["Final Cumulative Return"], label=f"λ={lambda_risk}")
        plt.title("Ritorno Cumulato Finale per diversi λ e configurazioni")
        plt.xlabel("Configurazioni")
        plt.ylabel("Rendimento Cumulativo Finale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "bar_lambda_cumulative_return.png"))
        plt.close()
    else:
        print("⚠️ Nessun risultato disponibile per il plot. Controlla che i file test_performance.csv esistano.")




def summarize_cumulative_returns_monthly(base_dir, top_configs_file="top_configs_by_lambda_monthly.json", model_type="monthly"):
    if not os.path.exists(top_configs_file):
        print(f"❌ File non trovato: {top_configs_file}")
        return

    with open(top_configs_file, "r") as f:
        best_configs = json.load(f)

    results = []

    for lambda_str, configs in best_configs.items():
        lambda_risk = float(lambda_str)
        for config in configs:
            result_dir = os.path.join("models", f"{model_type}_best_configs", f"lambda_{lambda_risk}", config,
                                      "results")
            performance_file = os.path.join(result_dir, "test_performance.csv")

            if os.path.exists(performance_file):
                df = pd.read_csv(performance_file)
                final_cum_return = df["Cumulative Return"].iloc[-1]
                results.append({
                    "Lambda Risk": lambda_risk,
                    "Config": config,
                    "Final Cumulative Return": final_cum_return
                })
            else:
                print(f"❌ File non trovato: {performance_file}")

    summary_df = pd.DataFrame(results)
    summary_csv = os.path.join(base_dir, "summary_cumulative_returns.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Riepilogo salvato in: {summary_csv}")

    if not summary_df.empty:
        plt.figure(figsize=(10, 6))
        for lambda_risk in summary_df["Lambda Risk"].unique():
            subset = summary_df[summary_df["Lambda Risk"] == lambda_risk]
            plt.bar([f"{lambda_risk}-{i}" for i in range(len(subset))], subset["Final Cumulative Return"], label=f"λ={lambda_risk}")
        plt.title("Ritorno Cumulato Finale per diversi λ e configurazioni")
        plt.xlabel("Configurazioni")
        plt.ylabel("Rendimento Cumulativo Finale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "bar_lambda_cumulative_return.png"))
        plt.close()
    else:
        print("⚠️ Nessun risultato disponibile per il plot. Controlla che i file test_performance.csv esistano.")
