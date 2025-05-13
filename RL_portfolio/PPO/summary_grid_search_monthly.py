import os
import pandas as pd
import json
import shutil

BASE_DIR = "models/monthly"
BEST_CONFIGS_DIR = "models/monthly_best_configs"
OUTPUT_CSV = "grid_search_summary_monthly.csv"
TOP_CONFIGS_JSON = "top_configs_by_lambda_monthly.json"

lambda_values = [0.0, 0.05, 0.1, 0.2]
summary = []
top_configs = {}

for lambda_risk in lambda_values:
    lambda_path = os.path.join(BASE_DIR, f"lambda_{lambda_risk}")
    if not os.path.exists(lambda_path):
        continue

    lambda_results = []

    for config in os.listdir(lambda_path):
        results_path = os.path.join(lambda_path, config, "results", "train_performance.csv")
        if not os.path.exists(results_path):
            continue

        df = pd.read_csv(results_path)
        if len(df) < 10:
            continue

        mean_reward = df["Total Reward"].tail(10).mean()
        mean_sharpe = df["Sharpe Ratio"].tail(10).mean()

        result = {
            "Lambda": lambda_risk,
            "Config": config,
            "Mean Reward": mean_reward,
            "Mean Sharpe": mean_sharpe
        }

        summary.append(result)
        lambda_results.append(result)

    # Ordina e salva top-3
    top_sorted = sorted(lambda_results, key=lambda x: x["Mean Sharpe"], reverse=True)[:3]
    top_configs[str(lambda_risk)] = [entry["Config"] for entry in top_sorted]

    # Copia nella directory dedicata
    for entry in top_sorted:
        config = entry["Config"]
        src = os.path.join(BASE_DIR, f"lambda_{lambda_risk}", config)
        dest = os.path.join(BEST_CONFIGS_DIR, f"lambda_{lambda_risk}", config)
        if os.path.exists(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)

# Salva CSV di riepilogo completo
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Riepilogo salvato in: {OUTPUT_CSV}")

# Salva JSON delle migliori configurazioni per lambda
with open(TOP_CONFIGS_JSON, "w") as f:
    json.dump(top_configs, f, indent=4)
print(f"✅ Configurazioni top salvate in: {TOP_CONFIGS_JSON}")
print(f"📁 Copiate le migliori configurazioni in: {BEST_CONFIGS_DIR}")
