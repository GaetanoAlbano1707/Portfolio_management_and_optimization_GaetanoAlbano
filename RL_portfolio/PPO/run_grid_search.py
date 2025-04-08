import os
import pandas as pd
from data_loader import load_data, get_rebalancing_dates
from simulate import simulate_portfolio
from logger_weights import PortfolioLogger
from validation_utils import validate_final_allocation

# === Dati
data_path = "./PPO/main_data_real.csv"
mu_path = "./PPO/expected_returns_real.csv"
sigma_path = "./PPO/forecasting_data_combined.csv"
results_dir = "./PPO/results/grid_search/"
os.makedirs(results_dir, exist_ok=True)

df_prices, df_mu, df_sigma = load_data(data_path, mu_path, sigma_path)
rebal_dates = get_rebalancing_dates(sorted(df_prices['date'].unique()), freq='Q')
tickers = df_prices['tic'].unique().tolist()

gammas = [0.5, 1.0, 2.0]
costs = [0.001, 0.005, 0.01]
summary = []

for gamma in gammas:
    for cost in costs:
        tag = f"g{gamma}_c{int(cost * 10000)}"
        print(f"🔁 Running simulation for gamma={gamma}, cost={cost}")

        logger = PortfolioLogger()
        series = simulate_portfolio(df_prices, df_mu, df_sigma, rebal_dates,
                                    gamma=gamma, cost_rate=cost, logger=logger)
        logger.save()

        val = validate_final_allocation(
            weights_df_path="./PPO/results/weights_log.csv",
            expected_return_path=mu_path,
            price_data_path=data_path,
            result_dir=results_dir
        )

        # Salvataggio portafoglio simulato
        file_path = os.path.join(results_dir, f"portfolio_values_{tag}.csv")
        series.to_frame(name="value").to_csv(file_path)

        # Riepilogo metriche
        summary.append({
            "gamma": gamma,
            "cost_rate": cost,
            "expected_return_%": val["expected_return_%"],
            "volatility_%": val["volatility_%"]
        })

# Salva tabella dei risultati
df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(results_dir, "grid_search_summary.csv"), index=False)
print("✅ Grid search completata.")
