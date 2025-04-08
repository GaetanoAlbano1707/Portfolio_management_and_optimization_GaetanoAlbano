import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns, EfficientFrontier

def validate_final_allocation(weights_df_path, expected_return_path, price_data_path,
                              result_dir="./PPO/results/", plot_frontier=True):
    # === Carica pesi finali
    df_weights = pd.read_csv(weights_df_path)
    final_row = df_weights.iloc[-1]
    tickers = final_row.index[1:]
    final_weights = final_row[tickers].values.astype(float)

    # === Carica dati
    mu_df = pd.read_csv(expected_return_path)
    mu_series = mu_df.groupby('tic')['predicted_t'].mean().reindex(tickers)
    df = pd.read_csv(price_data_path, parse_dates=['date'])
    returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()
    returns = returns[tickers]
    S = risk_models.sample_cov(returns)

    # === Calcolo performance portafoglio finale
    expected_ret = np.dot(final_weights, mu_series.values)
    volatility = np.sqrt(np.dot(final_weights, np.dot(S.values, final_weights)))

    # === Salva risultati
    results = {
        "expected_return_%": round(expected_ret * 100, 4),
        "volatility_%": round(volatility * 100, 4)
    }
    pd.Series(results).to_csv(f"{result_dir}/final_allocation_metrics.csv")

    # === Plot frontiera e punto finale
    if plot_frontier:
        gammas = np.linspace(1, 100, 50)
        rets, vols = [], []
        for g in gammas:
            ef = EfficientFrontier(mu_series, S)
            ef.max_quadratic_utility(risk_aversion=g)
            r, v, _ = ef.portfolio_performance()
            rets.append(r * 100)
            vols.append(v * 100)

        final_ret_pct = expected_ret * 100
        final_vol_pct = volatility * 100

        plt.figure(figsize=(10, 6))
        plt.plot(vols, rets, label="Efficient Frontier", color="blue")
        plt.scatter(final_vol_pct, final_ret_pct, color="red", label="Final Portfolio", zorder=5)
        plt.xlabel("Volatility (%)")
        plt.ylabel("Expected Return (%)")
        plt.title("Efficient Frontier + Final Portfolio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{result_dir}/final_point_on_frontier.png")
        plt.close()

    print(f"✅ Validazione completata: Rendimento atteso = {results['expected_return_%']}%, Volatilità = {results['volatility_%']}%")
    return results
