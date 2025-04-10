import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, EfficientFrontier

def validate_final_allocation(weights_df_path, expected_return_path, price_data_path,
                              result_dir="./PPO/results/", plot_frontier=True):
    df_weights = pd.read_csv(weights_df_path)

    # Se 'date' è presente come prima colonna, escludila
    if 'date' in df_weights.columns:
        tickers = df_weights.columns.drop('date').tolist()
    else:
        tickers = df_weights.columns.tolist()

    final_row = df_weights.iloc[-1]
    final_weights = final_row[tickers].astype(float).values
    final_weights /= final_weights.sum()

    # === Caricamento mu e rendimenti
    mu_df = pd.read_csv(expected_return_path)
    mu_series = mu_df.groupby('tic')['predicted_t'].mean().reindex(tickers)
    df = pd.read_csv(price_data_path, parse_dates=['date'])
    returns = df.pivot(index='date', columns='tic', values='adj_close').pct_change().dropna()
    returns = returns[tickers]

    # === Debug dati returns
    print(f"🔍 [DEBUG] Returns: min={returns.min().min():.6f}, max={returns.max().max():.6f}, mean={returns.mean().mean():.6f}")

    # === Covarianza + check volatilitá giornaliera
    S_daily = risk_models.sample_cov(returns)
    vol_daily = np.sqrt(np.dot(final_weights, np.dot(S_daily.values, final_weights)))
    print(f"📉 [DEBUG] Volatility daily: {vol_daily:.6f}")

    S = S_daily * 252  # ✅ Annualizza
    volatility = np.sqrt(np.dot(final_weights, np.dot(S.values, final_weights)))

    # === Calcolo rendimento
    expected_ret = np.dot(final_weights, mu_series.values)
    print(f"📈 [DEBUG] Expected return raw: {expected_ret:.6f}")
    print(f"📉 [DEBUG] Volatility annualized: {volatility:.6f}")

    results = {
        "expected_return_%": round(expected_ret * 100, 4),
        "volatility_%": round(volatility * 100, 4)
    }
    pd.Series(results).to_csv(f"{result_dir}/final_allocation_metrics.csv")

    # === Plot frontiera
    if plot_frontier:
        gammas = np.linspace(1, 100, 50)
        rets, vols = [], []
        for g in gammas:
            ef = EfficientFrontier(mu_series, S)
            ef.max_quadratic_utility(risk_aversion=g)
            r, v, _ = ef.portfolio_performance()
            rets.append(r * 100)
            vols.append(v * 100)

        plt.figure(figsize=(10, 6))
        plt.plot(vols, rets, label="Efficient Frontier", color="blue")
        plt.scatter(volatility * 100, expected_ret * 100, color="red", label="Final Portfolio", zorder=5)
        plt.xlabel("Volatility (%)")
        plt.ylabel("Expected Return (%)")
        plt.title("Efficient Frontier + Final Portfolio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{result_dir}/final_point_on_frontier.png")
        plt.close()

    print(f"✅ Validazione completata: Rendimento = {results['expected_return_%']}%, Volatilità = {results['volatility_%']}%")
    return results
