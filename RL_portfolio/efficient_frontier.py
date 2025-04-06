import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json


def calculate_efficient_frontier(price_data_path, config_path="config.json", save_path="efficient_frontier.png"):
    # === 1. Carica config per ottenere c_plus e c_minus
    with open(config_path, "r") as f:
        config = json.load(f)

    c_plus = config["costs"]["c_plus"]
    c_minus = config["costs"]["c_minus"]

    # === 2. Carica dati di prezzo
    df = pd.read_csv(price_data_path, parse_dates=['date'])
    df = df.pivot(index="date", columns="tic", values="adj_close").dropna()
    tickers = df.columns.tolist()

    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(tickers)

    # === 3. Funzioni portafoglio
    def portfolio_return(weights): return np.dot(weights, mean_returns)
    def portfolio_volatility(weights): return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def transaction_cost(weights):
        return sum(
            (c_plus if w > 1/num_assets else c_minus) * abs(w - 1/num_assets)
            for w in weights
        )

    # === 4. Funzioni obiettivo
    def min_vol_no_tc(weights): return portfolio_volatility(weights)
    def min_vol_with_tc(weights): return portfolio_volatility(weights) + transaction_cost(weights)

    # === 5. Ottimizzazione su diversi rendimenti attesi
    target_returns = np.linspace(min(mean_returns), max(mean_returns), 100)
    frontier_no_tc = []
    frontier_with_tc = []

    bounds = tuple((0, 1) for _ in range(num_assets))

    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )

        result_no_tc = minimize(min_vol_no_tc, num_assets * [1/num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        result_with_tc = minimize(min_vol_with_tc, num_assets * [1/num_assets], method='SLSQP', bounds=bounds, constraints=constraints)

        frontier_no_tc.append((portfolio_volatility(result_no_tc.x), target))
        frontier_with_tc.append((portfolio_volatility(result_with_tc.x), target - transaction_cost(result_with_tc.x)))

    # === 6. Plot
    vol_no_tc, ret_no_tc = zip(*frontier_no_tc)
    vol_tc, ret_tc = zip(*frontier_with_tc)

    plt.figure(figsize=(10, 7))
    plt.plot(vol_no_tc, ret_no_tc, label="No TC", color="blue")
    plt.plot(vol_tc, ret_tc, label="Linear TC", color="red", linestyle="--")
    plt.xlabel("Volatility (in %)")
    plt.ylabel("Net expected return (in %)")
    plt.title("Efficient Frontier with and without Transaction Costs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Efficient frontier salvata in: {save_path}")
