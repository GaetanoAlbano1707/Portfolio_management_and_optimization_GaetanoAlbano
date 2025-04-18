import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def run_equal_weights_strategy(df, tickers, rebalance_period=21):
    prices = df["adj_close"].unstack()
    returns = prices.pct_change().fillna(0)
    weights = np.ones(len(tickers)) / len(tickers)

    portfolio_values = []
    dates = []
    cum_value = 1.0

    for start in range(0, len(returns), rebalance_period):
        end = min(start + rebalance_period, len(returns))
        period_returns = returns.iloc[start:end]
        daily_returns = period_returns @ weights

        for ret, date in zip(daily_returns, period_returns.index):
            cum_value *= (1 + ret)
            portfolio_values.append(cum_value)
            dates.append(date)

    return pd.Series(portfolio_values, index=dates)

def run_minimum_variance_strategy(df, tickers, rebalance_period=21):
    prices = df["adj_close"].unstack()
    returns = prices.pct_change().dropna()
    weights = np.ones(len(tickers)) / len(tickers)

    portfolio_values = []
    dates = []
    cum_value = 1.0

    for start in range(0, len(returns), rebalance_period):
        end = min(start + rebalance_period, len(returns))
        cov_matrix = returns.iloc[start:end].cov().values

        def portfolio_variance(w):
            return w.T @ cov_matrix @ w

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(tickers)
        result = minimize(portfolio_variance, weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            continue

        weights = result.x
        period_returns = returns.iloc[start:end]
        daily_returns = period_returns @ weights

        for ret, date in zip(daily_returns, period_returns.index):
            cum_value *= (1 + ret)
            portfolio_values.append(cum_value)
            dates.append(date)

    return pd.Series(portfolio_values, index=dates)

def run_markowitz_strategy(df, tickers, rebalance_period=21):
    prices = df["adj_close"].unstack()
    returns = prices.pct_change().dropna()
    weights = np.ones(len(tickers)) / len(tickers)

    portfolio_values = []
    dates = []
    cum_value = 1.0

    for start in range(0, len(returns), rebalance_period):
        end = min(start + rebalance_period, len(returns))
        mean_returns = returns.iloc[start:end].mean().values
        cov_matrix = returns.iloc[start:end].cov().values

        def negative_sharpe(w):
            port_return = w @ mean_returns
            port_vol = np.sqrt(w.T @ cov_matrix @ w)
            return -port_return / (port_vol + 1e-6)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(tickers)
        result = minimize(negative_sharpe, weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            continue

        weights = result.x
        period_returns = returns.iloc[start:end]
        daily_returns = period_returns @ weights

        for ret, date in zip(daily_returns, period_returns.index):
            cum_value *= (1 + ret)
            portfolio_values.append(cum_value)
            dates.append(date)

    return pd.Series(portfolio_values, index=dates)

def run_random_strategy(df, tickers, rebalance_period=21):
    prices = df["adj_close"].unstack()
    returns = prices.pct_change().fillna(0)

    portfolio_values = []
    dates = []
    cum_value = 1.0

    for start in range(0, len(returns), rebalance_period):
        end = min(start + rebalance_period, len(returns))
        weights = np.random.rand(len(tickers))
        weights /= np.sum(weights)

        period_returns = returns.iloc[start:end]
        daily_returns = period_returns @ weights

        for ret, date in zip(daily_returns, period_returns.index):
            cum_value *= (1 + ret)
            portfolio_values.append(cum_value)
            dates.append(date)

    return pd.Series(portfolio_values, index=dates)

def run_buy_and_hold_strategy(df, tickers, asset_index=0):
    prices = df["adj_close"].unstack()
    returns = prices.pct_change().fillna(0)
    daily_returns = returns.iloc[:, asset_index]
    cum_returns = (1 + daily_returns).cumprod()
    return cum_returns

def run_buy_and_hold_portfolio(df, tickers):
    prices = df["adj_close"].unstack()
    returns = prices.pct_change().fillna(0)
    weights = np.ones(len(tickers)) / len(tickers)
    weighted_returns = returns @ weights
    cum_returns = (1 + weighted_returns).cumprod()
    return cum_returns

def compare_strategies(df, tickers, rebalance_period=21, output_csv="model_comparison_returns.csv", output_plot="compare_models_test.png"):
    strategies = {
        "Equal Weight": run_equal_weights_strategy(df, tickers, rebalance_period),
        "Minimum Variance": run_minimum_variance_strategy(df, tickers, rebalance_period),
        "Markowitz": run_markowitz_strategy(df, tickers, rebalance_period),
        "Random": run_random_strategy(df, tickers, rebalance_period),
        "Buy & Hold (Asset)": run_buy_and_hold_strategy(df, tickers, asset_index=0),
        "Buy & Hold (Portfolio)": run_buy_and_hold_portfolio(df, tickers)
    }

    comparison_df = pd.DataFrame(strategies).dropna()
    comparison_df.to_csv(output_csv, index=True)

    plt.figure(figsize=(12, 6))
    for name, series in comparison_df.items():
        plt.plot(series.index, series.values, label=name)
    plt.legend()
    plt.title("Confronto Strategie di Portafoglio")
    plt.xlabel("Data")
    plt.ylabel("Rendimento Cumulativo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
