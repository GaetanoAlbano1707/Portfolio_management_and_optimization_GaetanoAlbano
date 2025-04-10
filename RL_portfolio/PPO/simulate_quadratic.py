import numpy as np
import pandas as pd
from optimizer import optimize_portfolio


def simulate_portfolio_quadratic(df_prices, df_mu, df_sigma, rebal_dates,
                                 initial_amount=100000, gamma=1.0,
                                 c_linear=0.001, c_quad=0.002, logger=None):
    tickers = sorted(df_prices['tic'].unique())
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets
    current_value = initial_amount
    port_series = pd.Series(dtype=float)

    def quadratic_cost(w_new, w_old):
        delta = np.abs(w_new - w_old)
        return np.sum(c_linear * delta + c_quad * delta ** 2)

    for i in range(len(rebal_dates) - 1):
        start, end = rebal_dates[i], rebal_dates[i + 1]
        df_period = df_prices[(df_prices['date'] >= start) & (df_prices['date'] <= end)]

        mu = df_mu[df_mu['date'] == start].set_index('tic').reindex(tickers)['predicted_t'].fillna(0).values
        sigma_vals = df_sigma[df_sigma['date'] == start].set_index('tic').reindex(tickers)['LSTM_Vol'].fillna(0).values
        sigma = np.diag(sigma_vals ** 2)

        new_weights = optimize_portfolio(mu, sigma, weights, gamma, cost_rate=0.0)
        cost = quadratic_cost(new_weights, weights)

        prices = df_period.pivot(index='date', columns='tic', values='adj_close').reindex(columns=tickers).dropna()
        if prices.empty:
            continue

        first_prices = prices.iloc[0].values
        shares = (current_value * weights) / first_prices

        for date, row in prices.iterrows():
            value = np.dot(row.values, shares)
            port_series.at[date] = value
            if logger:
                logger.log_value(date, value)

        if logger:
            logger.log_weights(start, new_weights, tickers)

        current_value = np.dot(prices.iloc[-1].values, shares) * (1 - cost)
        weights = new_weights

    return port_series.dropna()
