import numpy as np
import pandas as pd
from optimizer import optimize_portfolio

def simulate_portfolio_quadratic(df_prices, df_mu, df_sigma, rebal_dates,
                                  initial_amount=100000, gamma=1.0,
                                  c_linear=0.001, c_quad=0.002, logger=None):
    tickers = df_prices['tic'].unique()
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets
    dates = sorted(df_prices['date'].unique())
    current_value = initial_amount
    port_series = pd.Series(index=dates, dtype=float)

    def quadratic_cost(w_new, w_old):
        delta = np.abs(w_new - w_old)
        return np.sum(c_linear * delta + c_quad * delta**2)

    for i in range(len(rebal_dates) - 1):
        start, end = rebal_dates[i], rebal_dates[i + 1]

        df_period = df_prices[(df_prices['date'] >= start) & (df_prices['date'] <= end)]
        period_dates = sorted(df_period['date'].unique())

        mu = df_mu[df_mu['date'] == start].set_index('tic').reindex(tickers)['predicted_t'].values
        sigma_vals = df_sigma[df_sigma['date'] == start].set_index('tic').reindex(tickers)['LSTM_Vol'].values
        sigma = np.diag(sigma_vals ** 2)

        # Ottimizzazione
        new_weights = optimize_portfolio(mu, sigma, weights, gamma, cost_rate=0.0)
        cost = quadratic_cost(new_weights, weights)

        prices = df_period.pivot(index='date', columns='tic', values='adj_close')
        prices = prices[tickers]
        first_prices = prices.iloc[0].values
        shares = (current_value * weights) / first_prices

        for date in prices.index:
            value = np.dot(prices.loc[date].values, shares)
            port_series[date] = value
            if logger:
                logger.log_value(date, value)

        if logger:
            logger.log_weights(start, new_weights, tickers)

        last_prices = prices.iloc[-1].values
        current_value = np.dot(last_prices, shares) * (1 - cost)
        weights = new_weights

    return port_series.dropna()
