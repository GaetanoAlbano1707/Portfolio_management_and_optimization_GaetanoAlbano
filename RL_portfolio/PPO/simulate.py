import numpy as np
import pandas as pd
from optimizer import optimize_portfolio
from optimizer_extended import optimize_portfolio_extended
def simulate_portfolio(df_prices, df_mu, df_sigma, rebal_dates, initial_amount=100000,
                       gamma=1.0, cost_rate=0.001, logger=None):
    tickers = df_prices['tic'].unique()
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets
    dates = sorted(df_prices['date'].unique())
    current_value = initial_amount
    port_series = pd.Series(index=dates, dtype=float)

    for i in range(len(rebal_dates) - 1):
        start, end = rebal_dates[i], rebal_dates[i + 1]

        df_period = df_prices[(df_prices['date'] >= start) & (df_prices['date'] <= end)]
        period_dates = sorted(df_period['date'].unique())

        mu = df_mu[df_mu['date'] == start].set_index('tic').reindex(tickers)['predicted_t'].values
        sigma_vals = df_sigma[df_sigma['date'] == start].set_index('tic').reindex(tickers)['LSTM_Vol'].values
        sigma = np.diag(sigma_vals ** 2)  # Approx cov matrix

        # Ottimizzazione
        new_weights = optimize_portfolio_extended(mu, sigma, weights, gamma=gamma)

        # Simulazione periodo
        prices = df_period.pivot(index='date', columns='tic', values='adj_close')
        prices = prices[tickers]
        first_prices = prices.iloc[0].values
        shares = (current_value * weights) / first_prices

        for date in prices.index:
            value = np.dot(prices.loc[date].values, shares)
            port_series[date] = value
            if logger:
                logger.log_value(date, value)

        # Log pesi post-ribilanciamento
        if logger:
            logger.log_weights(start, new_weights, tickers)

        last_prices = prices.iloc[-1].values
        current_value = np.dot(last_prices, shares)
        weights = new_weights

    return port_series.dropna()
