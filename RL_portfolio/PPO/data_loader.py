import pandas as pd
import numpy as np

def load_data(main_data_path, mu_path, sigma_path):
    df_prices = pd.read_csv(main_data_path, parse_dates=['date'])
    df_mu = pd.read_csv(mu_path, parse_dates=['date'])
    df_sigma = pd.read_csv(sigma_path, parse_dates=['date'])

    df_prices = df_prices[df_prices['date'] >= '2020-01-07']
    df_mu = df_mu[df_mu['date'] >= '2020-01-07']
    df_sigma = df_sigma[df_sigma['date'] >= '2020-01-07']

    return df_prices, df_mu, df_sigma

def get_rebalancing_dates(dates, freq='Q'):
    return pd.Series(dates).drop_duplicates().resample(freq).first().dropna().tolist()
