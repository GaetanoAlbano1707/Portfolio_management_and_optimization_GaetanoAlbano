import os
import pandas as pd
import numpy as np
from glob import glob

def load_volatility_data(vol_path_pattern: str) -> pd.DataFrame:
    """
    Carica i file CSV contenenti le volatilità previste (GARCH+LSTM) e li combina in un DataFrame.
    """
    all_vols = []

    for file in glob(vol_path_pattern):
        ticker = os.path.basename(file).split('_')[-1].split('.')[0]
        df = pd.read_csv(file, parse_dates=['Date'])
        df = df[['Date', 'LSTM_Vol']].copy()
        df['Ticker'] = ticker
        all_vols.append(df)

    combined = pd.concat(all_vols)
    combined = combined.set_index(['Date', 'Ticker']).sort_index()
    return combined


def compute_covariance_matrix(vol_df: pd.DataFrame, corr_matrix: pd.DataFrame = None) -> dict:
    """
    Costruisce una matrice di covarianza Σ(t) time-varying.
    """
    cov_matrices = {}

    for date, group in vol_df.groupby(level=0):
        vols = group['LSTM_Vol'].values
        tickers = group.index.get_level_values(1)

        sigma = np.diag(vols ** 2)
        if corr_matrix is None:
            corr = np.eye(len(vols))
        else:
            corr = corr_matrix.loc[tickers, tickers].values

        cov = np.multiply(np.outer(vols, vols), corr)
        cov_matrices[date] = pd.DataFrame(cov, index=tickers, columns=tickers)

    return cov_matrices


def load_expected_returns(path: str) -> pd.Series:
    """
    Carica un file CSV con rendimenti attesi stimati dalla distribuzione t location-scale.
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.set_index('Date')
    return df['predicted_t']
