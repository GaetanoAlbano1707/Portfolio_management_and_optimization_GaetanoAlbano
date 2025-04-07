import os
import pandas as pd
import numpy as np
from glob import glob

def load_volatility_data(file_path: str) -> pd.DataFrame:
    """
    Carica un singolo file CSV contenente tutte le volatilità previste combinate.
    Il file deve avere le colonne: Date, LSTM_Vol, tic
    """
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df[["Date", "LSTM_Vol", "tic"]]
    df = df.set_index(["Date", "tic"]).sort_index()
    return df


def compute_covariance_matrix(vol_df: pd.DataFrame, corr_matrix: pd.DataFrame = None) -> dict:
    """
    Costruisce una matrice di covarianza Σ(t) time-varying.
    """
    cov_matrices = {}

    for date, group in vol_df.groupby(level=0):
        vols = group["LSTM_Vol"].values
        tickers = group.index.get_level_values(1)

        print(f"[DEBUG] Covariance Matrix @ {date.strftime('%Y-%m-%d')}: Index = {group.index}")
        print(f"[DEBUG] Tickers estratti: {tickers.tolist()}")

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
