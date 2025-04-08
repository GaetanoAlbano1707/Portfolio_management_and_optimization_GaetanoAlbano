import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import pickle

def load_volatility_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df[["Date", "LSTM_Vol", "tic"]]
    df = df.set_index(["Date", "tic"]).sort_index()
    return df

def compute_covariance_matrix(vol_df: pd.DataFrame) -> dict:
    cov_matrices = {}
    for date, group in vol_df.groupby(level=0):
        tickers = group.index.get_level_values(1)
        vols = group["LSTM_Vol"].values
        try:
            returns_matrix = np.diagflat(vols)
            lw = LedoitWolf()
            lw_cov = lw.fit(returns_matrix).covariance_
            cov_df = pd.DataFrame(lw_cov, index=tickers, columns=tickers)
            cov_matrices[date] = cov_df
        except:
            continue
    return cov_matrices

def compute_rolling_covariance(file_path: str, window: int = 60) -> dict:
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.set_index("date", inplace=True)

    cov_matrices = {}
    lw = LedoitWolf()

    for i in range(window, len(df)):
        window_df = df.iloc[i - window:i]
        date = df.index[i]

        if window_df.isnull().values.any():
            continue

        try:
            lw_cov = lw.fit(window_df.values).covariance_
            cov_df = pd.DataFrame(lw_cov, index=window_df.columns, columns=window_df.columns)
            cov_matrices[date] = cov_df
        except:
            continue

    return cov_matrices

def load_expected_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    return df["predicted_t"]
