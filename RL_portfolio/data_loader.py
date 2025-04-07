import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

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
            # Costruisci una matrice diagonale di volatilità per simulare i ritorni
            returns_matrix = np.diagflat(vols)
            lw = LedoitWolf()
            lw_cov = lw.fit(returns_matrix).covariance_
            cov_df = pd.DataFrame(lw_cov, index=tickers, columns=tickers)
            cov_matrices[date] = cov_df
        except:
            continue
    return cov_matrices

def load_expected_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    return df["predicted_t"]
