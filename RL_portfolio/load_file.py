import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def download_tickers(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='./Tickers_file/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = pd.to_datetime(end_date, format='%d/%m/%Y')
    data_frames = {}

    for ticker in tickers:
        try:
            print(f"Elaborazione ticker: {ticker}")
            data = yf.download(ticker, period='max', auto_adjust= False)
            data = data.sort_index()
            data = data.ffill().bfill()

            # Appiattisci i nomi delle colonne (se MultiIndex)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns]

            # Rimuove il nome del ticker dalle colonne
            data.columns = [col.replace(f" {ticker}", "") for col in data.columns]

            # Controlla le colonne disponibili
            print(f"Colonne disponibili per {ticker}: {data.columns}")
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
            #Vedo solo i casi in cui c'è volume maggiore di 0
            data = data[data['Volume'] > 0]

            data['return'] = data['Adj Close'].pct_change() * 100
            data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

            data_frames[ticker] = data
            filename = f"{save_dir}{ticker}_data.csv"
            data.to_csv(filename, float_format='%.3f', index=True)
            print(f"Dati salvati in: {filename}")
        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")

    return data_frames


