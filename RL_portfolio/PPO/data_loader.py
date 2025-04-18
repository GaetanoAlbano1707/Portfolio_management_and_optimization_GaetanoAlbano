import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_financial_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} non trovato!")

    logging.info("Caricamento del file CSV...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    logging.info(f"File caricato con {df.shape[0]} righe e {df.shape[1]} colonne.")

    tickers = sorted(set(col.split('_')[0] for col in df.columns if '_' in col and col != 'date'))
    logging.info(f"Tickers rilevati: {tickers}")

    expected_suffixes = ['adj_close', 'high', 'low', 'open', 'volume', 'predicted_t', 'predicted_t_std', 'LSTM_Vol']
    rename_map = {'LSTM_Vol': 'volatility'}  # rinomina per compatibilità con environment.py

    for ticker in tickers:
        for suffix in expected_suffixes:
            col_name = f"{ticker}_{suffix}"
            if col_name not in df.columns:
                logging.warning(f"Colonna mancante: {col_name}")

    feature_frames = []
    for ticker in tickers:
        sub_df = df[['date'] + [f"{ticker}_{suf}" for suf in expected_suffixes]].copy()
        sub_df.columns = ['date'] + [rename_map.get(suf, suf) for suf in expected_suffixes]

        # Feature temporali aggiuntive
        sub_df['momentum_5d'] = sub_df['adj_close'].pct_change(periods=5)
        sub_df['mu_diff'] = sub_df['predicted_t'].diff()
        sub_df['vol_diff'] = sub_df['volatility'].diff()
        sub_df['volume_ma3'] = sub_df['volume'].rolling(window=3).mean()

        sub_df['ticker'] = ticker
        feature_frames.append(sub_df)

    full_df = pd.concat(feature_frames)
    full_df.set_index(['date', 'ticker'], inplace=True)
    full_df = full_df.dropna()  # rimuove righe incomplete dovute a differenze e rolling

    missing_perc = full_df.isna().mean() * 100
    logging.info("Percentuale valori NaN per colonna:\n" + str(missing_perc.round(2)))

    return full_df, tickers

if __name__ == "__main__":
    df, tickers = load_financial_data("merged_data_total_cleaned_wide_multifeatures.csv")
    print(df.head())
