import pandas as pd
from load_file import download_tickers
import os

def merge_ticker_csvs(tickers, output_file="./merged_tickers_data.csv"):
    csvs_dict = download_tickers(tickers)
    df_list = []

    for ticker, df in csvs_dict.items():
        if df.empty:
            print(f"⚠️ Nessun dato disponibile per {ticker}, salto...")
            continue

        # Seleziona le colonne di interesse
        df = df[['Date', 'Adj Close', 'High', 'Low', 'Open', 'Volume', 'Close', 'return', 'log_return']]
        # Rinomina le colonne per includere il ticker
        df.rename(columns={
            'Adj Close': f"{ticker}_AdjClose",
            'High': f"{ticker}_High",
            'Low': f"{ticker}_Low",
            'Open': f"{ticker}_Open",
            'Volume': f"{ticker}_Volume",
            'Close': f"{ticker}_Close",
            'return': f"{ticker}_return",
            'log_return': f"{ticker}_log_return"
        }, inplace=True)

        df_list.append(df)

    if not df_list:
        print("⚠️ Nessun file valido trovato, merge non eseguito.")
        return None


    # Merge su 'Date'
    merged_df = df_list[0]

    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    # Ordina per data
    merged_df.sort_values('Date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Salva il dataset unificato
    merged_df.to_csv(output_file, index=False)
    print(f"✅ File merged_tickers_data.csv salvato in: {os.path.abspath(output_file)}")

    return merged_df

tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
df = merge_ticker_csvs(tickers)
