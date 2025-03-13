# merge_data.py
import os
import glob
import pandas as pd


def merge_ticker_csvs(folder_path, output_file="./merged_tickers_data.csv"):
    csv_files = glob.glob(os.path.join(folder_path, "*_data.csv"))
    df_list = []
    for file in csv_files:
        # Estrai il ticker dal nome del file: ad es. "XLK_data.csv" -> "XLK"
        ticker = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file, parse_dates=['Date'])
        # Seleziona le colonne di interesse
        df = df[['Date', 'Adj Close','High','Low','Open','Volume','Close', 'return','log_return']]
        df.rename(columns={'Adj Close': f"{ticker}_AdjClose"}, inplace=True)
        df.rename(columns={'High': f"{ticker}_High"}, inplace=True)
        df.rename(columns={'Low': f"{ticker}_Low"}, inplace=True)
        df.rename(columns={'Open': f"{ticker}_Open"}, inplace=True)
        df.rename(columns={'Volume': f"{ticker}_Volume"}, inplace=True)
        df.rename(columns={'Close': f"{ticker}_Close"}, inplace=True)
        df.rename(columns={'return': f"{ticker}_return"}, inplace=True)
        df.rename(columns={'log_return': f"{ticker}_log_return"}, inplace=True)
        df_list.append(df)

    # Unisci tutti i DataFrame sul campo "Date"
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    merged_df.sort_values('Date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Dataset unificato salvato in {output_file}")
    return merged_df


if __name__ == '__main__':
    merge_ticker_csvs('./Tickers_file/')
