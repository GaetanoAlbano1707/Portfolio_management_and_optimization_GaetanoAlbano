import os
import pandas as pd


def merge_csv_files(input_folder, output_file, common_column='Date', log_return_column='log_return'):
    merged_df = None

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)

            if common_column not in df.columns or log_return_column not in df.columns:
                print(f"Skipping {filename} - Missing required columns")
                continue

            # Estrai il nome del ticker dal filename (escludendo '_data.csv')
            ticker = filename.replace('_data.csv', '')

            # Rinomina la colonna log_return e seleziona solo 'Date' e 'log_return_ticker'
            df = df[[common_column, log_return_column]]
            df = df.rename(columns={log_return_column: f'{log_return_column}_{ticker}'})

            # Merge
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=common_column, how='outer')

    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"File unito salvato in {output_file}")
    else:
        print("Nessun file valido trovato.")


# Esempio di utilizzo
merged_log_returns = merge_csv_files(input_folder='./Tickers_file', output_file='merged_log_returns.csv')
