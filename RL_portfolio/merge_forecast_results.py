# merge_forecast.py
import os
import glob
import pandas as pd


def merge_forecast_csvs(folder_path, output_file="merged_forecast_data.csv"):
    csv_files = glob.glob(os.path.join(folder_path, "risultati_forecasting_*.csv"))
    df_list = []
    for file in csv_files:
        # Estrai il ticker dal nome del file: es. "risultati_forecasting_ticker.csv" -> "ticker"
        ticker = os.path.basename(file).split('_')[-1].split('.')[0]
        df = pd.read_csv(file, parse_dates=['Date'])
        # Rinomina le colonne relative alla volatilità
        df.rename(columns={
            'Volatilità Reale (exp)': f"{ticker}_Vol_Reale",
            'Volatilità Predetta (exp)': f"{ticker}_Vol_Pred"
        }, inplace=True)
        df_list.append(df)

    merged_forecast = df_list[0]
    for df in df_list[1:]:
        merged_forecast = pd.merge(merged_forecast, df, on='Date', how='outer')

    merged_forecast.sort_values('Date', inplace=True)
    merged_forecast.reset_index(drop=True, inplace=True)
    merged_forecast.to_csv(output_file, index=False)
    print(f"Dataset forecast unificato salvato in {output_file}")
    return merged_forecast


if __name__ == '__main__':
    merge_forecast_csvs('./Forecast_Files/')
