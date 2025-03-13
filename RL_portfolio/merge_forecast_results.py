import pandas as pd
from forecast_data import directory_forecasting_results_csvs
import os

def merge_forecast_csvs(tickers, output_file="./Tickers_file/merged_forecast_results.csv"):

    lista_csvs= directory_forecasting_results_csvs(tickers)

    # Merge su 'Date'
    merged_forecast = lista_csvs[0]
    for df in lista_csvs[1:]:
        merged_forecast = pd.merge(merged_forecast, df, on='Date', how='outer')

    # Ordina per data e gestisce eventuali valori mancanti
    merged_forecast.sort_values('Date', inplace=True)
    merged_forecast.fillna(method='ffill', inplace=True)  # Forward fill
    merged_forecast.fillna(method='bfill', inplace=True)  # Backward fill

    # Salva il risultato in un CSV
    merged_forecast.to_csv(output_file, index=False)
    print(f"✅ File merged_forecast_results.csv salvato in: {os.path.abspath(output_file)}")

    return merged_forecast

tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
merge_forecast_csv= merge_forecast_csvs(tickers)