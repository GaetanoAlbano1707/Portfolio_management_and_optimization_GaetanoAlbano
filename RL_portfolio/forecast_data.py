import pandas as pd
import os

def directory_forecasting_results_csvs(tickers, folder_path = r"C:\Users\gaeta\Desktop\PORTFOLIO_MANAGEMENT_AND_OPTIMIZATION_GaetanoAlbano\Risultati_GARCH_LSTM_Forecasting"):
    if not os.path.exists(folder_path):
        print("Errore: la cartella non esiste!")
        return None

    csv_files = []
    for ticker in tickers:
        file_path = os.path.join(folder_path, f"risultati_forecasting_{ticker}.csv")  # ✅ Usa f-string
        if os.path.exists(file_path):
            csv_file = pd.read_csv(file_path, parse_dates=['Date'])
            print(f"File per {ticker} trovato:", file_path)

            # Rinomina le colonne relative alla volatilità
            csv_file.rename(columns={
                'Volatilità Reale (exp)': f"{ticker}_Vol_Reale",
                'Volatilità Predetta (exp)': f"{ticker}_Vol_Pred"
            }, inplace=True)

            csv_files.append(csv_file)
        else:
            print(f"❌ File non trovato per {ticker}: {file_path}")

    return csv_files