import os
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usa il backend non interattivo Agg
import matplotlib.pyplot as plt

def analyze_data_periodically_month(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='./Periodic_analysis/Periodic_analysis_month/'):
    """
    Analizza i dati di rendimento e prezzo su base periodica (mensile) per ogni ticker nella sua vita dal 2007 ad oggi.
    """
    # Controlla che la directory esista, altrimenti creala
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = pd.to_datetime(end_date, format='%d/%m/%Y')

    periodic_results_return = []
    periodic_results_AdjClose = []

    for ticker in tickers:
        try:
            # Scarica i dati storici
            data = yf.download(ticker, period='max')
            if data.empty:
                print(f"Errore per il ticker {ticker}: dati non disponibili.")
                continue

            data = data.sort_index()
            data = data.ffill().bfill()

            # Filtra i dati antecedenti alla data specificata
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]

            data['return'] = data['Adj Close'].pct_change() * 100
            # Arrotonda a 3 cifre decimali
            data['return'] = data['return'].round(3)

            # Controlla che ci siano dati sufficienti
            if data.empty or len(data) < 2:
                print(f"Errore per il ticker {ticker}: dati insufficienti dopo la pulizia.")
                continue

            data['Month'] = data.index.month

            # Statistiche per anno
            for month, group in data.groupby('Month'):
                if len(group) < 2:
                    continue

                # Calcola statistiche
                return_kurtosis = group['return'].dropna().kurt()
                return_skewness = group['return'].dropna().skew()
                return_mean = group['return'].dropna().mean()
                return_std = group['return'].dropna().std()

                adjclose_kurtosis = group['Adj Close'].dropna().kurt()
                if isinstance(adjclose_kurtosis, pd.Series):
                    adjclose_kurtosis = adjclose_kurtosis.iloc[0]

                adjclose_skewness = group['Adj Close'].dropna().skew()
                if isinstance(adjclose_skewness, pd.Series):
                    adjclose_skewness = adjclose_skewness.iloc[0]

                adjclose_mean = group['Adj Close'].dropna().mean()
                if isinstance(adjclose_mean, pd.Series):
                    adjclose_mean = adjclose_mean.iloc[0]

                adjclose_std = group['Adj Close'].dropna().std()
                if isinstance(adjclose_std, pd.Series):
                    adjclose_std = adjclose_std.iloc[0]

                # Salva i risultati
                periodic_results_return.append({
                    'Ticker': ticker,
                    'Month': month,
                    'Kurtosis': return_kurtosis,
                    'Skewness': return_skewness,
                    'Mean': return_mean,
                    'Std': return_std
                })
                periodic_results_AdjClose.append({
                    'Ticker': ticker,
                    'Month': month,
                    'Kurtosis': adjclose_kurtosis,
                    'Skewness': adjclose_skewness,
                    'Mean': adjclose_mean,
                    'Std': adjclose_std
                })

            # Grafico
            plt.figure(figsize=(10, 6))
            months = list(range(1, 13))  # Tutti i mesi da 1 a 12
            average_returns = data.groupby('Month')['return'].mean()  # Media dei rendimenti per mese
            plt.bar(average_returns.index, average_returns, color='blue', alpha=0.7)

            plt.title(f'Medie rendimenti mensili - {ticker}')
            plt.xlabel('Mesi')
            plt.ylabel('Media rendimenti')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Assicurati che tutti i mesi siano sull'asse x
            plt.xticks(ticks=months, labels=months, rotation=45)

            plt.savefig(f"{save_dir}{ticker}_medie_mensili.png")
            plt.close()


        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")

    # Crea DataFrame finali e salva i file CSV
    if periodic_results_return and periodic_results_AdjClose:
        df_return_stats = pd.DataFrame(periodic_results_return)
        df_adjclose_stats = pd.DataFrame(periodic_results_AdjClose)

        df_return_stats.to_csv(
            f"{save_dir}return_periodic_month_measures.csv",
            index=False,
            float_format='%.4f',
            sep=';',
            decimal=','
        )
        df_adjclose_stats.to_csv(
            f"{save_dir}adjclose_periodic_month_measures.csv",
            index=False,
            float_format='%.4f',
            sep=';',
            decimal=','
        )
        print("File 'return_periodic_month_measures.csv' e 'adjclose_periodic_month_measures.csv' salvati con successo.")
    else:
        print("Nessun risultato disponibile per creare i file CSV.")


def analyze_data_periodically_year(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='./Periodic_analysis/Periodic_analysis_year/'):
    """
    Analizza i dati di rendimento e prezzo su base periodica (annuale) di ogni ticker per tutta la sua vita dal 2007 a oggi.
    """
    # Controlla che la directory esista, altrimenti creala
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = pd.to_datetime(end_date, format='%d/%m/%Y')

    periodic_results_return = []
    periodic_results_AdjClose = []

    for ticker in tickers:
        try:
            # Scarica i dati storici
            data = yf.download(ticker, period='max')
            if data.empty:
                print(f"Errore per il ticker {ticker}: dati non disponibili.")
                continue

            data = data.sort_index()
            data = data.ffill().bfill()

            # Filtra i dati antecedenti alla data specificata
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
            data = data[data['Volume'] > 0]

            data['return'] = data['Adj Close'].pct_change()
            # Arrotonda a 3 cifre decimali
            data['return'] = data['return'].round(3)

            # Controlla che ci siano dati sufficienti
            if data.empty or len(data) < 2:
                print(f"Errore per il ticker {ticker}: dati insufficienti dopo la pulizia.")
                continue

            data['Year'] = data.index.year

            # Statistiche per anno
            for year, group in data.groupby('Year'):
                if len(group) < 2:
                    continue

                # Calcola statistiche
                return_kurtosis = group['return'].dropna().kurt()
                return_skewness = group['return'].dropna().skew()
                return_mean = group['return'].dropna().mean()
                return_std = group['return'].dropna().std()

                adjclose_kurtosis = group['Adj Close'].dropna().kurt()
                if isinstance(adjclose_kurtosis, pd.Series):
                    adjclose_kurtosis = adjclose_kurtosis.iloc[0]

                adjclose_skewness = group['Adj Close'].dropna().skew()
                if isinstance(adjclose_skewness, pd.Series):
                    adjclose_skewness = adjclose_skewness.iloc[0]

                adjclose_mean = group['Adj Close'].dropna().mean()
                if isinstance(adjclose_mean, pd.Series):
                    adjclose_mean = adjclose_mean.iloc[0]

                adjclose_std = group['Adj Close'].dropna().std()
                if isinstance(adjclose_std, pd.Series):
                    adjclose_std = adjclose_std.iloc[0]

                # Salva i risultati
                periodic_results_return.append({
                    'Ticker': ticker,
                    'Year': year,
                    'Kurtosis': return_kurtosis,
                    'Skewness': return_skewness,
                    'Mean': return_mean,
                    'Std': return_std
                })
                periodic_results_AdjClose.append({
                    'Ticker': ticker,
                    'Year': year,
                    'Kurtosis': adjclose_kurtosis,
                    'Skewness': adjclose_skewness,
                    'Mean': adjclose_mean,
                    'Std': adjclose_std
                })

            # Grafico
            plt.figure(figsize=(10, 6))
            years = list(range(start_date_dt.year, end_date_dt.year + 1))  # Tutti gli anni dal 2007 al 2024
            average_returns = data.groupby('Year')['return'].mean()  # Media dei rendimenti per anno
            plt.bar(average_returns.index, average_returns, color='blue', alpha=0.7)

            plt.title(f'Medie rendimenti Annuali - {ticker}')
            plt.xlabel('Anno')
            plt.ylabel('Media rendimenti')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Assicurati che tutti gli anni siano sull'asse x
            plt.xticks(ticks=years, labels=years, rotation=45)

            plt.savefig(f"{save_dir}{ticker}_medie_annuali.png")
            plt.close()

        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")

    # Crea DataFrame finali e salva i file CSV
    if periodic_results_return and periodic_results_AdjClose:
        df_return_stats = pd.DataFrame(periodic_results_return)
        df_adjclose_stats = pd.DataFrame(periodic_results_AdjClose)

        df_return_stats.to_csv(
            f"{save_dir}return_periodic_year_measures.csv",
            index=False,
            float_format='%.4f',
            sep=';',
            decimal=','
        )
        df_adjclose_stats.to_csv(
            f"{save_dir}adjclose_periodic_year_measures.csv",
            index=False,
            float_format='%.4f',
            sep=';',
            decimal=','
        )
        print("File 'return_periodic_year_measures.csv' e 'adjclose_periodic_measures.csv' salvati con successo.")
    else:
        print("Nessun risultato disponibile per creare i file CSV.")



# Lista dei ticker
tickers = [
    'XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI',
    'NVDA', 'LMT', 'WMT', 'XOM', 'NKE', 'AMZN', 'NFLX', 'AAPL'
]

# Esegui l'analisi
analyze_data_periodically_month(tickers)
# Esegui l'analisi
analyze_data_periodically_year(tickers)