import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def classify_distribution(kurtosis, skewness):
    """
    Classifica le distribuzioni in base a kurtosis e skewness.
    """
    if kurtosis > 3:
        kurtosis_type = "Leptocurtica"
    elif kurtosis < 3:
        kurtosis_type = "Platicurtica"
    else:
        kurtosis_type = "Mesocurtica"

    if skewness > 0:
        skewness_type = "Positivamente asimmetrica"
    elif skewness < 0:
        skewness_type = "Negativamente asimmetrica"
    else:
        skewness_type = "Simmetrica"

    return kurtosis_type, skewness_type


def kurtosis_skewness(data_frames, save_dir='./Kurtosis_skewness/'):
    """
    Calcola kurtosis e skewness per 'return', 'log_return' e 'Adj Close',
    classificando le distribuzioni e salvando i risultati in CSV.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []

    for ticker, data in data_frames.items():
        try:
            print(f"Calcolo kurtosis e skewness per il ticker: {ticker}")

            def filter_valid_values(series):
                return series.dropna().replace([np.inf, -np.inf], np.nan).dropna()

            valid_return = filter_valid_values(data['return'])
            valid_log_return = filter_valid_values(data['log_return'])
            valid_adjclose = filter_valid_values(data['Adj Close'])

            if len(valid_return) < 2 or len(valid_log_return) < 2 or len(valid_adjclose) < 2:
                print(f"ATTENZIONE: Dati insufficienti per calcolare kurtosis e skewness per {ticker}")
                continue

            return_kurtosis = data['return'].dropna().kurt()
            return_skewness = data['return'].dropna().skew()
            return_kurt_type, return_skew_type = classify_distribution(return_kurtosis, return_skewness)

            log_return_kurtosis = data['log_return'].dropna().kurt()
            log_return_skewness = data['log_return'].dropna().skew()
            log_kurt_type, log_skew_type = classify_distribution(log_return_kurtosis, log_return_skewness)

            adjclose_kurtosis = data['Adj Close'].dropna().kurt()
            adjclose_skewness = data['Adj Close'].dropna().skew()
            adj_kurt_type, adj_skew_type = classify_distribution(adjclose_kurtosis, adjclose_skewness)

            results.append({
                'ticker': ticker,
                'variable': 'return',
                'kurtosis': return_kurtosis,
                'skewness': return_skewness,
                'kurtosis_type': return_kurt_type,
                'skewness_type': return_skew_type
            })
            results.append({
                'ticker': ticker,
                'variable': 'log_return',
                'kurtosis': log_return_kurtosis,
                'skewness': log_return_skewness,
                'kurtosis_type': log_kurt_type,
                'skewness_type': log_skew_type
            })
            results.append({
                'ticker': ticker,
                'variable': 'Adj Close',
                'kurtosis': adjclose_kurtosis,
                'skewness': adjclose_skewness,
                'kurtosis_type': adj_kurt_type,
                'skewness_type': adj_skew_type
            })
        except Exception as e:
            print(f"Errore nel calcolo per il ticker {ticker}: {e}")

    df_stats = pd.DataFrame(results)
    df_stats.to_csv(
        f"{save_dir}kurtosis_skewness_classification.csv",
        index=False,
        float_format='%.4f',
        sep=';',
        decimal=','
    )
    print("File salvato: 'kurtosis_skewness_classification.csv'.")


def download_and_save_ticker_data_with_dividends_and_total_return(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='./Tickers_file/'):
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


def rolling_kurtosis_skewness(data_frames, window=30, save_dir='./Rolling_Kurtosis_Skewness/'):
    """
    Calcola kurtosis e skewness su finestre mobili (rolling window) per
    return, log_return e Adj Close, e salva i risultati in csv.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ticker, df in data_frames.items():
        # Copia locale per evitare di modificare l'originale
        data = df.copy()

        # Calcolo rolling kurt/skew per 'return'
        data['rolling_kurt_return'] = data['return'].rolling(window).kurt()
        data['rolling_skew_return'] = data['return'].rolling(window).skew()

        # Calcolo rolling kurt/skew per 'log_return'
        data['rolling_kurt_log'] = data['log_return'].rolling(window).kurt()
        data['rolling_skew_log'] = data['log_return'].rolling(window).skew()

        # Calcolo rolling kurt/skew per 'Adj Close'
        data['rolling_kurt_adj'] = data['Adj Close'].rolling(window).kurt()
        data['rolling_skew_adj'] = data['Adj Close'].rolling(window).skew()

        # Salva i risultati in un CSV
        out_cols = [
            'rolling_kurt_return','rolling_skew_return',
            'rolling_kurt_log','rolling_skew_log',
            'rolling_kurt_adj','rolling_skew_adj'
        ]
        data[out_cols].to_csv(
            f"{save_dir}{ticker}_rolling_kurt_skew.csv",
            float_format='%.4f',
            index=True,
            sep=';',
            decimal=','
        )
        print(f"Rolling kurtosis/skewness salvati per {ticker} in {ticker}_rolling_kurt_skew.csv")

def plot_distributions(data_frames, ticker, variables=['return','log_return','Adj Close'], bins=50, save_dir='./Plots/'):
    """
    Crea alcuni istogrammi di base per visualizzare la distribuzione di 'return',
    'log_return' e 'Adj Close' e, a scelta, grafici rolling. Salva i plot in PNG.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = data_frames[ticker].copy()

    # -- Istogrammi delle distribuzioni --
    fig, axes = plt.subplots(nrows=1, ncols=len(variables), figsize=(16, 4))
    if len(variables) == 1:
        # Se c'è una sola variabile, axes non è più una lista, lo forziamo
        axes = [axes]

    for ax, var in zip(axes, variables):
        data_var = df[var].dropna()
        ax.hist(data_var, bins=bins, alpha=0.7, color='g')
        ax.set_title(f"{ticker} - {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Frequenza")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{ticker}_histograms.png"))
    plt.close()
    print(f"Istogrammi salvati per {ticker} in {ticker}_histograms.png")


def plot_rolling_kurt_skew(data_frames, ticker, window=30, variable='return', save_dir='./Plots/'):
    """
    Crea un grafico dell'andamento rolling di kurtosis e skewness
    per una variabile a scelta ('return', 'log_return' o 'Adj Close').
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = data_frames[ticker].copy()

    # Calcolo le serie rolling
    rolling_kurt = df[variable].rolling(window).kurt()
    rolling_skew = df[variable].rolling(window).skew()

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(rolling_kurt.index, rolling_kurt, label='Rolling Kurtosis', color='blue')
    ax[0].set_title(f"{ticker} - {variable} - Rolling Kurtosis ({window})")
    ax[0].legend(loc='best')

    ax[1].plot(rolling_skew.index, rolling_skew, label='Rolling Skewness', color='red')
    ax[1].set_title(f"{ticker} - {variable} - Rolling Skewness ({window})")
    ax[1].legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{ticker}_{variable}_rolling_{window}.png"))
    plt.close()
    print(f"Grafico rolling kurtosis/skewness salvato: {ticker}_{variable}_rolling_{window}.png")


# Lista dei ticker
tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI', 'NVDA', 'LMT', 'WMT', 'XOM', 'NKE', 'AMZN', 'NFLX', 'AAPL']

# Scarica i dati e calcola rendimenti
data = download_and_save_ticker_data_with_dividends_and_total_return(tickers)

# Calcola kurtosis, skewness e classificazione
kurtosis_skewness(data)


for ticker in data.keys():
    # Plot istogrammi delle tre variabili
    plot_distributions(
        data_frames=data,
        ticker=ticker,
        variables=['return','log_return','Adj Close'],
        bins=50
    )

    # Plot rolling kurtosis/skewness per ciascuna variabile (finestra di 60 giorni)
    plot_rolling_kurt_skew(
        data_frames=data,
        ticker=ticker,
        window=60,
        variable='return'
    )
    plot_rolling_kurt_skew(
        data_frames=data,
        ticker=ticker,
        window=60,
        variable='log_return'
    )
    plot_rolling_kurt_skew(
        data_frames=data,
        ticker=ticker,
        window=60,
        variable='Adj Close'
    )
