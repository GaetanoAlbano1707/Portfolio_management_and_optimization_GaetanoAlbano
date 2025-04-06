import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import t, norm, skewnorm
from statsmodels.graphics.tsaplots import plot_acf  # Per l'ACF dei residui

#######################################################################
# FUNZIONI PER DOWNLOAD E MERGE DEI DATI
#######################################################################

def load_data(tickers, start_date='01/01/2007', end_date='23/12/2024',
              save_dir='/content/drive/MyDrive/Tickers_file'):
    """
    Scarica i dati per ciascun ticker utilizzando yfinance e salva i file CSV.
    Calcola anche 'return' e 'log_return' (in percentuale).
    """
    print("Inizio il download dei dati...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = pd.to_datetime(end_date, format='%d/%m/%Y')
    data_frames = {}
    for ticker in tickers:
        try:
            print(f"Download dati per il ticker: {ticker}")
            data = yf.download(ticker, period='max', auto_adjust=False)
            data = data.sort_index().ffill().bfill()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns]
            data.columns = [col.replace(f" {ticker}", "") for col in data.columns]
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
            data = data[data['Volume'] > 0]
            data['return'] = data['Adj Close'].pct_change() * 100
            data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)) * 100
            data.dropna(inplace=True)
            data_frames[ticker] = data
            filename = os.path.join(save_dir, f"{ticker}_data.csv")
            data.to_csv(filename, float_format='%.3f', index=True)
            print(f"[{ticker}] Dati salvati in: {filename}, {data.shape[0]} righe finali")
        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")
    return data_frames

def merge_csv_files(input_folder, output_file, common_column='Date', log_return_column='log_return'):
    """
    Unisce tutti i CSV nella cartella input_folder (contenenti le colonne 'Date' e 'log_return')
    e salva il file unito in output_file. Per ogni file, rinomina 'log_return' in 'log_return_<ticker>'.
    """
    merged_df = None
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)
            if common_column not in df.columns or log_return_column not in df.columns:
                print(f"Skipping {filename} - Missing required columns")
                continue
            ticker = filename.replace('_data.csv', '')
            df = df[[common_column, log_return_column]]
            df = df.rename(columns={log_return_column: f'{log_return_column}_{ticker}'})
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=common_column, how='outer')
    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"File unito salvato in {output_file}")
    else:
        print("Nessun file valido trovato.")
    return merged_df

#######################################################################
# DOWNLOAD E MERGE DEI DATI
#######################################################################

tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI', 'NVDA', 'ITA', 'WMT', 'XOM', 'NKE', 'AMZN', 'NFLX', 'AAPL']
save_dir = '/content/drive/MyDrive/Tickers_file'
data_frames = load_data(tickers, save_dir=save_dir)

merged_file = '/content/drive/MyDrive/Tickers_file/merged_log_returns.csv'
merged_log_returns = merge_csv_files(input_folder=save_dir, output_file=merged_file)

#######################################################################
# PREVISIONE CON SIMULAZIONE MONTE CARLO A BLOCCHI CON MISURA DELL'INCERTEZZA
#######################################################################

logging.basicConfig(level=logging.INFO)

data_path = merged_file
output_dir = '/content/drive/MyDrive/Rendimenti_Attesi'
os.makedirs(output_dir, exist_ok=True)

# Carica il file unito
data = pd.read_csv(data_path, parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Parametri
window_size = 126    # giorni per il fitting
block_size = 10      # blocchi di 10 giorni per il forecast
n_simulations = 1000 # numero di simulazioni Monte Carlo
cutoff_date = pd.to_datetime("2019-12-20")

def block_forecast_MC(series, window_size, block_size, n_simulations=1000):
    """
    Per ogni blocco:
      - Stima i parametri della distribuzione su una finestra di window_size giorni.
      - Esegue n_simulations Monte Carlo per generare forecast per block_size giorni.
      - Calcola la media e la deviazione standard delle simulazioni per ciascun giorno del blocco.
    Restituisce un DataFrame con:
       - predicted_t_MC: media dei rendimenti simulati (modello t)
       - predicted_t_std: deviazione standard (incertezza) del modello t
       (e analogamente per norm e skewnorm, se desiderato)
       - I parametri stimati, ripetuti per ogni giorno del blocco.
    """
    pred_dates = []
    predicted_t_MC = []
    predicted_t_std = []  # nuova colonna per l'incertezza del modello t
    predicted_norm_MC = []
    predicted_skew_MC = []

    # Liste per i parametri
    t_nu_list = []
    t_mu_list = []
    t_sigma_list = []
    norm_mu_list = []
    norm_sigma_list = []
    skew_shape_list = []
    skew_loc_list = []
    skew_scale_list = []

    for start in range(0, len(series) - window_size - block_size + 1, block_size):
        window = series.iloc[start: start + window_size]
        forecast_dates = series.index[start + window_size: start + window_size + block_size]

        # Fitting modello t
        try:
            params_t = t.fit(window)
            nu, mu_t, sigma_t = params_t
            if nu <= 1:
                nu = 1.1
        except Exception as e:
            logging.error(f"Errore nel fitting t per finestra terminante in {series.index[start+window_size-1]}: {e}")
            mu_t = np.nan
            nu, sigma_t = np.nan, np.nan

        # Fitting modello normale
        try:
            mu_norm, sigma_norm = norm.fit(window)
        except Exception as e:
            logging.error(f"Errore nel fitting norm per finestra terminante in {series.index[start+window_size-1]}: {e}")
            mu_norm, sigma_norm = np.nan, np.nan

        # Fitting modello skewnorm
        try:
            shape, loc_skew, scale_skew = skewnorm.fit(window)
            delta = shape / np.sqrt(1 + shape ** 2)
            expected_skew = loc_skew + scale_skew * delta * np.sqrt(2 / np.pi)
        except Exception as e:
            logging.error(f"Errore nel fitting skewnorm per finestra terminante in {series.index[start+window_size-1]}: {e}")
            expected_skew = np.nan
            shape, loc_skew, scale_skew = np.nan, np.nan, np.nan

        # Simulazioni Monte Carlo per il modello t
        sim_t = t.rvs(df=nu, loc=mu_t, scale=sigma_t, size=(n_simulations, block_size))
        forecast_t = sim_t.mean(axis=0)
        forecast_t_std = sim_t.std(axis=0)  # deviazione standard delle simulazioni

        # Per il modello normale
        sim_norm = norm.rvs(loc=mu_norm, scale=sigma_norm, size=(n_simulations, block_size))
        forecast_norm = sim_norm.mean(axis=0)

        # Per il modello skewnorm
        sim_skew = skewnorm.rvs(a=shape, loc=loc_skew, scale=scale_skew, size=(n_simulations, block_size))
        forecast_skew = sim_skew.mean(axis=0)

        pred_dates.extend(forecast_dates)
        predicted_t_MC.extend(forecast_t)
        predicted_t_std.extend(forecast_t_std)
        predicted_norm_MC.extend(forecast_norm)
        predicted_skew_MC.extend(forecast_skew)

        t_nu_list.extend([nu] * block_size)
        t_mu_list.extend([mu_t] * block_size)
        t_sigma_list.extend([sigma_t] * block_size)
        norm_mu_list.extend([mu_norm] * block_size)
        norm_sigma_list.extend([sigma_norm] * block_size)
        skew_shape_list.extend([shape] * block_size)
        skew_loc_list.extend([loc_skew] * block_size)
        skew_scale_list.extend([scale_skew] * block_size)

    df_pred = pd.DataFrame({
        'predicted_t_MC': predicted_t_MC,
        'predicted_t_std': predicted_t_std,
        'predicted_norm_MC': predicted_norm_MC,
        'predicted_skew_MC': predicted_skew_MC,
        't_nu': t_nu_list,
        't_mu': t_mu_list,
        't_sigma': t_sigma_list,
        'norm_mu': norm_mu_list,
        'norm_sigma': norm_sigma_list,
        'skew_shape': skew_shape_list,
        'skew_loc': skew_loc_list,
        'skew_scale': skew_scale_list
    }, index=pred_dates)

    return df_pred

results = {}

# Per ciascun ticker (ci aspettiamo colonne "log_return_<ticker>" nel file unito)
for ticker in tickers:
    col = f'log_return_{ticker}'
    if col not in data.columns:
        print(f"Colonna {col} non trovata nel file unito.")
        continue
    series = data[col].dropna().sort_index()

    df_pred = block_forecast_MC(series, window_size, block_size, n_simulations)
    df_merged = df_pred.join(series.rename('actual_return'), how='left')

    # Calcolo delle metriche per ciascun modello (qui uso il modello t MC come esempio)
    metrics = {}
    for model in ['predicted_t_MC', 'predicted_norm_MC', 'predicted_skew_MC']:
        valid = df_merged['actual_return'].notna() & df_merged[model].notna()
        if valid.sum() > 0:
            mae = np.mean(np.abs(df_merged.loc[valid, model] - df_merged.loc[valid, 'actual_return']))
            mse = np.mean((df_merged.loc[valid, model] - df_merged.loc[valid, 'actual_return']) ** 2)
        else:
            mae, mse = np.nan, np.nan
        metrics[model] = {'MAE': mae, 'MSE': mse}

    # Calcola residui
    df_merged['resid_t_MC'] = df_merged['actual_return'] - df_merged['predicted_t_MC']
    df_merged['resid_norm_MC'] = df_merged['actual_return'] - df_merged['predicted_norm_MC']
    df_merged['resid_skew_MC'] = df_merged['actual_return'] - df_merged['predicted_skew_MC']

    results[ticker] = {'df': df_merged, 'metrics': metrics}

    # Salva i risultati per il ticker
    df_merged.to_csv(os.path.join(output_dir, f'results_{ticker}.csv'))

    # Plot comparativo tra rendimenti reali e previsione (modello t MC)
    plt.figure(figsize=(12, 6))
    plt.plot(df_merged.index, df_merged['actual_return'], label='Rendimento Reale', color='black')
    plt.plot(df_merged.index, df_merged['predicted_t_MC'], label='Previsione t (MC)', linestyle='--')
    plt.fill_between(df_merged.index,
                     df_merged['predicted_t_MC'] - df_merged['predicted_t_std'],
                     df_merged['predicted_t_MC'] + df_merged['predicted_t_std'],
                     color='gray', alpha=0.3, label='Incertezza')
    plt.title(f'Confronto Rendimenti Reali e Previsioni: {ticker}')
    plt.xlabel('Data')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plot_{ticker}_comparison.png'))
    plt.close()

    # Altri plot (residui, evoluzione parametri, etc.)
    # Plot residui (modello t MC)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(df_merged.index, df_merged['resid_t_MC'], marker='o', linestyle='-', color='blue')
    ax[0, 0].set_title('Serie Temporale dei Residui (t MC)')
    ax[0, 0].grid(True)
    ax[0, 1].hist(df_merged['resid_t_MC'].dropna(), bins=30, color='blue', alpha=0.7)
    ax[0, 1].set_title('Istogramma dei Residui (t MC)')
    ax[0, 1].grid(True)
    plot_acf(df_merged['resid_t_MC'].dropna(), ax=ax[1, 0])
    ax[1, 0].set_title('ACF dei Residui (t MC)')
    ax[1, 1].scatter(df_merged['predicted_t_MC'], df_merged['actual_return'], alpha=0.5, color='blue')
    ax[1, 1].plot([df_merged['predicted_t_MC'].min(), df_merged['predicted_t_MC'].max()],
                  [df_merged['predicted_t_MC'].min(), df_merged['predicted_t_MC'].max()],
                  color='red', linestyle='--')
    ax[1, 1].set_title('Scatter: Previsione t MC vs Rendimento Reale')
    ax[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plot_{ticker}_residuals_t_MC.png'))
    plt.close()

    # Plot evoluzione dei parametri del modello t MC
    plt.figure(figsize=(12, 6))
    plt.plot(df_merged.index, df_merged['t_nu'], label='t_nu')
    plt.plot(df_merged.index, df_merged['t_mu'], label='t_mu')
    plt.plot(df_merged.index, df_merged['t_sigma'], label='t_sigma')
    plt.title(f'Evoluzione Parametri t (MC): {ticker}')
    plt.xlabel('Data')
    plt.ylabel('Parametro')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plot_{ticker}_params_t_MC.png'))
    plt.close()

    # Validazione Out-of-Sample
    df_oos = df_merged.loc[df_merged.index >= cutoff_date]
    if not df_oos.empty:
        metrics_oos = {}
        for model in ['predicted_t_MC', 'predicted_norm_MC', 'predicted_skew_MC']:
            valid = df_oos['actual_return'].notna() & df_oos[model].notna()
            if valid.sum() > 0:
                mae = np.mean(np.abs(df_oos.loc[valid, model] - df_oos.loc[valid, 'actual_return']))
                mse = np.mean((df_oos.loc[valid, model] - df_oos.loc[valid, 'actual_return']) ** 2)
            else:
                mae, mse = np.nan, np.nan
            metrics_oos[model] = {'MAE': mae, 'MSE': mse}
        pd.DataFrame(metrics_oos).to_csv(os.path.join(output_dir, f'Error_metrics_OOS_{ticker}.csv'))
        logging.info(f"{ticker} - Metriche Out-of-Sample (MC): {metrics_oos}")

# Riassunto delle metriche per tutti i ticker
metrics_all = []
for ticker in tickers:
    if ticker not in results:
        continue
    for model, vals in results[ticker]['metrics'].items():
        metrics_all.append({
            'Ticker': ticker,
            'Model': model,
            'MAE': vals['MAE'],
            'MSE': vals['MSE']
        })
df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(os.path.join(output_dir, 'Error_metrics_all_models_MC.csv'), index=False)
print(df_metrics)

# Concatenazione dei dati per tutti i ticker
concatenated_list = []
for ticker in tickers:
    if ticker not in results:
        continue
    df = results[ticker]['df'].copy()
    df = df.reset_index().rename(columns={'index': 'Date'})
    df_filtered = df[(df['Date'] >= pd.to_datetime("2020-01-07")) & (df['Date'] <= pd.to_datetime("2024-12-20"))]
    df_filtered.loc[:, 'ticker'] = ticker  # utilizzo .loc per evitare warning
    # In questo esempio, usiamo il forecast del modello t simulato (MC) e la deviazione standard come features aggiuntive
    df_filtered = df_filtered[['Date', 'actual_return', 'predicted_t_MC', 'predicted_t_std', 'ticker']]
    df_filtered = df_filtered.rename(columns={'predicted_t_MC': 'predicted_return'})
    concatenated_list.append(df_filtered)
final_df = pd.concat(concatenated_list, ignore_index=True)
final_df.to_csv(os.path.join(output_dir, 'concatenated_returns_MC.csv'), index=False)
print("CSV concatenato salvato in:", os.path.join(output_dir, 'concatenated_returns_MC.csv'))
