import pandas as pd
import numpy as np
import cvxpy as cp
import os

# -----------------------
# PARAMETRI DI CONFIGURAZIONE
# -----------------------
from config import Config

config = Config()
gamma = config.gamma

assets = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
n_assets = len(assets)

start_date = "2019-12-20"
end_date = "2024-12-20"

c_minus = 0.002 * np.ones(n_assets)
c_plus = 0.001 * np.ones(n_assets)


# -----------------------
# 1. CARICAMENTO DEI DATI DI RITORNO E CREAZIONE FILE MERGIATI DEI VARI TICKERS per ottenere R
# -----------------------
def merge_log_returns(tickers, directory="./Tickers_file/", output_file="./merged_log_returns.csv"):
    """
    CALCOLO R, vettore dei rendimenti storici sui dati storici reali

    """
    merged_df = None
    for ticker in tickers:
        file_path = os.path.join(directory, f"{ticker}_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Date', 'log_return'])
            df.rename(columns={'log_return': f'{ticker}_log_return'}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
        else:
            print(f"❌ File not found: {file_path}")

    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged file saved as {output_file}")
    else:
        print("❌ No data to merge.")

    return merged_df


log_returns_df = merge_log_returns(assets)

if log_returns_df is None:
    raise FileNotFoundError("❌ Errore: Nessun dato di ritorno logaritmico trovato. Controlla i file dei tickers.")

log_returns_df['Date'] = pd.to_datetime(log_returns_df['Date'])
log_returns_df.sort_values("Date", inplace=True)


'''
in questo punto ho tutto il file mergiato dei log_return ed ho R
'''


#2 CALCOLO mu con finestra mobile e stima delle tendenze recenti, in maniera differente da R
def calculate_mu(assets, merged_df, current_date, window_size=20):
    """
    Calcola il vettore dei ritorni attesi μ per ciascun asset usando una finestra mobile
    di log_return giornalieri precedenti alla data corrente.

    Parameters:
    - assets: lista dei ticker
    - merged_df: DataFrame giornaliero contenente le colonne 'Date' e per ogni asset '<TICKER>_log_return'
    - current_date: la data (string o datetime) per cui calcolare μ
    - window_size: numero di giorni da considerare nella finestra mobile

    Returns:
    - mu_vector: array numpy con la media mobile dei log_return per ogni asset
    """
    # Assicurati che current_date sia in formato datetime
    current_date = pd.to_datetime(current_date)

    # Filtra il DataFrame per le date precedenti a current_date
    df_window = merged_df[merged_df['Date'] < current_date].tail(window_size)

    mu_vector = []
    for ticker in assets:
        col_name = f"{ticker}_log_return"
        if col_name in df_window.columns and not df_window.empty:
            mu_val = df_window[col_name].mean()  # Media dei log_return della finestra
        else:
            mu_val = 0.0  # Se non c'è il dato, assegna 0
        mu_vector.append(mu_val)
    return np.array(mu_vector)


# -----------------------
# 3. CARICAMENTO DELLA VOLATILITÀ PREDETTA PER AVERE POI LA MATRICE DI COVARIANZA (IN QUESTO MOMENTO HO SOLO QUELLA PREDETTA DAL 2019 AL 2024)
#DEVO SOLO DECIDERE SE COLLEGARE I DATI DI VOLATILITA' REALI FINO AL 2019.12.20 E POI UNIRE QUELLI PREDETTI
# -----------------------
def merge_results_Garch_LSTM(tickers, directory= "./Risultati_GARCH_LSTM_Forecasting/",  output_file="./Tickers_file/merged_Results_Garch_LSTM.csv"):
    merged_df = None
    for ticker in tickers:
        file_path = os.path.join(directory, f"risultati_forecasting_{ticker}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Date', 'Volatilità Predetta (exp)'])
            df.rename(columns={'Volatilità Predetta (exp)': f'{ticker}_Vol_Pred'}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
        else:
            print(f"❌ File not found: {file_path}")

    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged file saved as {output_file}")
    else:
        print("❌ No data to merge.")

    return merged_df

merge_df_volatility = merge_results_Garch_LSTM(assets)

if merge_df_volatility is None or merge_df_volatility.empty:
    raise FileNotFoundError("❌ Errore: Nessun dato di volatilità trovato. Controlla i file GARCH-LSTM.")

merge_df_volatility.fillna(method='ffill', inplace=True)
merge_df_volatility.fillna(method='bfill', inplace=True)

merge_df_volatility['Date'] = pd.to_datetime(merge_df_volatility['Date'])
merge_df_volatility.sort_values("Date", inplace=True)




#MERGIO I FILE DI FORECAST VOLATILITà E LOG_RETURNS
# Supponiamo di aver fuso i dati così:
#merged_forecast_df = pd.merge(log_returns_df, merge_df_volatility, on="Date", how="inner")
#merged_forecast_df.sort_values("Date", inplace=True)


# -----------------------
# 5. CALCOLO DELLA MATRICE DI COVARIANZA
# -----------------------
def get_covariance_matrix(row, log_returns_df, assets):
    """
    Calcola la matrice di covarianza Σ per una data forecast.

    Parametri:
      - row: una riga del DataFrame dei forecast, che contiene le previsioni di volatilità per ogni asset.
      - log_returns_df: DataFrame dei log_return storici, usato per calcolare la matrice di correlazione.
      - assets: lista dei ticker (es. ['XLK', 'XLV', ...]).

    Restituisce:
      - Sigma: la matrice di covarianza (numpy array) per la data in 'row'.
    """
    # Estrai il vettore delle volatilità predette
    vol_vector = np.array([row.get(f"{ticker}_Vol_Pred", np.nan) for ticker in assets])

    # Se mancano dati, usa un fallback (ad esempio, 0.01)
    if np.isnan(vol_vector).any():
        print(f"⚠️ Dati di volatilità mancanti per la data {row['Date']}. Uso fallback = 0.01")
        vol_vector = np.full(len(assets), 0.01)

    # Costruisci la matrice diagonale D
    D = np.diag(vol_vector)

    # Calcola la matrice di correlazione. Qui usiamo i log_return storici
    # Assicurati di usare le colonne corrispondenti agli asset
    cols = [f"{ticker}_log_return" for ticker in assets]
    corr_matrix = log_returns_df[cols].corr().values
    # Sostituisci eventuali NaN con 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Calcola la matrice di covarianza Σ = D * corr_matrix * D
    Sigma = D @ corr_matrix @ D

    # Se la matrice è mal condizionata, aggiungi una piccola regolarizzazione
    if np.linalg.cond(Sigma) > 1e6:
        print("⚠️ Sigma mal condizionata, aggiungo regolarizzazione")
        Sigma += np.eye(len(assets)) * 0.0001

    return Sigma


# ----------------------
# -----------------------
# 6. OTTIMIZZAZIONE CON COSTI DI TRANSAZIONE
# -----------------------
def optimize_with_transaction_costs(mu, Sigma, w_tilde, c_minus, c_plus, delta_minus=0, delta_plus=0, gamma=0.95):
    n = len(mu)
    w = cp.Variable(n)
    delta_minus_var = cp.Variable(n, nonneg=True)
    delta_plus_var = cp.Variable(n, nonneg=True)

    linear_cost = cp.sum(cp.multiply(c_minus, delta_minus_var) + cp.multiply(c_plus, delta_plus_var))
    quad_cost = cp.sum(cp.multiply(delta_minus, cp.square(delta_minus_var)) + cp.multiply(delta_plus, cp.square(delta_plus_var)))
    total_transaction_cost = linear_cost + quad_cost

    constraints = [
        w == w_tilde + delta_plus_var - delta_minus_var,
        cp.sum(w) == 1,
        w >= 0,
        w <= 1,
        total_transaction_cost <= 0.01
    ]

    objective = cp.Minimize(0.5 * cp.quad_form(w, Sigma) - gamma * (w @ mu - total_transaction_cost))

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    if w.value is None:
        print(f"❌ Ottimizzazione fallita per mu={mu}, Sigma={Sigma}. Restituisco valori di default.")
        return np.ones(n) / n, np.zeros(n), np.zeros(n)

    return w.value, delta_minus_var.value, delta_plus_var.value
