import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Libreria ARCH per GARCH
from arch import arch_model

# TensorFlow e Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Keras Tuner
import keras_tuner as kt
from keras_tuner.tuners import RandomSearch

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")
import pickle

##############################################################################
# Callback per NaN
##############################################################################
class CheckNaNCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if np.isnan(logs.get('loss', 0)) or np.isnan(logs.get('val_loss', 0)):
            print(f"[Epoch {epoch}] NaN loss rilevato, interrompo il training.")
            self.model.stop_training = True

##############################################################################
# Download dati
##############################################################################
def load_data(tickers, start_date='01/01/2007', end_date='23/12/2024',
              save_dir='/content/drive/MyDrive/Prova_Garch_TUTTI_I_TICKERS'):
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

def add_vix_brent_gold_features(tickers_gold_vix_brent, start_date='01/01/2007', end_date='23/12/2024',
                                save_dir='/content/drive/MyDrive/Prova_Garch_TUTTI_I_TICKERS'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = pd.to_datetime(end_date, format='%d/%m/%Y')
    data_frames = {}
    for ticker in tickers_gold_vix_brent:
        try:
            print(f"Download dati per il ticker: {ticker}")
            data = yf.download(ticker, period='max', auto_adjust=False)
            data = data.sort_index().ffill().bfill()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns]
            data.columns = [col.replace(f" {ticker}", "") for col in data.columns]
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
            if ticker not in ['^VIX']:
                data = data[data['Volume'] > 0]
            data['return'] = data['Adj Close'].pct_change() * 100
            data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)) * 100
            data.dropna(inplace=True)
            data_frames[ticker] = data
            filename2 = os.path.join(save_dir, f"{ticker}_data.csv")
            data.to_csv(filename2, float_format='%.3f', index=True)
            print(f"[{ticker}] Dati salvati in: {filename2}, {data.shape[0]} righe finali")
        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")
    return data_frames

##############################################################################
# Funzioni di Feature Engineering
##############################################################################
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_MACD(series, short_window=12, long_window=26, signal_window=9):
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return sma, sma + rolling_std * num_std, sma - rolling_std * num_std

def compute_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(k_window).min()
    high_max = data['High'].rolling(k_window).max()
    stoch_k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(d_window).mean()
    return stoch_k, stoch_d

def add_features(data):
    print("Inizio aggiunta delle feature tecniche...")
    df = data.copy()
    df['gk_vol'] = np.sqrt(
        0.5 * (np.log(df['High'] / df['Low']))**2 -
        (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2
    )
    # Calcola HV10 (senza moltiplicare *100 in log_return, come già fatto in load_data)
    df['hv_10'] = df['log_return'].rolling(window=10).std()
    df['hv_10'].describe(percentiles=[0.9, 0.95, 0.99, 0.999])
    df['log_hv10'] = np.log(df['hv_10'] + 1e-8)
    df['target_vol'] = np.log1p(df['hv_10']).shift(-1)

    df['lag_log_return'] = df['log_return'].shift(1)
    df['lag_realized_vol'] = df['hv_10'].shift(1)
    df['open_to_close'] = df['Close'] - df['Open']
    df['hv_20'] = df['log_return'].rolling(window=20).std()
    df['log_hv20'] = np.log(df['hv_20'] + 1e-8)
    df['RSI'] = compute_RSI(df['Adj Close'])
    df['SMA_14'] = df['Adj Close'].rolling(window=14).mean()
    sma20, bb_up, bb_down = compute_bollinger_bands(df['Adj Close'], window=20, num_std=2)
    df['BB_mid'] = sma20
    df['BB_up'] = bb_up
    df['BB_down'] = bb_down
    macd_line, signal_line, macd_hist = compute_MACD(df['Adj Close'])
    df['MACD_line'] = macd_line
    df['MACD_signal'] = signal_line
    df['MACD_hist'] = macd_hist
    stoch_k, stoch_d = compute_stochastic_oscillator(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    df.dropna(inplace=True)
    print(f"Feature aggiunte. Righe finali: {df.shape[0]}")
    return df


##############################################################################
# GARCH day-by-day (1-step), rifit ogni 10 giorni (metodo manuale)
##############################################################################
def fit_garch(series, p=1, q=1, vol='Garch', dist='normal'):
    try:
        am = arch_model(series, vol=vol, p=p, q=q, dist=dist)
        res = am.fit(disp='off', show_warning=False)
        converged = hasattr(res, 'convergence') and (res.convergence == 0)
        return res, converged
    except Exception as e:
        return None, False

def grid_search_garch(series, p_vals=[1], q_vals=[1], vol_types=['Garch','EGARCH','GJR','TARCH'],
                      dists=['t','skewt','ged','normal'], criterion='aic'):
    best_score = np.inf
    best_params = (1,1,'Garch','normal')
    best_res = None
    for vol in vol_types:
        for p in p_vals:
            for q in q_vals:
                for dist in dists:
                    mod, conv = fit_garch(series.dropna(), p=p, q=q, vol=vol, dist=dist)
                    if mod is not None and conv:
                        score = mod.aic if criterion=='aic' else mod.bic
                        if score < best_score:
                            best_score = score
                            best_params = (p, q, vol, dist)
                            best_res = mod
    return best_params, best_res

def compute_next_sigma2(x_t, mu, last_sigma2, params, model_type):
    """
    Calcola sigma^2_{t+1} in base al modello selezionato.
    model_type è la stringa presente nei best_params[2] (es. 'Garch', 'EGARCH', 'GJR', 'TARCH').
    NOTA: Le formule per EGARCH, GJR e TARCH sono esempi; nella pratica dovresti implementare quella
    specifica se hai i relativi parametri (ad esempio gamma) ottenuti dal fitting.
    """
    if model_type == 'Garch':
        # Formula standard GARCH(1,1)
        return params['omega'] + params['alpha[1]'] * (x_t - mu)**2 + params['beta[1]'] * last_sigma2
    elif model_type == 'EGARCH':
        # Esempio semplificato: log(sigma^2_{t+1}) = omega + alpha*(|x_t - mu| - c) + beta*log(sigma^2_t)
        # c è una costante che potresti calcolare (qui la mettiamo a 0 come placeholder)
        c = 0.0
        return np.exp(params['omega'] + params['alpha[1]'] * (np.abs(x_t - mu) - c) + params['beta[1]'] * np.log(last_sigma2))
    elif model_type == 'GJR':
        # Esempio per GJR-GARCH: sigma^2_{t+1} = omega + alpha*(x_t - mu)^2 + gamma*(x_t - mu)^2*I(x_t < 0) + beta*sigma^2_t
        gamma = params.get('gamma[1]', 0)  # Se il parametro non è presente, lo considera 0
        indicator = 1 if (x_t - mu) < 0 else 0
        return params['omega'] + params['alpha[1]'] * (x_t - mu)**2 + gamma * (x_t - mu)**2 * indicator + params['beta[1]'] * last_sigma2
    elif model_type == 'TARCH':
        # Esempio semplificato per TARCH (le formulazioni variano)
        return params['omega'] + params['alpha[1]'] * np.abs(x_t - mu) + params['beta[1]'] * last_sigma2
    else:
        # Default: usa la formula standard GARCH
        return params['omega'] + params['alpha[1]'] * (x_t - mu)**2 + params['beta[1]'] * last_sigma2

def rolling_garch_forecast_daybyday_manual(df, refit_every=10):
    """
    Forecast GARCH 1-step con aggiornamento manuale della ricorsione.
    Ogni giorno:
       sigma²_{t+1} = formula specifica in base al modello selezionato.
    Ogni 'refit_every' giorni si rifitta il modello (grid search).
    """
    sr = df['log_return'].dropna()
    n = len(sr)
    forecast_vol = pd.Series(index=sr.index, dtype=float)

    pos = 100  # Finestra iniziale per il fitting
    train = sr.iloc[:pos]
    best_params, best_res = grid_search_garch(train)
    if best_res is None:
        best_res, _ = fit_garch(train)
    params = best_res.params
    mu    = params.get('mu', 0)
    # Inizializza last_sigma2 usando il forecast a 1-step
    fc = best_res.forecast(horizon=1)
    last_sigma2 = fc.variance.iloc[-1, 0]

    # Estrai il tipo di modello selezionato dal grid search
    model_type = best_params[2]

    steps_since_refit = 0
    print("Inizio forecast GARCH (manual update) con ricorsione giornaliera...")

    while pos < n:
        x_t = sr.iloc[pos]
        # Calcola sigma²_{t+1} usando la funzione dedicata
        sigma2_next = compute_next_sigma2(x_t, mu, last_sigma2, params, model_type)
        last_sigma2 = sigma2_next
        forecast_vol.iloc[pos] = np.sqrt(sigma2_next)

        pos += 1
        steps_since_refit += 1

        if pos % 50 == 0:
            print(f"GARCH forecast (manual update): elaborata {pos} di {n} osservazioni.")

        if steps_since_refit == refit_every and pos < n:
            print(f"Rifitting completo del modello GARCH a pos = {pos}.")
            train = sr.iloc[:pos].dropna()
            best_params, best_res = grid_search_garch(train)
            if best_res is None:
                best_res, _ = fit_garch(train)
            params = best_res.params
            mu    = params.get('mu', 0)
            fc = best_res.forecast(horizon=1)
            last_sigma2 = fc.variance.iloc[-1, 0]
            # Aggiorna il modello selezionato se necessario
            model_type = best_params[2]
            steps_since_refit = 0

    forecast_vol = forecast_vol.ffill()
    # Applica clipping sui percentili 1° e 99° per limitare valori estremi
    lower_bound, upper_bound = np.percentile(forecast_vol.dropna(), [1, 99])
    forecast_vol = np.clip(forecast_vol, lower_bound, upper_bound)
    # Usa shift(1) per allineare la previsione al giorno successivo
    df['garch_vol_forecast'] = forecast_vol.shift(1)
    df.dropna(subset=['garch_vol_forecast'], inplace=True)
    print("Forecast GARCH (manual update) completato.")
    return df


##############################################################################
# Funzioni per preparare i dati per la LSTM
##############################################################################
def create_sequences(df, features, target, window_size=10):
    X, y = [], []
    for i in range(len(df) - window_size):
        seq_x = df[features].iloc[i: i + window_size].values
        seq_y = df[target].iloc[i + window_size]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def scale_sequences(X_train, X_test):
    n_samples, window_size, n_features = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(n_samples, window_size, n_features)
    n_samples_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_samples_test, window_size, n_features)
    return X_train_scaled, X_test_scaled, scaler

##############################################################################
# Modello LSTM: Tuning con RandomSearch
##############################################################################
def build_model(hp):
    model = Sequential()
    # Primo strato LSTM: usa global_input_shape definito globalmente
    model.add(LSTM(
        hp.Int('units1', min_value=128, max_value=256, step=32),
        return_sequences=True,
        input_shape=global_input_shape,  # Correzione: uso di global_input_shape
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(Dropout(hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))

    # Secondo strato LSTM:
    model.add(LSTM(
        hp.Int('units2', min_value=64, max_value=128, step=16),
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))

    # Terzo strato LSTM:
    model.add(LSTM(
        hp.Int('units3', min_value=32, max_value=64, step=16),
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(Dropout(hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense layer:
    model.add(Dense(
        hp.Int('dense_units', min_value=32, max_value=64, step=16),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(Dense(1, activation='linear'))

    model.compile(
        optimizer=Adam(hp.Float('lr', min_value=0.0001, max_value=0.001, sampling='LOG')),
        loss='mse'
    )
    return model



##############################################################################
# Main: Training e fine tuning in modalità leave-one-out con split temporale
##############################################################################
if __name__ == "__main__":
    # Impostazioni iniziali e tickers
    tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI', 'NVDA', 'ITA', 'WMT', 'XOM', 'NKE', 'AMZN', 'NFLX', 'AAPL']
    # Definizione delle date per train, validation e test
    train_end_date = pd.to_datetime('2017-12-19')
    val_end_date   = pd.to_datetime('2019-12-19')  # validation fino al giorno prima del test
    test_start_date = pd.to_datetime('2019-12-20')
    test_end_date   = pd.to_datetime('2024-12-20')
    window_size = 10
    save_dir = "/content/drive/MyDrive/Prova_Garch_TUTTI_I_TICKERS"
    features_list = ['return', 'log_return', 'garch_vol_forecast',
                 'gk_vol', 'lag_log_return', 'open_to_close',
                 'hv_20', 'log_hv20', 'RSI', 'SMA_14', 'BB_mid', 'BB_up', 'BB_down',
                 'MACD_line', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D',
                 '^VIX_AdjClose', 'BZ=F_AdjClose', 'GC=F_AdjClose']

    target_column = 'target_vol'

    print("Inizio il download e preprocessing dei dati...")
    data = load_data(tickers, save_dir=save_dir)
    # Definisci i ticker di mercato per VIX, Brent e Gold
    market_tickers = ['^VIX', 'BZ=F', 'GC=F']

    # Scarica i dati di mercato per questi ticker
    market_data = add_vix_brent_gold_features(market_tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir=save_dir)

    for ticker in tickers:
        df = data[ticker]
        for m in market_tickers:
            m_df = market_data[m][['Adj Close']].rename(columns={'Adj Close': m + '_AdjClose'})
            # Se la colonna è già presente, saltala
            if m + '_AdjClose' in df.columns:
                print(f"La colonna {m + '_AdjClose'} è già presente in {ticker}.")
            else:
                df = df.join(m_df, how='left')
        data[ticker] = df

    for t in tickers:
        print(f"\nElaborazione per {t}...")
        df = data[t].copy()
        df = add_features(df)
        df = rolling_garch_forecast_daybyday_manual(df, refit_every=10)  # Assicurati di aver definito la funzione correttamente
        # Stampa un campione per verificare allineamento tra forecast e target
        print(df[['garch_vol_forecast', 'target_vol']].head(15))
        processed_path = os.path.join(save_dir, f"processed_{t}.csv")
        df.to_csv(processed_path, index=True)

        #prova plot di controllo ##DA ELIMINARE POI
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['garch_vol_forecast'], label='Forecast GARCH')
        plt.plot(df.index, df['target_vol'], label='Target Volatility')
        plt.title(f'Confronto Forecast vs Target per {t}')
        plt.legend()
        plt.show()
        # 2. Aggiungi una colonna per la differenza tra forecast e target e stampa un campione:
        df['diff'] = df['garch_vol_forecast'] - df['target_vol']
        print(df[['garch_vol_forecast', 'target_vol', 'diff']].head(15))

        print(f"File processato salvato in {processed_path}")
        data[t] = df

    # Creazione dei dataset per ogni ticker con split train/validation/test
    datasets = {}
    for t in tickers:
        df = data[t].copy()
        df.sort_index(inplace=True)
        train_df = df[df.index <= train_end_date].copy()
        val_df = df[(df.index > train_end_date) & (df.index <= val_end_date)].copy()
        test_df = df[(df.index >= test_start_date) & (df.index <= test_end_date)].copy()
        train_df = train_df[features_list + [target_column]].dropna()
        val_df = val_df[features_list + [target_column]].dropna()
        test_df = test_df[features_list + [target_column]].dropna()
        datasets[t] = {'train': train_df, 'val': val_df, 'test': test_df}
        print(f"Dataset per {t}: train {train_df.shape}, val {val_df.shape}, test {test_df.shape}")

    # --- Creazione delle sequenze per LSTM per ogni ticker ---
sequences = {}
for t in tickers:
    X_train, y_train = create_sequences(datasets[t]['train'], features_list, target_column, window_size)
    X_val, y_val     = create_sequences(datasets[t]['val'], features_list, target_column, window_size)
    X_test, y_test   = create_sequences(datasets[t]['test'], features_list, target_column, window_size)
    sequences[t] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'dates': datasets[t]['test'].index[window_size:]
    }
    print(f"Sequenze per {t}: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")

# --- Creazione dello scaler globale usando tutte le sequenze di training ---
# Concateno lo X_train di tutti i ticker e fissa la forma di input
global_X_train = np.concatenate([sequences[t]['X_train'] for t in tickers], axis=0)
n_samples, window_size, n_features = global_X_train.shape

global_scaler = StandardScaler()
global_scaler.fit(global_X_train.reshape(-1, n_features))
print("Scaler globale addestrato.")

# Definisci la forma di input globale
global_input_shape = (window_size, n_features)
print("Global input shape:", global_input_shape)

results = []
for left_out in tickers:
    print(f"\n------ Leave-One-Out: {left_out} ------")

    # 1) Concatena X_train e X_val di tutti i ticker tranne left_out
    X_train_all, y_train_all = [], []
    for t in tickers:
        if t != left_out:
            X_train_all.append(sequences[t]['X_train'])
            X_train_all.append(sequences[t]['X_val'])
            y_train_all.append(sequences[t]['y_train'])
            y_train_all.append(sequences[t]['y_val'])
    X_train_all = np.concatenate(X_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)

    print(f"Training base (no {left_out}): X={X_train_all.shape}, y={y_train_all.shape}")

    # 2) Fitta lo scaler solo su X_train_all (esclusi left_out)
    n_samples, window_size, n_features = X_train_all.shape
    X_train_all_2d = X_train_all.reshape(-1, n_features)
    local_scaler = StandardScaler()
    local_scaler.fit(X_train_all_2d)

    # 3) Trasforma X_train_all con local_scaler
    X_train_all_scaled = local_scaler.transform(X_train_all_2d).reshape(n_samples, window_size, n_features)

    # 4) Trasforma anche i dati del ticker left_out usando lo stesso local_scaler
    X_train_lo = sequences[left_out]['X_train']
    X_val_lo   = sequences[left_out]['X_val']
    X_test_lo  = sequences[left_out]['X_test']

    n_train_lo = X_train_lo.shape[0]
    n_val_lo   = X_val_lo.shape[0]
    n_test_lo  = X_test_lo.shape[0]

    X_train_lo_scaled = local_scaler.transform(X_train_lo.reshape(-1, n_features)).reshape(n_train_lo, window_size, n_features)
    X_val_lo_scaled   = local_scaler.transform(X_val_lo.reshape(-1, n_features)).reshape(n_val_lo, window_size, n_features)
    X_test_lo_scaled  = local_scaler.transform(X_test_lo.reshape(-1, n_features)).reshape(n_test_lo, window_size, n_features)

    y_train_lo = sequences[left_out]['y_train']
    y_val_lo   = sequences[left_out]['y_val']
    y_test_lo  = sequences[left_out]['y_test']

    # 5) Esegui il tuning del modello base sui dati esclusi (usando X_train_all_scaled)
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,  # puoi regolare questo parametro
        executions_per_trial=1,
        directory='tuner_dir',
        project_name=f'lstm_tuning_{left_out}'
    )
    tuner.search(X_train_all_scaled, y_train_all, epochs=50, validation_split=0.1,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    best_model = tuner.get_best_models(num_models=1)[0]
    print("Iperparametri migliori trovati:")
    print(tuner.get_best_hyperparameters(num_trials=1)[0].values)

    # (Opzionale) ulteriore training del modello base sui dati esclusi
    history = best_model.fit(X_train_all_scaled, y_train_all, epochs=100, batch_size=64, verbose=1,
                             callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                                        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5),
                                        CheckNaNCallback()],
                             validation_split=0.1)

    # 6) Fine tuning sul ticker left_out usando i dati scalati localmente
    model_ft = clone_model(best_model)
    model_ft.set_weights(best_model.get_weights())
    for layer in model_ft.layers[:-2]:
        layer.trainable = False
    model_ft.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    X_train_ft = X_train_lo_scaled
    X_val_ft   = X_val_lo_scaled

    ft_callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        CheckNaNCallback()
    ]
    print(f"Fine tuning sul ticker {left_out}...")
    model_ft.fit(X_train_ft, y_train_lo, epochs=50, batch_size=16, verbose=1,
                callbacks=ft_callbacks, validation_data=(X_val_ft, y_val_lo))

    # 7) Valutazione sul test del ticker left_out
    X_test_lo = sequences[left_out]['X_test']
    y_test_lo = sequences[left_out]['y_test']
    # NOTA: puoi decidere di usare X_test_lo_scaled al posto di X_test_lo se vuoi trasformare anche il test
    y_pred = model_ft.predict(X_test_lo_scaled).flatten()
    lstm_vol = np.expm1(y_pred)  # Riporta i valori predetti alla scala originale
    actual_vol = np.expm1(y_test_lo)  # Riporta anche il target vero alla scala originale

    test_dates = sequences[left_out]['dates']
    test_df = datasets[left_out]['test'].iloc[window_size:]
    garch_vol = test_df['garch_vol_forecast'].values

    mse_val_lstm = mean_squared_error(actual_vol, lstm_vol)
    mae_val_lstm = mean_absolute_error(actual_vol, lstm_vol)
    spearman_corr_lstm, _ = spearmanr(actual_vol, lstm_vol)

    mse_val_garch = mean_squared_error(actual_vol, garch_vol)
    mae_val_garch = mean_absolute_error(actual_vol, garch_vol)
    spearman_corr_garch, _ = spearmanr(actual_vol, garch_vol)

    results.append({
        'ticker': left_out,
        'MSE_lstm': mse_val_lstm,
        'MAE_lstm': mae_val_lstm,
        'Spearman_lstm': spearman_corr_lstm,
        'MSE_only_garch': mse_val_garch,
        'MAE_only_garch': mae_val_garch,
        'Spearman_only_garch': spearman_corr_garch
    })

    print(f"Metrics LSTM per {left_out}: MSE={mse_val_lstm:.4f}, MAE={mae_val_lstm:.4f}, Spearman={spearman_corr_lstm:.4f}")
    print(f"Metrics GARCH per {left_out}: MSE={mse_val_garch:.4f}, MAE={mae_val_garch:.4f}, Spearman={spearman_corr_garch:.4f}")

    model_path = os.path.join(save_dir, f"lstm_model_{left_out}.h5")
    model_ft.save(model_path)
    print(f"Modello salvato in {model_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_vol, label='Volatilità Reale')
    plt.plot(test_dates, garch_vol, label='GARCH Forecast')
    plt.plot(test_dates, lstm_vol, label='GARCH+LSTM Forecast')
    plt.yscale('log')
    plt.xlabel('Data')
    plt.ylabel('Volatilità')
    plt.title(f'Confronto Previsioni Volatilità - {left_out}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"forecast_comparison_{left_out}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot salvato in {plot_path}")
    # Salva i risultati di test per il ticker corrente in un CSV
    test_results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual_Vol': actual_vol,
        'LSTM_Vol': lstm_vol,
        'GARCH_Vol': garch_vol
    })
    test_results_csv_path = os.path.join(save_dir, f"test_results_{left_out}.csv")
    test_results_df.to_csv(test_results_csv_path, index=False)
    print(f"Test results CSV salvato in: {test_results_csv_path}")

results_df = pd.DataFrame(results)
csv_results = os.path.join(save_dir, "lstm_forecasting_metrics_leaveoneout.csv")
results_df.to_csv(csv_results, index=False)
print(f"\nCSV dei risultati salvato in {csv_results}")

# Calcola e salva le medie aggregate di MSE e MAE per LSTM e solo GARCH
avg_metrics = {
    'Avg_MSE_LSTM': results_df['MSE_lstm'].mean(),
    'Avg_MAE_LSTM': results_df['MAE_lstm'].mean(),
    'Avg_MSE_ONLY_GARCH': results_df['MSE_only_garch'].mean(),
    'Avg_MAE_ONLY_GARCH': results_df['MAE_only_garch'].mean()
}
avg_metrics_df = pd.DataFrame([avg_metrics])
summary_csv = os.path.join(save_dir, "summary_metrics.csv")
avg_metrics_df.to_csv(summary_csv, index=False)
print(f"\nCSV dei risultati medi salvato in {summary_csv}")