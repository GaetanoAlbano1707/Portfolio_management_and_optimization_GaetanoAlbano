import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Libreria ARCH per il GARCH
from arch import arch_model

# TensorFlow e Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, losses, metrics

# KerasTuner per il tuning degli iperparametri
import keras_tuner as kt

# Scikit-learn per scaling e metriche
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

import shap
import warnings
warnings.filterwarnings("ignore")

import pickle

##############################################################################
# 1. Callback per interrompere il training in caso di NaN
##############################################################################
class CheckNaNCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if np.isnan(logs.get('loss', 0)) or np.isnan(logs.get('val_loss', 0)):
            print(f"[Epoch {epoch}] NaN loss rilevato, interrompo il training.")
            self.model.stop_training = True

##############################################################################
# 2. Funzioni per il download e il preprocessing dei dati
##############################################################################
def load_data(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='/content/drive/MyDrive/Prova_Garch5'):
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
            data['return'] = data['Adj Close'].pct_change() * 100  # percentuale
            data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            data.dropna(inplace=True)
            data_frames[ticker] = data
            filename = os.path.join(save_dir, f"{ticker}_data.csv")
            data.to_csv(filename, float_format='%.3f', index=True)
            print(f"[{ticker}] Dati salvati in: {filename}, {data.shape[0]} righe finali")
        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")
    return data_frames

def add_vix_brent_gold_features(tickers_gold_vix_brent, start_date='01/01/2007', end_date='23/12/2024', save_dir='/content/drive/MyDrive/Prova_Garch5'):
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
            # Applica il filtro Volume solo se il ticker non è ^VIX
            if ticker not in ['^VIX']:
                data = data[data['Volume'] > 0]
            data['return'] = data['Adj Close'].pct_change() * 100  # percentuale
            data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            data.dropna(inplace=True)
            data_frames[ticker] = data
            filename2 = os.path.join(save_dir, f"{ticker}_data.csv")
            data.to_csv(filename2, float_format='%.3f', index=True)
            print(f"[{ticker}] Dati salvati in: {filename2}, {data.shape[0]} righe finali")
        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")
    return data_frames

##############################################################################
# 3. Funzioni di Feature Engineering
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
    #df['rolling_vol'] = df['log_return'].rolling(window=10).std()
    # Calcola la volatilità reale giornaliera come il valore assoluto del log_return (in percentuale)
    df['hv_10'] = df['log_return'].rolling(window=10).std() * 100
    # Trasforma HV10 in log (aggiungendo epsilon per evitare log(0))
    df['log_hv10'] = np.log(df['hv_10'] + 1e-8)
    # Il target sarà la log HV10 del giorno successivo (uso log(1+x))
    df['target_vol'] = np.log1p(df['hv_10'].shift(-1))

    # Calcolo di alcune lag per features
    df['lag_log_return'] = df['log_return'].shift(1)
    df['lag_realized_vol'] = df['hv_10'].shift(1)  # ora si usa HV10 come riferimento
    df['open_to_close'] = df['Close'] - df['Open']
    df['hv_20'] = df['log_return'].rolling(window=20).std()*100
    # Trasforma HV10 in log (aggiungendo epsilon per evitare log(0))
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
# 4. Funzioni GARCH: fit, grid search e forecast rolling con expanding window
##############################################################################
def is_converged(res):
    if hasattr(res, 'convergence'):
        return res.convergence
    elif hasattr(res, 'converged'):
        return res.converged
    else:
        print("Warning: nessun attributo di convergenza trovato, assumo converged=True")
        return True

def fit_garch_model(series, p=1, q=1, vol='Garch', dist='normal'):
    print(f"Fitting GARCH model: p={p}, q={q}, vol={vol}, dist={dist}")
    try:
        if vol == 'GJR':
            am = arch_model(series, vol='Garch', p=p, o=1, q=q, dist=dist)
        elif vol == 'TARCH':
            am = arch_model(series, vol='aparch', p=p, q=q, dist=dist, power=1.0)
        else:
            am = arch_model(series, vol=vol, p=p, q=q, dist=dist)
        res = am.fit(disp='off', show_warning=False)
        converged = is_converged(res)
        print("Model fit terminato. Converged:", converged)
        if converged:
            return res, True
        else:
            return res, False
    except Exception as e:
        print("Errore nel fitting GARCH:", e)
        return None, False

def grid_search_garch(series, p_values=[1,2], q_values=[1,2],
                      vol_types=['Garch','EGARCH','GJR','TARCH'],
                      dists=['t','skewt','ged','normal'],
                      criterion='aic'):
    print("Inizio grid search GARCH...")
    best_score = np.inf
    best_params = (None, None, None, None)
    best_res = None
    for vol in vol_types:
        for p in p_values:
            for q in q_values:
                for dist in dists:
                    print(f"Test: vol={vol}, p={p}, q={q}, dist={dist}")
                    res, conv = fit_garch_model(series.dropna(), p=p, q=q, vol=vol, dist=dist)
                    if res is not None and conv:
                        score = res.aic if criterion=='aic' else res.bic
                        print(f"Score ottenuto: {score}")
                        if score < best_score:
                            best_score = score
                            best_params = (p, q, vol, dist)
                            best_res = res
    print("Grid search completata. Best params:", best_params, "Score:", best_score)
    return best_params, best_res, best_score


def prepare_pipeline_data(raw_data, external_data, feature_cols, target_col):
    df = add_features(raw_data)
    for ext_ticker, ext_df in external_data.items():
        new_col = ext_ticker.replace('^', '').replace('=F', '').lower() + "_log_return"
        ext_feature = ext_df[['log_return']].rename(columns={'log_return': new_col})
        df = df.merge(ext_feature, left_index=True, right_index=True, how='left')
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    df = rolling_garch_forecast(df, step=10, use_expanding=True)
    fallback_mask = df['garch_vol_forecast'].isna()
    df.loc[fallback_mask, 'garch_vol_forecast'] = df['garch_vol_forecast'].ffill()
    df.dropna(subset=feature_cols + [target_col], inplace=True)
    return df


def rolling_garch_forecast(data, step=10, use_expanding=True,
                           p_values=[1,2], q_values=[1,2],
                           vol_types=['Garch','EGARCH','GJR','TARCH'],
                           dists=['t','skewt','ged','normal']):
    print(">>> Inizio rolling GARCH forecast...")
    log_ret_series = data['log_return'] * 100.0
    all_index = log_ret_series.index
    n = len(log_ret_series)
    forecast_vol = pd.Series(index=all_index, dtype=float)

    pos = 100  # Punto di inizio
    while pos < n:
        print(f"\n>>> Posizione corrente: {pos} (Training window: {pos} dati)")
        if use_expanding:
            train_data = log_ret_series.iloc[:pos]
        else:
            window = 1000
            train_data = log_ret_series.iloc[max(0, pos-window):pos]
            if len(train_data) < 50:
                pos += step
                continue

        best_params, best_res, score = grid_search_garch(train_data, p_values, q_values, vol_types, dists)
        if best_res is None:
            best_res, _ = fit_garch_model(train_data, p=1, q=1, vol='Garch', dist='t')
            if best_res is None:
                pos += step
                continue

        block_size = min(step, n - pos)
        # Usa 'simulation' per EGARCH e TARCH (che non supportano horizon>1 in analytic)
        forecast_method = "simulation" if best_params[2] in ['EGARCH', 'TARCH'] else "analytic"
        fc = best_res.forecast(horizon=block_size, method=forecast_method)

        for j in range(block_size):
            current_index = log_ret_series.index[pos + j]
            try:
                var_val = fc.variance.iloc[-block_size + j, 0]
                vol_forecast = np.sqrt(var_val) if var_val > 0 else np.nan
            except:
                vol_forecast = np.nan

            if np.isnan(vol_forecast):
                prev_valid = forecast_vol.dropna()
                vol_forecast = prev_valid.iloc[-1] if not prev_valid.empty else np.nan

            forecast_vol.loc[current_index] = vol_forecast

        pos += block_size

    forecast_vol = forecast_vol.ffill()
    valid_forecasts = forecast_vol.dropna()
    lower_bound, upper_bound = np.percentile(valid_forecasts, [1, 99])
    forecast_vol = np.clip(forecast_vol, lower_bound, upper_bound)
    data['garch_vol_forecast'] = forecast_vol
    data.dropna(subset=['garch_vol_forecast'], inplace=True)
    return data

##############################################################################
# 5. Preparazione dei dati per LSTM e split temporale
##############################################################################
def prepare_lstm_data_multivariate(data, feature_cols, target_col, look_back=10):
    print("Preparazione dei dati per LSTM...")
    df = data.copy()
    X, y, dates = [], [], []
    features = df[feature_cols].values
    target = df[target_col].values
    for i in range(look_back, len(df)):
        X.append(features[i-look_back:i])
        y.append(target[i])
        dates.append(df.index[i])
    X, y, dates = np.array(X), np.array(y), np.array(dates)
    print(f"Preparazione completata: {X.shape[0]} campioni creati.")
    return X, y, dates

def time_series_train_test_split(X, y, dates, test_years=5):
    print("Esecuzione dello split train/test...")
    last_date = pd.to_datetime(dates[-1])
    test_start_date = last_date - pd.DateOffset(years=test_years)
    mask_train = (dates < test_start_date)
    mask_test = (dates >= test_start_date)
    print(f"Train: {np.sum(mask_train)} campioni, Test: {np.sum(mask_test)} campioni")
    return X[mask_train], y[mask_train], X[mask_test], y[mask_test], dates[mask_train], dates[mask_test]

##############################################################################
# 6. Loss custom: QLIKE (opzionale)
##############################################################################
def qlike_loss(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 1e-8, 1e6)
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1e6)
    part1 = tf.square(y_true) / tf.square(y_pred)
    part2 = -tf.math.log(tf.square(y_pred))
    return tf.reduce_mean(part1 + part2)

##############################################################################
# 7. Costruzione del modello LSTM (per KerasTuner)
##############################################################################
def build_model_lstm(hp):
    model = Sequential([
        LSTM(
            units=hp.Choice('units1lstm', [16, 32, 64, 128, 256]),
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-5),
            input_shape=(None, hp.get('input_dim'))
        ),
        LSTM(
            units=hp.Choice('units2lstm', [16, 32, 64, 128, 256]),
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-5)
        ),
        LSTM(
            units=hp.Choice('units3lstm', [16, 32, 64, 128, 256]),
            return_sequences=False,
            kernel_regularizer=regularizers.l2(1e-5)
        ),
        Dense(
            units=hp.Choice('units1dense', [16, 32, 64, 128]),
            activation=hp.Choice('activation1', ['relu', 'tanh']),
            kernel_regularizer=regularizers.l2(1e-5)
        ),
        Dropout(rate=hp.Choice('dropout1', [0.25, 0.5])),
        Dense(
            units=hp.Choice('units2dense', [16, 32, 64, 128]),
            activation=hp.Choice('activation2', ['relu', 'tanh']),
            kernel_regularizer=regularizers.l2(1e-5)
        ),
        Dense(units=1)
    ])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate, clipnorm=1.0),
        loss=qlike_loss,
        metrics=[metrics.MeanAbsoluteError()]
    )
    return model

##############################################################################
# 8. Funzioni per il tuning e il fine tuning dell'LSTM
##############################################################################
def tune_lstm_model(X_train, y_train, X_val, y_val, input_dim, max_trials=10, epochs=100):
    hp = kt.HyperParameters()
    hp.Fixed('input_dim', input_dim)
    tuner = kt.RandomSearch(
        build_model_lstm,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='lstm_tuning',
        project_name='volatility_forecasting',
        overwrite=True,
        hyperparameters=hp
    )
    tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Migliori iperparametri trovati:", best_hps.values)
    return best_model, best_hps, tuner

def freeze_layers(model, freeze_until_idx):
    """
    Freeza tutti i layer fino all'indice specificato (non inclusi).
    Ad esempio, freeze_until_idx=4 freeza i layer 0,1,2,3.
    """
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= freeze_until_idx)
    return model

def fine_tune_model(model, X_train, y_train, X_val, y_val, epochs=100):
    # Esegui il freezing dei layer fino a un certo indice (es. 4)
    model = freeze_layers(model, freeze_until_idx=4)
    # Recompila con un learning rate ridotto per il fine tuning
    model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
                  loss=qlike_loss,
                  metrics=[metrics.MeanAbsoluteError()])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                                   CheckNaNCallback()],
                        verbose=1)
    return model, history

##############################################################################
# Funzione: Visualizza importanza delle feature con SHAP (DeepExplainer)
##############################################################################
def explain_lstm_with_shap(model, X_sample, feature_names, save_path=None):
    print("\nEsecuzione SHAP per LSTM...")
    explainer = shap.DeepExplainer(model, X_sample[:50])
    shap_values = explainer.shap_values(X_sample[:50])


    print("Generazione summary_plot SHAP...")
    shap.summary_plot(shap_values[0], X_sample[:50], feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path)
        print(f"Grafico SHAP salvato in: {save_path}")
    plt.show()


##############################################################################
# Funzione: Ensemble tra LSTM e GARCH (media pesata) + metriche
##############################################################################
def evaluate_ensemble(y_true, y_pred_lstm, y_pred_garch, alpha=0.7):
    offset = 1e-6
    max_val = 1e6

    y_true = np.clip(y_true.flatten(), a_min=offset, a_max=max_val)
    y_pred_lstm = np.clip(y_pred_lstm.flatten(), a_min=offset, a_max=max_val)
    y_pred_garch = np.clip(y_pred_garch.flatten(), a_min=offset, a_max=max_val)

    y_pred_ensemble = alpha * y_pred_lstm + (1 - alpha) * y_pred_garch
    y_pred_ensemble = np.clip(y_pred_ensemble, a_min=offset, a_max=max_val)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
    mae = mean_absolute_error(y_true, y_pred_ensemble)
    pearson = np.corrcoef(y_true.ravel().astype(float), y_pred_ensemble.ravel().astype(float))[0, 1]
    spearman = spearmanr(y_true.ravel().astype(float), y_pred_ensemble.ravel().astype(float))[0]
    qlike = np.mean((np.square(y_true) / np.square(y_pred_ensemble)) - np.log(np.square(y_pred_ensemble)))
    return {
        "Model": "GARCH + LSTM (ensemble)",
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
        "QLIKE": qlike
    }


##############################################################################
# 9. Pipeline Leave-One-Out: Pretraining e Fine Tuning per ogni ticker
##############################################################################
def run_leave_one_out_pipeline(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='/content/drive/MyDrive/Prova_Garch5'):
    os.makedirs(save_dir, exist_ok=True)
    # Scarica dati per tutti i tickers
    data_dict = load_data(tickers, start_date=start_date, end_date=end_date, save_dir=save_dir)
    tickers_gold_vix_brent = ['GC=F', '^VIX', 'BZ=F']
    external_data = add_vix_brent_gold_features(tickers_gold_vix_brent, start_date=start_date, end_date=end_date, save_dir=save_dir)

    # Definiamo le feature e il target per la LSTM
    feature_cols = [
        'garch_vol_forecast',   # forecast GARCH
        'gk_vol', 'log_return', 'lag_log_return', 'lag_realized_vol', 'hv_20',
        'RSI', 'SMA_14', 'BB_mid', 'BB_up', 'BB_down',
        'MACD_line', 'MACD_signal', 'MACD_hist',
        'Stoch_K', 'Stoch_D', 'open_to_close', 'hv_10',
        'vix_log_return', 'gc_log_return', 'bz_log_return'
    ]
    target_col = 'target_vol'  # target = log HV10 del giorno successivo
    look_back = 10

    all_metrics_global = []  # Lista per accumulare le metriche di tutti i ticker

    # Ciclo Leave-One-Out: per ogni ticker come target
    for target_ticker in tickers:
        print(f"\n\n=== Inizio Leave-One-Out per il ticker TARGET: {target_ticker} ===")
        pretrain_tickers = [t for t in tickers if t != target_ticker]

        # --- Pretraining sui tickers esclusi ---
        pretrain_dfs = []
        for ticker in pretrain_tickers:
            print(f"\n>>> Elaborazione per ticker {ticker} (Pretraining)")
            data_raw = data_dict[ticker].copy()
            data_feat = add_features(data_raw)
            # Unisci le feature esterne
            for ext_ticker, ext_df in external_data.items():
                new_col = ext_ticker.replace('^', '').replace('=F', '').lower() + "_log_return"
                ext_feature = ext_df[['log_return']].rename(columns={'log_return': new_col})
                data_feat = data_feat.merge(ext_feature, left_index=True, right_index=True, how='left')
            data_feat.fillna(method='ffill', inplace=True)
            data_feat.dropna(inplace=True)
            data_feat = rolling_garch_forecast(data_feat, step=10, use_expanding=True,
                                               p_values=[1,2], q_values=[1,2],
                                               vol_types=['Garch','EGARCH','GJR','TARCH'],
                                               dists=['t','skewt','ged','normal'])
            # Se rimangono NaN, puoi decidere di sostituirli con l'ultimo forecast valido
            fallback_mask = data_feat['garch_vol_forecast'].isna()
            if fallback_mask.sum() > 0:
                # Lascia i NaN o sostituisci con un valore neutro, ad esempio:
                data_feat.loc[fallback_mask, 'garch_vol_forecast'] = data_feat['garch_vol_forecast'].ffill()


            data_feat.dropna(subset=feature_cols + [target_col], inplace=True)
            pretrain_dfs.append(data_feat)

        # Aggrega i dati di pretraining
        pretrain_data = pd.concat(pretrain_dfs, axis=0).sort_index()
        print(f"\nDati aggregati per il pretraining: {pretrain_data.shape[0]} righe")
        # Prepara i dati per LSTM (pretraining)
        X_pre, y_pre, dates_pre = prepare_lstm_data_multivariate(pretrain_data, feature_cols, target_col, look_back=look_back)
        X_train_pre, y_train_pre, X_test_pre, y_test_pre, train_dates_pre, test_dates_pre = time_series_train_test_split(X_pre, y_pre, dates_pre, test_years=5)
        # Split train/validation 80/20
        train_size = int(0.8 * len(X_train_pre))
        X_train_pre_, X_val_pre = X_train_pre[:train_size], X_train_pre[train_size:]
        y_train_pre_, y_val_pre = y_train_pre[:train_size], y_train_pre[train_size:]
        print(f"Pretraining - Train: {X_train_pre_.shape[0]}, Val: {X_val_pre.shape[0]}")

        # Standardizzazione dei dati (pretraining)
        n_features = X_train_pre_.shape[2]
        scalerX_pre = StandardScaler()
        X_train_pre_flat = X_train_pre_.reshape(-1, n_features)
        X_train_pre_scaled_flat = scalerX_pre.fit_transform(X_train_pre_flat)
        X_train_pre_scaled = X_train_pre_scaled_flat.reshape(X_train_pre_.shape)

        X_val_pre_flat = X_val_pre.reshape(-1, n_features)
        X_val_pre_scaled_flat = scalerX_pre.transform(X_val_pre_flat)
        X_val_pre_scaled = X_val_pre_scaled_flat.reshape(X_val_pre.shape)

        X_test_pre_flat = X_test_pre.reshape(-1, n_features)
        X_test_pre_scaled_flat = scalerX_pre.transform(X_test_pre_flat)
        X_test_pre_scaled = X_test_pre_scaled_flat.reshape(X_test_pre.shape)

        scalerY_pre = StandardScaler()
        y_train_pre_reshaped = np.array(y_train_pre_).reshape(-1, 1)
        y_train_pre_scaled = scalerY_pre.fit_transform(y_train_pre_reshaped).ravel()
        y_val_pre_scaled = scalerY_pre.transform(np.array(y_val_pre).reshape(-1, 1)).ravel()
        y_test_pre_scaled = scalerY_pre.transform(np.array(y_test_pre).reshape(-1, 1)).ravel()

        # Tuning e pretraining LSTM
        input_dim = n_features
        base_model, best_hps, tuner = tune_lstm_model(X_train_pre_scaled, y_train_pre_scaled,
                                                      X_val_pre_scaled, y_val_pre_scaled,
                                                      input_dim=input_dim,
                                                      max_trials=10, epochs=100)
        print("Pretraining completato.")

        # Salva modello base e scaler se desiderato
        base_model.save(os.path.join(save_dir, f'modello_lstm_base_{target_ticker}.h5'))
        with open(os.path.join(save_dir, f'scalerX_pre_{target_ticker}.pkl'), 'wb') as f:
            pickle.dump(scalerX_pre, f)
        with open(os.path.join(save_dir, f'scalerY_pre_{target_ticker}.pkl'), 'wb') as f:
            pickle.dump(scalerY_pre, f)

        # --- Fine Tuning sul ticker target ---
        print(f"\n=== Fine Tuning sul ticker TARGET: {target_ticker} ===")
        data_raw_target = data_dict[target_ticker].copy()
        data_feat_target = add_features(data_raw_target)
        for ext_ticker, ext_df in external_data.items():
            new_col = ext_ticker.replace('^', '').replace('=F', '').lower() + "_log_return"
            ext_feature = ext_df[['log_return']].rename(columns={'log_return': new_col})
            data_feat_target = data_feat_target.merge(ext_feature, left_index=True, right_index=True, how='left')
        data_feat_target.fillna(method='ffill', inplace=True)
        data_feat_target.dropna(inplace=True)
        data_feat_target = rolling_garch_forecast(data_feat_target, step=10, use_expanding=True,
                                                  p_values=[1,2], q_values=[1,2],
                                                  vol_types=['Garch','EGARCH','GJR','TARCH'],
                                                  dists=['t','skewt','ged','normal'])
        # Se rimangono NaN, puoi decidere di sostituirli con l'ultimo forecast valido
        fallback_mask = data_feat_target['garch_vol_forecast'].isna()
        if fallback_mask.sum() > 0:
            # Lascia i NaN o sostituisci con un valore neutro, ad esempio:
            data_feat_target.loc[fallback_mask, 'garch_vol_forecast'] = data_feat_target['garch_vol_forecast'].ffill()

        data_feat_target.dropna(subset=feature_cols + [target_col], inplace=True)

        X_target, y_target, dates_target = prepare_lstm_data_multivariate(data_feat_target, feature_cols, target_col, look_back=look_back)
        X_train_target, y_train_target, X_test_target, y_test_target, train_dates_target, test_dates_target = time_series_train_test_split(X_target, y_target, dates_target, test_years=5)
        train_size_target = int(0.8 * len(X_train_target))
        X_train_target_, X_val_target = X_train_target[:train_size_target], X_train_target[train_size_target:]
        y_train_target_, y_val_target = y_train_target[:train_size_target], y_train_target[train_size_target:]
        print(f"Target - Train: {X_train_target_.shape[0]}, Val: {X_val_target.shape[0]}")

        # Standardizzazione dei dati (target)
        n_features_target = X_train_target_.shape[2]
        scalerX_target = StandardScaler()
        X_train_target_flat = X_train_target_.reshape(-1, n_features_target)
        X_train_target_scaled_flat = scalerX_target.fit_transform(X_train_target_flat)
        X_train_target_scaled = X_train_target_scaled_flat.reshape(X_train_target_.shape)

        X_val_target_flat = X_val_target.reshape(-1, n_features_target)
        X_val_target_scaled_flat = scalerX_target.transform(X_val_target_flat)
        X_val_target_scaled = X_val_target_scaled_flat.reshape(X_val_target.shape)

        X_test_target_flat = X_test_target.reshape(-1, n_features_target)
        X_test_target_scaled_flat = scalerX_target.transform(X_test_target_flat)
        X_test_target_scaled = X_test_target_scaled_flat.reshape(X_test_target.shape)

        scalerY_target = StandardScaler()
        y_train_target_reshaped = np.array(y_train_target_).reshape(-1, 1)
        y_train_target_scaled = scalerY_target.fit_transform(y_train_target_reshaped).ravel()
        y_val_target_scaled = scalerY_target.transform(np.array(y_val_target).reshape(-1, 1)).ravel()
        y_test_target_scaled = scalerY_target.transform(np.array(y_test_target).reshape(-1, 1)).ravel()

        # Clona il modello di base per il fine tuning
        model_target = clone_model(base_model)
        model_target.set_weights(base_model.get_weights())

        print("Inizio il fine tuning sul ticker target...")
        model_target, history_target = fine_tune_model(model_target, X_train_target_scaled, y_train_target_scaled,
                                                       X_val_target_scaled, y_val_target_scaled, epochs=100)
        test_loss = model_target.evaluate(X_test_target_scaled, y_test_target_scaled, verbose=0)
        print(f"[{target_ticker}] Test Loss: {test_loss[0]:.6f}")

        # Predizioni e inversione dello scaling
        y_pred_scaled = model_target.predict(X_test_target_scaled)

        # Protezione contro overflow/NaN prima di invertire lo scaling
        max_logval = 15.0    # evitiamo np.exp(>15)
        min_logval = -10.0   # evitiamo expm1(numeri negativi estremi)
        offset = 1e-6        # per evitare divisioni per zero

        # Inversione scaling + clipping
        y_pred_log = scalerY_target.inverse_transform(y_pred_scaled)
        y_pred_log = np.clip(y_pred_log, a_min=min_logval, a_max=max_logval)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, a_min=offset, a_max=1e6)

        # Idem per la volatilità reale
        y_test_log = scalerY_target.inverse_transform(y_test_target_scaled.reshape(-1, 1))
        y_test_log = np.clip(y_test_log, a_min=min_logval, a_max=max_logval)
        real_vol_test = np.expm1(y_test_log)
        real_vol_test = np.clip(real_vol_test, a_min=offset, a_max=1e6)
        # Sanity check: assicura che non ci siano inf, -inf o NaN nei dati
        assert np.all(np.isfinite(real_vol_test)), "[ERRORE] real_vol_test contiene inf o NaN"
        assert np.all(np.isfinite(y_pred)), "[ERRORE] y_pred contiene inf o NaN"



        lstm_rmse = np.sqrt(mean_squared_error(real_vol_test, y_pred))
        lstm_mae = mean_absolute_error(real_vol_test, y_pred)
        lstm_pearson_corr = np.corrcoef(real_vol_test.ravel().astype(float), y_pred.ravel().astype(float))[0, 1]
        lstm_spearman_corr = spearmanr(real_vol_test.ravel().astype(float), y_pred.ravel().astype(float))[0]
        lstm_qlike = np.mean((np.square(real_vol_test.flatten())/np.square(y_pred.flatten())) - np.log(np.square(y_pred.flatten())))
        metrics_dict_lstm = {
            "Ticker": target_ticker,
            "Model": "GARCH-LSTM (Transfer Learning)",
            "RMSE": lstm_rmse,
            "MAE": lstm_mae,
            "Pearson": lstm_pearson_corr,
            "Spearman": lstm_spearman_corr,
            "QLIKE": lstm_qlike
        }
        print(f"[{target_ticker}] Metriche GARCH-LSTM: {metrics_dict_lstm}")
        ticker_metrics = metrics_dict_lstm.copy()

        # Salva i risultati, la history e il modello
        results_df = pd.DataFrame({
            'Date': test_dates_target,
            'Volatilità Reale': real_vol_test.flatten(),
            'Volatilità Predetta': y_pred.flatten()
        })
        results_csv_path = os.path.join(save_dir, f"risultati_forecasting_{target_ticker}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"[{target_ticker}] Risultati salvati in: {results_csv_path}")

        history_csv_path = os.path.join(save_dir, f"training_history_{target_ticker}.csv")
        pd.DataFrame(history_target.history).to_csv(history_csv_path, index=False)
        model_save_path = os.path.join(save_dir, f"modello_lstm_{target_ticker}.h5")
        model_target.save(model_save_path.replace('.h5', '.keras'))
        print(f"[{target_ticker}] Modello LSTM salvato in: {model_save_path}")

        # --- Calcolo metriche e confronto grafico con Solo GARCH ---
        # --- Calcolo metriche e confronto grafico con Solo GARCH ---
        # Allinea GARCH con le date del test
        garch_only_test = data_feat_target['garch_vol_forecast'].reindex(test_dates_target)

        # Rimuovi eventuali NaN dovuti al reindex
        valid_mask = ~garch_only_test.isna()
        garch_pred = garch_only_test[valid_mask].values.flatten()
        real_vol_aligned = real_vol_test[valid_mask]
        y_pred_aligned = y_pred[valid_mask]

        # Salva anche i test_dates allineati per i grafici
        test_dates_aligned = test_dates_target[valid_mask]

        if len(garch_pred) == len(real_vol_aligned) == len(y_pred_aligned):
            garch_rmse = np.sqrt(mean_squared_error(real_vol_aligned, garch_pred))
            garch_mae = np.mean(np.abs(real_vol_aligned - garch_pred))
            garch_pearson_corr = np.corrcoef(real_vol_aligned.ravel().astype(float), garch_pred.ravel().astype(float))[0, 1]
            garch_spearman_corr = spearmanr(real_vol_aligned.ravel().astype(float), garch_pred.ravel().astype(float))[0]
            garch_qlike = np.mean(((real_vol_aligned + offset)**2 / (garch_pred + offset)**2) -
                                  np.log((garch_pred + offset)**2))

            metrics_dict_garch = {
                "Ticker": target_ticker,
                "Model": "Solo GARCH",
                "RMSE": garch_rmse,
                "MAE": garch_mae,
                "Pearson": garch_pearson_corr,
                "Spearman": garch_spearman_corr,
                "QLIKE": garch_qlike
            }
            print(f"[{target_ticker}] Metriche SOLO GARCH: {metrics_dict_garch}")
        else:
            print("ATTENZIONE: Dimensioni diverse tra GARCH e test set.")
        mask_valid_ensemble = np.isfinite(real_vol_aligned) & np.isfinite(y_pred_aligned) & np.isfinite(garch_pred)
        real_vol_aligned = real_vol_aligned[mask_valid_ensemble]
        y_pred_aligned = y_pred_aligned[mask_valid_ensemble]
        garch_pred = garch_pred[mask_valid_ensemble]
        # ➕ Ensemble: stesso allineamento
        metrics_ensemble = evaluate_ensemble(real_vol_aligned, y_pred_aligned, garch_pred)
        print(f"[{target_ticker}] Metriche ENSEMBLE: {metrics_ensemble}")

        ticker_metrics.update({f"ENS_{k}": v for k, v in metrics_ensemble.items() if k != "Model"})
        all_metrics_global.append(ticker_metrics)

        # Plot comparativo: volatilità reale, GARCH-LSTM e Solo GARCH
        try:
            plt.figure(figsize=(12,6))
            plt.plot(test_dates_aligned, real_vol_aligned, label='Volatilità Reale')
            plt.plot(test_dates_aligned, y_pred_aligned, label='GARCH-LSTM Predetta')
            plt.plot(test_dates_aligned, garch_pred, label='Solo GARCH Forecast')
            plt.title(f"Forecasting della Volatilità per {target_ticker} (ultimi 5 anni)")
            plt.xlabel("Date")
            plt.ylabel("Volatilità")
            plt.legend()
            plot_path = os.path.join(save_dir, f"Reale_vs_Pred_{target_ticker}.png")
            plt.savefig(plot_path)
            plt.show()
            plt.close()
        except Exception as e:
            print(f"[{target_ticker}] Errore durante il salvataggio del grafico: {e}")

        print(f"=== Fine iterazione per il ticker {target_ticker} completata ===\n\n")
        if target_ticker == "AAPL":
            explain_lstm_with_shap(model_target, X_test_target_scaled, feature_cols,
                                  save_path=os.path.join(save_dir, "SHAP_AAPL.png"))

    # Salva le metriche complessive in un file CSV, una volta fuori dal ciclo
    metrics_df = pd.DataFrame(all_metrics_global)
    metrics_csv_path = os.path.join(save_dir, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metriche complessive salvate in: {metrics_csv_path}")
    print("\n=== METRICHE FINALI (TUTTI I TICKER) ===")
    print(metrics_df)



##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    # Imposta il seme per la riproducibilità
    np.random.seed(42)
    tf.random.set_seed(42)

    #test su 3 ticker
    tickers=['XLF','AAPL', 'XLY']

    # Lista di tickers per la pipeline leave-one-out
    #tickers = ['AAPL', 'NVDA', 'NFLX', 'AMZN', 'XOM', 'WMT', 'NKE', 'XLE', 'XLI', 'XLK', 'XLV', 'XLY', 'XLF', 'ITA']
    run_leave_one_out_pipeline(tickers, start_date='01/01/2007', end_date='23/12/2024', save_dir='/content/drive/MyDrive/Prova_Garch5')
