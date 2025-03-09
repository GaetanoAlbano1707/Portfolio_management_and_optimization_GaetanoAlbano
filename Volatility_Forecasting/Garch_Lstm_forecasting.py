import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# GARCH
from arch import arch_model

# LSTM & Deep Learning con TensorFlow Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber  # Huber Loss per robustezza agli outlier

# Scaling e Metriche
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Hyperparameter tuning con KerasTuner
import keras_tuner as kt

##########################################
# Callback per fermare il training se compare NaN
##########################################
class CheckNaNCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if np.isnan(logs.get('loss', 0)) or np.isnan(logs.get('val_loss', 0)):
            print("NaN loss rilevato, interrompo il trial.")
            self.model.stop_training = True
            raise ValueError("NaN loss encountered.")
            # Non lanciamo l'eccezione; il trial terminerà e il tuner registrerà il risultato

##########################################
# 1. Caricamento dei dati e preprocessing
##########################################
def load_data(tickers, start_date='01/01/2007', end_date='23/12/2024',
              save_dir='./Volatility_forecasting/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = pd.to_datetime(end_date, format='%d/%m/%Y')
    data_frames = {}
    for ticker in tickers:
        try:
            print(f"Elaborazione ticker: {ticker}")
            data = yf.download(ticker, period='max', auto_adjust=False)
            data = data.sort_index().ffill().bfill()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns]
            data.columns = [col.replace(f" {ticker}", "") for col in data.columns]
            print(f"Colonne disponibili per {ticker}: {data.columns}")
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
            data = data[data['Volume'] > 0]
            data['return'] = data['Adj Close'].pct_change() * 100  # percentuale
            data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            data_frames[ticker] = data
            filename = os.path.join(save_dir, f"{ticker}_data.csv")
            data.to_csv(filename, float_format='%.3f', index=True)
            print(f"Dati salvati in: {filename}")
        except Exception as e:
            print(f"Errore per il ticker {ticker}: {e}")
    return data_frames

##########################################
# 2. Feature Engineering: aggiunta di feature extra
##########################################
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, short_window=12, long_window=26, signal_window=9):
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return sma, upper_band, lower_band

def compute_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(k_window).min()
    high_max = data['High'].rolling(k_window).max()
    stoch_k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(d_window).mean()
    return stoch_k, stoch_d

def add_features(data):
    if 'log_return' not in data.columns:
        data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['gk_vol'] = np.sqrt(
        0.5 * (np.log(data['High'] / data['Low'])) ** 2 -
        (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open'])) ** 2
    )
    data['rolling_vol'] = data['log_return'].rolling(window=10).std()
    data['realized_vol'] = data['log_return'].rolling(window=5).std()
    data['log_realized_vol'] = np.log(data['realized_vol'] + 1e-6)
    data['target_vol'] = data['log_realized_vol'].shift(-1)
    data['lag_log_return'] = data['log_return'].shift(1)
    data['lag_realized'] = data['realized_vol'].shift(1)
    data['RSI'] = compute_RSI(data['Adj Close'])
    data['SMA'] = data['Adj Close'].rolling(window=14).mean()
    sma_20, bb_up, bb_down = compute_bollinger_bands(data['Adj Close'], window=20, num_std=2)
    data['BB_mid'] = sma_20
    data['BB_up'] = bb_up
    data['BB_down'] = bb_down
    macd_line, signal_line, hist = compute_MACD(data['Adj Close'])
    data['MACD_line'] = macd_line
    data['MACD_signal'] = signal_line
    data['MACD_hist'] = hist
    stoch_k, stoch_d = compute_stochastic_oscillator(data)
    data['Stoch_K'] = stoch_k
    data['Stoch_D'] = stoch_d
    data.dropna(inplace=True)
    return data

##########################################
# 3. Ottimizzazione del modello GARCH e forecasting a blocchi (ricorsivo)
##########################################
def grid_search_garch(sample_data, p_values, q_values, dists, model_type, criterion='aic'):
    best_crit_value = np.inf
    best_params = None
    best_dist = None
    best_o = 0
    if model_type == 'GJR':
        vol_type = 'Garch'
        o_param = 1
    elif model_type == 'TARCH':
        vol_type = 'aparch'
        o_param = 0
    else:
        vol_type = model_type
        o_param = 0
    for p in p_values:
        for q in q_values:
            for dist in dists:
                try:
                    if o_param > 0:
                        am = arch_model(sample_data, vol=vol_type, p=p, o=o_param, q=q, dist=dist)
                    elif model_type == 'TARCH':
                        am = arch_model(sample_data, vol='aparch', p=p, q=q, dist=dist, power=1.0)
                    else:
                        am = arch_model(sample_data, vol=vol_type, p=p, q=q, dist=dist)
                    res = am.fit(update_freq=0, disp='off')
                    crit_value = res.aic if criterion == 'aic' else res.bic
                    if crit_value < best_crit_value:
                        best_crit_value = crit_value
                        best_params = (p, q)
                        best_dist = dist
                        best_o = o_param
                except Exception:
                    continue
    return best_params, best_dist, best_crit_value, best_o

def optimize_garch_models(data, p_values=[1,2,3], q_values=[1,2,3], dists=['t','skewt','ged'],
                          windows=[250,500,750,1000,1500], criterion='aic', ticker=None):
    log_ret = data['log_return'].dropna() * 100
    best_overall = np.inf
    best_model_type = None
    best_model_params = None
    best_model_dist = None
    best_window = None
    model_list = ['Garch','EGARCH','GJR','TARCH']
    for w in windows:
        sample_data = log_ret.iloc[-w:]
        for m in model_list:
            params, dist, crit_val, _ = grid_search_garch(sample_data, p_values, q_values, dists, m, criterion)
            print(f"Modello {m}: best_params={params}, dist={dist}, {criterion.upper()}={crit_val:.4f}")
            if params is not None and crit_val < best_overall:
                best_overall = crit_val
                best_model_type = m
                best_model_params = params
                best_model_dist = dist
                best_window = w
    print(f"Selezionato {best_model_type} con p={best_model_params[0]}, q={best_model_params[1]}, "
          f"dist={best_model_dist}, window={best_window} e {criterion.upper()}={best_overall:.4f}")

    # Forecasting ricorsivo in blocchi di 10 giorni (1-step alla volta)
    forecast_vol = pd.Series(index=log_ret.index, dtype=float)
    step = 10
    n = len(log_ret)
    pos = best_window  # partiamo dopo la finestra ottimale
    while pos < n:
        block_size = min(step, n - pos)
        block_forecasts = []
        for j in range(block_size):
            current_window = log_ret.iloc[pos - best_window: pos]
            try:
                if best_model_type == 'GJR':
                    am = arch_model(current_window, vol='Garch', p=best_model_params[0], o=1, q=best_model_params[1],
                                    dist=best_model_dist)
                elif best_model_type == 'TARCH':
                    am = arch_model(current_window, vol='aparch', p=best_model_params[0], q=best_model_params[1],
                                    dist=best_model_dist, power=1.0)
                else:
                    am = arch_model(current_window, vol=best_model_type, p=best_model_params[0], q=best_model_params[1],
                                    dist=best_model_dist)
                res = am.fit(update_freq=0, disp='off')
                fc = res.forecast(horizon=1)
                forecast_value = np.sqrt(fc.variance.iloc[-1, 0])
            except Exception as e:
                print(f"Errore nel forecasting a partire da indice {pos}: {e}")
                forecast_value = np.nan
            block_forecasts.append(forecast_value)
            pos += 1
        indices_block = log_ret.index[pos - block_size: pos]
        forecast_vol.loc[indices_block] = block_forecasts

    # Se ci sono NaN, applichiamo un forward-fill
    forecast_vol = forecast_vol.fillna(method='ffill')
    # Clipping per evitare valori estremi nella feature forecast
    forecast_vol = forecast_vol.clip(lower=0, upper=np.percentile(forecast_vol.dropna(), 99))

    data = data.copy()
    data['garch_vol_forecast'] = np.nan
    common_idx = data.index.intersection(forecast_vol.index)
    data.loc[common_idx, 'garch_vol_forecast'] = forecast_vol.loc[common_idx]
    data.dropna(inplace=True)
    return data

##########################################
# 4. Preparazione dei dati per il modello LSTM
##########################################
def prepare_lstm_data_multivariate(data, feature_cols, target_col, look_back=10):
    df = data.copy().dropna()
    X, y, dates = [], [], []
    features = df[feature_cols].values
    target = df[target_col].values
    for i in range(look_back, len(df)):
        X.append(features[i-look_back:i])
        y.append(target[i])
        dates.append(df.index[i])
    return np.array(X), np.array(y), np.array(dates)

def split_data_by_date(X, y, dates, test_period_years=5):
    last_date = pd.to_datetime(dates.max())
    test_start_date = last_date - pd.DateOffset(years=test_period_years)
    train_mask = dates < test_start_date
    test_mask = dates >= test_start_date
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask], dates[train_mask], dates[test_mask]

def scale_data(X_train, X_val, X_test):
    ns, ts, nf = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, nf)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(ns, ts, nf)

    ns_val = X_val.shape[0]
    X_val_flat = X_val.reshape(-1, nf)
    X_val_scaled = scaler.transform(X_val_flat).reshape(ns_val, ts, nf)

    ns_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, nf)
    X_test_scaled = scaler.transform(X_test_flat).reshape(ns_test, ts, nf)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# Funzione di default per hyperparameters (fallback)
def default_hyperparameters(num_features, look_back=10):
    hp = kt.HyperParameters()
    hp.Fixed('look_back', look_back)
    hp.Fixed('num_features', num_features)
    hp.Int('units_1', 32, 128, step=32, default=64)
    hp.Float('dropout_1', 0.2, 0.4, step=0.1, default=0.3)
    hp.Int('units_2', 32, 128, step=32, default=64)
    hp.Float('dropout_2', 0.1, 0.3, step=0.1, default=0.1)
    hp.Boolean('use_third_layer', default=False)
    hp.Float('learning_rate', 1e-4, 3e-4, sampling='log', default=1.5e-4)
    return hp

##########################################
# 5. Costruzione del modello LSTM
##########################################
# Modifica dei model builder per ridurre instabilità numerica

def advanced_model_builder(hp, look_back=10, num_features=15):
    reg = l2(1e-4)
    units_1 = hp.Int('units_1', 32, 256, step=32, default=64)
    dropout_1 = hp.Float('dropout_1', 0.1, 0.5, step=0.1, default=0.3)
    units_2 = hp.Int('units_2', 32, 256, step=32, default=64)
    dropout_2 = hp.Float('dropout_2', 0.1, 0.5, step=0.1, default=0.1)
    use_third_layer = hp.Boolean('use_third_layer', default=False)
    if use_third_layer:
        units_3 = hp.Int('units_3', 32, 256, step=32, default=64)
        dropout_3 = hp.Float('dropout_3', 0.1, 0.5, step=0.1, default=0.1)
    learning_rate = hp.Float('learning_rate', 1e-5, 3e-4, sampling='log', default=1e-4)
    model = Sequential()
    model.add(LSTM(units_1, return_sequences=True, input_shape=(look_back, num_features), kernel_regularizer=reg))
    model.add(Dropout(dropout_1))
    model.add(LSTM(units_2, return_sequences=use_third_layer, kernel_regularizer=reg))
    model.add(Dropout(dropout_2))
    if use_third_layer:
        model.add(LSTM(units_3, kernel_regularizer=reg))
        model.add(Dropout(dropout_3))
    model.add(Dense(1, kernel_regularizer=reg))
    opt = Adam(learning_rate=learning_rate, clipvalue=0.5)
    model.compile(optimizer=opt, loss=Huber())
    return model

def advanced_model_builder_fallback(hp, look_back=10, num_features=15):
    reg = l2(1e-4)
    units_1 = hp.Int('units_1', 32, 128, step=32, default=64)
    dropout_1 = hp.Float('dropout_1', 0.1, 0.3, step=0.1, default=0.2)
    units_2 = hp.Int('units_2', 32, 128, step=32, default=64)
    dropout_2 = hp.Float('dropout_2', 0.1, 0.2, step=0.1, default=0.1)
    learning_rate = 1e-4
    model = Sequential()
    model.add(LSTM(units_1, return_sequences=True, input_shape=(look_back, num_features), kernel_regularizer=reg))
    model.add(Dropout(dropout_1))
    model.add(LSTM(units_2, kernel_regularizer=reg))
    model.add(Dropout(dropout_2))
    model.add(Dense(1, kernel_regularizer=reg))
    opt = Adam(learning_rate=learning_rate, clipvalue=0.5)
    model.compile(optimizer=opt, loss=Huber())
    return model

# Alias per il builder standard
advanced_model_builder_standard = advanced_model_builder

##########################################
# Funzioni di training: standard e fallback
##########################################
def train_with_tuner(ticker, X_train, y_train, X_val, y_val, conservative=False, max_epochs=40):
    check_nan_cb = CheckNaNCallback()
    if conservative:
        builder = advanced_model_builder_fallback
        dir_name = f"./kt_dir_fallback"
        proj_name = f"fallback_{ticker}"
    else:
        builder = advanced_model_builder_standard
        dir_name = f"./kt_dir_standard"
        proj_name = f"standard_{ticker}"
    try:
        tuner = kt.Hyperband(
            lambda hp: builder(hp, look_back=X_train.shape[1], num_features=X_train.shape[2]),
            objective='val_loss',
            max_epochs=max_epochs,
            factor=3,
            directory=dir_name,
            project_name=proj_name,
            max_consecutive_failed_trials=20
        )
        tuner.search(
            X_train, y_train,
            epochs=max_epochs,
            validation_data=(X_val, y_val),
            callbacks=[check_nan_cb],
            verbose=1
        )
        best_hp_list = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hp_list:
            print(f"[{ticker}] Nessun iperparametro valido trovato, uso default.")
            return default_hyperparameters(X_train.shape[2], X_train.shape[1])
        return best_hp_list[0]
    except Exception as e:
        print(f"[{ticker}] Tuner search fallito con errore: {e}. Uso default hyperparameters.")
        return default_hyperparameters(X_train.shape[2], X_train.shape[1])

def train_lstm_for_ticker(ticker, X_train, y_train, X_val, y_val, max_epochs=40):
    try:
        print(f"[{ticker}] Provo pipeline standard.")
        best_hp = train_with_tuner(ticker, X_train, y_train, X_val, y_val, conservative=False, max_epochs=max_epochs)
        print(f"[{ticker}] Pipeline standard completata. Parametri: {best_hp.values}")
        used_fallback = False
    except Exception as e:
        print(f"[{ticker}] Pipeline standard fallita: {e}")
        print(f"[{ticker}] Uso pipeline fallback.")
        best_hp = train_with_tuner(ticker, X_train, y_train, X_val, y_val, conservative=True, max_epochs=max_epochs)
        print(f"[{ticker}] Pipeline fallback completata. Parametri: {best_hp.values}")
        used_fallback = True
    return best_hp, used_fallback

##########################################
# MAIN
##########################################
if __name__ == "__main__":
    np.random.seed(42)
    tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI', 'NVDA', 'LMT', 'WMT', 'XOM', 'NKE', 'AMZN', 'NFLX', 'AAPL']
    data_dict = load_data(tickers)
    for ticker in tickers:
        print(f"\n=== Processing {ticker} ===")
        data = data_dict[ticker].copy()
        data = add_features(data)
        # Applichiamo il modello GARCH per ottenere le previsioni in blocchi (ricorsivo 1-step)
        data = optimize_garch_models(data, ticker=ticker)

        feature_cols = [
            'garch_vol_forecast', 'gk_vol', 'rolling_vol', 'log_return',
            'lag_log_return', 'RSI', 'SMA', 'BB_mid', 'BB_up', 'BB_down',
            'MACD_line', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D'
        ]
        target_col = 'target_vol'
        look_back = 10

        X, y, dates = prepare_lstm_data_multivariate(data, feature_cols, target_col, look_back=look_back)
        X_train, y_train, X_test, y_test, train_dates, test_dates = split_data_by_date(X, y, dates, test_period_years=5)

        train_split = int(0.8 * len(X_train))
        X_train_final, y_train_final = X_train[:train_split], y_train[:train_split]
        X_val, y_val = X_train[train_split:], y_train[train_split:]

        X_train_scaled, X_val_scaled, X_test_scaled, scalerX = scale_data(X_train_final, X_val, X_test)

        scalerY = StandardScaler()
        y_train_scaled = scalerY.fit_transform(y_train_final.reshape(-1,1)).ravel()
        y_val_scaled = scalerY.transform(y_val.reshape(-1,1)).ravel()
        y_test_scaled = scalerY.transform(y_test.reshape(-1,1)).ravel()
        y_train_scaled = np.clip(y_train_scaled, -3, 3)
        y_val_scaled = np.clip(y_val_scaled, -3, 3)
        y_test_scaled = np.clip(y_test_scaled, -3, 3)

        # Verifica eventuali NaN nei dati scalati
        print("X_train_scaled NaNs:", np.isnan(X_train_scaled).sum())
        print("y_train_scaled NaNs:", np.isnan(y_train_scaled).sum())

        num_features = X_train_scaled.shape[2]

        best_hp, used_fallback = train_lstm_for_ticker(ticker, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, max_epochs=40)
        best_hp_dict = best_hp.values
        print("Best Hyperparameters:", best_hp_dict)

        if used_fallback:
            final_model = advanced_model_builder_fallback(best_hp, look_back, num_features)
        else:
            final_model = advanced_model_builder_standard(best_hp, look_back, num_features)

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        nan_cb = CheckNaNCallback()

        try:
            history = final_model.fit(
                X_train_scaled, y_train_scaled,
                epochs=best_hp_dict.get('tuner/epochs', 40),
                batch_size=16,
                validation_data=(X_val_scaled, y_val_scaled),
                callbacks=[es, rlrop, nan_cb],
                verbose=1
            )
        except ValueError as e:
            print(f"[{ticker}] Training finale fallito con NaN loss: {e}")
            print(f"[{ticker}] Riparto in modalità fallback con training semplificato.")
            final_model = advanced_model_builder_fallback(best_hp, look_back, num_features)
            # Riparto con un numero minore di epoche per stabilizzare
            history = final_model.fit(
                X_train_scaled, y_train_scaled,
                epochs=20,
                batch_size=16,
                validation_data=(X_val_scaled, y_val_scaled),
                callbacks=[es, rlrop],
                verbose=1
            )

        test_loss = final_model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        print(f"[{ticker}] Test Loss (MSE in log-scale): {test_loss:.4f}")

        y_pred_scaled = final_model.predict(X_test_scaled)
        y_pred_log = scalerY.inverse_transform(y_pred_scaled)
        y_pred = np.exp(y_pred_log) - 1e-6

        y_test_log = scalerY.inverse_transform(y_test_scaled.reshape(-1,1))
        real_vol_test = np.exp(y_test_log) - 1e-6

        rmse = np.sqrt(mean_squared_error(real_vol_test, y_pred))
        mae = mean_absolute_error(real_vol_test, y_pred)
        print(f"[{ticker}] RMSE (scala reale): {rmse:.6f}, MAE (scala reale): {mae:.6f}")

        results_df = pd.DataFrame({
            'Date': test_dates,
            'Volatilità Reale (exp)': real_vol_test.flatten(),
            'Volatilità Predetta (exp)': y_pred.flatten()
        })
        results_csv_path = os.path.join('./Volatility_forecasting/', f"risultati_forecasting_{ticker}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"[{ticker}] Risultati salvati in: {results_csv_path}")

        history_csv_path = os.path.join('/Volatility_forecasting/', f"training_history_{ticker}.csv")
        pd.DataFrame(history.history).to_csv(history_csv_path, index=False)
        model_save_path = os.path.join('/Volatility_forecasting/', f"modello_lstm_{ticker}.h5")
        final_model.save(model_save_path)
        print(f"[{ticker}] Modello LSTM salvato in: {model_save_path}")

        plt.figure(figsize=(12, 6))
        plt.plot(real_vol_test, label='Volatilità Reale (exp)')
        plt.plot(y_pred, label='Volatilità Predetta (exp)')
        plt.title(f"Forecasting della Volatilità per {ticker} (ultimi 5 anni) - fallback={used_fallback}")
        plt.xlabel("Campioni")
        plt.ylabel("Volatilità")
        plt.legend()
        plot_path = os.path.join('/Volatility_forecasting/', f"Reale_vs_Pred_{ticker}.png")
        plt.savefig(plot_path)
        plt.show()
