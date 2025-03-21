import numpy as np
import gym
from gym import spaces
from transaction_costs import calculate_mu, get_covariance_matrix, optimize_with_transaction_costs # importiamo la nostra funzione di ottimizzazione
import scipy.special
import pandas as pd
from collections import deque

class TickersPortfolioEnv(gym.Env):
    def __init__(self, config, data, forecast_data, log_returns_df, mode='train'):
        super(TickersPortfolioEnv, self).__init__()
        self.config = config
        self.mode = mode
        self.data = data.reset_index(drop=True)
        print("📊 DEBUG: Prime date disponibili nel dataset:")
        print(self.data[['Date']].head(10))
        self.forecast_data = forecast_data.reset_index(drop=True)
        self.log_returns_df = log_returns_df
        self.current_day = 0
        self.stock_num = len(self.config.tickers)
        self.capital = config.initial_asset
        self.capital_hist = deque(maxlen=100)
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.performance_log = []

        # **💡 FISSA IL PROBLEMA QUI**
        try:
            self.current_prices = self.data.loc[self.current_day, [f"{ticker}_AdjClose" for ticker in self.config.tickers]].values.astype(float)
        except KeyError as e:
            raise KeyError(f"❌ Errore: Le colonne dei prezzi non sono presenti nel dataset. Dettaglio errore: {e}")

        print(f"🔍 DEBUG: self.current_prices inizializzato con {self.current_prices.shape} valori (dovrebbe essere 6)")
        print(f"📊 DEBUG: Prezzi iniziali assegnati: {self.current_prices}")
        # Spazio degli stati e delle azioni
        self.expected_obs_size = 100 + self.stock_num + self.stock_num * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.expected_obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_num,), dtype=np.float32)

    def weights_normalization(self, action):
        if np.sum(action) == 0:
            print("⚠️ Attenzione: Azione con tutti zeri, assegno una distribuzione uniforme!")
            return np.ones(self.stock_num) / self.stock_num
        return np.clip(action, 0, 1) / np.sum(action)

    def get_next_prices(self):
        next_day = self.current_day + 1
        if next_day >= len(self.data):
            print("⚠️ Fine dei dati: manteniamo i prezzi dell'ultimo giorno.")
            return self.current_prices  # Non aggiorniamo nulla se siamo all'ultimo giorno.

        new_prices = self.data.loc[next_day, [f"{ticker}_AdjClose" for ticker in self.config.tickers]].values.astype(
            float)

        print(f"🔄 DEBUG: Giorno attuale {self.current_day} → Giorno successivo {next_day}")
        print(f"📊 DEBUG: Prezzi correnti: {self.current_prices}")
        print(f"📊 DEBUG: Prezzi futuri: {new_prices}")
        print(f"📊 DEBUG: Differenza (futuri - correnti): {new_prices - self.current_prices}")

        return new_prices  # NON aggiornare self.current_prices qui

    def seed(self, seed=None):
        np.random.seed(seed)

    def _get_base_state(self):
        last_n_days = 5  # Può essere regolato
        start_idx = max(0, self.current_day - last_n_days)

        vol_forecast_series = self.forecast_data.iloc[start_idx:self.current_day].filter(
            like='_Vol_Pred').mean(axis=0)
        pred_return_series = self.forecast_data.iloc[start_idx:self.current_day].filter(
            like='_pred_return').mean(axis=0)

        # Creiamo un vettore di lunghezza 100 riempiendolo con dati o 0 se mancano
        base_state = np.zeros(100)
        base_values = np.concatenate([vol_forecast_series.values, pred_return_series.values])

        # Inseriamo i dati reali nel base_state (fino a 100 valori)
        base_state[:len(base_values)] = base_values

        print(f"🔍 DEBUG: Nuova dimensione base_state={base_state.shape[0]}")

        return base_state

    def _get_forecast_features(self):
        current_date = self.data.loc[self.current_day, 'Date']
        row = self.forecast_data[self.forecast_data['Date'] == current_date]
        if not row.empty:
            vol_forecast = np.array([row.iloc[0].get(f"{ticker}_Vol_Pred", 0) for ticker in self.config.tickers])
        else:
            print(f"⚠️ ATTENZIONE: Previsioni di volatilità mancanti per la data {current_date}, riempite con 0.")
            vol_forecast = np.zeros(self.stock_num)
        window_size = 10
        start_idx = max(0, self.current_day - window_size)
        recent_data = self.data.iloc[start_idx:self.current_day]
        expected_returns = []
        for ticker in self.config.tickers:
            col_name = f"{ticker}_log_return"
            if col_name in recent_data.columns:
                mu_ticker = recent_data[col_name].mean()
            else:
                mu_ticker = 0.0
            expected_returns.append(mu_ticker)
        expected_returns = np.array(expected_returns)
        print(
            f"🔍 DEBUG: vol_forecast dimensione = {vol_forecast.shape}, expected_returns dimensione = {expected_returns.shape}")
        return vol_forecast, expected_returns

    def _get_state(self):
        base_state = self._get_base_state()
        alloc_state = self.current_allocation
        vol_forecast, pred_return = self._get_forecast_features()
        capital_scaled = np.array([self.capital / self.config.initial_asset])

        state_parts = {
            "base_state": base_state.shape[0],
            "alloc_state": alloc_state.shape[0],
            "vol_forecast": vol_forecast.shape[0],
            "pred_return": pred_return.shape[0],
            "capital_scaled": capital_scaled.shape[0],
        }

        print(f"🔍 DEBUG: Componenti dello stato - {state_parts}")

        state = np.concatenate([base_state, alloc_state, vol_forecast, pred_return, capital_scaled])

        if state.shape[0] != self.expected_obs_size:
            raise ValueError(f"Dimensione stato errata! Prevista: {self.expected_obs_size}, Ottenuta: {state.shape[0]}")

        return state

    def _compute_portfolio_return(self, new_prices):
        tickers_adjclose_cols = [f"{ticker}_AdjClose" for ticker in self.config.tickers]

        new_prices_df = pd.DataFrame([new_prices], columns=tickers_adjclose_cols)

        new_prices_selected = new_prices_df[tickers_adjclose_cols].values.flatten()

        # 🔥 **Fix: Controlliamo se i nuovi prezzi sono effettivamente diversi**
        print(f"📊 DEBUG: Confronto prezzi (vecchi vs nuovi):")
        print(pd.DataFrame({'Vecchi': self.current_prices, 'Nuovi': new_prices_selected,
                            'Diff': new_prices_selected - self.current_prices}))

        if np.allclose(new_prices_selected, self.current_prices, atol=1e-10):
            print(
                f"❌ ERRORE: I nuovi prezzi sono troppo vicini ai vecchi! Differenze reali: {new_prices_selected - self.current_prices}")

        # **🔥 FIX: Ricalcoliamo bene i ritorni**
        returns = new_prices_selected / self.current_prices - 1
        if np.all(returns == 0):
            print("❌ ERRORE: Tutti i ritorni sono zero! Controlla il dataset.")

        portfolio_return = np.dot(self.current_allocation, returns)

        print(f"📊 DEBUG: Ritorni calcolati: {returns}")
        print(f"📊 DEBUG: Allocazione attuale: {self.current_allocation}")
        print(f"📊 DEBUG: Rendimento del portafoglio: {portfolio_return}")

        return portfolio_return

    def _check_done(self):
        if self.current_day >= len(self.data) - 1:
            print("🏁 Fine dei dati raggiunta!")
            return True

    def step(self, action):
        self.current_allocation = self.weights_normalization(action)

        new_prices = self.get_next_prices()
        day_return = self._compute_portfolio_return(new_prices)
        print(f"📉 DEBUG: Ritorni giornalieri: {day_return}")
        self.current_prices = new_prices
        self.current_day += 1
        old_capital = self.capital  # 🔍 Salva il capitale prima dell'aggiornamento
        self.capital *= (1 + day_return)
        print(f"📊 DEBUG: Capitale prima: {old_capital}, Capitale dopo: {self.capital}, Rendimento: {day_return}")

        self.capital_hist.append(self.capital)


        current_date = self.data.loc[self.current_day, 'Date']

        self.performance_log.append({
            "Date": current_date,
            "Capital": self.capital,
            "Return": day_return,
            "Allocation": self.current_allocation.copy()
        })

        if self.current_day % self.config.correlation_update_frequency == 0: #il valore del parametro è 60 perchè si intende giorni di trading (ogni 3 mesi circa=
            print(f"🔄 Aggiornamento parametri per la data {current_date}")
            new_mu = calculate_mu(self.config.tickers, self.log_returns_df, current_date, window_size=20)
            row_forecast = self.forecast_data[self.forecast_data['Date'] == current_date]

            if not row_forecast.empty:
                Sigma = get_covariance_matrix(row_forecast.iloc[0], self.log_returns_df, self.config.tickers)
            else:
                print(f"⚠️ Nessun forecast trovato per la data {current_date}, uso fallback per Σ")
                Sigma = np.eye(self.stock_num) * 0.01

            w_opt, delta_minus_opt, delta_plus_opt = optimize_with_transaction_costs(
                mu=new_mu,
                Sigma=Sigma,
                w_tilde=self.current_allocation,
                c_minus=self.config.c_minus,
                c_plus=self.config.c_plus,
                delta_minus=self.config.delta_minus,
                delta_plus=self.config.delta_plus,
                gamma=self.config.gamma
            )

            print(f"📊 DEBUG: Nuova allocazione ottimizzata: {w_opt}")
            self.current_allocation = w_opt.copy()

        reward = (self.config.lambda_profit * day_return - self.config.lambda_risk * np.std(
            list(self.capital_hist)[-5:]))
        reward = -1 if np.isnan(reward) or np.isinf(reward) else reward

        return self._get_state(), reward, self._check_done(), {"capital": self.capital, "day_return": day_return}

    def calculate_metrics(self):
        df = pd.DataFrame(self.performance_log)

        print("📊 DEBUG: Contenuto performance_log:")
        print(df.head())  # Stampa le prime righe per verificare la struttura

        if 'Date' not in df.columns:
            print("❌ ERRORE: La colonna 'Date' non è presente nel DataFrame!")
            print(f"Colonne disponibili: {df.columns}")
            return  # Evita l'errore bloccando l'esecuzione

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        returns = df["Return"]

        if returns.std() == 0:  # Evita divisione per zero
            print("⚠️ Sharpe Ratio non calcolabile: deviazione standard nulla.")
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        max_drawdown = np.min(df["Capital"]) / np.max(df["Capital"]) - 1

        last_return = self.performance_log[-1]["Return"] if self.performance_log else 0.0
        if self.current_day % 10 == 0:
            print(f"📈 MONITOR: Giorno {self.current_day}, Capitale: {self.capital:.2f}, Ritorno ultimo giorno: {last_return:.5f}")

        print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "📊 Sharpe Ratio: N/A")
        print(f"📉 Max Drawdown: {max_drawdown:.2%}")

    def reset(self):
        self.current_day = 0
        self.capital = self.config.initial_asset
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.performance_log = []

        try:
            self.current_prices = self.data.loc[self.current_day, [f"{ticker}_AdjClose" for ticker in self.config.tickers]].values.astype(float)
        except KeyError:
            raise KeyError("❌ Errore: Le colonne dei prezzi non sono presenti nel dataset.")

        return self._get_state()

