import numpy as np
import gym
from gym import spaces
from transaction_costs import calculate_mu, get_covariance_matrix, optimize_with_transaction_costs # importiamo la nostra funzione di ottimizzazione
import scipy.special



class TickersPortfolioEnv(gym.Env):
    def __init__(self, config, data, forecast_data, log_returns_df, mode='train'):
        super(TickersPortfolioEnv, self).__init__()
        self.config = config
        self.mode = mode
        self.data = data.reset_index(drop=True)
        self.forecast_data = forecast_data.reset_index(drop=True)
        self.log_returns_df = log_returns_df  # Salva il DataFrame dei log_return
        self.current_day = 0
        self.stock_num = len(self.config.tickers)
        print(f"🔍 DEBUG: Numero asset (stock_num): {self.stock_num}")
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_num,), dtype=np.float32)
        self.expected_obs_size = 100 + self.stock_num + self.stock_num * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.expected_obs_size,), dtype=np.float32)
        self.capital = config.initial_asset
        self.capital_hist = [self.capital]
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.capital = config.initial_asset
        self.current_day = 0
        self.current_prices = self.data.loc[self.current_day, [f"{ticker}_AdjClose" for ticker in self.config.tickers]].values.astype(float)


    def weights_normalization(self, action):
        normalized_action = np.clip(action, 0, 1)  # Restringe tra 0 e 1
        return normalized_action / np.sum(normalized_action)  # Normalizza a somma 1

    def get_next_prices(self):
        self.current_day += 1
        if self.current_day >= len(self.data):
            self.current_day = len(self.data) - 1
        self.current_prices = self.data.loc[self.current_day].drop('Date').values.astype(float)
        return self.current_prices

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
        # Calcola il rendimento del portafoglio in base alla variazione dei prezzi
        returns = new_prices / self.current_prices - 1
        return np.dot(self.current_allocation, returns)

    def _check_done(self):
        return self.current_day >= len(self.data) - 1

    def step(self, action):
        # Aggiorna il giorno corrente e ottieni i prezzi correnti
        current_date = self.data.loc[self.current_day, 'Date']

        # Calcola il rendimento del portafoglio per il giorno corrente
        new_prices = self.get_next_prices()  # Questo metodo aggiorna self.current_day e self.current_prices
        day_return = self._compute_portfolio_return(new_prices)

        # Aggiorna il capitale in base al rendimento del giorno
        self.capital *= (1 + day_return)
        self.capital_hist.append(self.capital)
        if len(self.capital_hist) > 100:
            self.capital_hist.pop(0)

        # Trigger: Aggiorna i parametri ogni 10 giorni
        if self.current_day % 10 == 0:
            print(f"🔄 Aggiornamento parametri per la data {current_date}")
            # Calcola μ usando i log_return storici con una finestra mobile
            new_mu = calculate_mu(self.config.tickers, self.log_returns_df, current_date, window_size=20)

            # Estrai le previsioni di volatilità per la data corrente dal forecast_data
            row_forecast = self.forecast_data[self.forecast_data['Date'] == current_date]
            if not row_forecast.empty:
                Sigma = get_covariance_matrix(row_forecast.iloc[0], self.log_returns_df, self.config.tickers)
            else:
                print(f"⚠️ Nessun forecast trovato per la data {current_date}, uso fallback per Σ")
                Sigma = np.eye(self.stock_num) * 0.01

            # Esegui l'ottimizzazione con costi di transazione per aggiornare i pesi
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
            self.current_allocation = w_opt.copy()  # Aggiorna l'allocazione
        else:
            # Nei giorni in cui non aggiorni, mantieni l'allocazione corrente
            w_opt = self.current_allocation.copy()

        # Calcola il reward (puoi aggiustare il calcolo in base alla tua logica)
        reward = (self.config.lambda_profit * day_return
                  - self.config.lambda_risk * np.std(self.capital_hist[-5:])
                  # Puoi aggiungere altri termini se vuoi penalizzare i costi di transazione
                  )
        if np.isnan(reward) or np.isinf(reward):
            reward = -1  # Penalizzazione in caso di errore

        next_state = self._get_state()
        done = self._check_done()
        info = {"capital": self.capital, "day_return": day_return}
        return next_state, reward, done, info

    def reset(self):
        self.current_day = 0
        self.capital = self.config.initial_asset
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.current_prices = self.data.loc[self.current_day].drop('Date').values.astype(float)
        state = self._get_state()
        print(f"🔍 DEBUG: Dimensione stato: {state.shape}, Prevista: {self.expected_obs_size}")
        return state
