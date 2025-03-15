import numpy as np
import gym
from gym import spaces
from transaction_costs import optimize_with_transaction_costs  # importiamo la nostra funzione di ottimizzazione
import scipy.special



class TickersPortfolioEnv(gym.Env):
    def __init__(self, config, data, forecast_data, mode='train'):
        """
        data: DataFrame contenente i dati storici. Deve avere almeno le colonne 'Date' e i prezzi (es. 'Adj Close')
        forecast_data: DataFrame con le previsioni. Deve avere una colonna 'Date' e per ogni ticker le colonne
                       '<TICKER>_vol_forecast' e '<TICKER>_pred_return'
        """
        super(TickersPortfolioEnv, self).__init__()

        self.config = config
        self.mode = mode
        self.data = data.reset_index(drop=True)
        self.forecast_data = forecast_data.reset_index(drop=True)
        self.current_day = 0
        self.stock_num = len(self.config.tickers)
        print(f"🔍 DEBUG: Numero asset (stock_num): {self.stock_num}")
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_num,), dtype=np.float32)
        # Definiamo uno state space esteso (qui includiamo un vettore base, l'allocazione corrente, le previsioni e lo scaling del capitale)
        self.expected_obs_size = 100 + self.stock_num + self.stock_num * 2 + 1  # diventa 100 + 6 + 12 + 1 = 119
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.expected_obs_size,), dtype=np.float32)

        self.capital = config.initial_asset
        self.capital_hist = [self.capital]

        # Stato iniziale del portafoglio: pesi uguali
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.capital = config.initial_asset
        self.current_day = 0

        # Inizializza i prezzi per il primo giorno
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
        """
        Estrae le previsioni di volatilità dal forecast_data e calcola i ritorni attesi
        (μ) usando una finestra storica dei log_return dalla serie storica (self.data).
        """
        current_date = self.data.loc[self.current_day, 'Date']
        # Ottieni la previsione di volatilità dal forecast_data
        row = self.forecast_data[self.forecast_data['Date'] == current_date]
        if not row.empty:
            vol_forecast = np.array([row.iloc[0].get(f"{ticker}_Vol_Pred", 0) for ticker in self.config.tickers])
        else:
            print(f"⚠️ ATTENZIONE: Previsioni di volatilità mancanti per la data {current_date}, riempite con 0.")
            vol_forecast = np.zeros(self.stock_num)

        # Calcola i ritorni attesi (μ) usando i log_return storici.
        # Ad esempio, usa una finestra mobile degli ultimi N giorni (qui impostiamo N=10)
        window_size = 10
        start_idx = max(0, self.current_day - window_size)
        recent_data = self.data.iloc[start_idx:self.current_day]

        expected_returns = []
        for ticker in self.config.tickers:
            col_name = f"{ticker}_log_return"
            if col_name in recent_data.columns:
                mu_ticker = recent_data[col_name].mean()  # Media dei log_return della finestra
            else:
                mu_ticker = 0.0
            expected_returns.append(mu_ticker)
        expected_returns = np.array(expected_returns)

        # Debug: stampa le dimensioni
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
        # L'agente propone un'azione: una nuova allocazione (non ancora normalizzata)
        proposed_weights = self.weights_normalization(action)

        vol_forecast, expected_returns = self._get_forecast_features()
        mu = expected_returns

        # Costruiamo una matrice Sigma semplificata come: Sigma = diag(vol_forecast^2)
        # (Assumiamo correlazioni nulle; in una versione più avanzata potresti usare una matrice di correlazione)
        D = np.diag(vol_forecast)
        Sigma = D @ D  # equivalente a diag(vol_forecast^2)

        # Ora, invece di applicare direttamente l'azione, usiamo il nostro modulo di ottimizzazione con costi:
        # Il portafoglio corrente è self.current_allocation, e i costi e gamma sono passati da config.
        w_opt, delta_minus_opt, delta_plus_opt = optimize_with_transaction_costs(
            mu=mu,
            Sigma=Sigma,
            w_tilde=self.current_allocation,
            c_minus=self.config.c_minus,
            c_plus=self.config.c_plus,
            delta_minus=self.config.delta_minus,
            delta_plus=self.config.delta_plus,
            gamma=self.config.gamma
        )

        # Calcola il costo totale (somma dei costi unitari * quantità)
        total_cost = np.sum(self.config.c_minus * delta_minus_opt + self.config.c_plus * delta_plus_opt)

        # Aggiorna il capitale sottraendo il costo (ipotizzando che il costo sia una percentuale)
        self.capital *= (1 - total_cost)
        self.capital_hist.append(self.capital)
        if len(self.capital_hist) > 100:  # Mantiene solo gli ultimi 100 valori
            self.capital_hist.pop(0)

        # Aggiorna il portafoglio corrente con i nuovi pesi ottenuti dall'ottimizzazione
        self.current_allocation = w_opt.copy()

        # Aggiorna i prezzi: simula il passaggio al giorno successivo
        old_prices = self.current_prices.copy()
        new_prices = self.get_next_prices()
        day_return = self._compute_portfolio_return(new_prices)
        # Aggiorna il capitale in base al rendimento ottenuto
        self.capital *= (1 + day_return)

        quadratic_penalty = np.sum(self.config.delta_minus * (np.array(delta_minus_opt) ** 2)) + \
                            np.sum(self.config.delta_plus * (np.array(delta_plus_opt) ** 2))

        reward = (
                self.config.lambda_profit * day_return
                - self.config.lambda_cost * (total_cost + quadratic_penalty)
                - self.config.lambda_risk * np.std(self.capital_hist[-5:])
                + 0.1 * (day_return / (np.std(self.capital_hist[-5:]) + 1e-6))
        )
        if np.isnan(reward) or np.isinf(reward):
            reward = -1  # Penalizzazione in caso di errore numerico

        next_state = self._get_state()
        done = self._check_done()
        info = {"total_cost": total_cost, "capital": self.capital, "day_return": day_return}
        return next_state, reward, done, info

    def reset(self):
        self.current_day = 0
        self.capital = self.config.initial_asset
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.current_prices = self.data.loc[self.current_day].drop('Date').values.astype(float)
        state = self._get_state()
        print(f"🔍 DEBUG: Dimensione stato: {state.shape}, Prevista: {self.expected_obs_size}")
        return state
