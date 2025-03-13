import numpy as np
import gym
from gym import spaces
import pandas as pd
from transaction_costs import optimize_with_transaction_costs  # importiamo la nostra funzione di ottimizzazione
import scipy.special

class TickersPortfolioEnv(gym.Env):
    def __init__(self, config, data, forecast_data, mode='train'):
        """
        data: DataFrame contenente i dati storici. Deve avere almeno le colonne 'date' e i prezzi (es. 'Adj Close')
        forecast_data: DataFrame con le previsioni. Deve avere una colonna 'date' e per ogni ticker le colonne
                       '<TICKER>_vol_forecast' e '<TICKER>_pred_return'
        """
        super(TickersPortfolioEnv, self).__init__()

        self.config = config
        self.mode = mode
        self.data = data.reset_index(drop=True)
        self.forecast_data = forecast_data.reset_index(drop=True)
        self.current_day = 0
        self.stock_num = self.data.drop(columns=['date']).shape[1]  # si assume che le colonne dopo 'date' siano i prezzi
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_num,), dtype=np.float32)
        # Definiamo uno state space esteso (qui includiamo un vettore base, l'allocazione corrente, le previsioni e lo scaling del capitale)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100 + self.stock_num + self.stock_num * 2 + 1,), dtype=np.float32)
        self.capital = config.initial_asset
        self.capital_hist = [self.capital]

        # Stato iniziale del portafoglio: pesi uguali
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.capital = config.initial_asset
        self.current_day = 0

        # Inizializza i prezzi per il primo giorno
        self.current_prices = self.data.loc[self.current_day].drop('date').values.astype(float)

    def weights_normalization(self, action):
        normalized_action = scipy.special.softmax(action)
        return normalized_action / np.sum(normalized_action)  # Assicura che sommi a 1

    def get_next_prices(self):
        self.current_day += 1
        if self.current_day >= len(self.data):
            self.current_day = len(self.data) - 1
        self.current_prices = self.data.loc[self.current_day].drop('date').values.astype(float)
        return self.current_prices

    def _get_base_state(self):
        last_n_days = 5  # Può essere regolato
        start_idx = max(0, self.current_day - last_n_days)

        vol_forecast_mean = self.forecast_data.iloc[start_idx:self.current_day].filter(
            like='_vol_forecast').mean().mean()
        pred_return_mean = self.forecast_data.iloc[start_idx:self.current_day].filter(like='_pred_return').mean().mean()

        return np.array([vol_forecast_mean, pred_return_mean])

    def _get_forecast_features(self):
        """
        Estrae le previsioni per il giorno corrente dal forecast_data.
        Ci aspettiamo che forecast_data abbia:
          - 'date'
          - per ogni asset, '<TICKER>_vol_forecast'
          - per ogni asset, '<TICKER>_pred_return'
        Restituisce due vettori:
          - vol_forecast: dimensione (stock_num,)
          - pred_return: dimensione (stock_num,)
        """
        current_date = self.data.loc[self.current_day, 'date']
        row = self.forecast_data[self.forecast_data['date'] == current_date]
        if not row.empty:
            vol_forecast = np.array([row.iloc[0][f"{ticker}_vol_forecast"] for ticker in self.config.tickers])
            pred_return = np.array([row.iloc[0][f"{ticker}_pred_return"] for ticker in self.config.tickers])
        else:
            vol_forecast = np.zeros(self.stock_num)
            pred_return = np.zeros(self.stock_num)
        return vol_forecast, pred_return

    def _get_state(self):
        # Combina il base state, l'allocazione corrente, le previsioni (vol e ritorni per ciascun asset) e il capitale scalato
        base_state = self._get_base_state()  # dimensione 100
        alloc_state = self.current_allocation  # dimensione stock_num
        vol_forecast, pred_return = self._get_forecast_features()  # ciascuno di dimensione stock_num
        capital_scaled = np.array([self.capital / self.config.initial_asset])
        state = np.concatenate([base_state, alloc_state, vol_forecast, pred_return, capital_scaled])
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

        # Estrai le previsioni dal forecast_data per il giorno corrente per costruire mu e Sigma:
        vol_forecast, pred_return = self._get_forecast_features()
        # Usiamo pred_return come vettore dei ritorni attesi (mu)
        mu = pred_return
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
            delta_minus=self.config.delta_minus,  # Passiamo i parametri corretti
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

        next_state = self._get_state()
        done = self._check_done()
        info = {"total_cost": total_cost, "capital": self.capital, "day_return": day_return}
        return next_state, reward, done, info

    def reset(self):
        self.current_day = 0
        self.capital = self.config.initial_asset
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.current_prices = self.data.loc[self.current_day].drop('date').values.astype(float)
        return self._get_state()
