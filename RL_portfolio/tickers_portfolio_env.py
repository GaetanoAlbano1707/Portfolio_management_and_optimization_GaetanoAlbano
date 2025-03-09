# extended_stock_portfolio_env.py
import numpy as np
import gym
from gym import spaces
from transaction_cost import quadratic_transaction_cost


class TickersPortfolioEnv(gym.Env):
    def __init__(self, config, data, forecast_data, mode='train'):
        """
        data: DataFrame contenente i dati storici. Deve avere almeno le colonne 'date' e 'Adj Close'
        forecast_data: DataFrame con le previsioni, colonna 'date' allineata a data, e colonne 'vol_forecast' e 'pred_return'
        """
        super(TickersPortfolioEnv, self).__init__()
        self.config = config
        self.mode = mode
        self.data = data.reset_index(drop=True)
        self.forecast_data = forecast_data.reset_index(drop=True)
        self.current_day = 0
        self.stock_num = self.data.drop(columns=['date']).shape[1]  # assumiamo che le colonne (es. XLK, XLV, ...) siano dopo 'date'
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_num,), dtype=np.float32)
        # Definiamo uno state space esteso: esempio dimensione 100+stock_num+2+1 (base_state + allocazione + previsioni + capital scaling)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100 + self.stock_num + 2 + 1,),
                                            dtype=np.float32)

        # Stato iniziale del portafoglio
        self.current_allocation = np.array([1.0 / self.stock_num] * self.stock_num)
        self.capital = config.initial_asset

        # Inizializza i prezzi per il primo giorno (si assume che ogni riga di data rappresenti i prezzi di chiusura per quel giorno)
        self.current_prices = self.data.loc[self.current_day].drop('date').values.astype(float)

    def weights_normalization(self, action):
        # Puoi utilizzare softmax oppure una normalizzazione semplice
        action = np.array(action)
        if np.sum(np.abs(action)) == 0:
            return np.array([1.0 / self.stock_num] * self.stock_num)
        return action / np.sum(np.abs(action))

    def get_next_prices(self):
        self.current_day += 1
        if self.current_day >= len(self.data):
            self.current_day = len(self.data) - 1
        self.current_prices = self.data.loc[self.current_day].drop('date').values.astype(float)
        return self.current_prices

    def _get_base_state(self):
        # Per semplicità, creiamo un vettore base casuale. Nella tua implementazione questo includerebbe indicatori tecnici, etc.
        return np.random.randn(100)

    def _get_forecast_features(self):
        # Ottieni il forecast per la data corrente
        current_date = self.data.loc[self.current_day, 'date']
        forecast_row = self.forecast_data[self.forecast_data['date'] == current_date]
        if not forecast_row.empty:
            vol_forecast = forecast_row['vol_forecast'].values[0]
            pred_return = forecast_row['pred_return'].values[0]
        else:
            vol_forecast = 0.0
            pred_return = 0.0
        return np.array([vol_forecast, pred_return])

    def _get_state(self):
        # Combina il base state, la normalizzazione del capitale, l'allocazione corrente e le feature di forecast
        base_state = self._get_base_state()
        capital_scaled = np.array([self.capital / self.config.initial_asset])
        alloc_state = self.current_allocation  # dimensione stock_num
        forecast_features = self._get_forecast_features()  # 2 elementi
        state = np.concatenate([base_state, alloc_state, forecast_features, capital_scaled])
        return state

    def _compute_portfolio_return(self, new_prices):
        # Calcola il rendimento del portafoglio: (new_prices / current_prices - 1) * allocation
        returns = new_prices / self.current_prices - 1
        return np.dot(self.current_allocation, returns)

    def _check_done(self):
        return self.current_day >= len(self.data) - 1

    def step(self, action):
        # Normalizza l'azione (i pesi)
        weights = self.weights_normalization(action)
        # Calcola i costi di transazione in base alla differenza tra la nuova allocazione e quella attuale
        total_cost, _ = quadratic_transaction_cost(
            w_new=weights,
            w_old=self.current_allocation,
            c_minus=self.config.c_minus,
            c_plus=self.config.c_plus,
            delta_minus=self.config.delta_minus,
            delta_plus=self.config.delta_plus
        )
        # Aggiorna il capitale sottraendo il costo (per semplicità, lo consideriamo come una percentuale di capitale)
        self.capital *= (1 - total_cost)

        # Aggiorna l'allocazione (dopo aver pagato il costo, normalizziamo per rispettare il vincolo di budget)
        self.current_allocation = weights.copy()  # Potresti anche normalizzare con: weights / (1+total_cost) se preferisci

        # Calcola il rendimento in base ai prezzi: prima memorizziamo i prezzi correnti per il calcolo
        old_prices = self.current_prices.copy()
        new_prices = self.get_next_prices()
        day_return = self._compute_portfolio_return(new_prices)
        # Aggiorna il capitale con il rendimento ottenuto
        self.capital *= (1 + day_return)

        # Calcola un "reward" che tenga conto del profitto, dei costi e del rischio (qui penalizziamo in base alla volatilità forecast)
        forecast_features = self._get_forecast_features()
        vol_forecast = forecast_features[0]
        reward = (self.config.lambda_profit * day_return
                  - self.config.lambda_cost * total_cost
                  - self.config.lambda_risk * vol_forecast)

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
