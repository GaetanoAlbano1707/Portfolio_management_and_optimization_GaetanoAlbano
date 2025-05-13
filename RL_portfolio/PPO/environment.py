import numpy as np
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, df, tickers, window_size=5, sharpe_window=50,
                 c_plus=0.001, c_minus=0.001, delta_plus=0.001, delta_minus=0.001,
                 rebalance_period=63):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.tickers = tickers
        self.window_size = window_size
        self.sharpe_window = sharpe_window
        self.rebalance_period = rebalance_period

        self.num_assets = len(tickers)
        self.feature_names = [
            'adj_close', 'predicted_t', 'predicted_t_std', 'volatility',
            'open', 'high', 'low', 'volume',
            'momentum_5d', 'mu_diff', 'vol_diff', 'volume_ma3'
        ]

        self.data = df[self.feature_names].unstack('ticker')
        self.steps = len(self.data)
        self.current_step = self.window_size
        self.weights = np.ones(self.num_assets) / self.num_assets
        self.prev_weights = self.weights.copy()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size, self.num_assets, len(self.feature_names)),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

        self.feature_mean = self.data.mean()
        self.feature_std = self.data.std().replace(0, 1)

        self.portfolio_returns = []
        self.rebalance_steps = []

        self.c_plus = c_plus
        self.c_minus = c_minus
        self.delta_plus = delta_plus
        self.delta_minus = delta_minus

    def _get_observation(self):
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        obs = window_data.values.reshape(self.window_size, self.num_assets, len(self.feature_names))
        obs = (obs - self.feature_mean.values.reshape(1, self.num_assets, len(self.feature_names))) / \
              self.feature_std.values.reshape(1, self.num_assets, len(self.feature_names))
        return obs

    def reset(self):
        self.current_step = self.window_size
        self.weights = np.ones(self.num_assets) / self.num_assets
        self.prev_weights = self.weights.copy()
        self.portfolio_returns = []
        self.rebalance_steps = []
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / np.sum(action) if np.sum(action) != 0 else self.prev_weights

        is_rebalance_day = ((self.current_step - self.window_size) % self.rebalance_period == 0)
        if is_rebalance_day:
            self.prev_weights = action
            self.rebalance_steps.append(self.current_step)
            print(f"🔁 Ribilanciamento effettuato al passo {self.current_step}")

        if self.current_step + 1 >= self.steps:
            return self._get_observation(), 0.0, True, {}

        prices_today = self.data.iloc[self.current_step].xs('adj_close', level=0).values
        prices_next = self.data.iloc[self.current_step + 1].xs('adj_close', level=0).values
        relative_price = prices_next / prices_today
        portfolio_return = np.dot(self.prev_weights, relative_price) - 1.0

        safe_net_return = max(portfolio_return, -0.999)
        log_return = np.log(1 + safe_net_return)
        self.portfolio_returns.append(log_return)

        delta_w = action - self.prev_weights
        cost = (
            self.c_plus * np.sum(np.clip(delta_w, 0, None)) +
            self.c_minus * np.sum(np.clip(-delta_w, 0, None)) +
            self.delta_plus * np.sum(np.clip(delta_w, 0, None)**2) +
            self.delta_minus * np.sum(np.clip(-delta_w, 0, None)**2)
        )

        reward = log_return - cost
        if len(self.portfolio_returns) >= self.sharpe_window:
            mean = np.mean(self.portfolio_returns[-self.sharpe_window:])
            std = np.std(self.portfolio_returns[-self.sharpe_window:]) + 1e-6
            reward = mean / std - cost

        self.current_step += 1
        return self._get_observation(), reward, False, {}
