import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioOptimizationEnv(gym.Env):
    def __init__(self, data_path, expected_return_path, forecasting_path, initial_amount=100000, transaction_cost=0.001, reward_type='log_return'):
        super(PortfolioOptimizationEnv, self).__init__()

        self.df = pd.read_csv(data_path, parse_dates=['date'])
        self.expected_returns = pd.read_csv(expected_return_path, parse_dates=['date'])
        self.volatility_forecast = pd.read_csv(forecasting_path, parse_dates=['date'])

        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type

        self.tickers = self.df['tic'].unique()
        self.num_assets = len(self.tickers)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets * 7,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.weights = np.array([1.0 / self.num_assets] * self.num_assets)
        return self._next_observation(), {}

    def step(self, action):
        weights = action / np.sum(action)

        last_day_prices = self._get_prices(self.current_step)
        current_day_prices = self._get_prices(self.current_step + 1)
        portfolio_return = np.dot(weights, (current_day_prices - last_day_prices) / last_day_prices)

        self.portfolio_value *= (1 + portfolio_return)

        transaction_cost = self.transaction_cost * np.sum(np.abs(weights - self.weights))

        if self.reward_type == 'log_return':
            reward = np.log(self.portfolio_value / self.initial_amount) - transaction_cost
        else:
            raise ValueError(f"Reward type {self.reward_type} not recognized")

        self.weights = weights
        self.current_step += 1
        terminated = self.current_step >= len(self.df['date'].unique()) - 1

        return self._next_observation(), reward, terminated, False, {}

    def _next_observation(self):
        date = self.df['date'].unique()[self.current_step]
        frame = self.df[self.df['date'] == date].set_index('tic').reindex(self.tickers)

        adj_close = frame['adj_close'].values
        high = frame['high'].values
        low = frame['low'].values
        open_ = frame['open'].values
        volume = frame['volume'].values

        expected_return = self.expected_returns[self.expected_returns['date'] == date].set_index('tic').reindex(self.tickers)['predicted_t'].values
        volatility_forecast = self.volatility_forecast[self.volatility_forecast['date'] == date].set_index('tic').reindex(self.tickers)['LSTM_Vol'].values

        obs = np.concatenate([adj_close, high, low, open_, volume, expected_return, volatility_forecast])
        return obs

    def _get_prices(self, step):
        date = self.df['date'].unique()[step]
        frame = self.df[self.df['date'] == date].set_index('tic').reindex(self.tickers)
        return frame['adj_close'].values
