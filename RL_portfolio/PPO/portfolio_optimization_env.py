import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from datetime import datetime

class PortfolioOptimizationEnv(gym.Env):
    def __init__(self, data_path, expected_return_path, forecasting_path,
                 initial_amount=100000, transaction_cost=0.001,
                 reward_type='log_return', risk_penalty_coeff=0.01):
        super().__init__()

        self.log_file = os.path.join("./PPO/logs/", "env_log.txt")
        self.csv_log_file = os.path.join("./PPO/logs/", "episode_metrics.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._log("\n🔄 Avvio ambiente PortfolioOptimizationEnv...")

        # === Caricamento dati
        self.df = pd.read_csv(data_path, parse_dates=['date']).dropna()
        self.expected_returns = pd.read_csv(expected_return_path, parse_dates=['date']).dropna()
        self.volatility_forecast = pd.read_csv(forecasting_path, parse_dates=['date']).dropna()

        # === Intersezione delle date comuni
        common_dates = set(self.df['date']) & set(self.expected_returns['date']) & set(self.volatility_forecast['date'])
        self.dates = sorted([d for d in self.df['date'].unique() if d in common_dates])
        self.df = self.df[self.df['date'].isin(self.dates)]
        self.expected_returns = self.expected_returns[self.expected_returns['date'].isin(self.dates)]
        self.volatility_forecast = self.volatility_forecast[self.volatility_forecast['date'].isin(self.dates)]

        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type
        self.risk_penalty_coeff = risk_penalty_coeff

        self.tickers = sorted(self.df['tic'].unique())
        self.num_assets = len(self.tickers)
        self.num_steps = len(self.dates)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets * 7,), dtype=np.float32)

        # === Inizializza log CSV
        with open(self.csv_log_file, "w") as f:
            f.write("episode,total_reward,total_cost,total_risk_penalty,final_value\n")

        self.episode_count = 0
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.weights = np.ones(self.num_assets) / self.num_assets
        self.reward_cumulative = 0
        self.risk_penalty_cumulative = 0
        self.cost_cumulative = 0

        obs = self._next_observation()
        obs = np.nan_to_num(obs)

        self._log(f"\n🚀 RESET episodio {self.episode_count} - Data iniziale: {self.dates[self.current_step].date()}")
        self._log(f"  ▸ Valore iniziale portafoglio: {self.portfolio_value}")
        self._log(f"  ▸ Somma pesi: {np.sum(self.weights):.4f}")
        self._log(f"  ▸ Osservazione: shape={obs.shape}, NaN={np.isnan(obs).sum()}, min={obs.min():.4f}, max={obs.max():.4f}")

        return obs, {}

    def step(self, action):
        weights = action / (np.sum(action) + 1e-8)

        last_prices = self._get_prices(self.current_step)
        next_prices = self._get_prices(self.current_step + 1)
        gross_return = np.dot(weights, (next_prices - last_prices) / (last_prices + 1e-8))

        cost = self.transaction_cost * np.sum(np.abs(weights - self.weights))
        self.cost_cumulative += cost

        self.portfolio_value *= (1 + gross_return)

        vol_series = self.volatility_forecast[self.volatility_forecast['date'] == self.dates[self.current_step]].set_index('tic').reindex(self.tickers)['LSTM_Vol'].fillna(0).values
        weighted_risk = np.dot(weights, vol_series)
        risk_penalty = self.risk_penalty_coeff * weighted_risk
        self.risk_penalty_cumulative += risk_penalty

        if self.reward_type == 'log_return':
            base_reward = np.log((self.portfolio_value + 1e-8) / self.initial_amount)
            reward = base_reward - cost - risk_penalty
        else:
            raise ValueError(f"Reward type {self.reward_type} not recognized")

        self.reward_cumulative += reward

        self._log(
            f"🪙 Step {self.current_step:04d} | Portafoglio: {self.portfolio_value:.2f} | "
            f"Rend. lordo: {gross_return:.5f} | Costo: {cost:.5f} | Rischio: {risk_penalty:.5f} | "
            f"Reward: {reward:.5f}"
        )

        self.weights = weights
        self.current_step += 1
        terminated = self.current_step >= self.num_steps - 1

        if terminated:
            self._save_episode_metrics()

        obs = self._next_observation()
        return np.nan_to_num(obs), reward, terminated, False, {}

    def _next_observation(self):
        date = self.dates[self.current_step]
        frame = self.df[self.df['date'] == date].set_index('tic').reindex(self.tickers)

        data = [
            frame['adj_close'].fillna(0).values,
            frame['high'].fillna(0).values,
            frame['low'].fillna(0).values,
            frame['open'].fillna(0).values,
            frame['volume'].fillna(0).values,
            self.expected_returns[self.expected_returns['date'] == date].set_index('tic').reindex(self.tickers)['predicted_t'].fillna(0).values,
            self.volatility_forecast[self.volatility_forecast['date'] == date].set_index('tic').reindex(self.tickers)['LSTM_Vol'].fillna(0).values
        ]

        obs = np.concatenate(data)
        obs = (obs - np.mean(obs)) / (np.std(obs) + 1e-8)
        return obs

    def _get_prices(self, step):
        date = self.dates[step]
        frame = self.df[self.df['date'] == date].set_index('tic').reindex(self.tickers)
        return frame['adj_close'].fillna(1.0).values

    def _log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save_episode_metrics(self):
        with open(self.csv_log_file, "a") as f:
            f.write(
                f"{self.episode_count},{self.reward_cumulative:.6f},{self.cost_cumulative:.6f},"
                f"{self.risk_penalty_cumulative:.6f},{self.portfolio_value:.2f}\n"
            )
        self._log(f"📋 Fine episodio {self.episode_count} | Totale reward: {self.reward_cumulative:.4f} | "
                  f"Costo totale: {self.cost_cumulative:.4f} | Penalità rischio: {self.risk_penalty_cumulative:.4f} | "
                  f"Valore finale: {self.portfolio_value:.2f}")
        self.episode_count += 1
