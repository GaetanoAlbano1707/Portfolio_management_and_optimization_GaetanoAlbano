from data_loader import load_volatility_data, load_expected_returns
from policy_gradient import PolicyGradient
from portfolio_optimization_env import PortfolioOptimizationEnv

import torch
import numpy as np
import pandas as pd
from torch.optim import Adam

# === Esempio Setup ===
df = pd.read_csv("your_main_dataframe.csv")
costs = {
    "cost_c_plus": [0.001] * len(df["tic"].unique()),
    "cost_c_minus": [0.001] * len(df["tic"].unique()),
    "cost_delta_plus": [0.01] * len(df["tic"].unique()),
    "cost_delta_minus": [0.01] * len(df["tic"].unique()),
}

# Dummy model (sostituire con il tuo)
class DummyPolicy(torch.nn.Module):
    def forward(self, state, last_action):
        return torch.nn.functional.softmax(torch.rand(state.shape[2] + 1), dim=0).unsqueeze(0)

policy = DummyPolicy()
optimizer = Adam(policy.parameters(), lr=0.001)

# Dummy buffer & memory (sostituire con i tuoi reali)
class DummyBuffer:
    def __init__(self): self._data = []
    def add(self, x): self._data.append(x)
    def sample(self, n): return self._data[-n:]
    def __len__(self): return len(self._data)

class DummyMemory:
    def __init__(self, n): self._a = np.array([1] + [0]*n, dtype=np.float32)
    def retrieve(self): return self._a
    def add(self, a): self._a = a

buffer = DummyBuffer()
memory = DummyMemory(portfolio_size=len(df["tic"].unique()))

trainer = PolicyGradient(
    env_class=PortfolioOptimizationEnv,
    policy_net=policy,
    optimizer=optimizer,
    buffer=buffer,
    memory=memory,
    reward_scaling=1.0,
    rebalancing_period=75,
    **costs
)

trainer.train(df=df, initial_amount=100000, episodes=5)
