import os
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam

from data_loader import load_volatility_data, load_expected_returns
from models import EIIE, GPM, EI3
from policy_gradient import PolicyGradient
from portfolio_optimization_env import PortfolioOptimizationEnv
from evaluate_policy import evaluate_policy
from cost_optimization import grid_search_transaction_costs
from rebalance_comparison import compare_rebalancing_periods
from logger import ExperimentLogger
from utils import set_seed, load_config

# === Configurazione ===
config = load_config("config.json")
set_seed(config["seed"])
os.makedirs("results/models", exist_ok=True)

# === Dati ===
df = pd.read_csv("your_main_dataframe.csv")  # Rinomina al tuo CSV
tickers = df["tic"].unique().tolist()

# === Costi ===
cost_c_plus = [config["costs"]["c_plus"]] * len(tickers)
cost_c_minus = [config["costs"]["c_minus"]] * len(tickers)
cost_delta_plus = [config["costs"]["delta_plus"]] * len(tickers)
cost_delta_minus = [config["costs"]["delta_minus"]] * len(tickers)

# === Modello ===
model_type = config["model_type"]
num_assets = len(tickers)
time_window = 50
num_features = 3

if model_type == "EIIE":
    model = EIIE(num_assets, time_window, num_features)
elif model_type == "GPM":
    model = GPM(num_assets * num_features, 64, num_assets + 1)
elif model_type == "EI3":
    model = EI3(num_assets, time_window, num_features)
else:
    raise ValueError("Modello non valido!")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = Adam(model.parameters(), lr=config["learning_rate"])

# === Buffer e Memory Dummy ===
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
memory = DummyMemory(num_assets)

# === Trainer RL ===
trainer = PolicyGradient(
    env_class=PortfolioOptimizationEnv,
    policy_net=model,
    optimizer=optimizer,
    buffer=buffer,
    memory=memory,
    batch_size=config["batch_size"],
    reward_scaling=config["reward_scaling"],
    rebalancing_period=config["rebalancing_period"],
    cost_c_plus=cost_c_plus,
    cost_c_minus=cost_c_minus,
    cost_delta_plus=cost_delta_plus,
    cost_delta_minus=cost_delta_minus,
    device=device
)

# === Logging ===
logger = ExperimentLogger()
logger.log_config(config)

# === Training ===
trainer.train(
    df=df,
    initial_amount=config["initial_amount"],
    episodes=config["episodes"],
    features=["close", "high", "low"],
    valuation_feature="close",
    time_column="date",
    tic_column="tic",
    tics_in_portfolio="all",
    time_window=time_window,
    data_normalization="by_previous_time",
)

# === Salvataggio Modello ===
torch.save(model.state_dict(), f"results/models/{model_type.lower()}_final.pt")
print(f"💾 Modello salvato!")

# === Ricarica per valutazione (opzionale) ===
model.load_state_dict(torch.load(f"results/models/{model_type.lower()}_final.pt"))
model.eval()

# === Valutazione ===
metrics = evaluate_policy(
    policy_net=model,
    env_class=PortfolioOptimizationEnv,
    df=df,
    initial_amount=config["initial_amount"],
    device=device,
    cost_c_plus=cost_c_plus,
    cost_c_minus=cost_c_minus,
    cost_delta_plus=cost_delta_plus,
    cost_delta_minus=cost_delta_minus,
    reward_scaling=config["reward_scaling"],
    features=["close", "high", "low"],
    valuation_feature="close",
    time_column="date",
    tic_column="tic",
    tics_in_portfolio="all",
    time_window=time_window,
    data_normalization="by_previous_time",
)

logger.log_metrics(metrics)
logger.save()

# === Grid Search (opzionale) ===
if config.get("optimize_costs", False):
    best_combo, _ = grid_search_transaction_costs(
        policy_net=model,
        df=df,
        cost_grid=config["grid_costs"],
        evaluation_metric="fapv",
        device=device,
        model_name=model_type,
        reward_scaling=config["reward_scaling"],
        env_kwargs={
            "features": ["close", "high", "low"],
            "valuation_feature": "close",
            "time_column": "date",
            "tic_column": "tic",
            "tics_in_portfolio": "all",
            "time_window": time_window,
            "data_normalization": "by_previous_time",
        }
    )

# === Rebalance Comparison (opzionale) ===
if config.get("compare_rebalancing", False):
    from rebalance_comparison import compare_rebalancing_periods
    periods = [21, 42, 63, 75, 100]
    compare_rebalancing_periods(model, df, periods,
        initial_amount=config["initial_amount"],
        device=device,
        cost_c_plus=cost_c_plus,
        cost_c_minus=cost_c_minus,
        cost_delta_plus=cost_delta_plus,
        cost_delta_minus=cost_delta_minus,
        reward_scaling=config["reward_scaling"],
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=time_window,
        data_normalization="by_previous_time",
    )
