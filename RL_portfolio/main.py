import pandas as pd
import numpy as np
import torch
from torch.optim import Adam

from data_loader import load_volatility_data, load_expected_returns
from policy_gradient import PolicyGradient
from portfolio_optimization_env import PortfolioOptimizationEnv
from evaluate_policy import evaluate_policy
from models import EIIE, GPM, EI3
from utils import set_seed, load_config, save_metrics
from plot_utils import compute_buy_and_hold, plot_portfolio_comparison

# === Carica configurazione ===
config = load_config("config.json")
set_seed(config["seed"])

# === Carica dati ===
df = pd.read_csv("your_main_dataframe.csv")  # sostituisci con il tuo dataframe principale
tickers = df["tic"].unique().tolist()

# === Imposta costi transazione ===
cost_c_plus = [config["costs"]["c_plus"]] * len(tickers)
cost_c_minus = [config["costs"]["c_minus"]] * len(tickers)
cost_delta_plus = [config["costs"]["delta_plus"]] * len(tickers)
cost_delta_minus = [config["costs"]["delta_minus"]] * len(tickers)

# === Seleziona modello ===
model_type = config["model_type"]
num_assets = len(tickers)
time_window = 50
num_features = 3

if model_type == "EIIE":
    model = EIIE(num_assets, time_window, num_features)
elif model_type == "GPM":
    model = GPM(input_size=num_assets * num_features, hidden_size=64, output_size=num_assets + 1)
elif model_type == "EI3":
    model = EI3(num_assets, time_window, num_features)
else:
    raise ValueError("Modello non valido!")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = Adam(model.parameters(), lr=config["learning_rate"])

# === Dummy Buffer e Memory ===
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

# === Inizializza Trainer ===
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

# === Allena ===
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

save_metrics(metrics, "results/evaluation/metrics.json")

# === Confronto con Buy & Hold ===
benchmark = compute_buy_and_hold(df, tickers)
plot_portfolio_comparison(metrics["final_value"], benchmark, output_path="results/evaluation/comparison.png")
