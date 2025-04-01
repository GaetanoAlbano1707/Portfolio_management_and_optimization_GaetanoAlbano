import os
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from pathlib import Path

from data_loader import (
    load_volatility_data,
    load_expected_returns,
    compute_covariance_matrix
)
from models import EIIE, GPM, EI3
from policy_gradient import PolicyGradient
from portfolio_optimization_env import PortfolioOptimizationEnv
from evaluate_policy import evaluate_policy
from cost_optimization import grid_search_transaction_costs
from rebalance_comparison import compare_rebalancing_periods
from logger import ExperimentLogger
from utils import set_seed, load_config
from grid_search_plot import plot_grid_search_results
from correlation_utils import calculate_rolling_correlation
from plot_utils import plot_covariance_evolution


# === Config ===
config = load_config("config.json")
set_seed(config["seed"])

# === Directory risultati
result_dir = Path("results/test" if config.get("test_mode", False) else "results/experiment")
result_dir.mkdir(parents=True, exist_ok=True)
(result_dir / "models").mkdir(parents=True, exist_ok=True)

# === Dati principali ===
df = pd.read_csv("./TEST/main_data_fake.csv", parse_dates=["date"])
features = ["adj_close", "close", "high", "low", "open", "volume", "return", "log_return"]
tickers = df["tic"].unique().tolist()

# === Volatilità e rendimenti attesi
vol_df = load_volatility_data("./TEST/test_results_*.csv")
expected_returns = load_expected_returns("./TEST/expected_returns_FAKE.csv")
corr_matrices = calculate_rolling_correlation(df, window=63)
# === Covarianza dinamica
cov_matrices = compute_covariance_matrix(vol_df)
plot_covariance_evolution(cov_matrices, save_dir=result_dir)



# === Costi transazione
n_assets_plus_cash = len(tickers) + 1

cost_c_plus = [config["costs"]["c_plus"]] * n_assets_plus_cash
cost_c_minus = [config["costs"]["c_minus"]] * n_assets_plus_cash
cost_delta_plus = [config["costs"]["delta_plus"]] * n_assets_plus_cash
cost_delta_minus = [config["costs"]["delta_minus"]] * n_assets_plus_cash

# === Costruzione modello
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

# === Dummy buffer/memory
class DummyBuffer:
    def __init__(self): self._data = []
    def add(self, x): self._data.append(x)
    def sample(self, n): return self._data[-n:]
    def __len__(self): return len(self._data)

class DummyMemory:
    def __init__(self, n_assets):
        self._n_assets = n_assets
        self._a = self._random_action()

    def _random_action(self):
        a = np.random.rand(self._n_assets + 1)
        return a / np.sum(a)

    def retrieve(self):
        return self._a

    def add(self, a):
        self._a = a

buffer = DummyBuffer()
memory = DummyMemory(num_assets)

# === Trainer RL
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

# === Logging
logger = ExperimentLogger(log_dir=result_dir)
logger.log_config(config)

# === Train
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

# === Salvataggio modello
model_path = result_dir / "models" / f"{model_type.lower()}_final.pt"
torch.save(model.state_dict(), model_path)
print(f"💾 Modello salvato in: {model_path}")

# === Valutazione
model.load_state_dict(torch.load(model_path))
model.eval()

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

# === Grid search
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
        },
        save_path=result_dir / "grid_search_results.json"
    )
    plot_grid_search_results(result_dir / "grid_search_results.json")

# === Confronto ribilanciamento
if config.get("compare_rebalancing", False):
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
        cwd=str(result_dir)
    )
