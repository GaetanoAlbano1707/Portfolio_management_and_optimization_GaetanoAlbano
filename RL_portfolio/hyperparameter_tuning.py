import optuna
import torch
import pandas as pd
from pathlib import Path
from models import EIIE, GPM, EI3
from policy_gradient import PolicyGradient
from evaluate_policy import evaluate_policy
from portfolio_optimization_env import PortfolioOptimizationEnv
from utils import set_seed

def objective(trial):
    # === Parametri suggeriti ===
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    lambda_risk = trial.suggest_float("diversification_lambda", 0.0, 0.5)
    noise = trial.suggest_float("exploration_noise", 0.0, 0.2)
    model_type = trial.suggest_categorical("model_type", ["EIIE", "EI3", "GPM"])

    # === Dati
    df = pd.read_csv("./TEST/main_data_real.csv", parse_dates=["date"])
    features = ["adj_close", "high", "low"]
    tickers = df["tic"].unique().tolist()
    num_assets = len(tickers)
    time_window = 50
    num_features = len(features)

    # === Modello
    if model_type == "EIIE":
        model = EIIE(num_assets, time_window, num_features)
    elif model_type == "GPM":
        model = GPM(num_assets * num_features, 64, num_assets + 1)
    else:
        model = EI3(num_assets, time_window, num_features)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Dummy buffer e memoria
    buffer = []
    memory = lambda: [1.0] + [0.0] * num_assets

    trainer = PolicyGradient(
        env_class=PortfolioOptimizationEnv,
        policy_net=model,
        optimizer=optimizer,
        buffer=buffer,
        memory=memory,
        batch_size=64,
        reward_scaling=1.0,
        rebalancing_period=75,
        exploration_noise=noise,
        diversification_lambda=lambda_risk,
        cost_c_plus=[0.01] * (num_assets + 1),
        cost_c_minus=[0.01] * (num_assets + 1),
        cost_delta_plus=[0.02] * (num_assets + 1),
        cost_delta_minus=[0.02] * (num_assets + 1),
        device=device,
        verbose=False
    )

    trainer.train(
        df=df,
        initial_amount=100000,
        episodes=3,
        features=features,
        valuation_feature="adj_close",
        time_column="date",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=time_window,
        data_normalization="by_previous_time",
    )

    metrics = evaluate_policy(
        model,
        PortfolioOptimizationEnv,
        df,
        initial_amount=100000,
        device=device,
        reward_scaling=1.0,
        cost_c_plus=[0.01] * (num_assets + 1),
        cost_c_minus=[0.01] * (num_assets + 1),
        cost_delta_plus=[0.02] * (num_assets + 1),
        cost_delta_minus=[0.02] * (num_assets + 1),
        features=features,
        valuation_feature="adj_close",
        time_column="date",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=time_window,
        data_normalization="by_previous_time",
    )

    # Obiettivo di ottimizzazione (puoi cambiarlo in 'sharpe' o 'fapv')
    return -metrics["sharpe"]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Salvataggio risultati
    df_trials = study.trials_dataframe()
    df_trials.to_csv("results/optuna_tuning_results.csv", index=False)

    print("✅ Best hyperparameters:", study.best_params)
