import pandas as pd
from config import Config
from tickers_portfolio_env import TickersPortfolioEnv
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback,  CheckpointCallback
import os
import torch
from transaction_costs import log_returns_df


def main():

    # Carica i dati storici dei prezzi
    data = pd.read_csv("./Tickers_file/merged_tickers_data.csv", parse_dates=['Date'])
    # Per semplicità, assumiamo che il CSV sia già nel formato: ogni riga una data e ogni colonna (oltre 'Date') è il prezzo di chiusura di un asset.

    # Carica i forecast ottenuti dal modello GARCH-LSTM (CSV con colonne: 'Date', '_Vol_Pred', 'pred_return')
    forecast_data = pd.read_csv("./Tickers_file/merged_forecast_results.csv", parse_dates=['Date'])

    config = Config(seed_num=2022)
    env = TickersPortfolioEnv(config=config, data=data, forecast_data=forecast_data, log_returns_df=log_returns_df, mode='train')

    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=10000, deterministic=True)

    model = TD3("MlpPolicy", env, verbose=1, learning_rate=config.learning_rate, seed=config.seed_num)

    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="./checkpoints/",
                                             name_prefix="TD3_transaction_cost")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.learn(total_timesteps=500000, callback=[eval_callback, checkpoint_callback])

    # Salva il modello addestrato
    model.save("TD3_extended_transaction_cost_model")

    print("Training completato. Capitale finale:", env.capital)


if __name__ == '__main__':
    main()
