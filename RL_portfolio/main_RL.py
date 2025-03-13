# main_rl_extended.py
import pandas as pd
from config import Config
from tickers_portfolio_env import TickersPortfolioEnv
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback,  CheckpointCallback
from merge_tickers import merge_ticker_csvs
from merge_forecast_results import merge_forecast_csvs
import os
import torch


def main():
    tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']

    # Carica i dati storici dei prezzi (assicurati che il CSV abbia colonna 'date' e colonne per ogni ticker)
    data = merge_ticker_csvs(tickers, './Tickers_file/')
    # Per semplicità, assumiamo che il CSV sia già nel formato: ogni riga una data e ogni colonna (oltre 'date') è il prezzo di chiusura di un asset.

    # Carica i forecast ottenuti dal modello GARCH-LSTM (CSV con colonne: 'date', 'vol_forecast', 'pred_return')
    forecast_data = merge_forecast_csvs(tickers, './PORTFOLIO_MANAGEMENT_AND_OPTIMIZATION_GaetanoAlbano/Risultati_GARCH_LSTM_Forecasting/')

    if data is None or forecast_data is None:
        print("Errore: I dati storici o i forecast non sono stati generati correttamente.")
        exit(1)

    config = Config(seed_num=2022)
    env = TickersPortfolioEnv(config=config, data=data, forecast_data=forecast_data, mode='train')

    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=10000, deterministic=True)

    model = TD3("MlpPolicy", env, verbose=1, learning_rate=config.learning_rate, seed=config.seed_num)

    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="./checkpoints/",
                                             name_prefix="TD3_transaction_cost")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.learn(total_timesteps=100000, callback=[eval_callback, checkpoint_callback])

    # Salva il modello addestrato
    model.save("TD3_extended_transaction_cost_model")

    print("Training completato. Capitale finale:", env.capital)


if __name__ == '__main__':
    main()
