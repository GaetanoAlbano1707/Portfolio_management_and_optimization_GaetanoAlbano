# main_rl_extended.py
import pandas as pd
from config import Config
from tickers_portfolio_env import TickersPortfolioEnv
from stable_baselines3 import TD3
from load_file import download_tickers


def main():
    # Carica i dati storici dei prezzi (assicurati che il CSV abbia colonna 'date' e colonne per ogni ticker)
    data = pd.read_csv("dati_storici.csv", parse_dates=['date'])
    # Per semplicità, assumiamo che il CSV sia già nel formato: ogni riga una data e ogni colonna (oltre 'date') è il prezzo di chiusura di un asset.

    # Carica i forecast ottenuti dal modello GARCH-LSTM (CSV con colonne: 'date', 'vol_forecast', 'pred_return')
    forecast_data = pd.read_csv("forecast_data.csv", parse_dates=['date'])

    config = Config(seed_num=2022)
    env = ExtendedStockPortfolioEnv(config=config, data=data, forecast_data=forecast_data, mode='train')

    # Crea il modello RL: usiamo TD3 con policy MLP (modifica in base alle tue esigenze)
    model = TD3("MlpPolicy", env, verbose=1, learning_rate=config.learning_rate, seed=config.seed_num)

    # Esegui il training per un numero di timesteps (ad esempio 100000, da regolare)
    model.learn(total_timesteps=100000)

    # Salva il modello addestrato
    model.save("TD3_extended_transaction_cost_model")

    print("Training completato. Capitale finale:", env.capital)


if __name__ == '__main__':
    main()
