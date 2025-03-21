import pandas as pd
from config import Config
from tickers_portfolio_env import TickersPortfolioEnv
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback,  CheckpointCallback
import os
import torch
from transaction_costs import log_returns_df
import matplotlib.pyplot as plt

def plot_performance(env):
    df = pd.DataFrame(env.performance_log)
    print("📊 DEBUG: Contenuto performance_log:")
    print(df.head())

    if 'Date' not in df.columns:
        print("❌ ERRORE: La colonna 'Date' non esiste! Colonne disponibili:", df.columns)
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Capital"], label="RL Portfolio", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.title("Evoluzione del Capitale del Portafoglio")
    plt.legend()
    plt.grid()
    plt.show()




'''def main():

    # Carica i dati storici dei prezzi
    data = pd.read_csv("./Tickers_file/merged_tickers_data.csv", parse_dates=['Date'])
    # Per semplicità, assumiamo che il CSV sia già nel formato: ogni riga una data e ogni colonna (oltre 'Date') è il prezzo di chiusura di un asset.

    # Carica i forecast ottenuti dal modello GARCH-LSTM (CSV con colonne: 'Date', '_Vol_Pred', 'pred_return')
    forecast_data = pd.read_csv("./Tickers_file/merged_forecast_results.csv", parse_dates=['Date'])

    print(data.head())
    print(forecast_data.head())

    config = Config(seed_num=2022)
    config.print_config()
    env = TickersPortfolioEnv(config=config, data=data, forecast_data=forecast_data, log_returns_df=log_returns_df, mode='train')

    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=10000, deterministic=True)

    model = TD3("MlpPolicy", env, verbose=1, learning_rate=config.learning_rate, seed=config.seed_num)

    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="./checkpoints/",
                                             name_prefix="TD3_transaction_cost")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Avvia l'addestramento con RL
    print("🚀 Avvio dell'addestramento...")
    #provo con 10000 solo per test
    model.learn(total_timesteps=10000, callback=[eval_callback, checkpoint_callback], log_interval=10)


    # Salva il modello addestrato
    model.save("TD3_extended_transaction_cost_model")

    print("Training completato. Capitale finale:", env.capital)
    plot_performance(env)
    env.calculate_metrics()'''

def quick_test():
    # Carica i dati
    data = pd.read_csv("./Tickers_file/merged_tickers_data.csv", parse_dates=['Date'])
    forecast_data = pd.read_csv("./Tickers_file/merged_forecast_results.csv", parse_dates=['Date'])

    config = Config(seed_num=2022)
    env = TickersPortfolioEnv(config=config, data=data, forecast_data=forecast_data, log_returns_df=log_returns_df, mode='test')

    obs = env.reset()
    for i in range(100):  # Esegui solo 10 passi
        print(f"\n🔄 STEP {i+1} 🔄")
        action = env.action_space.sample()  # Azione casuale
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Capitale: {info['capital']}, Giorno: {env.current_day}")
        if done:
            break  # Se finisce prima, interrompi

    env.calculate_metrics()
    pd.DataFrame(env.performance_log).to_csv("log_completo.csv", index=False)

if __name__ == '__main__':
    quick_test()
