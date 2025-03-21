import pandas as pd
from stable_baselines3 import PPO
from tickers_portfolio_env import TickersPortfolioEnv
from config import Config



# Carica i dati
data = pd.read_csv("./Tickers_file/merged_tickers_data.csv", parse_dates=['Date'])
forecast_data = pd.read_csv("./Tickers_file/merged_forecast_results.csv", parse_dates=['Date'])
log_returns_df = pd.read_csv("./merged_log_returns.csv")  # Usa il tuo metodo


# Inizializza ambiente in modalità test
config = Config(seed_num=2022)
env = TickersPortfolioEnv(config, data, forecast_data, log_returns_df, mode='test')

# Carica il modello addestrato
model = PPO.load("ppo_portfolio_final")
print("✅ Modello PPO caricato")

# Reset iniziale dell’ambiente
obs = env.reset()
done = False
step_count = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    step_count += 1

    print(f"Step {step_count} | Capitale: {info['capital']:.2f} | Rendimento: {info['day_return']:.5f}")

# Calcola e mostra metriche finali
env.calculate_metrics()
# Salva l'intero performance log su CSV
performance_df = pd.DataFrame(env.performance_log)
performance_df.to_csv("test_results.csv", index=False)
print("📁 Risultati del test salvati in test_results.csv")

