import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from tickers_portfolio_env import TickersPortfolioEnv
from config import Config

from custom_policy import CustomPolicyNetwork  # Import della policy custom

# Carica i dati
data = pd.read_csv("./Tickers_file/merged_tickers_data.csv", parse_dates=['Date'])
forecast_data = pd.read_csv("./Tickers_file/merged_forecast_results.csv", parse_dates=['Date'])
log_returns_df = pd.read_csv("./merged_log_returns.csv")  # Usa il tuo metodo


# Crea l'ambiente di training
config = Config(seed_num=2022)
env = TickersPortfolioEnv(config, data, forecast_data, log_returns_df, mode='train')

# Ambiente di valutazione
eval_env = TickersPortfolioEnv(config, data, forecast_data, log_returns_df, mode='test')

# Callback per salvataggi periodici
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path='./models/',
    name_prefix='ppo_portfolio'
)

# Callback per early stopping
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=5000,
    best_model_save_path='./models/best_model/',
    verbose=1
)

# Inizializza il modello con la policy custom
model = PPO(
    policy=CustomPolicyNetwork,
    env=env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# Addestramento
model.learn(
    total_timesteps=100_000,
    callback=[checkpoint_callback, eval_callback]
)

# Salva il modello finale
model.save("ppo_portfolio_final")
print("✅ Modello salvato come 'ppo_portfolio_final.zip'")
