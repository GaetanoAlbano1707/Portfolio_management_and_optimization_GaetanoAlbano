import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from portfolio_optimization_env import PortfolioOptimizationEnv

data_path = "./PPO/main_data_real.csv"
expected_return_path = "./PPO/expected_returns_real.csv"
forecasting_path = "./PPO/forecasting_data_combined.csv"
log_dir = "./PPO/logs/"
model_dir = "./PPO/models/"
import os
os.makedirs("./PPO/results", exist_ok=True)

# Creazione delle directory se non esistono
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Configurazione del logger
new_logger = configure(log_dir, ["stdout", "csv"])

# Creazione dell'ambiente
def make_env():
    env = PortfolioOptimizationEnv(data_path=data_path, initial_amount=100000, reward_type='log_return')
    return env

env = DummyVecEnv([
    lambda: PortfolioOptimizationEnv(
        data_path=data_path,
        expected_return_path=expected_return_path,
        forecasting_path=forecasting_path,
        initial_amount=100000,
        reward_type='log_return'
    )
])

# Inizializzazione del modello PPO
model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

# Addestramento del modello
model.learn(total_timesteps=100000)

# Salvataggio del modello addestrato
model_path = os.path.join(model_dir, "ppo_portfolio_optimization")
model.save(model_path)

print(f"✅ Modello salvato in: {model_path}")
