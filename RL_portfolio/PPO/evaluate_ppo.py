import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio_optimization_env import PortfolioOptimizationEnv

# === Percorsi dei file
data_path = "./PPO/main_data_real.csv"
expected_return_path = "./PPO/expected_returns_real.csv"
forecasting_path = "./PPO/forecasting_data_combined.csv"
model_path = "./PPO/models/ppo_portfolio_optimization.zip"
os.makedirs("./PPO/results", exist_ok=True)

# === Funzione per creare l'ambiente
def make_env():
    return PortfolioOptimizationEnv(
        data_path=data_path,
        expected_return_path=expected_return_path,
        forecasting_path=forecasting_path,
        initial_amount=100000,
        reward_type='log_return'
    )

# === Inizializzazione ambiente vettorizzato
env = DummyVecEnv([make_env])

# === Caricamento modello PPO addestrato
model = PPO.load(model_path, env=env)

# === Loop di valutazione
obs = env.reset()
terminated = False
total_reward = 0
step = 0

while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    print(f"[STEP {step}] Reward: {reward:.6f} | Total Reward: {total_reward:.6f}")

print(f"\n✅ Valutazione completata. Reward totale: {total_reward:.6f}")
