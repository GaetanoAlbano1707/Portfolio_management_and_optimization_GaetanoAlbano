import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio_optimization_env import PortfolioOptimizationEnv

# === Percorsi
data_path = "./main_data_real.csv"
expected_return_path = "./expected_returns_real.csv"
forecasting_path = "./forecasting_data_combined.csv"
model_path = "./PPO/models/ppo_portfolio_optimization"
os.makedirs("./PPO/results", exist_ok=True)

# === Ambiente
def make_env():
    return PortfolioOptimizationEnv(
        data_path=data_path,
        expected_return_path=expected_return_path,
        forecasting_path=forecasting_path,
        initial_amount=100000,
        reward_type='log_return'
    )

env = DummyVecEnv([make_env])
model = PPO.load(model_path, env=env)

obs = env.reset()
obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
done = False
total_reward = 0.0
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done_vec, info = env.step(action)
    done = done_vec[0]  # 🔥 aggiorna qui per fermare correttamente
    reward = reward[0]  # reward è array
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    total_reward += reward
    step += 1
    print(f"[STEP {step}] Reward: {reward:.6f} | Total: {total_reward:.6f}")

print(f"\n✅ Valutazione completata. Reward totale: {total_reward:.6f}")
