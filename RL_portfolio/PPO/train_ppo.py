import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from portfolio_optimization_env import PortfolioOptimizationEnv

# === Percorsi
data_path = "./main_data_real.csv"
expected_return_path = "./expected_returns_real.csv"
forecasting_path = "./forecasting_data_combined.csv"
log_dir = "./PPO/logs/"
model_dir = "./PPO/models/"
os.makedirs("./PPO/results", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# === Logger
new_logger = configure(log_dir, ["stdout", "csv"])

# === INFO DATI INIZIALI
df_prices = pd.read_csv(data_path, parse_dates=['date'])
tickers = df_prices['tic'].unique()
start_date, end_date = df_prices['date'].min(), df_prices['date'].max()
print(f"📅 Dati disponibili da {start_date.date()} a {end_date.date()}")
print(f"💼 Ticker unici nel dataset: {len(tickers)} => {sorted(tickers.tolist())}")

# Controllo NaN
pivot_prices = df_prices.pivot(index="date", columns="tic", values="adj_close")
nan_perc = pivot_prices.isna().mean().sort_values(ascending=False)
if nan_perc.any():
    print("⚠️ Ticker con dati mancanti:")
    print(nan_perc[nan_perc > 0].apply(lambda x: f"{x:.1%}"))

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

# === Callback personalizzato per tracciare reward cumulato
class RewardTrackingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if any(self.locals["dones"]):
            self.episode_rewards.append(self.env.envs[0].reward_cumulative)
        return True

    def _on_training_end(self) -> None:
        with open("./PPO/logs/episode_rewards.csv", "w") as f:
            f.write("episode,reward_cumulative\n")
            for i, r in enumerate(self.episode_rewards):
                f.write(f"{i},{r:.6f}\n")

# === Validazione osservazione iniziale
obs = env.reset()
obs = np.nan_to_num(obs)

print(f"🧠 Osservazione iniziale - Shape: {obs.shape}")
print(f"🔎 Valori osservazione: min={obs.min():.4f}, max={obs.max():.4f}, NaN rimossi={np.isnan(obs).sum()}")

assert not np.isnan(obs).any(), "❌ NaN rilevato nell'osservazione iniziale!"

# === PPO

# === PPO
model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

print(f"🛠️ Inizio training PPO con {model.n_steps * model.n_envs} passi per aggiornamento...")
callback = RewardTrackingCallback(env)
model.learn(total_timesteps=100_000, callback=callback)

# === Salvataggio modello
model_path = os.path.join(model_dir, "ppo_portfolio_optimization")
model.save(model_path)
print(f"✅ Modello PPO salvato in: {model_path}")

# === Estrazione pesi finali dalla policy
def extract_policy_weights(model, env):
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    weights = action.flatten()
    weights /= weights.sum()
    return weights

final_weights = extract_policy_weights(model, env)
tickers = env.get_attr("tickers")[0]
weights_df = pd.DataFrame([final_weights], columns=tickers)
weights_df.to_csv("./PPO/results/weights_log.csv", index=False)
print(f"✅ Pesi finali policy salvati in weights_log.csv: {np.round(final_weights, 4)}")
