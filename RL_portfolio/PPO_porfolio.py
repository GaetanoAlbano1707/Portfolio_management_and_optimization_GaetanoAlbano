from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tickers_portfolio_env import TickersPortfolioEnv
from config import Config


# 1. Carica i dati
data, forecast_data, log_returns_df = load_data_and_forecasts()
config = Config()

# 2. Crea l'ambiente
env = DummyVecEnv([lambda: TickersPortfolioEnv(config, data, forecast_data, log_returns_df)])

# 3. Crea il modello PPO
model = PPO("MlpPolicy", env, verbose=1)

# 4. Allena il modello
model.learn(total_timesteps=10000)

# 5. Salva il modello
model.save("PPO_Portfolio")
