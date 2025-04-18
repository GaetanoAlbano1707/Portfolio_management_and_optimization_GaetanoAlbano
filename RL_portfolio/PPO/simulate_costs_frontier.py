import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_financial_data
from environment import PortfolioEnv
from ppo_agent import PPOAgent

CSV_PATH = "merged_data_total_cleaned_wide_multifeatures.csv"
df, tickers = load_financial_data(CSV_PATH)

test_configs = [
    {"c_plus": 0.001, "c_minus": 0.001, "delta_plus": 0.001, "delta_minus": 0.001},
    {"c_plus": 0.01,  "c_minus": 0.02,  "delta_plus": 0.005, "delta_minus": 0.005},
    {"c_plus": 0.02,  "c_minus": 0.01,  "delta_plus": 0.01,  "delta_minus": 0.01}
]

results = []
EPISODES = 5
WINDOW_SIZE = 5

for config in test_configs:
    env = PortfolioEnv(df, tickers, window_size=WINDOW_SIZE, sharpe_window=50, **config)
    input_dim = np.prod(env.observation_space.shape)
    agent = PPOAgent(input_dim=input_dim, num_assets=len(tickers))

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        rewards = []
        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        total = sum(rewards)
        sharpe = np.mean(rewards) / (np.std(rewards) + 1e-6)
        results.append({
            "c+": config["c_plus"], "c-": config["c_minus"],
            "Δ+": config["delta_plus"], "Δ-": config["delta_minus"],
            "Reward": total, "Sharpe": sharpe
        })

df = pd.DataFrame(results)
df.to_csv("costs_sweep_results.csv", index=False)

plt.figure()
for key, grp in df.groupby(["c+", "c-"]):
    plt.plot(grp["Sharpe"], grp["Reward"], marker='o', label=f"c+={key[0]}, c-={key[1]}")
plt.title("Simulated Efficient Frontier (RL)")
plt.xlabel("Sharpe Ratio")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.savefig("simulated_frontier.png")
plt.show()
