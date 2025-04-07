import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_policy(
    policy_net,
    env_class,
    df,
    initial_amount,
    device,
    results_path="./results/test",
    sharpe_window=63,
    **env_kwargs
):
    env = env_class(df=df, initial_amount=initial_amount, print_metrics=False, plot_graphs=False, **env_kwargs)
    state, _ = env.reset()
    done = False
    last_action = torch.tensor([1] + [0] * env.portfolio_size, dtype=torch.float32).unsqueeze(0).to(device)

    portfolio_values = []
    rewards = []
    dates = []
    actions = []

    while not done:
        observation = torch.tensor(state["state"] if isinstance(state, dict) else state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(observation, last_action)
        action_np = action.squeeze(0).cpu().numpy()

        state, reward, done, _, info = env.step(action_np)
        last_action = action

        portfolio_values.append(env._portfolio_value)
        rewards.append(reward)
        actions.append(action_np)
        dates.append(info["end_time"])

    reward_std = np.std(rewards)
    reward_mean = np.mean(rewards)
    sharpe = reward_mean / reward_std if reward_std != 0 else 0

    metrics = {
        "agent_type": "policy",
        "final_value": portfolio_values[-1],
        "fapv": portfolio_values[-1] / portfolio_values[0],
        "reward_std": reward_std,
        "mean_reward": reward_mean,
        "sharpe": sharpe,
        "max_drawdown": max(1 - np.array(portfolio_values) / np.maximum.accumulate(portfolio_values)),
    }

    results_dir = Path(results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    # === Log
    log_df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "portfolio_value": portfolio_values,
        "reward": rewards
    })
    log_df.to_csv(results_dir / "evaluation_log_policy.csv", index=False)

    # === Salvataggio metriche
    pd.DataFrame([metrics]).to_csv(results_dir / "evaluation_results_policy.csv", index=False)

    # === Grafici
    plt.figure(figsize=(10, 4))
    plt.plot(dates, portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / "portfolio_value_policy.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(dates, rewards, label="Reward")
    plt.title("Reward per Step")
    plt.xlabel("Date")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / "reward_policy.png")
    plt.close()

    # === Rolling Sharpe Ratio
    log_df["rolling_mean"] = log_df["reward"].rolling(window=sharpe_window).mean()
    log_df["rolling_std"] = log_df["reward"].rolling(window=sharpe_window).std()
    log_df["sharpe_ratio"] = log_df["rolling_mean"] / log_df["rolling_std"]

    plt.figure(figsize=(12, 5))
    plt.plot(log_df["date"], log_df["sharpe_ratio"], label=f"Sharpe Ratio Rolling ({sharpe_window} giorni)")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rolling_sharpe_ratio.png")
    plt.close()

    print(f"📁 Risultati policy salvati in: {results_dir / 'evaluation_results_policy.csv'}")
    print(f"✅ Policy agent valutato. FAPV: {metrics['fapv']:.4f}, Valore Finale: {metrics['final_value']:.2f}")
    return metrics
