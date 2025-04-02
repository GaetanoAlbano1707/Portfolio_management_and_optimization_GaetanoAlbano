import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def evaluate_policy(
    policy_net,
    env_class,
    df: pd.DataFrame,
    initial_amount: float = 100000,
    device: str = "cpu",
    plot_results: bool = True,
    results_path: str = "./results/test",
    cost_c_plus=None,
    cost_c_minus=None,
    cost_delta_plus=None,
    cost_delta_minus=None,
    reward_scaling: float = 1.0,
    **env_kwargs,
) -> dict:
    env = env_class(
        df=df,
        initial_amount=initial_amount,
        reward_scaling=reward_scaling,
        cost_c_plus=cost_c_plus,
        cost_c_minus=cost_c_minus,
        cost_delta_plus=cost_delta_plus,
        cost_delta_minus=cost_delta_minus,
        print_metrics=False,
        plot_graphs=False,
        **env_kwargs,
    )

    state, _ = env.reset()
    done = False
    last_action = torch.tensor(np.array([1] + [0] * env.portfolio_size)).unsqueeze(0).to(device)

    portfolio_values = []
    rewards = []

    while not done:
        observation = torch.tensor(state["state"] if isinstance(state, dict) else state).unsqueeze(0).to(device)

        with torch.no_grad():
            action = policy_net(observation, last_action)
        state, reward, done, _, info = env.step(action.squeeze().cpu().numpy())
        last_action = action

        portfolio_values.append(env._portfolio_value)
        rewards.append(reward)

    # === Calcolo metriche ===
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

    # === Save CSV con metriche
    results_dir = Path(results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(results_dir / "evaluation_results.csv", index=False)

    # === Plot
    if plot_results:
        plt.figure(figsize=(10, 4))
        plt.plot(portfolio_values, label="Portfolio Value")
        plt.title("Portfolio Value During Evaluation")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(results_dir / "portfolio_value.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(rewards, label="Reward")
        plt.title("Reward per Step")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(results_dir / "reward.png")
        plt.close()

    print(f"✅ Valutazione completata. FAPV: {metrics['fapv']:.4f}, Final Value: {metrics['final_value']:.2f}")
    return metrics
