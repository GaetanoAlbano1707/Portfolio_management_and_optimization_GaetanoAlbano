import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from portfolio_optimization_env import PortfolioOptimizationEnv


def evaluate_random_agent(
    df: pd.DataFrame,
    initial_amount: float = 100000,
    reward_scaling: float = 1.0,
    results_path: str = "./results/test",
    **env_kwargs
):
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=initial_amount,
        reward_scaling=reward_scaling,
        print_metrics=False,
        plot_graphs=False,
        **env_kwargs
    )

    state, _ = env.reset()
    done = False

    portfolio_values = []
    rewards = []

    while not done:
        action = np.random.dirichlet(np.ones(env.portfolio_size + 1))
        state, reward, done, _, info = env.step(action)

        portfolio_values.append(env._portfolio_value)
        rewards.append(reward)

    reward_std = np.std(rewards)
    reward_mean = np.mean(rewards)
    sharpe = reward_mean / reward_std if reward_std != 0 else 0

    metrics = {
        "agent_type": "random",
        "final_value": portfolio_values[-1],
        "fapv": portfolio_values[-1] / portfolio_values[0],
        "reward_std": reward_std,
        "mean_reward": reward_mean,
        "sharpe": sharpe,
        "max_drawdown": max(1 - np.array(portfolio_values) / np.maximum.accumulate(portfolio_values)),
    }

    results_dir = Path(results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(results_dir / "evaluation_results_random.csv", index=False)

    print(f"✅ Random agent valutato. FAPV: {metrics['fapv']:.4f}, Final Value: {metrics['final_value']:.2f}")
