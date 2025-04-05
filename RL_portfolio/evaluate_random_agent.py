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

    portfolio_values, rewards, dates, actions = [], [], [], []
    log_data = []

    step = 0
    while not done:
        action = np.random.dirichlet(np.ones(env.portfolio_size + 1))
        state, reward, done, _, info = env.step(action)

        portfolio_values.append(env._portfolio_value)
        rewards.append(reward)
        dates.append(info["end_time"])
        actions.append(action)

        if step % 63 == 0:
            log_data.append({
                "step": step,
                "date": info["end_time"],
                "portfolio_value": env._portfolio_value,
                "reward": reward,
                **{f"alloc_{i}": w for i, w in enumerate(action)}
            })

        step += 1

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
    # Compatibilità per compare_agents.py
    # Salvataggio standard
    random_path = results_dir / "evaluation_results_random.csv"
    print(f"📁 File salvato in: {random_path.resolve()}")
    pd.DataFrame([metrics]).to_csv(random_path, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_values)
    plt.title("Random Agent Portfolio Value")
    plt.grid(True)
    for l in range(0, len(portfolio_values), 63):
        plt.axvline(x=l, color='red', linestyle='--', alpha=0.3)
    plt.savefig(results_dir / "portfolio_value_random.png")
    plt.close()

    print(f"✅ Random agent valutato. FAPV: {metrics['fapv']:.4f}, Final Value: {metrics['final_value']:.2f}")

if __name__ == "__main__":

    df = pd.read_csv("./TEST/main_data_fake.csv", parse_dates=["date"])
    evaluate_random_agent(
        df=df,
        initial_amount=100000,
        reward_scaling=1.0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=50,
        data_normalization="by_previous_time"
    )
