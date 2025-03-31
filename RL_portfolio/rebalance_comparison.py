from evaluate_policy import evaluate_policy
from portfolio_optimization_env import PortfolioOptimizationEnv
import matplotlib.pyplot as plt


def compare_rebalancing_periods(policy_net, df, periods, **env_kwargs):
    results = {}

    for period in periods:
        print(f"⏱️ Testing rebalancing every {period} steps")
        metrics = evaluate_policy(
            policy_net=policy_net,
            env_class=PortfolioOptimizationEnv,
            df=df,
            rebalancing_period=period,
            **env_kwargs
        )
        results[period] = metrics

    # Plot confronto FAPV
    plt.figure()
    fapvs = [results[p]["fapv"] for p in periods]
    plt.plot(periods, fapvs, marker="o")
    plt.title("FAPV vs Rebalancing Period")
