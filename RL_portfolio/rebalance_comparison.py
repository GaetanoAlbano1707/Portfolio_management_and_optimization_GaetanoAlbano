from evaluate_policy import evaluate_policy
from portfolio_optimization_env import PortfolioOptimizationEnv
import matplotlib.pyplot as plt


def compare_rebalancing_periods(policy_net, df, periods, **env_kwargs):
    results = {}

    for period in periods:
        print(f"⏱️ Testing rebalancing every {period} steps")

        # ✅ Iniettiamo nel dizionario il valore del rebalancing period
        env_kwargs_modified = env_kwargs.copy()
        env_kwargs_modified["rebalancing_period"] = period

        metrics = evaluate_policy(
            policy_net=policy_net,
            env_class=PortfolioOptimizationEnv,
            df=df,
            **env_kwargs_modified
        )
        results[period] = metrics

    # 📈 Plot confronto FAPV
    plt.figure()
    fapvs = [results[p]["fapv"] for p in periods]
    plt.plot(periods, fapvs, marker="o")
    plt.title("FAPV vs Rebalancing Period")
    plt.xlabel("Rebalancing Period")
    plt.ylabel("FAPV")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/rebalancing_comparison.png")
    plt.close()
    print("✅ Confronto ribilanciamento salvato in: results/rebalancing_comparison.png")
