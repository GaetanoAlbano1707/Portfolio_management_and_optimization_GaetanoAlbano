import pandas as pd
import numpy as np

def compute_performance_metrics(series):
    returns = series.pct_change().dropna()
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_dd = (series / series.cummax() - 1).min()
    total_return = (series.iloc[-1] / series.iloc[0]) - 1

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2)
    }

if __name__ == "__main__":
    path = "./PPO/results/policy_values.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    metrics = compute_performance_metrics(df['value'])

    result_path = "./PPO/results/rl_policy_metrics.csv"
    pd.Series(metrics).to_csv(result_path)
    print(f"✅ Metriche salvate in: {result_path}\n", metrics)
