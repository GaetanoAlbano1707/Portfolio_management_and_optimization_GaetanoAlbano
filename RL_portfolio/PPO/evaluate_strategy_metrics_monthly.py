import pandas as pd
import numpy as np

def evaluate_strategies(strategy_returns: pd.DataFrame, output_path="strategy_risk_metrics_monthly.csv"):
    """
    Calcola metriche di rischio e performance per ogni strategia.

    Metriche incluse:
    - Sharpe Ratio
    - Sortino Ratio
    - Volatilità (Std Dev)
    - Max Drawdown
    - Calmar Ratio
    """

    metrics = {}

    for strategy in strategy_returns.columns:
        series = strategy_returns[strategy].dropna()
        if (series <= 0).any():
            series = (series / series.iloc[0]) - 1  # se sono cumulative, calcola i rendimenti

        daily_returns = series.pct_change().dropna()
        if daily_returns.std() == 0:
            continue

        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
        downside_std = daily_returns[daily_returns < 0].std()
        sortino = daily_returns.mean() / (downside_std + 1e-8) * np.sqrt(252)
        volatility = daily_returns.std() * np.sqrt(252)

        cum_returns = (1 + daily_returns).cumprod()
        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        max_drawdown = drawdown.min()
        calmar = daily_returns.mean() * 252 / abs(max_drawdown + 1e-8)

        metrics[strategy] = {
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar
        }

    metrics_df = pd.DataFrame(metrics).T.round(4)
    metrics_df.to_csv(output_path)
    print(f"✅ Metriche salvate in: {output_path}")
