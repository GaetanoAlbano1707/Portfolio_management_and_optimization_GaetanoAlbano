import pandas as pd
import numpy as np

def compute_quarterly_metrics(returns, dates, save_path="results/quarterly_metrics.csv"):
    """
    Calcola Sharpe Ratio, Max Drawdown, Volatilità e Rendimento medio per ogni trimestre.
    """
    returns = pd.Series(returns, index=pd.to_datetime(dates))
    returns = returns.asfreq('B').fillna(0)  # Frequenza giornaliera
    quarters = returns.resample("Q")

    metrics = {
        "Quarter": [],
        "Mean Return": [],
        "Volatility": [],
        "Sharpe Ratio": [],
        "Max Drawdown": []
    }

    for period, data in quarters:
        q_returns = data
        mean_return = q_returns.mean()
        volatility = q_returns.std()
        sharpe_ratio = mean_return / (volatility + 1e-6)
        cumulative = (q_returns + 1).cumprod()
        max_drawdown = (cumulative.cummax() - cumulative).max()

        metrics["Quarter"].append(str(period.date()))
        metrics["Mean Return"].append(mean_return)
        metrics["Volatility"].append(volatility)
        metrics["Sharpe Ratio"].append(sharpe_ratio)
        metrics["Max Drawdown"].append(max_drawdown)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_path, index=False)
    print(f"✅ Metriche trimestrali salvate in: {save_path}")
    return df_metrics


def compute_monthly_metrics(returns, dates, save_path="results/monthly_metrics.csv"):
    returns = pd.Series(returns, index=pd.to_datetime(dates))
    returns = returns.asfreq('B').fillna(0)
    months = returns.resample("M")

    metrics = {
        "Month": [],
        "Mean Return": [],
        "Volatility": [],
        "Sharpe Ratio": [],
        "Max Drawdown": []
    }

    for period, data in months:
        m_returns = data
        mean_return = m_returns.mean()
        volatility = m_returns.std()
        sharpe_ratio = mean_return / (volatility + 1e-6)
        cumulative = (m_returns + 1).cumprod()
        max_drawdown = (cumulative.cummax() - cumulative).max()

        metrics["Month"].append(str(period.date()))
        metrics["Mean Return"].append(mean_return)
        metrics["Volatility"].append(volatility)
        metrics["Sharpe Ratio"].append(sharpe_ratio)
        metrics["Max Drawdown"].append(max_drawdown)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_path, index=False)
    print(f"✅ Metriche mensili salvate in: {save_path}")
    return df_metrics