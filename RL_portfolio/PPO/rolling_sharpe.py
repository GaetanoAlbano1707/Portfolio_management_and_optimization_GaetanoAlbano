import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_rolling_sharpe(performance_file="test_performance.csv", window=63, output_csv="rolling_sharpe.csv",
                           output_plot="rolling_sharpe.png"):
    df = pd.read_csv(performance_file)

    if "Reward" not in df.columns:
        raise ValueError("Il file deve contenere una colonna chiamata 'Reward'.")

    returns = df["Reward"].values
    dates = pd.date_range(start="2020-01-01", periods=len(returns), freq="B")

    returns_series = pd.Series(returns, index=dates)
    rolling_mean = returns_series.rolling(window).mean()
    rolling_std = returns_series.rolling(window).std() + 1e-6
    rolling_sharpe = rolling_mean / rolling_std

    # Salva i risultati
    rolling_sharpe.to_csv(output_csv, header=["Rolling Sharpe"])

    # Plot
    plt.figure(figsize=(10, 6))
    rolling_sharpe.plot(title="Rolling Sharpe Ratio (63-day window)", grid=True)
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
