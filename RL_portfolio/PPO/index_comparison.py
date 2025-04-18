# index_comparison.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib
matplotlib.use('Agg')  # backend non interattivo per salvare immagini

def run_index_comparison(result_dir, test_performance_path, start_date="2024-01-01", end_date="2024-12-20"):
    benchmark_tickers = {
        "S&P 500": "^GSPC",
        "MSCI World": "URTH",
        "EURO STOXX Banks": "EXX1.DE",
        "Fidelity Global Quality": "FGQI.L"
    }

    benchmark_data = {}
    for name, ticker in benchmark_tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if not data.empty:
            data['Daily Return'] = data['Adj Close'].pct_change()
            data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
            benchmark_data[name] = data
        else:
            print(f"⚠️ Dati non disponibili per {name} ({ticker})")

    ppo_df = pd.read_csv(test_performance_path)
    ppo_df['Date'] = pd.date_range(start=start_date, periods=len(ppo_df), freq='B')
    ppo_df.set_index('Date', inplace=True)
    ppo_df['Daily Return'] = ppo_df['Cumulative Return'].pct_change()
    ppo_df['Cumulative Return'] = (1 + ppo_df['Daily Return'].fillna(0)).cumprod()

    def calculate_metrics(returns, cumulative=None):
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = mean_return / (volatility + 1e-6)
        if cumulative is None:
            cumulative = (1 + returns.fillna(0)).cumprod()
        max_dd = (cumulative.cummax() - cumulative).max()
        return mean_return, volatility, sharpe_ratio, max_dd

    metrics = {}
    ppo_returns = ppo_df['Daily Return'].dropna()
    ppo_cum = ppo_df['Cumulative Return']
    metrics['Reinforcement Learning'] = calculate_metrics(ppo_returns, ppo_cum)

    for name, data in benchmark_data.items():
        daily_ret = data['Daily Return'].dropna()
        cum_ret = data['Cumulative Return']
        metrics[name] = calculate_metrics(daily_ret, cum_ret)

    metrics_df = pd.DataFrame(metrics, index=["Mean Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]).T
    metrics_df.to_csv(os.path.join(result_dir, "benchmark_comparison_metrics.csv"))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_df['Cumulative Return'], label="Reinforcement Learning", linewidth=2)
    for name, data in benchmark_data.items():
        plt.plot(data['Cumulative Return'], label=name)
    plt.title("Confronto Rendimento Cumulativo - Benchmark")
    plt.xlabel("Data")
    plt.ylabel("Rendimento Cumulativo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "benchmark_comparison_plot.png"))
    plt.close()
