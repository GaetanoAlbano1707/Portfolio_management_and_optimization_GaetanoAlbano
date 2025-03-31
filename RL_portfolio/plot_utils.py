import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_portfolio_comparison(portfolio_values, benchmark_values, output_path="comparison.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_values, label="RL Portfolio")
    plt.plot(benchmark_values, label="Buy & Hold")
    plt.title("Portfolio Value Comparison")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def compute_buy_and_hold(df, tickers, valuation_col="close", initial_amount=100000):
    last_prices = df[df["tic"].isin(tickers)].pivot(index="date", columns="tic", values=valuation_col)
    initial_prices = last_prices.iloc[0]
    weights = np.array([1.0 / len(tickers)] * len(tickers))
    holdings = (initial_amount * weights) / initial_prices

    portfolio_values = (last_prices * holdings).sum(axis=1)
    return portfolio_values
