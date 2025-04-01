import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

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


def plot_covariance_evolution(cov_matrices: dict, save_dir="results/test/"):
    """
    Plotta alcune heatmap dell'evoluzione di Σ(t) nel tempo.

    Args:
        cov_matrices: dict con chiave date, valore DataFrame matrice di covarianza
        save_dir: cartella dove salvare le immagini
    """
    os.makedirs(save_dir, exist_ok=True)

    # Seleziona 5 date equidistanti
    dates = list(sorted(cov_matrices.keys()))
    sample_dates = [dates[i] for i in np.linspace(0, len(dates) - 1, 5, dtype=int)]

    for date in sample_dates:
        cov = cov_matrices[date]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cov, annot=False, cmap="coolwarm", square=True, fmt=".2f")
        plt.title(f"Covariance Matrix - {date.strftime('%Y-%m-%d')}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cov_matrix_{date.strftime('%Y%m%d')}.png")
        plt.close()