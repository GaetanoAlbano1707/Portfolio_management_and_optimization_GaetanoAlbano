import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def compute_equal_weight(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    weights = np.ones(len(tickers)) / len(tickers)
    initial_prices = prices.iloc[0]
    shares = (initial_amount * weights) / initial_prices
    portfolio = prices.dot(shares)
    return portfolio

def compute_risk_parity(df, tickers, initial_amount=100000):
    prices = df[df['tic'].isin(tickers)].pivot(index='date', columns='tic', values='close')
    returns = prices.pct_change().dropna()
    vol = returns.std()
    inv_vol_weights = 1 / vol
    weights = inv_vol_weights / inv_vol_weights.sum()
    initial_prices = prices.iloc[0]
    shares = (initial_amount * weights) / initial_prices
    portfolio = prices.dot(shares)
    return portfolio

def plot_strategies(policy_portfolio, equal_weight, risk_parity, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(policy_portfolio.index, policy_portfolio.values, label='RL Policy')
    plt.plot(equal_weight.index, equal_weight.values, label='Equal Weight')
    plt.plot(risk_parity.index, risk_parity.values, label='Risk Parity')
    plt.title("Confronto Strategie di Allocazione")
    plt.xlabel("Data")
    plt.ylabel("Valore Portafoglio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    df = load_data("./TEST/main_data_real.csv")
    policy = pd.read_csv("results/test/evaluation_log_policy.csv", parse_dates=["date"]).set_index("date")["portfolio_value"]
    tickers = df["tic"].unique().tolist()

    equal_weight = compute_equal_weight(df, tickers)
    risk_parity = compute_risk_parity(df, tickers)
    plot_strategies(policy, equal_weight, risk_parity, "results/test/passive_strategy_comparison.png")
