import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_policy_data(path):
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["portfolio_value"]

def add_noise(series, scale):
    return series * (1 + pd.Series(np.random.normal(0, scale, len(series)), index=series.index))

def plot_robustness(base, variations, labels, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(base.index, base, label='Original', linewidth=2)
    for v, l in zip(variations, labels):
        plt.plot(base.index, v, label=l, linestyle='--')
    plt.legend()
    plt.title("Test di Robustezza su Volatilità")
    plt.xlabel("Data")
    plt.ylabel("Valore Portafoglio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    base = load_policy_data("results/test/evaluation_log_policy.csv")
    var1 = add_noise(base, 0.05)
    var2 = add_noise(base, 0.1)
    var3 = add_noise(base, 0.15)
    plot_robustness(base, [var1, var2, var3], ["+5% noise", "+10% noise", "+15% noise"], "results/test/robustness_volatility_test.png")
