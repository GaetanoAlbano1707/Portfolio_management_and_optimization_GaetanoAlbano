import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_strategies(rl_path, opt_path, passive_equal, passive_risk, save_path):
    rl = pd.read_csv(rl_path, parse_dates=['date'], index_col='date')
    opt = pd.read_csv(opt_path, parse_dates=['date'], index_col='date')
    ew = pd.read_csv(passive_equal, parse_dates=['date'], index_col='date')
    rp = pd.read_csv(passive_risk, parse_dates=['date'], index_col='date')

    df = pd.DataFrame({
        "RL Policy": rl["value"],
        "Optimized": opt["value"],
        "Equal Weight": ew["value"],
        "Risk Parity": rp["value"]
    }).dropna()

    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df[col], label=col)

    plt.title("Confronto Strategie: RL vs. Ottimizzato vs. Passive")
    plt.xlabel("Data")
    plt.ylabel("Valore Portafoglio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
