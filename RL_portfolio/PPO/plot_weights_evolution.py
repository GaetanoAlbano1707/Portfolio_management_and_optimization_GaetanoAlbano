import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_weights_evolution(weights_path, save_path):
    df = pd.read_csv(weights_path, parse_dates=['date'])
    df.set_index('date', inplace=True)

    plt.figure(figsize=(14, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title("Evoluzione dei pesi del portafoglio nel tempo")
    plt.xlabel("Data")
    plt.ylabel("Peso")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    weights_log = "./PPO/results/weights_log.csv"
    output_img = "./PPO/results/weights_evolution.png"
    plot_weights_evolution(weights_log, output_img)
    print(f"✅ Grafico evoluzione pesi salvato in: {output_img}")
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_weights_evolution(weights_path, save_path):
    df = pd.read_csv(weights_path, parse_dates=['date'])
    df.set_index('date', inplace=True)

    plt.figure(figsize=(14, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title("Evoluzione dei pesi del portafoglio nel tempo")
    plt.xlabel("Data")
    plt.ylabel("Peso")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    weights_log = "./PPO/results/weights_log.csv"
    output_img = "./PPO/results/weights_evolution.png"
    plot_weights_evolution(weights_log, output_img)
    print(f"✅ Grafico evoluzione pesi salvato in: {output_img}")
