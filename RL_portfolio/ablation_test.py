import pandas as pd
import matplotlib.pyplot as plt

def load_results(file_paths, labels):
    results = {}
    for path, label in zip(file_paths, labels):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date")["portfolio_value"]
        results[label] = df
    return results

def plot_ablation(results, save_path):
    plt.figure(figsize=(10, 6))
    for label, values in results.items():
        plt.plot(values.index, values, label=label)
    plt.title("Ablation Test: Impatto delle Componenti")
    plt.xlabel("Data")
    plt.ylabel("Valore Portafoglio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    file_paths = [
        "results/test/evaluation_log_policy.csv",
        "results/test/evaluation_log_policy_novol.csv",
        "results/test/evaluation_log_policy_noret.csv"
    ]
    labels = ["Full Model", "No Volatility Forecast", "No Expected Returns"]
    results = load_results(file_paths, labels)
    plot_ablation(results, "results/test/ablation_test.png")
