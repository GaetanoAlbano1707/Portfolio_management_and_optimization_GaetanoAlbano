import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_grid_search_results(json_path, x_param="c_plus", y_param="delta_plus", metric="metric"):
    """
    Crea una heatmap dai risultati della grid search.

    Args:
        json_path: percorso al file JSON (es. results/grid_search_results.json)
        x_param, y_param: parametri da visualizzare sull'asse X e Y
        metric: metrica da plottare (default: 'metric' = FAPV)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Arrotonda i valori per evitare problemi di pivot
    df[x_param] = df[x_param].round(6)
    df[y_param] = df[y_param].round(6)

    pivot = df.pivot_table(index=y_param, columns=x_param, values=metric)

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title(f"Grid Search - {metric.upper()} ({y_param} vs {x_param})")
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.tight_layout()
    plt.savefig("results/grid_search_heatmap.png")
    plt.close()
    print("✅ Heatmap salvata in: results/grid_search_heatmap.png")
