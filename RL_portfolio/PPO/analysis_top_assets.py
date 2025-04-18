import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dominant_weights(actions, tickers, top_k=3):
    """
    Analizza quali asset compaiono più spesso tra i top_k pesi.
    """
    dominant_counter = Counter()

    for action in actions:
        top_indices = np.argsort(action)[-top_k:]
        dominant_counter.update([tickers[i] for i in top_indices])

    # Ordina risultati
    sorted_dominants = dominant_counter.most_common()
    total = sum([v for _, v in sorted_dominants])

    print("Asset Dominanti:")
    for asset, count in sorted_dominants:
        perc = 100 * count / total
        print(f"  ➤ {asset}: {perc:.2f}% delle allocazioni top-{top_k}")

    # Plot
    labels, values = zip(*sorted_dominants)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45)
    plt.title(f"Frequenza Top-{top_k} Allocazioni")
    plt.ylabel("Conteggio")
    plt.tight_layout()
    plt.savefig("dominant_weights.png")