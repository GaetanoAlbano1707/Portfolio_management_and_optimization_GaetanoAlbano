import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # <-- assicurati di avere seaborn installato (pip install seaborn)

def plot_portfolio_weights(weights_series, tickers):
    df_weights = pd.DataFrame(weights_series, columns=tickers)
    df_weights.index.name = 'Step'
    df_weights.plot.area(figsize=(12, 6), colormap='tab20')
    plt.title("Evoluzione dei pesi del portafoglio")
    plt.ylabel("Peso (%)")
    plt.xlabel("Step temporale")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.grid(True)

def plot_cumulative_returns(rewards_list, label='Strategia', color='green'):
    cumulative = np.cumsum(rewards_list)
    plt.plot(cumulative, label=label, color=color)
    plt.title('Rendimento Cumulato')
    plt.xlabel('Step')
    plt.ylabel('Log-Return cumulato')
    plt.grid(True)
    plt.legend()


def plot_grouped_bar_weights(filepath, output_path="grouped_bar_weights_by_quarter.png"):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period("Q").astype(str)

    # Prendi solo le colonne dei ticker
    ticker_columns = [col for col in df.columns if col not in ['step', 'date', 'quarter']]
    df_grouped = df.groupby('quarter')[ticker_columns].mean() * 100  # percentuali

    # Reorganizza i dati per plotting
    quarters = df_grouped.index.tolist()
    tickers = df_grouped.columns.tolist()
    n_quarters = len(quarters)
    x = np.arange(len(tickers))
    width = 0.2  # larghezza delle barre

    # Setup figura
    plt.figure(figsize=(16, 6))

    for i, quarter in enumerate(quarters):
        plt.bar(x + i * width, df_grouped.loc[quarter], width=width, label=quarter)

    plt.xticks(x + width * (n_quarters - 1) / 2, tickers, rotation=45)
    plt.ylabel("Peso medio (%)")
    plt.title("Distribuzione Pesi Medi per Trimestre")
    plt.legend(title="Trimestre")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Grafico barre raggruppate salvato in: {output_path}")
