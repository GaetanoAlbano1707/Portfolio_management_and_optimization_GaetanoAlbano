import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_plot(log_path, title_prefix="Agent"):
    df = pd.read_csv(log_path, parse_dates=["date"])

    # Plot portfolio value evolution
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["portfolio_value"], marker='o', label="Portfolio Value")
    plt.title(f"{title_prefix} - Portfolio Value at Rebalance Steps")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_path.parent / f"{title_prefix.lower().replace(' ', '_')}_rebalance_values.png")
    plt.close()

    # Plot reward evolution
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["reward"], marker='o', label="Reward", color="green")
    plt.title(f"{title_prefix} - Reward at Rebalance Steps")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_path.parent / f"{title_prefix.lower().replace(' ', '_')}_rebalance_rewards.png")
    plt.close()

    # Show allocazioni per asset
    alloc_cols = [col for col in df.columns if col.startswith("alloc_")]
    alloc_df = df[["date"] + alloc_cols].set_index("date")
    alloc_df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    plt.title(f"{title_prefix} - Allocazioni per Asset (ogni 3 mesi circa)")
    plt.ylabel("Proporzione")
    plt.tight_layout()
    plt.savefig(log_path.parent / f"{title_prefix.lower().replace(' ', '_')}_rebalance_allocations.png")
    plt.close()

    print(f"📊 Report creato per {title_prefix} con {len(df)} ribilanciamenti.")
    print(df[["date", "portfolio_value", "reward"]].to_string(index=False))

if __name__ == "__main__":
    # Cambia i path in base all'agente che vuoi analizzare
    base_path = Path("results/test")
    load_and_plot(base_path / "evaluation_log.csv", title_prefix="Policy Agent")
    load_and_plot(base_path / "evaluation_log_random.csv", title_prefix="Random Agent")
