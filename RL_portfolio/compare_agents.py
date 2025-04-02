import pandas as pd
import matplotlib.pyplot as plt
import os

# === Percorsi ===
base_path = "results/test"
rl_path = os.path.join(base_path, "evaluation_results.csv")
random_path = os.path.join(base_path, "evaluation_results_random.csv")

# === Caricamento risultati ===
rl_df = pd.read_csv(rl_path)
random_df = pd.read_csv(random_path)

# === Calcolo metriche ===
def compute_metrics(df, agent_name):
    fapv = df["Portfolio Value"].iloc[-1] / df["Portfolio Value"].iloc[0]
    reward_std = df["Reward"].std()
    return {"agent": agent_name, "fapv": fapv, "reward_std": reward_std}

rl_metrics = compute_metrics(rl_df, "RL Agent")
random_metrics = compute_metrics(random_df, "Random Agent")

# === Confronto DataFrame ===
comparison_df = pd.DataFrame([rl_metrics, random_metrics])
comparison_df.to_csv(os.path.join(base_path, "agents_comparison.csv"), index=False)
print("📄 Confronto salvato in agents_comparison.csv")

# === Plot FAPV e Reward STD
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
comparison_df.plot(x="agent", y="fapv", kind="bar", ax=axes[0], legend=False, title="FAPV")
comparison_df.plot(x="agent", y="reward_std", kind="bar", ax=axes[1], legend=False, title="Reward Std Dev")
for ax in axes: ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "agents_comparison.png"))
plt.close()
print("📊 Grafico confronto agenti salvato in agents_comparison.png")

# === Determina vincitore
winner = comparison_df.sort_values("fapv", ascending=False).iloc[0]
print(f"\n🏆 L'agente con miglior FAPV è: {winner['agent']} (FAPV = {winner['fapv']:.4f})")
