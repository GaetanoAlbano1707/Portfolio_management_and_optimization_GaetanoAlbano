import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# === Percorsi ===
base_path = "results/test"
file_pg = os.path.join(base_path, "evaluation_results_policy.csv")
file_random = os.path.join(base_path, "evaluation_results_random.csv")

# === Caricamento metriche ===
def load_metrics(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        fapv = df["fapv"].iloc[-1] if "fapv" in df.columns else df["FAPV"].iloc[-1]
        std_rewards = df["reward_std"].iloc[-1] if "reward_std" in df.columns else None
        return {"label": label, "fapv": fapv, "std_reward": std_rewards}
    else:
        raise FileNotFoundError(f"⚠️ File non trovato: {path}")

pg_metrics = load_metrics(file_pg, "Policy Gradient")
random_metrics = load_metrics(file_random, "Random Agent")

# === Confronto FAPV ===
metrics_df = pd.DataFrame([pg_metrics, random_metrics])

plt.figure(figsize=(8, 5))
sns.barplot(x="label", y="fapv", data=metrics_df, palette="pastel")
plt.title("📊 FAPV Comparison: Policy Gradient vs Random Agent")
plt.ylabel("FAPV (Final Accumulated Portfolio Value)")
plt.xlabel("Agent")
for i, row in metrics_df.iterrows():
    plt.text(i, row["fapv"] + 0.005, f"{row['fapv']:.4f}", ha="center")
plt.tight_layout()

output_path = os.path.join(base_path, "fapv_comparison.png")
plt.savefig(output_path)
plt.close()
print(f"✅ Grafico FAPV salvato in: {output_path}")

# === Print instabilità se disponibile ===
for agent in [pg_metrics, random_metrics]:
    if agent["std_reward"] is not None:
        print(f"🔄 {agent['label']} - Std reward (instabilità): {agent['std_reward']:.6f}")
