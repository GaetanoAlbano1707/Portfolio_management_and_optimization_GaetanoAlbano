import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === Percorsi ===
base_path = Path("results/test")
rl_path = base_path / "evaluation_results_policy.csv"    # <- già corretto
random_path = base_path / "evaluation_results_random.csv"  # <- già corretto

# === Controllo esistenza file ===
if not rl_path.exists() or not random_path.exists():
    print("❌ Uno dei file di valutazione non è stato trovato.")
    print(f"Verifica che esistano:\n - {rl_path}\n - {random_path}")
    exit(1)

# === Caricamento metriche già salvate nei CSV ===
rl_df = pd.read_csv(rl_path)
random_df = pd.read_csv(random_path)

# === Confronto metrica: già pronta nei CSV ===
comparison_df = pd.concat([rl_df, random_df], ignore_index=True)
comparison_df.to_csv(base_path / "agents_comparison.csv", index=False)
print("📄 Confronto salvato in agents_comparison.csv")

# === Plot confronto
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

comparison_df.plot(x="agent_type", y="fapv", kind="bar", ax=axes[0], legend=False, title="FAPV")
comparison_df.plot(x="agent_type", y="reward_std", kind="bar", ax=axes[1], legend=False, title="Reward Std Dev")

for ax in axes: ax.set_ylabel("")
plt.tight_layout()
plt.savefig(base_path / "agents_comparison.png")
plt.close()
print("📊 Grafico confronto agenti salvato in agents_comparison.png")

# === Determina vincitore
winner = comparison_df.sort_values("fapv", ascending=False).iloc[0]
print(f"\n🏆 L'agente con miglior FAPV è: {winner['agent_type']} (FAPV = {winner['fapv']:.4f})")
