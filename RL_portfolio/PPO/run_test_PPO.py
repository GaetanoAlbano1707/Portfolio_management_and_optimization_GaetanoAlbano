import subprocess

def run_step(description, script_path):
    print(f"\n🛠️ {description}...")
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"✅ Completato: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore in {script_path}: {e}")
        exit(1)

# === Esecuzione step-by-step
run_step("Addestramento modello PPO", "train_ppo.py")
run_step("Valutazione del modello", "evaluate_ppo.py")
run_step("Confronto con strategie passive", "evaluate_and_compare.py")
run_step("Generazione frontiera efficiente", "efficient_frontier.py")

print("\n🎯 Test completato. Risultati in 'PPO/results/'")
