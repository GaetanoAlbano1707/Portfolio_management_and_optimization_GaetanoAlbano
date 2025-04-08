import subprocess
import os

def run_step(description, script_path):
    print(f"\n🛠️ {description}...")
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"✅ Completato: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione di {script_path}: {e}")
        exit(1)

# === Step 1: Addestramento PPO ===
run_step("Avvio addestramento con PPO", "train_ppo.py")

# === Step 2: Valutazione del modello ===
run_step("Valutazione del modello addestrato", "evaluate_ppo.py")

# === Step 3: Confronto con strategie passive ===
run_step("Confronto con strategie passive", "evaluate_and_compare.py")

# === Step 4: Generazione della frontiera efficiente ===
run_step("Generazione della frontiera efficiente", "efficient_frontier.py")

print("\n🎉 Test PPO completato. Controlla la cartella 'PPO/results/' per i risultati.")
