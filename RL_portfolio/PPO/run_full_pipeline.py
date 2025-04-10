# run_full_pipeline.py

import subprocess
import sys

STEPS = [
    ("🔁 1. Avvio training RL con PPO...", "train_ppo.py"),
    ("📊 2. Valutazione e confronto strategie...", "evaluate_and_compare.py"),
    ("📈 3. Simulazione ribilanciamenti e confronto...", "main_compare.py"),
    ("📉 4. Visualizzazione frontiera efficiente con ultimo punto...", "efficient_frontier_final_point.py"),
    ("🔍 5. Validazione allocazione finale vs frontiera...", "validate_vs_frontier.py"),
]

def run_step(description, script):
    print(description)
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore nell'esecuzione di `{script}`. Dettagli: {e}")
        sys.exit(1)

def main():
    print("🚀 Inizio esecuzione della pipeline completa...\n")
    for description, script in STEPS:
        run_step(description, script)
    print("\n✅ Flusso completato con successo.")

if __name__ == "__main__":
    main()
