import os
import subprocess

def run_step(description, script_name):
    print(f"\n🛠️ {description}...")
    try:
        subprocess.run(["python", script_name], check=True)
        print(f"✅ Completato: {script_name}")
    except subprocess.CalledProcessError:
        print(f"❌ Errore durante l'esecuzione di {script_name}")
        exit(1)

# === Step 1: Esecuzione del training e valutazione policy ===
run_step("Avvio esperimento principale (training + valutazione policy)", "main.py")

# === Step 2: Analisi trimestrale del log della policy ===
run_step("Analisi dettagliata trimestrale del log della policy", "analyze_evaluation_log.py")

# === Step 3: Confronto con strategie passive ===
run_step("Confronto con strategie passive", "passive_strategies_comparison.py")

# === Step 4: Test di robustezza con volatilità variabile ===
run_step("Test di robustezza", "robustness_tests.py")

# === Step 5: Ablation test per confrontare configurazioni semplificate ===
run_step("Ablation test", "ablation_test.py")

# === Step 6: Generazione della Frontiera Efficiente ===
print("\n📈 Generazione della Frontiera Efficiente...")
subprocess.run(["python", "efficient_frontier.py"], check=True)
print("✅ Frontiera Efficiente generata e salvata.")

print("\n🎉 Tutto completato. Controlla la cartella 'results/test/' per tutti i risultati!")
