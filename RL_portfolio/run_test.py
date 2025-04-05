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

# === Step 1: Generazione dei dati fittizi ===
run_step("Generazione dei dati fittizi", "generate_fake_data_for_test.py")

# === Step 2: Esecuzione del training e valutazione policy ===
run_step("Avvio esperimento principale (training + valutazione policy)", "main.py")

# === Step 3: Valutazione agente random ===
run_step("Valutazione agente random", "evaluate_random_agent.py")

# === Step 4: Confronto tra agenti ===
run_step("Analisi comparativa tra agenti", "compare_agents.py")

# === Step 5: Analisi trimestrale del log della policy ===
run_step("Analisi dettagliata trimestrale del log della policy", "analyze_evaluation_log.py")

print("\n🎉 Tutto completato. Controlla la cartella 'results/test/' per tutti i risultati!")
