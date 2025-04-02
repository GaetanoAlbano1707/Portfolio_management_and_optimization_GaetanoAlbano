import os
import subprocess

# === Step 1: Generazione dei dati fittizi ===
print("📊 Generazione dei dati fittizi...")
subprocess.run(["python", "generate_fake_data_for_test.py"], check=True)

# === Step 2: Esecuzione del training e valutazione ===
print("\n🚀 Avvio esperimento principale...")
subprocess.run(["python", "main.py"], check=True)

# === Step 3: Confronto con agente random ===
print("\n🎲 Confronto con agente random...")
subprocess.run(["python", "evaluate_random_agent.py"], check=True)

# === Step 4: Analisi comparativa
print("\n📈 Analisi comparativa dei risultati...")
subprocess.run(["python", "compare_agents.py"], check=True)

print("\n✅ Tutto completato. Controlla la cartella 'results/test/' per i risultati.")
