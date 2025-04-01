import os
import subprocess

# === Step 1: Generazione dei dati fittizi ===
print("📊 Generazione dei dati fittizi...")
subprocess.run(["python", "generate_fake_data_for_test.py"], check=True)

# === Step 2: Esecuzione del training e valutazione ===
print("\n🚀 Avvio esperimento principale...")
subprocess.run(["python", "main.py"], check=True)

print("\n✅ Tutto completato. Controlla la cartella 'results/test/' per i risultati.")
