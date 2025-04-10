import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_strategies(rl_path, opt_path, passive_equal, passive_risk, save_path):
    # === Funzione helper
    def load_series(path, name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File '{path}' ({name}) non trovato.")
        df = pd.read_csv(path)
        if 'date' not in df.columns or 'value' not in df.columns:
            raise ValueError(f"❌ Il file '{path}' ({name}) deve contenere colonne 'date' e 'value'.")
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['date', 'value']).set_index('date')
        df = df[~df.index.duplicated(keep='first')]
        return df.rename(columns={'value': name})

    # === Caricamento dati
    rl = load_series(rl_path, "RL Policy")
    opt = load_series(opt_path, "Optimized")
    ew = load_series(passive_equal, "Equal Weight")
    rp = load_series(passive_risk, "Risk Parity")

    # === Merge e filtro
    df = rl.join([opt, ew, rp], how='inner').dropna()

    if df.empty:
        raise ValueError("❌ Nessuna data comune trovata tra le strategie per il confronto.")

    # === Salvataggio
    df.to_csv(save_path.replace(".png", ".csv"))

    # === Grafico
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title("Confronto Strategie: RL vs. Ottimizzato vs. Passive")
    plt.xlabel("Data")
    plt.ylabel("Valore Portafoglio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Grafico confronto strategie salvato in: {save_path}")
