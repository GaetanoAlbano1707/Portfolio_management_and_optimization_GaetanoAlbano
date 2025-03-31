---

## 📦 Moduli e Funzioni del Progetto

Questa sezione descrive in modo dettagliato il ruolo di ciascun file nel progetto.

### 🔁 `main.py`
Gestisce l'intera pipeline:
- Carica configurazione
- Inizializza modello e ambiente
- Addestra il modello con Policy Gradient
- Valuta la performance del portafoglio
- Salva il modello
- (Opzionale) Esegue grid search dei costi
- (Opzionale) Confronta diverse frequenze di ribilanciamento

---

### ⚙️ `config.json`
Contiene tutti i parametri configurabili:
- Episodi, batch size, learning rate
- Tipo di modello (`EIIE`, `GPM`, `EI3`)
- Costi di transazione (lineari e quadratici)
- Flag per attivare grid search e confronto ribilanciamento

---

### 🧠 `models.py`
Contiene l'implementazione delle reti neurali:
- `EIIE`: modello convoluzionale ispirato all'originale paper di JPMorgan
- `GPM`: modello LSTM per apprendere pattern temporali
- `EI3`: estensione con conv + FC più profonda

---

### 🧮 `policy_gradient.py`
Algoritmo di ottimizzazione della policy tramite:
- Policy gradient classico
- Memoria delle azioni
- Normalizzazione dei reward

---

### 🧾 `evaluate_policy.py`
Valuta un modello già addestrato su un dataset di test.
Restituisce:
- Valore finale portafoglio
- FAPV (Final Accumulative Portfolio Value)
- Sharpe Ratio
- Max Drawdown

---

### 🧰 `utils.py`
- Set del seed per riproducibilità
- Salvataggio/caricamento di metriche e config

---

### 📋 `logger.py`
- Salva un log completo dell'esperimento in formato JSON
- Include: configurazione, metriche, parametri

---

### 📉 `plot_utils.py`
- Genera grafici per il portafoglio RL e Buy & Hold
- Visualizza confronto tra strategie

---

### 💰 `portfolio_optimization_env.py`
Ambiente RL Gym per simulare portafogli finanziari:
- Azione = allocazione del capitale tra n asset + cash
- Stato = finestra temporale di caratteristiche (es. close, high, low)
- Reward = log-return penalizzato dai costi di transazione

Supporta:
- Costi lineari e quadratici
- 3 modelli di commissione: `trf`, `trf_approx`, `wvm`
- Ribilanciamenti flessibili

---

### 🔍 `cost_optimization.py`
Esegue **Grid Search** su `c_plus`, `c_minus`, `delta_plus`, `delta_minus`:
- Prova tutte le combinazioni definite in `config.json`
- Usa una metrica (FAPV, Sharpe…) per scegliere la migliore
- Salva i risultati in `results/grid_search_results.json`

---

### 🌡 `grid_search_plot.py`
Visualizza i risultati della Grid Search:
- Crea heatmap da `grid_search_results.json`
- Evidenzia graficamente le combinazioni di costo migliori

---

### 🔄 `rebalance_comparison.py`
Confronta l'effetto di diverse frequenze di ribilanciamento:
- Esegue valutazione RL su vari `rebalancing_period`
- Plotta FAPV per ogni periodo in un grafico

---

## 🧪 Esperimenti Consigliati

1. Avviare il sistema con `optimize_costs = true` per ottenere costi ottimali
2. Valutare con `compare_rebalancing = true` per confrontare 1, 2, 3 mesi
3. Usare `logger` per archiviare esperimenti in modo tracciabile

---
