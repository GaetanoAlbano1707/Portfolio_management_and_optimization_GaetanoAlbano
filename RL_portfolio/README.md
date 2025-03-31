# 📈 Reinforcement Learning per Ottimizzazione di Portafoglio con Costi di Transazione

Questo progetto implementa una pipeline completa di **portfolio optimization** tramite **reinforcement learning**. È progettato per lavorare con dati di **azioni ed ETF**, incorporando costi di transazione realistici (lineari e quadratici) e strategie di ribilanciamento.

---

## 🚀 Funzionalità

- RL Environment basato su `Gymnasium`
- Modelli: `EIIE`, `GPM`, `EI3`
- Addestramento con Policy Gradient
- Valutazione con metriche (FAPV, Sharpe, MDD)
- **Costi di transazione** con funzione quadratica \( C_i(w|\tilde{w}) \)
- **Ottimizzazione automatica dei costi** (Grid Search)
- **Confronto tra ribilanciamenti** (mensile, trimestrale, ecc.)
- Salvataggio/caricamento modelli `.pt`
- Logging strutturato esperimenti

---

## 📁 Struttura del Progetto

```bash
.
├── main.py
├── config.json
├── models.py
├── policy_gradient.py
├── evaluate_policy.py
├── data_loader.py
├── portfolio_optimization_env.py
├── cost_optimization.py
├── rebalance_comparison.py
├── plot_utils.py
├── logger.py
├── utils.py
└── results/
