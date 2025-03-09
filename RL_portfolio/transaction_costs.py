# transaction_cost.py
import numpy as np


def quadratic_transaction_cost(w_new, w_old, c_minus, c_plus, delta_minus, delta_plus):
    """
    Calcola i costi di transazione quadratici per ogni asset.

    Parametri:
      - w_new: array dei nuovi pesi (allocazione proposta)
      - w_old: array dei pesi correnti (allocazione attuale)
      - c_minus: costo unitario per vendite (bid)
      - c_plus: costo unitario per acquisti (ask)
      - delta_minus: coefficiente quadratico per vendite
      - delta_plus: coefficiente quadratico per acquisti

    Ritorna:
      - total_cost: costo totale (somma dei costi per asset)
      - cost_per_asset: array dei costi per ogni asset
    """
    w_new = np.array(w_new)
    w_old = np.array(w_old)
    diff = w_new - w_old
    cost_per_asset = np.where(diff < 0, (-diff * c_minus) + (diff ** 2) * delta_minus, np.where(diff > 0, (diff * c_plus) + (diff ** 2) * delta_plus, 0.0))
    total_cost = np.sum(cost_per_asset)
    return total_cost, cost_per_asset
