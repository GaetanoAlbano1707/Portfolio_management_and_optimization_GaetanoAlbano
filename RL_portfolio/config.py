# config.py
import numpy as np
import os
import pandas as pd
import datetime

class Config():
    def __init__(self, seed_num=2022, c_minus=0.002, c_plus=0.001, delta_minus=0.005, delta_plus=0.005, current_date=None):

        self.tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
        # Parametri di base
        self.initial_asset = 1000000
        self.tradeDays_per_year = 252
        self.seed_num = seed_num

        # Parametri RL
        self.learning_rate = 0.0001
        self.num_epochs = 50

        # PARAMETRI COSTI DI TRANSAZIONE (in percentuale)
        # Costi lineari
        self.c_minus = c_minus   # 0.2% per vendite
        self.c_plus  = c_plus   # 0.1% per acquisti
        # Coefficienti quadratici
        self.delta_minus = delta_minus  # coefficiente quadratico per vendite
        self.delta_plus  = delta_plus  # coefficiente quadratico per acquisti

        # Parametri per il reward: bilanciamento tra profitto, costi e rischio
        self.lambda_profit = 1.0  # peso del rendimento
        self.lambda_cost   = 1.0  # peso del costo di transazione
        self.lambda_risk   = 0.5  # peso della penalizzazione per elevata volatilità forecast

        # Altri parametri utili (ad es. per le feature extra derivate da GARCH/LSTM)
        # Questi potrebbero essere utilizzati per normalizzare il capitale o per scalare gli input.
        self.vol_forecast_scale = 1.0
        self.pred_return_scale  = 1.0

        if current_date is None:
            self.cur_datetime = datetime.datetime.now().strftime('%d/%m/%Y')
        else:
            self.cur_datetime = current_date

    def print_config(self):
        print("Configurazione RL e transazioni:")
        print("Initial Asset:", self.initial_asset)
        print("c_minus:", self.c_minus, "c_plus:", self.c_plus)
        print("delta_minus:", self.delta_minus, "delta_plus:", self.delta_plus)
        print("λ_profit:", self.lambda_profit, "λ_cost:", self.lambda_cost, "λ_risk:", self.lambda_risk)
        # ... stampare altri parametri se necessario

if __name__ == '__main__':
    config = Config()
    config.print_config()
