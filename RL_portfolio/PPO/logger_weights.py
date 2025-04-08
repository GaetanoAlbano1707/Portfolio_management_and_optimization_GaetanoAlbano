import pandas as pd
import os

class PortfolioLogger:
    def __init__(self, save_dir="./PPO/results/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.weights_log = []
        self.portfolio_values = []

    def log_weights(self, date, weights, tickers):
        log = {'date': date}
        for i, t in enumerate(tickers):
            log[t] = weights[i]
        self.weights_log.append(log)

    def log_value(self, date, value):
        self.portfolio_values.append({'date': date, 'value': value})

    def save(self):
        df_weights = pd.DataFrame(self.weights_log)
        df_weights.to_csv(os.path.join(self.save_dir, "weights_log.csv"), index=False)

        df_values = pd.DataFrame(self.portfolio_values)
        df_values.to_csv(os.path.join(self.save_dir, "portfolio_values.csv"), index=False)
