import json
import os
from datetime import datetime
import numpy as np  # 🔁 Assicurati che sia importato

class ExperimentLogger:
    def __init__(self, log_dir="results/experiments"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.data = {}

    def log_config(self, config: dict):
        self.data["config"] = config

    def log_metrics(self, metrics: dict):
        self.data["metrics"] = metrics

    def log_results(self, label: str, values: dict):
        self.data[label] = values

    def save(self):
        def convert(obj):
            if isinstance(obj, np.generic):  # converte numpy float, int, ecc.
                return obj.item()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=4, default=convert)
        print(f"📁 Log salvato in: {self.log_file}")
