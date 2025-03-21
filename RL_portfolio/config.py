import datetime

class Config():
    def __init__(self, seed_num=2022,learning_rate=0.0003, c_minus=0.002, c_plus=0.001, delta_minus=0.0005, delta_plus=0.0005, gamma=0.95, current_date=None):

        self.tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
        # Parametri di base
        self.initial_asset = 1000000
        self.trading_days_per_year = 252
        self.seed_num = seed_num
        # Validazioni parametri RL
        if not (0 < gamma <= 1):
            raise ValueError("gamma deve essere tra 0 e 1.")

        if not (0 < c_minus < 1 and 0 < c_plus < 1):
            raise ValueError("I costi di transazione devono essere frazioni tra 0 e 1.")

        # Parametri RL
        self.learning_rate = max(1e-6, learning_rate)  # Evita LR troppo piccolo o negativo
        self.num_epochs = max(1, int(50))  # Assicura che sia almeno 1

        # PARAMETRI COSTI DI TRANSAZIONE (in percentuale)
        # Costi lineari
        self.c_minus = c_minus   # 0.2% per vendite
        self.c_plus  = c_plus   # 0.1% per acquisti
        # Coefficienti quadratici
        self.delta_minus = delta_minus  # coefficiente quadratico per vendite
        self.delta_plus  = delta_plus  # coefficiente quadratico per acquisti
        self.gamma = gamma
        # Parametri per il reward: bilanciamento tra profitto, costi e rischio
        self.lambda_profit = 1.0  # peso del rendimento
        self.lambda_cost   = 1.0  # peso del costo di transazione
        self.lambda_risk   = 0.5  # peso della penalizzazione per elevata volatilità forecast

        #Frequenza per calcolo delle correlazioni (3 mesi circa 60 di trading)
        self.correlation_update_frequency = 60

        if current_date is None:
            self.cur_datetime = datetime.datetime.now().strftime('%d/%m/%Y')
        else:
            self.cur_datetime = current_date

    def print_config(self):
        print("\n===== Configurazione RL e Transazioni =====")
        print(f"Data corrente: {self.cur_datetime}")
        print("\n--- Parametri generali ---")
        print(f"Initial Asset: {self.initial_asset}")
        print(f"Trading Days per Year: {self.trading_days_per_year}")
        print(f"Seed Number: {self.seed_num}")

        print("\n--- Parametri RL ---")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Numero di Epoche: {self.num_epochs}")

        print("\n--- Costi di Transazione ---")
        print(f"Costi lineari -> Vendite: {self.c_minus * 100}%, Acquisti: {self.c_plus * 100}%")
        print(f"Costi quadratici -> Vendite: {self.delta_minus}, Acquisti: {self.delta_plus}")

        print("\n--- Parametri per il Reward ---")
        print(f"λ_profit: {self.lambda_profit}")
        print(f"λ_cost: {self.lambda_cost}")
        print(f"λ_risk: {self.lambda_risk}")
        print(f"gamma = {self.gamma}")

        print("\n--- Tickers Considerati ---")
        print(", ".join(self.tickers))
        print("=" * 40)

if __name__ == '__main__':
    config = Config()
    config.print_config()
