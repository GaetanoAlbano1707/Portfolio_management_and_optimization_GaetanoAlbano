import pandas as pd
import numpy as np
import cvxpy as cp
import os

# -----------------------
# PARAMETRI DI CONFIGURAZIONE
# -----------------------

gamma = 1.0  # Parametro di avversione al rischio (da calibrare)
assets = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
n_assets = len(assets)

# Date per la stima dei ritorni attesi e per la volatilità predetta
start_date = "2019-12-20"
end_date = "2024-12-20"

# Costi di transazione (esempio: 0.002 per vendere, 0.001 per comprare)
c_minus = 0.002 * np.ones(n_assets)  # costo di vendita per ogni asset
c_plus = 0.001 * np.ones(n_assets)  # costo di acquisto per ogni asset


# -----------------------
# 1. CARICAMENTO DEI DATI DI RITORNO
# (Il codice è già implementato, lo riutilizziamo)
def merge_log_returns(directory: str, tickers: list, output_file: str):
    merged_df = None
    for ticker in tickers:
        file_path = os.path.join(directory, f"{ticker}_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Date', 'log_return'])
            df.rename(columns={'log_return': f'{ticker}_log_return'}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
        else:
            print(f"File not found: {file_path}")
    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged file saved as {output_file}")
    else:
        print("No data to merge.")
    return merged_df


directory_returns = "./Data_Analysis/Tickers_file/"
tickers = assets  # già definito
output_file_returns = "merged_log_returns.csv"
log_returns_df = merge_log_returns(directory_returns, tickers, output_file_returns)

log_returns_df['Date'] = pd.to_datetime(log_returns_df['Date'])
log_returns_df.sort_values("Date", inplace=True)


# -----------------------
# 2. AGGREGAZIONE A BLOCCO DI 10 GIORNI
# -----------------------
def group_blocks(df, block_size=10):
    df = df.copy()
    df['block'] = np.repeat(np.arange(len(df) // block_size), block_size)
    grouped = df.groupby('block').agg({col: 'sum' for col in df.columns if col.endswith('_log_return')})
    grouped['Date'] = df.groupby('block')['Date'].first().values
    return grouped


blocks_df = group_blocks(log_returns_df, block_size=10)

# -----------------------
# 3. CALCOLO DEL VETTORE DEI RITORNI ATTESI (μ)
# -----------------------
mask_mu = (blocks_df["Date"] >= pd.to_datetime(start_date)) & (blocks_df["Date"] <= pd.to_datetime(end_date))
blocks_recent = blocks_df[mask_mu]
mu = blocks_recent[[f"{ticker}_log_return" for ticker in assets]].mean().values
print("Vettore dei ritorni attesi (μ):", mu)


# -----------------------
# 4. CARICAMENTO DEI DATI DI VOLATILITÀ PREDETTA
# -----------------------
def merge_results_Garch_LSTM(directory: str, tickers: list, output_file: str):
    merged_df = None
    for ticker in tickers:
        file_path = os.path.join(directory, f"risultati_forecasting_{ticker}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Date', 'Volatilità Predetta (exp)'])
            df.rename(columns={'Volatilità Predetta (exp)': f'{ticker}_pred_volatility'}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
        else:
            print(f"File not found: {file_path}")
    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged file saved as {output_file}")
    else:
        print("No data to merge.")
    return merged_df


directory_vol = "./Risultati_GARCH_LSTM_Forecasting/"
output_file_vol = "merged_Results_Garch_LSTM.csv"
merge_df_volatility = merge_results_Garch_LSTM(directory_vol, tickers, output_file_vol)
merge_df_volatility['Date'] = pd.to_datetime(merge_df_volatility['Date'])
merge_df_volatility.sort_values("Date", inplace=True)

vol_columns = [f"{ticker}_pred_volatility" for ticker in assets]
blocks_df = blocks_df.merge(merge_df_volatility[["Date"] + vol_columns], on="Date", how="left")

# -----------------------
# 5. CALCOLO DELLA MATRICE DI CORRELAZIONE (ogni 3 mesi)
# -----------------------
log_returns_df['quarter'] = log_returns_df['Date'].dt.to_period("Q")
quarter_corr = {}
for quarter, group in log_returns_df.groupby("quarter"):
    corr_matrix = group[[f"{ticker}_log_return" for ticker in assets]].corr().values
    quarter_corr[str(quarter)] = corr_matrix


# -----------------------
# 6. CALCOLO DELLA MATRICE DI COVARIANZA (per ciascun blocco di 10 giorni)
# -----------------------
def get_covariance_matrix(row):
    vol_vector = np.array([row[f"{ticker}_pred_volatility"] for ticker in assets])
    D = np.diag(vol_vector)
    quarter = pd.to_datetime(row["Date"]).to_period("Q")
    corr = quarter_corr.get(str(quarter))
    if corr is None:
        corr = np.eye(n_assets)
    Sigma = D @ corr @ D
    return Sigma


blocks_df["Sigma"] = blocks_df.apply(get_covariance_matrix, axis=1)


# -----------------------
# 7. OTTIMIZZAZIONE CON COSTI DI TRANSAZIONE
# -----------------------
# Definiamo una funzione per risolvere l'ottimizzazione con costi di transazione.
def optimize_with_transaction_costs(mu, Sigma, w_tilde, c_minus, c_plus, gamma=1.0):
    n = len(mu)
    # Variabili: nuovo portafoglio, quantità da vendere (delta_minus) e da comprare (delta_plus)
    w = cp.Variable(n)
    delta_minus = cp.Variable(n)
    delta_plus = cp.Variable(n)

    # Vincolo di relazione
    constraints = [
        cp.sum(w) <= 1,  # Evita sovrainvestimento
        cp.sum(delta_minus) <= cp.sum(delta_plus),  # Evita squilibri
        cp.sum(cp.multiply(c_minus, delta_minus) + cp.multiply(c_plus, delta_plus)) <= 0.01  # Limita i costi
    ]


    # Costo totale di transazione
    transaction_cost = cp.sum(cp.multiply(c_minus, delta_minus) + cp.multiply(c_plus, delta_plus))

    # Funzione obiettivo: minimizzare 1/2 * w^T Sigma w - gamma*(w^T mu - transaction_cost)
    objective = cp.Minimize(0.5 * cp.quad_form(w, Sigma) - gamma * (w @ mu - transaction_cost))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value, delta_minus.value, delta_plus.value


# Inizialmente, il portafoglio corrente (w_tilde) è uguale per tutti gli asset
w_tilde = np.ones(n_assets) / n_assets

results_with_costs = []

# Per ogni blocco di 10 giorni, esegui l'ottimizzazione con costi di transazione
for idx, row in blocks_df.iterrows():
    Sigma = row["Sigma"]
    # Ottimizza il portafoglio considerando i costi di transazione
    w_opt, delta_minus_opt, delta_plus_opt = optimize_with_transaction_costs(mu, Sigma, w_tilde, c_minus, c_plus, gamma)

    exp_ret = mu @ w_opt
    var_port = w_opt.T @ Sigma @ w_opt

    results_with_costs.append({
        "Date": row["Date"],
        **{f"{assets[i]}_weight": w_opt[i] for i in range(n_assets)},
        "Expected_Return": exp_ret,
        "Variance": var_port,
        "Transaction_Cost": np.sum(c_minus * delta_minus_opt + c_plus * delta_plus_opt)
    })

    # Aggiorna il portafoglio corrente per il blocco successivo
    w_tilde = w_opt

opt_results_costs_df = pd.DataFrame(results_with_costs)
opt_results_costs_df.to_csv("optimization_results_with_costs.csv", index=False)
print("Ottimizzazione con costi completata e risultati salvati in 'optimization_results_with_costs.csv'.")
