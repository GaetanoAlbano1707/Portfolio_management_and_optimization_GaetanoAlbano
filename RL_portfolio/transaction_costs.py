import pandas as pd
import numpy as np
import cvxpy as cp
import os

# -----------------------
# PARAMETRI DI CONFIGURAZIONE
# -----------------------
from config import Config

config = Config()
gamma = config.gamma

assets = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI']
n_assets = len(assets)

start_date = "2019-12-20"
end_date = "2024-12-20"

c_minus = 0.002 * np.ones(n_assets)
c_plus = 0.001 * np.ones(n_assets)


# -----------------------
# 1. CARICAMENTO DEI DATI DI RITORNO
# -----------------------
def merge_log_returns(tickers, directory="./Tickers_file/", output_file="merged_log_returns.csv"):
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
            print(f"❌ File not found: {file_path}")

    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged file saved as {output_file}")
    else:
        print("❌ No data to merge.")

    return merged_df


directory_returns = "./Tickers_file/"
output_file_returns = "merged_log_returns.csv"
log_returns_df = merge_log_returns(assets, directory_returns, output_file_returns)

if log_returns_df is None:
    raise FileNotFoundError("❌ Errore: Nessun dato di ritorno logaritmico trovato. Controlla i file dei tickers.")

log_returns_df['Date'] = pd.to_datetime(log_returns_df['Date'])
log_returns_df.sort_values("Date", inplace=True)


# -----------------------
# 2. AGGREGAZIONE A BLOCCO DI 10 GIORNI
# -----------------------
def group_blocks(df, block_size=10):
    df = df.copy()

    # Rimuove righe in eccesso per evitare problemi di dimensione
    rows_to_keep = (len(df) // block_size) * block_size
    df = df.iloc[:rows_to_keep]

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
print("✅ Vettore dei ritorni attesi (μ) calcolato:", mu)


# -----------------------
# 4. CARICAMENTO DELLA VOLATILITÀ PREDETTA
# -----------------------
def merge_results_Garch_LSTM(tickers, directory= "./Risultati_GARCH_LSTM_Forecasting/",  output_file="merged_Results_Garch_LSTM.csv"):
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
            print(f"❌ File not found: {file_path}")

    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged file saved as {output_file}")
    else:
        print("❌ No data to merge.")

    return merged_df


directory_vol = "./Risultati_GARCH_LSTM_Forecasting/"
output_file_vol = "merged_Results_Garch_LSTM.csv"
merge_df_volatility = merge_results_Garch_LSTM(assets, directory_vol, output_file_vol)

if merge_df_volatility is None or merge_df_volatility.empty:
    raise FileNotFoundError("❌ Errore: Nessun dato di volatilità trovato. Controlla i file GARCH-LSTM.")

merge_df_volatility['Date'] = pd.to_datetime(merge_df_volatility['Date'])
merge_df_volatility.sort_values("Date", inplace=True)

blocks_df = blocks_df.merge(merge_df_volatility, on="Date", how="left")


# -----------------------
# 5. CALCOLO DELLA MATRICE DI COVARIANZA
# -----------------------
def get_covariance_matrix(row, alpha=0.1):
    vol_vector = np.array([row.get(f"{ticker}_pred_volatility", np.nan) for ticker in assets])

    if np.isnan(vol_vector).any():
        print(f"❌ ATTENZIONE: Dati di volatilità mancanti per la data {row['Date']}.")
        return None

    D = np.diag(vol_vector)
    Sigma = D @ np.eye(n_assets) @ D

    return Sigma


blocks_df["Sigma"] = blocks_df.apply(get_covariance_matrix, axis=1)


# -----------------------
# 6. OTTIMIZZAZIONE CON COSTI DI TRANSAZIONE
# -----------------------
def optimize_with_transaction_costs(mu, Sigma, w_tilde, c_minus, c_plus, gamma=1.0):
    n = len(mu)

    w = cp.Variable(n)
    delta_minus_var = cp.Variable(n)
    delta_plus_var = cp.Variable(n)

    total_transaction_cost = cp.sum(cp.multiply(c_minus, delta_minus_var) +
                                    cp.multiply(c_plus, delta_plus_var))

    constraints = [
        cp.sum(w) == 1,
        total_transaction_cost <= 0.01
    ]

    objective = cp.Minimize(0.5 * cp.quad_form(w, Sigma) - gamma * (w @ mu - total_transaction_cost))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value, delta_minus_var.value, delta_plus_var.value


w_tilde = np.ones(n_assets) / n_assets

results_with_costs = []

for idx, row in blocks_df.iterrows():
    if "Sigma" not in row or row["Sigma"] is None:
        print(f"⚠️ Saltata ottimizzazione per la data {row['Date']} perché Sigma è mancante.")
        continue

    Sigma = row["Sigma"]
    w_opt, delta_minus_opt, delta_plus_opt = optimize_with_transaction_costs(
        mu, Sigma, w_tilde, c_minus, c_plus, gamma)

    results_with_costs.append({
        "Date": row["Date"],
        **{f"{assets[i]}_weight": w_opt[i] for i in range(n_assets)},
        "Expected_Return": mu @ w_opt,
        "Variance": w_opt.T @ Sigma @ w_opt,
        "Transaction_Cost": np.sum(c_minus * delta_minus_opt + c_plus * delta_plus_opt)
    })

    w_tilde = w_opt

opt_results_costs_df = pd.DataFrame(results_with_costs)
opt_results_costs_df.to_csv("optimization_results_with_costs.csv", index=False)
print("✅ Ottimizzazione con costi completata e risultati salvati in 'optimization_results_with_costs.csv'.")
